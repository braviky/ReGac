import torch

from ._base import _mask_select_data, register_transform
from ..protein import constants


def _mask_to_segments(mask):
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return []

    segments = []
    start = end = idx[0].item()
    for value in idx[1:].tolist():
        if value == end + 1:
            end = value
        else:
            segments.append((start, end))
            start = end = value
    segments.append((start, end))
    return segments


def _merge_segments(segments, max_gap):
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end - 1 <= max_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _expand_segment(start, end, target_len, upper):
    cur_len = end - start + 1
    if cur_len >= target_len:
        return start, end

    extra = target_len - cur_len
    left_extra = extra // 2
    right_extra = extra - left_extra
    start = max(0, start - left_extra)
    end = min(upper, end + right_extra)

    while (end - start + 1) < target_len:
        if start > 0:
            start -= 1
        elif end < upper:
            end += 1
        else:
            break
    return start, end


def _refine_soft_mask(seed_local, ag_chain_nb, min_segment_len=5, merge_gap=1):
    if not seed_local.any():
        return seed_local

    refined = torch.zeros_like(seed_local, dtype=torch.bool)
    if ag_chain_nb is None:
        segments = _merge_segments(_mask_to_segments(seed_local), merge_gap)
        upper = seed_local.numel() - 1
        for start, end in segments:
            start, end = _expand_segment(start, end, min_segment_len, upper)
            refined[start:end + 1] = True
        return refined

    for chain in ag_chain_nb.unique(sorted=True).tolist():
        chain_positions = torch.where(ag_chain_nb == chain)[0]
        local_seed = seed_local[chain_positions]
        segments = _merge_segments(_mask_to_segments(local_seed), merge_gap)
        upper = chain_positions.numel() - 1
        for start, end in segments:
            start, end = _expand_segment(start, end, min_segment_len, upper)
            refined[chain_positions[start:end + 1]] = True
    return refined


def _build_min_soft_fragment(best_idx, ag_chain_nb, min_segment_len=5):
    fragment = torch.zeros_like(ag_chain_nb, dtype=torch.bool)
    chain = ag_chain_nb[best_idx]
    chain_positions = torch.where(ag_chain_nb == chain)[0]
    center = torch.where(chain_positions == best_idx)[0].item()

    left = center
    right = center
    while (right - left + 1) < min(min_segment_len, chain_positions.numel()):
        can_left = left > 0
        can_right = right < chain_positions.numel() - 1
        if can_left:
            left -= 1
        if (right - left + 1) >= min_segment_len:
            break
        if can_right:
            right += 1
        if not can_left and not can_right:
            break
    fragment[chain_positions[left:right + 1]] = True
    return fragment


def _build_antigen_adjacency(pos_bb, mask_bb, ag_idx, chain_nb=None, spatial_cutoff=4.0):
    if ag_idx.numel() == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=pos_bb.device)

    ag_bb_pos = pos_bb[ag_idx].reshape(-1, 3)
    ag_bb_mask = mask_bb[ag_idx].reshape(-1)
    n_ag = ag_idx.numel()

    ag_ag_dist = torch.cdist(ag_bb_pos, ag_bb_pos)
    valid_pair = ag_bb_mask[:, None] & ag_bb_mask[None, :]
    ag_ag_dist = torch.where(valid_pair, ag_ag_dist, torch.full_like(ag_ag_dist, 1e6))
    spatial_adj = (
        ag_ag_dist.view(n_ag, 4, n_ag, 4).min(dim=1)[0].min(dim=2)[0] < spatial_cutoff
    )

    seq_adj = torch.abs(torch.arange(n_ag, device=pos_bb.device).unsqueeze(0) -
                        torch.arange(n_ag, device=pos_bb.device).unsqueeze(1)) == 1
    if chain_nb is not None:
        ag_chain_nb = chain_nb[ag_idx]
        seq_adj = seq_adj & (ag_chain_nb.unsqueeze(0) == ag_chain_nb.unsqueeze(1))

    adj = torch.clamp(spatial_adj.float() + seq_adj.float(), 0, 1)
    return adj


def get_inward_expansion_epitope_mask(antigen_mask, seed_mask, anchor_flag, pos_heavyatom, mask_heavyatom,
                                      chain_nb=None, seed_cutoff=5.0, soft_steps=1, rigid_steps=2,
                                      min_soft_segment_len=5, soft_merge_gap=1):
    device = antigen_mask.device
    L = antigen_mask.shape[0]
    bb_idx = [constants.BBHeavyAtom.N, constants.BBHeavyAtom.CA, constants.BBHeavyAtom.C, constants.BBHeavyAtom.O]

    pos_bb = pos_heavyatom[:, bb_idx, :]  # (L, 4, 3)
    mask_bb = mask_heavyatom[:, bb_idx]

    ag_idx = torch.where(antigen_mask)[0]
    seed_idx = torch.where(seed_mask)[0]
    anchor_idx = torch.where(anchor_flag)[0]

    if ag_idx.numel() == 0 or (seed_idx.numel() == 0 and anchor_idx.numel() == 0):
        return torch.zeros(L, dtype=torch.bool, device=device), torch.zeros(L, dtype=torch.bool, device=device)

    ag_bb_pos = pos_bb[ag_idx].reshape(-1, 3)
    if seed_idx.numel() > 0:
        ref_idx = seed_idx
    else:
        ref_idx = anchor_idx
    ref_bb_pos = pos_bb[ref_idx].reshape(-1, 3)
    ref_bb_mask = mask_bb[ref_idx].reshape(-1)

    dist_matrix = torch.cdist(ref_bb_pos, ag_bb_pos)
    valid_dist_mask = ref_bb_mask[:, None] & mask_bb[ag_idx].reshape(-1)[None, :]
    dist_matrix = torch.where(valid_dist_mask, dist_matrix, torch.full_like(dist_matrix, 1e6))

    dist_res = dist_matrix.min(dim=0)[0].reshape(-1, 4).min(dim=1)[0]
    is_seed_local = (dist_res <= seed_cutoff) & (mask_bb[ag_idx].any(dim=1))
    ag_chain_nb = chain_nb[ag_idx] if chain_nb is not None else None
    is_seed_local = _refine_soft_mask(
        seed_local=is_seed_local,
        ag_chain_nb=ag_chain_nb,
        min_segment_len=min_soft_segment_len,
        merge_gap=soft_merge_gap,
    )
    if not is_seed_local.any() and ag_chain_nb is not None:
        best_idx = dist_res.argmin()
        is_seed_local = _build_min_soft_fragment(
            best_idx=best_idx,
            ag_chain_nb=ag_chain_nb,
            min_segment_len=min_soft_segment_len,
        )

    adj = _build_antigen_adjacency(
        pos_bb=pos_bb,
        mask_bb=mask_bb,
        ag_idx=ag_idx,
        chain_nb=chain_nb,
    )

    x_soft = is_seed_local.float().unsqueeze(1)
    for _ in range(soft_steps):
        x_soft = (torch.mm(adj, x_soft) > 0).float()

    is_active_local = x_soft.squeeze().bool()

    x_rigid = x_soft.clone()
    for _ in range(rigid_steps):
        x_rigid = (torch.mm(adj, x_rigid) > 0).float()

    is_rigid_local = x_rigid.squeeze().bool() & (~is_active_local)

    active_mask = torch.zeros(L, dtype=torch.bool, device=device)
    active_mask[ag_idx[is_active_local]] = True

    rigid_mask = torch.zeros(L, dtype=torch.bool, device=device)
    rigid_mask[ag_idx[is_rigid_local]] = True

    return active_mask, rigid_mask

@register_transform('patch_around_anchor')
class PatchAroundAnchor(object):  # 围绕anchor（关键AA）提取其临近（top k）区域的AA，并对局部区域+anchor去中心化。即以anchor的中心为整个局部区域的中心
    def __init__(self, initial_patch_size=128, antigen_size=128, antigen_patch_mode='soft_rigid',
                 epitope_definition_mode=None, keep_rim_in_soft_only=None, surface_source=None,
                 surface_exposure_radius=None, surface_exposed_ratio=None, surface_min_count=None,
                 core_seed_cutoff=None, min_core_segment_len=None, core_merge_gap=None,
                 rim_knn=None, rim_max_distance=None, rim_seq_window=None):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.antigen_size = antigen_size
        if antigen_patch_mode not in ('soft_rigid', 'soft_only'):
            raise ValueError(f"Unsupported antigen_patch_mode: {antigen_patch_mode}")
        self.antigen_patch_mode = antigen_patch_mode
        self.epitope_definition_mode = epitope_definition_mode
        self.keep_rim_in_soft_only = keep_rim_in_soft_only
        self.surface_source = surface_source
        self.surface_exposure_radius = surface_exposure_radius
        self.surface_exposed_ratio = surface_exposed_ratio
        self.surface_min_count = surface_min_count
        self.core_seed_cutoff = core_seed_cutoff
        self.min_core_segment_len = min_core_segment_len
        self.core_merge_gap = core_merge_gap
        self.rim_knn = rim_knn
        self.rim_max_distance = rim_max_distance
        self.rim_seq_window = rim_seq_window

    def _center(self, data, origin):
        origin = origin.reshape(1, 1, 3)
        data['pos_heavyatom'] -= origin # (L, A, 3)
        data['pos_heavyatom'] = data['pos_heavyatom'] * data['mask_heavyatom'][:, :, None]
        data['origin'] = origin.reshape(3)
        return data

    def __call__(self, data): # data此时已经是合并后的数据了

        pos_heavyatom = data['pos_heavyatom']  # 按照H L antigen的顺序连成一共整体
        antigen_mask = (data['fragment_type'] == constants.Fragment.Antigen)  # eg True的有个555，也可以为0
        antibody_mask = torch.logical_not(antigen_mask).to(torch.bool)  # eg True 的有 128，不会为0


        anchor_flag = data['anchor_flag'].to(torch.bool)

        if antigen_mask.sum() > 0:  # 有抗原区域
           antigen_soft_mask, antigen_rigid_mask = get_inward_expansion_epitope_mask(
               antigen_mask=antigen_mask,
               seed_mask=data['generate_flag'].to(torch.bool),
               anchor_flag=anchor_flag,
               pos_heavyatom=pos_heavyatom,
               mask_heavyatom=data['mask_heavyatom'],
               chain_nb=data.get('chain_nb', None))
        else:
           antigen_soft_mask = torch.zeros_like(antigen_mask, dtype=torch.bool)
           antigen_rigid_mask = torch.zeros_like(antigen_mask, dtype=torch.bool)

        if self.antigen_patch_mode == 'soft_only':
            antigen_rigid_mask_runtime = torch.zeros_like(antigen_rigid_mask, dtype=torch.bool)
            antigen_mask_full = antigen_soft_mask
        else:
            antigen_rigid_mask_runtime = antigen_rigid_mask
            antigen_mask_full = antigen_soft_mask | antigen_rigid_mask_runtime

        data['antigen_patch_mode'] = self.antigen_patch_mode
        data['antigen_mask'] = antigen_mask_full  # 整个抗原区域，全是0 如果没有antigen
        data['antigen_mask_raw'] = antigen_mask_full.clone()
        data['antigen_soft_mask'] = antigen_soft_mask  # 柔软的抗原区域（表位），全是0 如果没有antigen
        data['antigen_soft_mask_raw'] = antigen_soft_mask.clone()
        data['structure_generate_flag'] = torch.logical_or(data['structure_generate_flag'], antigen_soft_mask)
        data['antigen_rigid_mask'] = antigen_rigid_mask_runtime  # 刚性的抗原区域（非表位），全是0 如果没有antigen
        data['antigen_rigid_mask_raw'] = antigen_rigid_mask_runtime.clone()

        if antigen_mask.sum() > 0:
            region_type_updated = data['region_type'].clone().long()
            region_type_updated[antigen_soft_mask] = constants.AG.EPI  # 15
            region_type_updated[antigen_rigid_mask_runtime] = constants.AG.NON_EPI
            data['region_type'] = region_type_updated
        patch_mask = torch.logical_or(
           antibody_mask,
           antigen_mask_full,
        )  # eg 50



        data_patch = _mask_select_data(data, patch_mask)  # 得到复合物的中patch部分的数据 data的所有key都只是patch部分的数据了
        patch_idx = torch.arange(0, patch_mask.shape[0])[
           patch_mask]  # 整个复合物中 生成区域+对应的anchor AAs+离其最近的128(self.initial_patch_size)个AA+离其最近的(self.antigen_size)个抗原AA 的索引
        data_patch['patch_idx'] = patch_idx  # 局部区域的在原复合物中的AA索引

        anchor_points = pos_heavyatom[anchor_flag, constants.BBHeavyAtom.CA]  # anchor点对应的Cα坐标   (n_anchors, 3)

        data_patch = self._center(  # 局部区域去中心化（中心是anchor points的Cα的均值）
            data_patch,
            origin = anchor_points.mean(dim=0)  # anchor的CA坐标的平均，表示生成区域的中心坐标
        )

        patch_ag_mask = (data_patch['fragment_type'] == constants.Fragment.Antigen)
        assert 'antigen_mask' in data_patch, "antigen_mask not found in data_patch"
        patch_ab_mask = torch.logical_not(patch_ag_mask)

        fr_mask = (data_patch['fr_flag'] > 0)
        fix_cdr_mask = data_patch['fix_cdr_flag'].to(torch.bool)
        gen_mask = data_patch['generate_flag'].to(torch.bool)
        fr_fix_overlap = (fr_mask & fix_cdr_mask).sum()
        fr_gen_overlap = (fr_mask & gen_mask).sum()
        fix_gen_overlap = (fix_cdr_mask & gen_mask).sum()
        if fr_fix_overlap > 0 or fr_gen_overlap > 0 or fix_gen_overlap > 0:
            print(f"[V1.2 DEBUG] FR & fix_cdr overlap: {fr_fix_overlap}")
            print(f"[V1.2 DEBUG] FR & generate overlap: {fr_gen_overlap}")
            print(f"[V1.2 DEBUG] fix_cdr & generate overlap: {fix_gen_overlap}")
            print(f"[V1.2 DEBUG] fr_flag values: {data_patch['fr_flag'].unique()}")
            print(f"[V1.2 DEBUG] cdr_flag values: {data_patch['cdr_flag'].unique()}")

        ab_mask_computed = fr_mask | fix_cdr_mask | gen_mask
        if not torch.equal(patch_ab_mask, ab_mask_computed):
            print(f"[V1.2 DEBUG] patch_ab_mask.sum = {patch_ab_mask.sum()}, computed.sum = {ab_mask_computed.sum()}")
            print(f"[V1.2 DEBUG] Difference: {(patch_ab_mask != ab_mask_computed).sum()}")

        assert torch.equal(patch_ab_mask, (data_patch['fr_flag'] > 0) | data_patch['fix_cdr_flag'].to(torch.bool)\
                           | data_patch['generate_flag'].to(torch.bool)), f"ab_mask split into fr and fix_cdr ERROR!!!!"

        s = (data_patch['fr_flag'] > 0).sum() + data_patch['fix_cdr_flag'].to(torch.bool).sum() + \
            data_patch['generate_flag'].to(torch.bool).sum() + \
            data_patch['antigen_mask'].sum()

        if data_patch['aa'].shape[0] != s:
            print(f"[V1.2 DEBUG] aa.shape[0] = {data_patch['aa'].shape[0]}, s = {s}")
            print(f"[V1.2 DEBUG] fr_flag.sum = {(data_patch['fr_flag'] > 0).sum()}")
            print(f"[V1.2 DEBUG] fix_cdr_flag.sum = {data_patch['fix_cdr_flag'].to(torch.bool).sum()}")
            print(f"[V1.2 DEBUG] generate_flag.sum = {data_patch['generate_flag'].to(torch.bool).sum()}")
            print(f"[V1.2 DEBUG] antigen_mask.sum = {data_patch['antigen_mask'].sum()}")
            print(f"[V1.2 DEBUG] fix_cdr_flag values: {data_patch['fix_cdr_flag'].unique()}")
            print(f"[V1.2 DEBUG] generate_flag values: {data_patch['generate_flag'].unique()}")
            overlap = (data_patch['fix_cdr_flag'].to(torch.bool) & data_patch['generate_flag'].to(torch.bool)).sum()
            print(f"[V1.2 DEBUG] overlap between fix_cdr and generate: {overlap}")

        assert data_patch['aa'].shape[0] == s, f"patch split ERROR!!!! aa={data_patch['aa'].shape[0]}, s={s}"

        cdr_bool = data_patch['cdr_flag'].to(torch.bool)
        gen_bool = data_patch['generate_flag'].to(torch.bool)
        fix_bool = data_patch['fix_cdr_flag'].to(torch.bool)

        if not (torch.equal(cdr_bool, gen_bool | fix_bool) or ((gen_bool | fix_bool) >= cdr_bool).all()):
            print(f"[V1.2 DEBUG cdr] cdr_bool.sum = {cdr_bool.sum()}, gen_bool.sum = {gen_bool.sum()}, fix_bool.sum = {fix_bool.sum()}")
            print(f"[V1.2 DEBUG cdr] cdr_flag unique: {data_patch['cdr_flag'].unique()}")
            print(f"[V1.2 DEBUG cdr] generate_flag unique: {data_patch['generate_flag'].unique()}")
            print(f"[V1.2 DEBUG cdr] fix_cdr_flag unique: {data_patch['fix_cdr_flag'].unique()}")
            diff = cdr_bool != (gen_bool | fix_bool)
            print(f"[V1.2 DEBUG cdr] diff positions: {diff.sum()}")
            if diff.sum() > 0:
                print(f"[V1.2 DEBUG cdr] cdr_bool[diff]: {data_patch['cdr_flag'][diff]}")
                print(f"[V1.2 DEBUG cdr] gen_bool[diff]: {data_patch['generate_flag'][diff]}")
                print(f"[V1.2 DEBUG cdr] fix_bool[diff]: {data_patch['fix_cdr_flag'][diff]}")

        assert (gen_bool & fix_bool).sum() == 0, "generate and fix_cdr overlap!"
        assert ((gen_bool | fix_bool) >= cdr_bool).all(), f"cdr split ERROR!!! gen|fix doesn't cover all CDRs"


        return data_patch
