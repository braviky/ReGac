import torch
from diffab_251166.utils.protein import constants


def build_fake_epitope_mask(true_soft_mask, antigen_mask, chain_nb, res_nb, mode="contiguous_fake"):
    true_soft_mask = true_soft_mask.to(torch.bool)
    antigen_mask = antigen_mask.to(torch.bool)
    out = torch.zeros_like(true_soft_mask, dtype=torch.bool)
    batch_size = true_soft_mask.shape[0]
    for b in range(batch_size):
        true_soft = true_soft_mask[b]
        full_antigen = antigen_mask[b]
        true_count = int(true_soft.sum().item())
        antigen_idx = full_antigen.nonzero(as_tuple=True)[0]
        if true_count == 0 or antigen_idx.numel() == 0:
            continue
        if mode != "contiguous_fake":
            raise ValueError(f"unsupported mode: {mode}")
        chain_nb_b = chain_nb[b]
        res_nb_b = res_nb[b]
        best = None
        antigen_chain = chain_nb_b[antigen_idx]
        for chain in antigen_chain.unique():
            chain_mask = antigen_idx[antigen_chain == chain]
            chain_mask = chain_mask[torch.argsort(res_nb_b[chain_mask])]
            if chain_mask.numel() < true_count:
                continue
            for start in range(0, chain_mask.numel() - true_count + 1):
                seg = chain_mask[start:start + true_count]
                span = int((res_nb_b[seg[-1]] - res_nb_b[seg[0]]).item())
                score = (span, int(seg[0].item()))
                if best is None or score < best[0]:
                    best = (score, seg)
        if best is not None:
            out[b, best[1]] = True
        else:
            perm = torch.randperm(antigen_idx.numel(), device=antigen_idx.device)
            chosen = antigen_idx[perm[:true_count]]
            out[b, chosen] = True
    return out


def split_fake_core_rim(fake_soft_mask, pos_heavyatom, true_core_mask=None):
    fake_soft_mask = fake_soft_mask.to(torch.bool)
    out_core = torch.zeros_like(fake_soft_mask, dtype=torch.bool)
    out_rim = torch.zeros_like(fake_soft_mask, dtype=torch.bool)
    batch_size = fake_soft_mask.shape[0]
    for b in range(batch_size):
        idx = fake_soft_mask[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if true_core_mask is not None:
            core_count = int(true_core_mask[b].to(torch.bool).sum().item())
        else:
            core_count = max(1, int(round(0.4 * idx.numel())))
        core_count = max(1, min(idx.numel(), core_count))
        ca = pos_heavyatom[b, idx, constants.BBHeavyAtom.CA]
        center = ca.mean(dim=0, keepdim=True)
        dist = torch.norm(ca - center, dim=-1)
        core_idx = idx[torch.argsort(dist)[:core_count]]
        out_core[b, core_idx] = True
        out_rim[b, idx] = True
        out_rim[b, core_idx] = False
    return out_core, out_rim


def build_alt_region_type(
    region_type,
    antigen_mask,
    fake_soft_mask,
    pos_heavyatom=None,
    true_core_mask=None,
    epi_core_index=int(constants.AG.EPI_CORE),
    epi_rim_index=int(constants.AG.EPI_RIM),
    non_epi_index=int(constants.AG.NON_EPI),
):
    antigen_mask = antigen_mask.to(torch.bool)
    fake_soft_mask = fake_soft_mask.to(torch.bool)
    out = region_type.clone()
    out[antigen_mask] = non_epi_index
    if pos_heavyatom is None:
        out[fake_soft_mask] = epi_core_index
        return out
    fake_core_mask, fake_rim_mask = split_fake_core_rim(
        fake_soft_mask, pos_heavyatom, true_core_mask=true_core_mask
    )
    out[fake_core_mask] = epi_core_index
    out[fake_rim_mask] = epi_rim_index
    return out
