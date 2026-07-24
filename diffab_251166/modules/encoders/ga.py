import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffab_251166.modules.common.geometry import global_to_local, local_to_global, normalize_vector, construct_3d_basis, angstrom_to_nm
from diffab_251166.modules.common.layers import mask_zero, LayerNorm
from diffab_251166.utils.protein.constants import BBHeavyAtom, REGION_NUM
from diffab_251166.modules.common.region_feat_encode import RegionConditioner, Updater
from diffab_251166.modules.common.hier_region_residue_coupler import HierarchicalRegionResidueCoupler

def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)  # (N, L, L, n_head)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)  # (N, L, L, n_head) 两个AA都真实存在对应的head菜都尉true，否则 都为false

    logits = torch.where(mask_pair, logits, logits - inf)  # 真实的AA位置保留原值，涉及padding AA的都设为无穷小
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))  # (N, L, L, num_heads) 真实的AA位置保留原值，涉及padding AA的都为0
    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)

class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=32, query_key_dim=32, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points * num_heads * 3, bias=bias)

        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim) + (num_heads * value_dim) + (
                    num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim))
        self.layer_norm_2 = LayerNorm(node_feat_dim)


    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)  self.proj_query(x) 线性层映射，维度变化->最后一维拆分成(N, L, feat2)->(N, L, head, qk)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) *
                       (1 / np.sqrt(self.query_key_dim))).sum(-1)  # (N, L, L, num_heads)
        return logits_node

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)  # 线性层转化 (N, L, feat1)->(N, L, num_heads)
        return logits_pair

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()

        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points,
                              3)  # (N, L, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points)  # 用噪音数据进行旋转平移，得到全局坐标 Global query coordinates, (N, L, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)

        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points,
                            3)  # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points)  # Global key coordinates, (N, L, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)

        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)  # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points)))
                                        / 2)  # (N, L, L, n_heads)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2)  # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)  # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2)  # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points,
                              3)  # (N, L, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points,
                                                                  3))  # (N, L, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * \
                      value_points.unsqueeze(1)  # (N, *, L, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(dim=2)  # (N, L, n_heads, n_pnts, 3)
        feat_points = global_to_local(R, t, aggr_points)  # (N, L, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1)  # (N, L, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)  # (N, L, n_heads, n_pnts, 3)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, x, z, mask, return_attention=False):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
            return_attention: If True, return (x_updated, alpha) for interpretability.
        Returns:
            x': Updated node-wise features, (N, L, F).
            alpha (optional): Attention weights, (N, L, L, n_heads).
        """

        logits_node = self._node_logits(x)  # 加噪后的序列特征。(N, L, L, n_heads) 仅仅基于加噪后的序列特征进行转化
        logits_pair = self._pair_logits(z)  # 原始pair特征。(N, L, L, feat)->(N, L, L, n_heads) 仅仅经过线性层转化
        logits_spatial = self._spatial_logits(R, t, x)  # 融入了噪音结构特征的AA特征(N, L, L, feat)->(N, L, L, n_heads)  对加噪后的序列特征融入加噪的结构特征
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads) 真实的AA位置保留原值，涉及padding AA的都为0

        feat_p2n = self._pair_aggregation(alpha, z)  # (N, L, F(768)) 将原始pair特征和apha进行融合
        feat_node = self._node_aggregation(alpha, x)  # (N, L, F(384)) 将加噪后的AA特征和apha进行融合
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)  # (N, L, F(672))将结合加噪后的AA特征、局部信息 将其转化为全局数据，再和apha进行融合后，重新转化为回局部信息

        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))  # 线性层，对pair、AA、空间AA数据特征融合(N, L, F(128))
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)  # padding处的特征置为0
        x_updated = self.layer_norm_1(x + feat_all)  # LayerNorm
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))  # 经理线性层后，再layernorm

        if return_attention:
            return x_updated, alpha  # (N, L, F(128)), (N, L, L, n_heads)
        return x_updated  # (N, L, F(128))


class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, region_dim, num_layers, ga_block_opt={}, region_relation_opt=None):
        super(GAEncoder, self).__init__()
        region_relation_opt = region_relation_opt or {}
        self.num_layers = num_layers
        self.region_aux_layer = region_relation_opt.get('aux_loss_layer', 'last')
        self.hierarchical_region_residue_opt = region_relation_opt.get('hierarchical_region_residue', {})
        self.hierarchical_region_residue_enabled = bool(
            self.hierarchical_region_residue_opt.get('enabled', False)
        )
        self.blocks = nn.ModuleList([
            GABlock(node_feat_dim, pair_feat_dim, **ga_block_opt)
            for _ in range(num_layers)
        ])
        self.region_conditioners = nn.ModuleList([
            RegionConditioner(
            D_res=node_feat_dim,
            D_region=region_dim,
            relation_opt=region_relation_opt)
            for _ in range(num_layers)])

        self.updaters = nn.ModuleList([
            Updater(
            D_res=node_feat_dim,
            D_region=region_dim
        )for _ in range(num_layers)])
        self.hierarchical_couplers = nn.ModuleList([
            HierarchicalRegionResidueCoupler(
                node_dim=node_feat_dim,
                region_dim=region_dim,
                num_regions=REGION_NUM,
                opt=self.hierarchical_region_residue_opt,
            )
            for _ in range(num_layers)
        ])

    def _get_region_aux_layer_idx(self):
        layer = self.region_aux_layer
        if layer == 'first':
            return 0
        if layer == 'last':
            return self.num_layers - 1
        if isinstance(layer, int):
            return max(0, min(self.num_layers - 1, layer))
        raise ValueError(f'Unsupported aux_loss_layer: {layer}')

    def get_hierarchical_async_state(self):
        if not self.hierarchical_region_residue_enabled:
            return None
        for coupler in reversed(self.hierarchical_couplers):
            state = getattr(coupler, 'last_state', None)
            if state is not None:
                return state
        return None

    def forward(self, R_t, p_t, res_feat, pair_feat, mask, region_type, structure_generate_flag, beta_embed, b_factor=None, region_aux_inputs=None, p_contact=None):  # 对res_feat更新：整合加噪局部坐标、加噪位置信息、包含加噪序列信息的res_feat、真实条件的pair——feat(无噪音)、真实AA的标记
        kl_loss_dict = {}
        p_contact = p_t if p_contact is None else p_contact

        B_from_res_feat = res_feat.shape[0]
        B_from_region_type = region_type.shape[0]
        if B_from_res_feat != B_from_region_type:
            print(f"[V3 DEBUG] Shape mismatch: res_feat B={B_from_res_feat}, region_type B={B_from_region_type}")
            print(f"  res_feat shape: {res_feat.shape}")
            print(f"  region_type shape: {region_type.shape}")
            if b_factor is not None:
                print(f"  b_factor shape: {b_factor.shape}")

        B, L = region_type.shape
        N_regions = REGION_NUM

        region_type_clamped = region_type.long().clamp(min=0, max=N_regions)  # 转为long并防止越界
        region_onehot = torch.nn.functional.one_hot(region_type_clamped, num_classes=N_regions + 1)  # (B, L, 17)
        region_onehot = region_onehot[:, :, 1:]  # 去掉第0维（padding），得到 (B, L, 16)

        region_valid_mask = (region_onehot.sum(dim=1) > 0).bool()  # (B, 16)

        aux_layer_idx = self._get_region_aux_layer_idx()
        for i, module_set in enumerate(zip(self.blocks, self.region_conditioners, self.updaters, self.hierarchical_couplers)):
            block, region_conditioner, updater, hierarchical_coupler = module_set

            region_guide, R_sample_info, reg_kl_loss, region_aux = region_conditioner(
                res_feat, region_type, structure_generate_flag, valid_mask=mask, t=beta_embed,
                region_valid_mask=region_valid_mask,
                b_factor=b_factor,
                region_aux_inputs=region_aux_inputs if i == aux_layer_idx else None,
            )

            res_feat, _ = updater(structure_generate_flag, res_feat, region_guide, mask)
            if self.hierarchical_region_residue_enabled:
                res_feat, hgacd_aux = hierarchical_coupler(
                    res_feat,
                    region_type,
                    structure_generate_flag,
                    mask,
                    p_t=p_contact,
                    region_aux_inputs=region_aux_inputs if i == aux_layer_idx else None,
                )
                for key, value in hgacd_aux.items():
                    if isinstance(value, torch.Tensor):
                        out_key = key if key.startswith('region_') else f'{key}_l{i}'
                        kl_loss_dict[out_key] = value
            res_feat = block(R_t, p_t, res_feat, pair_feat, mask)
            kl_loss_dict.update({f"ib_kl_l{i}": reg_kl_loss})

            if i == aux_layer_idx and region_aux is not None:
                for key, value in region_aux.items():
                    if isinstance(value, torch.Tensor):
                        kl_loss_dict[key] = value

        return res_feat, kl_loss_dict

