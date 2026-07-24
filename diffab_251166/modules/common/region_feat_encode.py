import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional
from diffab_251166.utils.protein.constants import CDR, FR, AG, BBHeavyAtom
from .epitope_negative_masks import build_fake_epitope_mask, build_alt_region_type

from ...utils.protein.constants import REGION_NUM

DEBUG_MODE = False  # 暂时关闭，只保留B因子相关调试

def _create_antibody_relationship_dict():
    relationships = {}
    for i in range(1, REGION_NUM + 1):
        relationships[i] = []

    relationships[FR.H1].extend([
        CDR.H1
    ])

    relationships[FR.H2].extend([
        CDR.H1, CDR.H2
    ])

    relationships[FR.H3].extend([
        CDR.H2, CDR.H3
    ])

    relationships[FR.H4].extend([
        CDR.H3
    ])

    relationships[FR.L1].extend([
        CDR.L1
    ])

    relationships[FR.L2].extend([
        CDR.L1, CDR.L2
    ])

    relationships[FR.L3].extend([
        CDR.L2, CDR.L3
    ])

    relationships[FR.L4].extend([
        CDR.L3,
    ])

    relationships[CDR.H1].extend([
        FR.H1, FR.H2
    ])

    relationships[CDR.H2].extend([
        FR.H2, FR.H3
    ])

    relationships[CDR.H3].extend([
        FR.H3, FR.H4
    ])

    relationships[CDR.L1].extend([
        FR.L1, FR.L2
    ])

    relationships[CDR.L2].extend([
        FR.L2, FR.L3
    ])

    relationships[CDR.L3].extend([
        FR.L3, FR.L4
    ])

    all_frs = list(FR)
    for fr_i in all_frs:
        relationships[fr_i].extend([c for c in all_frs if c != fr_i])

    all_cdrs = list(CDR)
    for cdr_i in all_cdrs:
        relationships[cdr_i].extend([c for c in all_cdrs if c != cdr_i])
        relationships[cdr_i].extend([AG.EPI_CORE, AG.EPI_RIM])

    relationships[AG.EPI_CORE].extend(all_cdrs)
    relationships[AG.EPI_CORE].extend([AG.EPI_RIM, AG.NON_EPI])
    relationships[AG.EPI_RIM].extend(all_cdrs)
    relationships[AG.EPI_RIM].extend([AG.EPI_CORE, AG.NON_EPI])
    relationships[AG.NON_EPI].extend([AG.EPI_CORE, AG.EPI_RIM])

    final_relationships = {}
    for key, neighbors in relationships.items():
        for neighbor in neighbors:
            if key not in relationships[neighbor]:
                relationships[neighbor].append(key)
        final_relationships[key] = list(set(relationships[key]))

    return final_relationships


RELATION_DICT = _create_antibody_relationship_dict()

class VIBEncoder(nn.Module):
    """
    变分信息瓶颈（VIB）编码器：
    将高维的氨基酸池化特征压缩为区域级的平均值（mu）和方差（log_var）。
    """

    def __init__(self, D_res: int, D_region: int):
        super().__init__()
        self.conv_mu = nn.Linear(D_res, D_region)
        self.conv_log_var = nn.Linear(D_res, D_region)

    def forward(self, f_pooled_region: torch.Tensor, region_valid_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_pooled_region: (B*N, D_res) 区域池化特征
            region_valid_mask: (B, N) 区域有效性掩码，True表示该区域有残基
        """
        mu = self.conv_mu(f_pooled_region)
        log_var = self.conv_log_var(f_pooled_region)

        if region_valid_mask is not None:
            B, N = region_valid_mask.shape
            valid_flat = region_valid_mask.reshape(B * N, 1).float()  # (B*N, 1)，转为float
            mu = mu * valid_flat
            log_var = log_var * valid_flat

        return mu, log_var

class RegionInteractionLayer(nn.Module):
    """
    可学习的、时间依赖的区域交互层（动态 GNN 聚合），硬约束仅用于初始化。

    div7 增强点：
    1. 使用多层 region interaction（由上层控制）
    2. 将硬先验图改为“可学习图 + 可调先验强度”
    3. 使用 edge-type-specific 时间门控，而不只是全局单标量 gamma_t
    """

    def __init__(self, D_region: int, M_matrix_hard: torch.Tensor, relation_opt: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.D_region = D_region
        relation_opt = relation_opt or {}
        self.enable_runtime_debug_cache = bool(relation_opt.get('enable_runtime_debug_cache', False))
        N = M_matrix_hard.shape[0]
        self.register_buffer('M_matrix_hard', M_matrix_hard)
        identity_mask = torch.eye(N)
        self.register_buffer('M_non_self', (1.0 - identity_mask))
        self.register_buffer('self_mask', identity_mask.bool())
        connected = (M_matrix_hard > 0.5) & (1.0 - identity_mask).bool()
        self.register_buffer('connected_mask', connected.float())
        edge_type_ids = self._build_edge_type_ids(M_matrix_hard)
        self.register_buffer('edge_type_ids', edge_type_ids)
        self.num_edge_types = int(edge_type_ids.max().item()) + 1
        self.A_learnable = nn.Parameter(M_matrix_hard * 0.5)
        self.connected_prior_bias = float(relation_opt.get('connected_prior_bias', 0.2))
        self.non_connected_prior_bias = float(relation_opt.get('non_connected_prior_bias', -0.7))
        self.edge_type_logit_scale = nn.Parameter(torch.ones(self.num_edge_types))
        self.edge_type_logit_bias = nn.Parameter(torch.zeros(self.num_edge_types))
        self.graph_time_bias_strength = float(relation_opt.get('graph_time_bias_strength', 1.0))
        init = float(relation_opt.get('graph_time_edge_init', 0.0))
        self.graph_time_edge_weight = nn.Parameter(torch.full((self.num_edge_types,), init))
        self.graph_time_edge_offset = nn.Parameter(torch.zeros(self.num_edge_types))
        self.t_mod_net = nn.Sequential(
            nn.Linear(D_region * 2, D_region),
            nn.SiLU(),
            nn.Linear(D_region, 1)
        )
        self.message_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D_region, D_region),
                nn.SiLU()
            )
            for _ in range(self.num_edge_types)
        ])
        self.gate_net = nn.Sequential(
            nn.Linear(D_region * 2, D_region * 2),
            nn.SiLU()
        )
        self.candidate_net = nn.Sequential(
            nn.Linear(D_region * 2, D_region),
            nn.SiLU()
        )
        self.norm = nn.LayerNorm(D_region)

    def _cache_debug_tensor(self, name: str, tensor: Optional[torch.Tensor]) -> None:
        if not self.enable_runtime_debug_cache:
            return
        setattr(self, name, None if tensor is None else tensor.detach().cpu())

    def _build_edge_type_ids(self, M_matrix_hard: torch.Tensor) -> torch.Tensor:
        def _cat(idx_0based: int) -> str:
            idx = idx_0based + 1
            if 1 <= idx <= 6:
                return 'cdr'
            if 7 <= idx <= 14:
                return 'fr'
            if idx in (15, 16):
                return 'epi'
            if idx == 17:
                return 'non_epi'
            return 'other'
        N = M_matrix_hard.shape[0]
        edge_types = torch.zeros(N, N, dtype=torch.long)
        for i in range(N):
            for j in range(N):
                if i == j:
                    edge_types[i, j] = 0
                    continue
                if M_matrix_hard[i, j] <= 0.5:
                    edge_types[i, j] = 0
                    continue
                ci, cj = _cat(i), _cat(j)
                if {ci, cj} == {'cdr', 'fr'}:
                    edge_types[i, j] = 1
                elif ci == 'cdr' and cj == 'cdr':
                    edge_types[i, j] = 2
                elif ci == 'fr' and cj == 'fr':
                    edge_types[i, j] = 3
                elif {ci, cj} == {'cdr', 'epi'}:
                    edge_types[i, j] = 4
                elif {ci, cj} == {'epi', 'non_epi'}:
                    edge_types[i, j] = 5
                else:
                    edge_types[i, j] = 6
        return edge_types

    def _graph_time_edge_bias(self, graph_time_beta: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if graph_time_beta is None or self.graph_time_bias_strength == 0.0:
            return None
        beta = torch.clamp(graph_time_beta.float(), min=0.0)
        denom = torch.log1p(beta.new_tensor(100.0))
        noise_level = torch.log1p(beta * 100.0) / denom
        noise_level = torch.clamp(noise_level, 0.0, 1.0)
        pair_noise = 0.5 * (noise_level.unsqueeze(2) + noise_level.unsqueeze(1))
        centered_pair_noise = pair_noise - 0.5
        edge_type = self.edge_type_ids.to(beta.device).unsqueeze(0)
        weight = self.graph_time_edge_weight.to(beta.device)[edge_type]
        offset = self.graph_time_edge_offset.to(beta.device)[edge_type]
        bias = centered_pair_noise * weight + offset
        bias = bias.masked_fill(self.self_mask.to(beta.device).unsqueeze(0), 0.0)
        return bias * self.graph_time_bias_strength

    def _get_soft_adj_matrix(self, t_mod: torch.Tensor, region_valid_mask: torch.Tensor = None, graph_time_beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = t_mod.shape[0]
        A_logits = self.A_learnable
        A_logits = 0.5 * (A_logits + A_logits.T)  # symmetrize
        if t_mod.dim() == 2:
            t_mod = t_mod.unsqueeze(1)
        node_gate_raw = self.t_mod_net(t_mod).squeeze(-1)  # (B, N_regions) or (B, 1)
        gamma_nodes = 0.5 + torch.sigmoid(node_gate_raw)  # range [0.5, 1.5]
        if gamma_nodes.size(1) == 1:
            pair_gamma = gamma_nodes.unsqueeze(-1)  # (B, 1, 1)
        else:
            pair_gamma = 0.5 * (gamma_nodes.unsqueeze(2) + gamma_nodes.unsqueeze(1))  # (B, N, N)
        edge_type_scale = self.edge_type_logit_scale[self.edge_type_ids].unsqueeze(0)
        edge_type_bias = self.edge_type_logit_bias[self.edge_type_ids].unsqueeze(0)
        A_logits_t = A_logits.unsqueeze(0) * pair_gamma * edge_type_scale + edge_type_bias
        self._cache_debug_tensor('_last_gamma_t', gamma_nodes)

        prior_bias = torch.where(
            self.connected_mask > 0.5,
            torch.full_like(A_logits, self.connected_prior_bias),
            torch.full_like(A_logits, self.non_connected_prior_bias),
        )
        graph_time_bias = self._graph_time_edge_bias(graph_time_beta)
        if graph_time_bias is not None:
            A_logits_t = A_logits_t + graph_time_bias.to(dtype=A_logits_t.dtype)
            self._cache_debug_tensor('_last_graph_time_bias', graph_time_bias)
        else:
            self._cache_debug_tensor('_last_graph_time_bias', None)

        A_masked = A_logits_t + prior_bias.unsqueeze(0)
        A_masked = A_masked.masked_fill(self.self_mask.unsqueeze(0), float('-inf'))
        if region_valid_mask is not None:
            valid_mask_2d = region_valid_mask.unsqueeze(1) & region_valid_mask.unsqueeze(-1)
            A_masked = A_masked.masked_fill(~valid_mask_2d, float('-inf'))
        A_norm = F.softmax(A_masked, dim=-1)
        all_masked = A_masked.isinf().all(dim=-1, keepdim=True)
        A_norm = A_norm.masked_fill(all_masked, 0.0)
        self._cache_debug_tensor('_last_adj', A_norm)
        return A_norm

    def _build_edge_typed_messages(self, R_norm: torch.Tensor) -> torch.Tensor:
        message_bank = torch.stack(
            [message_net(R_norm) for message_net in self.message_nets],
            dim=2,
        )  # (B, N, T, D)
        edge_type_onehot = F.one_hot(
            self.edge_type_ids,
            num_classes=self.num_edge_types,
        ).float().to(R_norm.device)
        return torch.einsum('ijt,bjtd->bijd', edge_type_onehot, message_bank)

    def forward(self, R_in: torch.Tensor, t_mod: torch.Tensor = None, region_valid_mask: torch.Tensor = None, graph_time_beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        R_norm = self.norm(R_in)
        if t_mod is not None:
            adj_matrix = self._get_soft_adj_matrix(t_mod, region_valid_mask, graph_time_beta)
        else:
            adj_matrix = self.M_matrix_hard * self.M_non_self
            if region_valid_mask is not None:
                valid_mask_2d = region_valid_mask.unsqueeze(1) & region_valid_mask.unsqueeze(-1)
                adj_matrix = adj_matrix.unsqueeze(0) * valid_mask_2d.float()
            else:
                adj_matrix = adj_matrix.unsqueeze(0).expand(R_in.size(0), -1, -1)
            self._cache_debug_tensor('_last_adj', adj_matrix)
            self._cache_debug_tensor('_last_gamma_t', torch.ones(R_in.size(0), 1, device=R_in.device))
        edge_typed_messages = self._build_edge_typed_messages(R_norm)
        M_i = torch.einsum('bij,bijd->bid', adj_matrix, edge_typed_messages)
        gate_input = torch.cat([R_norm, M_i], dim=-1)
        z_r = self.gate_net(gate_input)
        z, r = torch.chunk(z_r, 2, dim=-1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        candidate_input = torch.cat([r * R_norm, M_i], dim=-1)
        H_tilde = torch.tanh(self.candidate_net(candidate_input))
        R_out = (1.0 - z) * R_in + z * H_tilde
        if region_valid_mask is not None:
            valid_mask = region_valid_mask.unsqueeze(-1).float()
            R_out = R_out * valid_mask + R_in * (1 - valid_mask)
        return R_out

class SelfGuide(nn.Module):
    """
    内区域引导：将区域 R_i 的特征 R_i 映射回区域 i 内的所有残基。
    """

    def __init__(self, D_region):
        super().__init__()
        self.proj = nn.Linear(D_region, D_region)

    def forward(self, R_region, region_type_onehot, region_valid_mask=None):
        """
        Args:
            R_region: (B, N_regions, D_region)
            region_type_onehot: (B, L, N_regions) - 残基所属区域的 One-Hot 编码
            region_valid_mask: (B, N_regions)
        """
        self_region_feat = torch.bmm(region_type_onehot, R_region)

        self_guide = self.proj(self_region_feat)

        if region_valid_mask is not None:
            valid_mask = torch.bmm(region_type_onehot, region_valid_mask.float().unsqueeze(-1)).squeeze(-1)  # (B, L)
            self_guide = self_guide * valid_mask.unsqueeze(-1)

        return self_guide


class CrossGuide(nn.Module):
    """
    跨区域引导：使用 residue query 与 region key/value 做 cross attention。

    div7 增强点：
    - 显式加入 region-relation bias
    - 不再只是简单的单路 dot-product 到区域均值
    """

    def __init__(self, D_res, D_region, num_regions, relation_matrix=None, relation_opt: Optional[Dict[str, Any]] = None):
        super().__init__()
        relation_opt = relation_opt or {}
        self.num_regions = num_regions
        self.query_proj = nn.Linear(D_res, D_region)
        self.key_proj = nn.Linear(D_region, D_region)
        self.value_proj = nn.Linear(D_region, D_region)
        self.out_proj = nn.Linear(D_region, D_region)
        self.layer_norm = nn.LayerNorm(D_region)
        self.use_relation_bias = bool(relation_opt.get('cross_attention_use_relation_bias', False))
        self.relation_bias_strength = nn.Parameter(
            torch.tensor(float(relation_opt.get('cross_attention_relation_bias_strength', 0.15)))
        )
        self.guide_semantics = 'content_cross_region_readout'
        if relation_matrix is None:
            relation_matrix = torch.zeros(num_regions, num_regions)
        self.register_buffer('relation_matrix', relation_matrix.float())

    def forward(self, f_res, R_region, region_type, region_valid_mask=None):
        aa_queries = self.layer_norm(self.query_proj(f_res))
        region_keys = self.key_proj(R_region)
        region_values = self.value_proj(R_region)
        content_logits = torch.bmm(aa_queries, region_keys.transpose(1, 2)) / (R_region.shape[-1] ** 0.5)
        relation_bias_logits = torch.zeros_like(content_logits)
        attn_logits = content_logits
        if self.use_relation_bias:
            query_region = region_type.to(torch.long).clamp(min=0, max=self.num_regions)
            query_region_idx = torch.clamp(query_region - 1, min=0)
            relation_bias = self.relation_matrix[query_region_idx]
            relation_bias = torch.where(
                (query_region > 0).unsqueeze(-1),
                relation_bias,
                torch.zeros_like(relation_bias)
            )
            bias_strength = torch.clamp(self.relation_bias_strength, min=0.0)
            relation_bias_logits = bias_strength * relation_bias
            attn_logits = attn_logits + relation_bias_logits
        pre_mask_logits = attn_logits
        if region_valid_mask is not None:
            valid_mask = region_valid_mask.unsqueeze(1).float()
            attn_logits = attn_logits.masked_fill(valid_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        if region_valid_mask is not None:
            valid_mask = region_valid_mask.unsqueeze(1).float()
            attn_weights = attn_weights * valid_mask
            attn_sum = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            attn_weights = attn_weights / attn_sum
        self._last_content_logits = content_logits
        self._last_relation_bias_logits = relation_bias_logits
        self._last_pre_mask_logits = pre_mask_logits
        self._last_attn_logits = attn_logits
        self._last_attn_weights = attn_weights
        self._last_region_valid_mask = region_valid_mask
        cross_guide = torch.bmm(attn_weights, region_values)
        cross_guide = self.out_proj(cross_guide)
        return cross_guide


class FinedRegionModel(nn.Module):
    """
    区域特征交互与氨基酸引导模块。
    """

    def __init__(self, D_res: int, D_region: int, num_regions: int = REGION_NUM, num_layers: int = 1, relation_opt: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.D_res = D_res
        self.D_region = D_region
        self.num_regions = num_regions
        self.relation_opt = relation_opt or {}
        num_layers = int(self.relation_opt.get('num_layers', num_layers))
        self.context_update_mode = str(self.relation_opt.get('context_update_mode', 'all')).lower()

        self.M_matrix_hard = self._get_interaction_matrix_hard()

        self.region_interaction_layers = nn.ModuleList([
            RegionInteractionLayer(D_region, self.M_matrix_hard, relation_opt=self.relation_opt)
            for _ in range(num_layers)
        ])

        self.self_guide = SelfGuide(D_region)
        self.cross_guide = CrossGuide(D_res, D_region, num_regions, relation_matrix=self.M_matrix_hard, relation_opt=self.relation_opt)

        self.gate_net = nn.Sequential(
            nn.Linear(D_region * 2 + 1, D_region),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_region, D_region // 2),
            nn.ReLU(),
            nn.Linear(D_region // 2, 1),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(D_region, D_region)

    def _build_region_update_mask(self, region_type_onehot, structure_generate_mask, region_valid_mask=None):
        if self.context_update_mode in ('all', 'full', 'default', ''):
            return None
        if self.context_update_mode not in ('generated_only', 'freeze_context', 'cdr_generated_only'):
            raise ValueError(f'Unknown context_update_mode: {self.context_update_mode}')

        generated_residue = structure_generate_mask.squeeze(-1).float()
        generated_counts = torch.bmm(
            region_type_onehot.transpose(1, 2),
            generated_residue.unsqueeze(-1),
        ).squeeze(-1)
        update_mask = generated_counts > 0
        if region_valid_mask is not None:
            update_mask = update_mask & region_valid_mask.bool()
        return update_mask

    def _get_interaction_matrix_hard(self) -> torch.Tensor:
        """从 RELATION_DICT 构建初始硬约束矩阵 M_hard"""
        interactions = RELATION_DICT
        M = torch.zeros(self.num_regions, self.num_regions, dtype=torch.float)
        for region_i, neighbors in interactions.items():
            i = region_i - 1
            for region_j in neighbors:
                j = region_j - 1
                M[i, j] = 1.0

        M_sym = torch.max(M, M.T)
        return M_sym

    def forward(self, f_res, R_region, region_type, structure_generate_flag, valid_mask, t_mod, region_valid_mask=None, graph_time_beta=None):
        region_type_clamped = region_type.to(torch.long).clamp(min=0, max=self.num_regions)
        region_type_onehot = F.one_hot(region_type_clamped, num_classes=self.num_regions + 1)[:, :, 1:].float()
        structure_generate_mask = (structure_generate_flag > 0).unsqueeze(-1)

        R_prime = R_region
        region_update_mask = self._build_region_update_mask(region_type_onehot, structure_generate_mask, region_valid_mask)
        for layer in self.region_interaction_layers:
            R_next = layer(R_prime, t_mod, region_valid_mask, graph_time_beta)
            if region_update_mask is None:
                R_prime = R_next
            else:
                update_mask = region_update_mask.unsqueeze(-1).to(dtype=R_next.dtype)
                R_prime = update_mask * R_next + (1.0 - update_mask) * R_prime

        self_guide = self.self_guide(R_prime, region_type_onehot, region_valid_mask)
        cross_guide = self.cross_guide(f_res, R_prime, region_type, region_valid_mask)

        self_generating = structure_generate_mask.float()

        gate_input = torch.cat([
            self_guide,
            cross_guide,
            self_generating,
        ], dim=-1)

        gate_weights = self.gate_net(gate_input)
        fused_guide = gate_weights * self_guide + (1 - gate_weights) * cross_guide
        final_guide = self.output_proj(fused_guide)

        final_guide = final_guide * valid_mask.unsqueeze(-1)
        self._last_R_prime = R_prime
        self._last_self_guide = self_guide
        self._last_cross_guide = cross_guide
        self._last_gate_weights = gate_weights
        self._last_final_guide = final_guide
        self._last_region_update_mask = region_update_mask

        return final_guide



class ConditionerEncoder(nn.Module):
    """
    通用层级条件编码器基类：负责从低层级特征中提取高层级特征。

    功能:
    1. 注意力池化: 低层级 -> 高层级
    2. VIB 编码: 压缩为 mu/log_var
    3. 时间调制: 根据时间步 t 调整高层级特征
    4. 重参数化采样: 得到高层级特征

    可用于:
    - Residue -> Region (底层)
    - Region -> Chain (高层)
    """

    def __init__(self, D_low: int, D_high: int, N_total_high: int, lambda_cond: float = 0.3):
        super().__init__()
        self.N_total_high = N_total_high
        self.D_low = D_low
        self.D_high = D_high
        self.lambda_target = 1.0  # CDRs KL loss 权重
        self.lambda_cond = lambda_cond  # FRs/Ag KL loss 权重

        self.E_embed = nn.Embedding(N_total_high + 1, D_high)

        self.attn_score_head = nn.Sequential(
            nn.Linear(D_low + D_high, max(D_low // 4, 32)),
            nn.SiLU(),
            nn.Linear(max(D_low // 4, 32), 1)
        )

        self.vib_encoder = VIBEncoder(D_low, D_high)

        self.time_mlp = nn.Sequential(
            nn.Linear(3, D_high),
            nn.SiLU(),
            nn.Linear(D_high, D_high)
        )
        self.scale_shift_proj = nn.Linear(D_high, D_high * 2)

    def _compute_kl_loss(self, mu, std, log_var, high_valid_mask):
        """计算 KL 散度损失"""
        if high_valid_mask is not None:
            valid_expanded = high_valid_mask.unsqueeze(-1).float()
            mu = mu * valid_expanded
            std = std * valid_expanded + (1 - valid_expanded) * 1.0
            log_var = log_var * valid_expanded

        kl_div = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1 - log_var, dim=-1)
        kl_loss = kl_div.sum(dim=-1)
        return kl_loss

    def _attention_pooling(self, f_low, high_type_onehot, valid_mask=None):
        """使用注意力机制将低层级特征池化为高层级特征

        Args:
            f_low: (B, L, D_low) 低层级特征
            high_type_onehot: (B, L, N_high) - 所属高层级的 One-Hot 编码
                注意：region_type=0 (padding) 对应 high_type_onehot[:, :, 0]=0
            valid_mask: (B, L) - 有效掩码
        Returns:
            f_pooled: (B, N_high, D_low) 池化后的高层级特征
            high_valid_mask: (B, N_high) 高层级有效掩码
        """
        B, L, D_low = f_low.shape[:3]
        N_high = self.N_total_high

        E_embed_high = self.E_embed.weight[1:].unsqueeze(0)

        f_low_exp = f_low.unsqueeze(2).expand(-1, -1, N_high, -1)
        E_embed_exp = E_embed_high.unsqueeze(1).expand(B, L, -1, -1)

        attn_input = torch.cat([f_low_exp, E_embed_exp], dim=-1)
        attn_scores = self.attn_score_head(attn_input).squeeze(-1)

        masked_scores = attn_scores.masked_fill(high_type_onehot == 0, float('-inf'))

        if valid_mask is not None:
            masked_scores = masked_scores.masked_fill(~valid_mask.unsqueeze(-1), float('-inf'))

        W_attn = F.softmax(masked_scores, dim=1)
        W_attn = W_attn.masked_fill(torch.isnan(W_attn), 0.0)

        f_pooled = torch.bmm(W_attn.transpose(1, 2), f_low)

        if valid_mask is not None:
            high_type_masked = high_type_onehot * valid_mask.unsqueeze(-1).float()
            high_valid_mask = high_type_masked.sum(dim=1) > 0
        else:
            high_valid_mask = high_type_onehot.sum(dim=1) > 0

        self._last_attention_pool_scores = attn_scores
        self._last_attention_pool_masked_scores = masked_scores
        self._last_attention_pool_weights = W_attn
        self._last_attention_pool_high_valid_mask = high_valid_mask
        return f_pooled, high_valid_mask

    def forward(self,
                f_low: torch.Tensor,
                high_type: torch.Tensor,
                structure_generate_flag: torch.Tensor,
                valid_mask: torch.Tensor,
                time_embed: torch.Tensor,
                high_valid_mask: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_low: (B, L, D_low) 低层级特征
            high_type: (B, L) 高层级类型 1-N_high
            structure_generate_flag: (B, L) 生成标志
            valid_mask: (B, L) 有效掩码
            time_embed: (B, D_high) 或 (B, N_high, D_high) 时间嵌入
                - (B, D_high): 全局时间，广播到所有节点
                - (B, N_high, D_high): per-node 时间，每个节点有自己的时间
            high_valid_mask: (B, N_high) 高层级有效掩码
        Returns:
            f_high: (B, N_high, D_high) 高层级特征
            sample_info: dict with mu, log_var, std
            kl_loss: (B,)
            time_conditioning: (B, 2*D_high) 或 (B, N_high, 2*D_high)
            high_valid_mask: (B, N_high)
        """
        B, L, D_low = f_low.shape

        if DEBUG_MODE:
            print(f"\n[ConditionerEncoder-Debug] ========== Forward Start ==========")
            print(f"  f_low.shape = {f_low.shape}, expected (B={B}, L={L}, D_low={D_low})")
            print(f"  f_low min/max/mean = {f_low.min():.4f} / {f_low.max():.4f} / {f_low.mean():.4f}")
            print(f"  high_type.shape = {high_type.shape}")
            print(f"  high_type values (first 10, batch 0): {high_type[0, :10].tolist()}")
            print(f"  time_embed.shape = {time_embed.shape}, time_embed.dim() = {time_embed.dim()}")
            print(f"  time_embed min/max/mean = {time_embed.min():.6f} / {time_embed.max():.6f} / {time_embed.mean():.6f}")
            if time_embed.dim() == 3:
                print(f"    per-node time (B, N_high, D_high), expected (B={B}, N_high={self.N_total_high}, D_high={self.D_high})")
                sample_b = 0
                print(f"    Sample batch {sample_b} time values (first 5 nodes, first 3 dims):")
                for n_id in range(min(5, time_embed.size(1))):
                    vals = time_embed[sample_b, n_id, :3].detach().cpu().numpy()
                    print(f"      Node {n_id}: [{vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}]")
            elif time_embed.dim() == 2:
                print(f"    global time (B, D_high), expected (B={B}, D_high={self.D_high})")
                sample_b = 0
                vals = time_embed[sample_b, :3].detach().cpu().numpy()
                print(f"    Sample batch {sample_b} time values: [{vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}]")
            else:
                print(f"    scalar time (B,), expected (B={B})")
                sample_b = 0
                print(f"    Sample batch {sample_b} time value: {time_embed[sample_b].item():.6f}")
            print(f"  valid_mask.shape = {valid_mask.shape}")
            print(f"  valid_mask sum per batch: {valid_mask.sum(dim=1).tolist()}")
            if high_valid_mask is not None:
                print(f"  high_valid_mask.shape = {high_valid_mask.shape}")
                print(f"  high_valid_mask (valid nodes, batch 0): {high_valid_mask[0].nonzero().squeeze().tolist()}")

        high_type_clamped = high_type.to(torch.long).clamp(0, self.N_total_high)  # 16 3
        high_type_onehot = F.one_hot(high_type_clamped, num_classes=self.N_total_high + 1)
        high_type_onehot_nozero = high_type_onehot[:, :, 1:].float()  # (B, L, N_high)


        if time_embed.dim() == 1:
            beta_embed = torch.stack([time_embed, torch.sin(time_embed), torch.cos(time_embed)], dim=-1)
            time_embed_encoded = self.time_mlp(beta_embed)  # (B, D_high)
        elif time_embed.dim() == 2:
            if time_embed.size(-1) == 3:
                time_embed_encoded = self.time_mlp(time_embed)  # (B, D_high)
            else:
                time_embed_encoded = time_embed
        else:
            if time_embed.size(-1) == 3:
                B, L, _ = time_embed.shape


                beta_embed_expanded = time_embed.unsqueeze(2).expand(-1, -1, self.N_total_high, -1)

                mask_expanded = high_type_onehot_nozero.unsqueeze(-1)  # (B, L, N_high, 1)

                if valid_mask is not None:
                    valid_expanded = valid_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
                    mask_expanded = mask_expanded * valid_expanded

                region_time_sum = (beta_embed_expanded * mask_expanded).sum(dim=1)  # (B, N_high, 3)

                region_count = mask_expanded.sum(dim=1).clamp(min=1)  # (B, N_high, 1)

                region_time_avg = region_time_sum / region_count  # (B, N_high, 3)
                region_time_flat = region_time_avg.reshape(B * self.N_total_high, 3)  # (B*N_high, 3)
                time_embed_flat = self.time_mlp(region_time_flat)  # (B*N_high, D_high)
                time_embed_encoded = time_embed_flat.reshape(B, self.N_total_high, self.D_high)  # (B, N_high, D_high)
            else:
                time_embed_encoded = time_embed

        time_conditioning = self.scale_shift_proj(time_embed_encoded)  # (B, 2*D_high) 或 (B, N, 2*D_high)
        if time_conditioning.dim() == 2:
            time_scale = time_conditioning[:, :self.D_high].unsqueeze(1)  # (B, 1, D_high)
            time_shift = time_conditioning[:, self.D_high:].unsqueeze(1)  # (B, 1, D_high)
        else:
            time_scale = time_conditioning[:, :, :self.D_high]  # (B, N, D_high)
            time_shift = time_conditioning[:, :, self.D_high:]  # (B, N, D_high)

        pooled_features, detected_valid_mask = self._attention_pooling(f_low, high_type_onehot_nozero, valid_mask)

        if DEBUG_MODE:
            print(f"\n[ConditionerEncoder-Debug] After attention pooling:")
            print(f"  pooled_features.shape = {pooled_features.shape}, expected (B={B}, N_high={self.N_total_high}, D_low={D_low})")
            print(f"  pooled_features min/max/mean = {pooled_features.min():.4f} / {pooled_features.max():.4f} / {pooled_features.mean():.4f}")
            sample_b = 0
            print(f"  Sample batch {sample_b} pooled_features (first 3 nodes, first 3 dims):")
            for n_id in range(min(3, pooled_features.size(1))):
                vals = pooled_features[sample_b, n_id, :3].detach().cpu().numpy()
                print(f"    Node {n_id}: [{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}]")

        if high_valid_mask is None:
            high_valid_mask = detected_valid_mask

        f_vib_input = pooled_features.reshape(B * self.N_total_high, D_low)
        mu_flat, log_var_flat = self.vib_encoder(f_vib_input, high_valid_mask)

        mu = mu_flat.reshape(B, self.N_total_high, self.D_high)
        log_var = log_var_flat.reshape(B, self.N_total_high, self.D_high)

        if DEBUG_MODE:
            print(f"\n[ConditionerEncoder-Debug] After VIB encoding:")
            print(f"  mu.shape = {mu.shape}")
            print(f"  mu min/max/mean = {mu.min():.4f} / {mu.max():.4f} / {mu.mean():.4f}")
            print(f"  log_var.shape = {log_var.shape}")
            print(f"  log_var min/max = {log_var.min():.4f} / {log_var.max():.4f}")
            sample_b = 0
            print(f"  Sample batch {sample_b} VIB outputs (first 3 nodes, first 3 dims):")
            for n_id in range(min(3, mu.size(1))):
                mu_vals = mu[sample_b, n_id, :3].detach().cpu().numpy()
                logvar_vals = log_var[sample_b, n_id, :3].detach().cpu().numpy()
                std_vals = torch.exp(0.5 * log_var)[sample_b, n_id, :3].detach().cpu().numpy()
                print(f"    Node {n_id}: mu=[{mu_vals[0]:.3f}, {mu_vals[1]:.3f}, {mu_vals[2]:.3f}], " +
                      f"logvar=[{logvar_vals[0]:.3f}, {logvar_vals[1]:.3f}, {logvar_vals[2]:.3f}], " +
                      f"std=[{std_vals[0]:.3f}, {std_vals[1]:.3f}, {std_vals[2]:.3f}]")

        if high_valid_mask is not None:
            valid_mask_expanded = high_valid_mask.unsqueeze(-1).float()
            time_scale = time_scale * valid_mask_expanded
            time_shift = time_shift * valid_mask_expanded
        time_conditioned_mu = mu * (1 + time_scale) + time_shift
        std = torch.exp(0.5 * log_var)

        if DEBUG_MODE:
            print(f"\n[ConditionerEncoder-Debug] After time modulation:")
            print(f"  time_conditioned_mu.shape = {time_conditioned_mu.shape}")
            print(f"  time_conditioned_mu min/max/mean = {time_conditioned_mu.min():.4f} / {time_conditioned_mu.max():.4f} / {time_conditioned_mu.mean():.4f}")
            print(f"  std.shape = {std.shape}")
            print(f"  std min/max = {std.min():.4f} / {std.max():.4f}")
            sample_b = 0
            print(f"  Sample batch {sample_b} time-conditioned mu (first 3 nodes, first 3 dims):")
            for n_id in range(min(3, time_conditioned_mu.size(1))):
                before_vals = mu[sample_b, n_id, :3].detach().cpu().numpy()
                after_vals = time_conditioned_mu[sample_b, n_id, :3].detach().cpu().numpy()
                scale_vals = time_scale[sample_b, n_id, :3].detach().cpu().numpy()
                shift_vals = time_shift[sample_b, n_id, :3].detach().cpu().numpy()
                print(f"    Node {n_id}: before=[{before_vals[0]:.3f}, {before_vals[1]:.3f}, {before_vals[2]:.3f}], " +
                      f"scale=[{scale_vals[0]:.3f}, {scale_vals[1]:.3f}, {scale_vals[2]:.3f}], " +
                      f"shift=[{shift_vals[0]:.3f}, {shift_vals[1]:.3f}, {shift_vals[2]:.3f}], " +
                      f"after=[{after_vals[0]:.3f}, {after_vals[1]:.3f}, {after_vals[2]:.3f}]")

        min_std = 0.1
        std_for_sample = torch.where(
            high_valid_mask.unsqueeze(-1),
            torch.clamp(std, min=min_std),
            std
        )

        eps = torch.randn_like(std_for_sample)
        sampled_features = time_conditioned_mu + eps * std_for_sample

        high_embed = self.E_embed.weight[1:].unsqueeze(0).expand(B, -1, -1)

        if high_valid_mask is not None:
            valid_mask_expanded = high_valid_mask.unsqueeze(-1).float()
        f_high = high_embed + sampled_features * valid_mask_expanded
        f_high = f_high * valid_mask_expanded

        sample_info = {'mu': time_conditioned_mu, 'log_var': log_var, 'std': std}

        kl_loss = self._compute_kl_loss(time_conditioned_mu, std, log_var, high_valid_mask)

        return f_high, sample_info, kl_loss, time_conditioning, high_valid_mask



class RegionConditionerEncoder(ConditionerEncoder):
    """
    区域调节器编码器：从残基特征中提取区域级特征。
    继承自 ConditionerEncoder，专门用于 Residue -> Region。

    Args:
        D_res: int, 残基特征维度
        D_region: int, 区域特征维度
        lambda_cond: float, KL loss 权重
        N_total_regions: int, 区域数量 (默认16)
    """

    def __init__(self, D_res: int, D_region: int, lambda_cond: float = 0.3, N_total_regions=REGION_NUM,
                 v3_config: Optional[Dict] = None):
        super().__init__(D_res, D_region, N_total_regions, lambda_cond)

    def _get_kl_weighted_loss(self, structure_generate_flag: torch.Tensor,
                              region_type: torch.Tensor,
                              mu_mod: torch.Tensor, std: torch.Tensor,
                              log_var_mod: torch.Tensor, region_valid_mask=None):
        """计算加权的 KL 散度损失，CDR 区域权重更高

        Args:
            structure_generate_flag: (B, L) 残基级别的生成标志
            region_type: (B, L) 残基所属区域
            mu_mod, std, log_var_mod: (B, N_regions, D_region) VIB参数
        """
        B, N_regions, D_region = mu_mod.shape

        if region_valid_mask is not None:
            valid_expanded = region_valid_mask.unsqueeze(-1).float()
            mu_mod = mu_mod * valid_expanded
            std = std * valid_expanded + (1 - valid_expanded) * 1.0
            log_var_mod = log_var_mod * valid_expanded

        kl_div_full = 0.5 * torch.sum(mu_mod.pow(2) + std.pow(2) - 1 - log_var_mod, dim=-1)

        gen_mask = structure_generate_flag > 0  # (B, L) bool mask for generating residues
        region_ids = region_type - 1  # (B, L) convert to 0-based region IDs

        L = structure_generate_flag.shape[1]
        batch_indices = torch.arange(B, device=mu_mod.device).unsqueeze(1).expand(-1, L)  # (B, L)

        gen_mask_flat = gen_mask.reshape(-1)
        batch_flat = batch_indices.reshape(-1)
        region_ids_flat = region_ids.reshape(-1)

        gen_indices = gen_mask_flat.nonzero().squeeze(-1)  # 生成残基在flat数组中的索引
        if gen_indices.numel() > 0:
            gen_batch_ids = batch_flat[gen_indices]  # 这些残基的batch ID
            gen_region_ids = region_ids_flat[gen_indices]  # 这些残基的region ID

            valid_region_mask = (gen_region_ids >= 0) & (gen_region_ids < N_regions)
            gen_batch_ids = gen_batch_ids[valid_region_mask]
            gen_region_ids = gen_region_ids[valid_region_mask]

            is_VIB_target_cdr = torch.zeros(B, N_regions, device=mu_mod.device, dtype=torch.bool)
            if gen_batch_ids.numel() > 0:
                is_VIB_target_cdr[gen_batch_ids, gen_region_ids] = True
        else:
            is_VIB_target_cdr = torch.zeros(B, N_regions, device=mu_mod.device, dtype=torch.bool)

        W_kl_weights = torch.full_like(kl_div_full, self.lambda_cond)
        W_kl_weights = torch.masked_fill(W_kl_weights, is_VIB_target_cdr, self.lambda_target)

        kl_divs = kl_div_full * W_kl_weights
        kl_loss = kl_divs.sum(dim=-1)  # 只在区域维度求和

        return kl_loss

    def forward(self,
                f_res: torch.Tensor,
                region_type: torch.Tensor,
                structure_generate_flag: torch.Tensor,
                valid_mask: torch.Tensor,
                t: torch.Tensor,
                region_valid_mask: torch.Tensor = None,
                b_factor: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            f_res: (B, L, D_res) 残基特征
            region_type: (B, L) 区域类型 1-16
            structure_generate_flag: (B, L) 生成标志
            valid_mask: (B, L) 有效掩码
            t: (B,) 时间步
            region_valid_mask: (B, N_regions) 区域有效掩码
            b_factor: (B, L, n_atom) B-factor (V3, 可选)

        Returns:
            R_all: (B, N_regions, D_region) 区域级特征
            R_sample_info: dict 包含 mu, log_var, std
            reg_kl_loss: (B,) KL损失
            region_valid_mask: (B, N_regions) 区域有效掩码
        """
        R_all, R_sample_info, _, time_conditioning, region_valid_mask_out = super().forward(
            f_res, region_type, structure_generate_flag, valid_mask, t, region_valid_mask
        )

        mu = R_sample_info['mu']
        std = R_sample_info['std']
        log_var = R_sample_info['log_var']
        reg_kl_loss = self._get_kl_weighted_loss(
            structure_generate_flag, region_type,
            mu, std, log_var, region_valid_mask_out
        )

        return R_all, R_sample_info, reg_kl_loss, None, time_conditioning, region_valid_mask_out


class ChainConditionerEncoder(ConditionerEncoder):
    """
    链调节器编码器：从区域特征中提取链级特征。
    继承自 ConditionerEncoder，专门用于 Region -> Chain。

    Args:
        D_region: int, 区域特征维度
        D_chain: int, 链特征维度
        lambda_cond: float, KL loss 权重
        N_total_chains: int, 链数量 (默认3)
    """

    def __init__(self, D_region: int, D_chain: int, lambda_cond: float = 0.3, N_total_chains=3):
        super().__init__(D_region, D_chain, N_total_chains, lambda_cond)

    def forward(self,
                f_region: torch.Tensor,
                chain_type: torch.Tensor,
                valid_mask: torch.Tensor,
                time_embed: torch.Tensor,
                chain_valid_mask: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_region: (B, N_regions, D_region) 区域特征
            chain_type: (B, N_regions) 链类型 1-3
            valid_mask: (B, N_regions) 区域有效掩码
            time_embed: (B, D_chain) 或 (B, N_chains, D_chain) 时间嵌入
                - (B, D_chain): 全局时间，广播到所有节点
                - (B, N_chains, D_chain): per-node 时间，每个节点有自己的时间
            chain_valid_mask: (B, N_chains) 链有效掩码

        Returns:
            C_all: (B, N_chains, D_chain) 链级特征
            C_sample_info: dict 包含 mu, log_var, std
            chain_kl_loss: (B,) KL损失
            time_conditioning: (B, 2*D_chain) 或 (B, N_chains, 2*D_chain)
            chain_valid_mask: (B, N_chains) 链有效掩码
        """
        C_all, C_sample_info, kl_loss, time_conditioning, chain_valid_mask_out = super().forward(
            f_region, chain_type, torch.zeros_like(chain_type), valid_mask, time_embed, chain_valid_mask
        )

        return C_all, C_sample_info, kl_loss, time_conditioning, chain_valid_mask_out



class ChainConditionerDecoder(nn.Module):
    """
    链调节器解码器：将链级特征解码为区域级引导特征。

    功能:
    1. 链间交互: 可选的 GNN 或 MHA
    2. 区域引导: 将链特征广播到区域

    Args:
        D_region: int, 区域特征维度
        D_chain: int, 链特征维度
        N_total_chains: int, 链数量 (默认3)
    """

    def __init__(self, D_region: int, D_chain: int, N_total_chains: int = 3):
        super().__init__()
        self.D_region = D_region
        self.D_chain = D_chain
        self.N_total_chains = N_total_chains

        self.chain_interaction = nn.MultiheadAttention(D_chain, num_heads=4, batch_first=True)
        self.chain_norm = nn.LayerNorm(D_chain)

        self.chain_to_region_proj = nn.Sequential(
            nn.Linear(D_chain, D_region),
            nn.SiLU(),
            nn.Linear(D_region, D_region)
        )

        region_to_chain = torch.tensor([
            1, 1, 1,  # H CDRs -> Heavy
            2, 2, 2,  # L CDRs -> Light
            1, 1, 1, 1,  # H FRs -> Heavy
            2, 2, 2, 2,  # L FRs -> Light
            3, 3,  # Ag -> Antigen
        ], dtype=torch.long)
        self.register_buffer('region_to_chain', region_to_chain)

    def forward(self,
                f_region: torch.Tensor,
                C_all: torch.Tensor,
                region_to_chain_idx: torch.Tensor,
                valid_mask: torch.Tensor,
                t_mod: torch.Tensor,
                chain_valid_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        Args:
            f_region: (B, N_regions, D_region) 区域特征
            C_all: (B, N_chains, D_chain) 链级特征
            region_to_chain_idx: (B, N_regions) 区域到链的映射 1-3
            valid_mask: (B, N_regions) 区域有效掩码
            t_mod: (B, 2, D_chain) 时间调制参数
            chain_valid_mask: (B, N_chains) 链有效掩码

        Returns:
            chain_guide: (B, N_regions, D_region) 区域级链引导特征
        """
        B = C_all.shape[0]

        C_interacted = C_all
        if chain_valid_mask is not None:
            key_padding_mask = ~chain_valid_mask  # (B, N_chains)
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False  # 临时解除，避免error

            C_interacted, _ = self.chain_interaction(
                C_all, C_all, C_all,
                key_padding_mask=key_padding_mask
            )
            if all_masked.any():
                C_interacted = C_interacted.clone()
                C_interacted[all_masked] = 0
        else:
            C_interacted, _ = self.chain_interaction(C_all, C_all, C_all)

        C_interacted = self.chain_norm(C_interacted + C_all)

        if chain_valid_mask is not None:
            C_interacted = C_interacted * chain_valid_mask.unsqueeze(-1).float()

        region_to_chain_0based = self.region_to_chain.clamp(0, 4)  # 0-2
        chain_feat_expanded = C_interacted[:, region_to_chain_0based, :]  # (B, N_regions, D_chain)

        chain_guide = self.chain_to_region_proj(chain_feat_expanded)

        if valid_mask is not None:
            chain_guide = chain_guide * valid_mask.unsqueeze(-1).float()

        return chain_guide


class Updater(nn.Module):
    """
    基于仿射耦合层（Affaine Coupling Layer）实现可逆变换，用于扩散模型的去噪步骤。
    """

    def __init__(self, D_res, D_region):
        super().__init__()
        self.split_dim = D_res // 2
        self.affine_net = nn.Sequential(
            nn.Linear(D_region, D_region // 2),
            nn.SiLU(),
            nn.Linear(D_region // 2, self.split_dim * 2),
            nn.Tanh()
        )

        self.gen_scale = nn.Parameter(torch.tensor(0.1))
        self.gen_shift = nn.Parameter(torch.tensor(0.05))

    def forward(self, structure_generate_flag, f_res, region_guide, valid_mask):
        f1, f2 = torch.split(f_res, self.split_dim, dim=-1)

        affine_params = self.affine_net(region_guide)
        s, t = torch.chunk(affine_params, 2, dim=-1)

        s = 0.3 * torch.tanh(s)  # div9: 0.3 instead of 0.1, allow stronger region conditioning

        gen_mask = (structure_generate_flag > 0).float().unsqueeze(-1)
        s = s * (1.0 + self.gen_scale * gen_mask)
        t = t * (1.0 + self.gen_shift * gen_mask)

        f2_out = f2 * (1.0 + s) + 0.3 * t  # div9: 0.3 instead of 0.1
        f_out = torch.cat([f1, f2_out], dim=-1)

        f_out = f_out * valid_mask.unsqueeze(-1)

        log_det = torch.log(torch.clamp(1.0 + s, min=1e-8)).sum(dim=[1, 2])

        self._last_region_guide = region_guide
        self._last_affine_scale = s
        self._last_affine_shift = t
        self._last_f_res_in = f_res
        self._last_f_res_out = f_out

        return f_out, log_det

    def inverse(self, structure_generate_flag, f_res, region_guide, valid_mask):
        f1, f2 = torch.split(f_res, self.split_dim, dim=-1)
        affine_params = self.affine_net(region_guide)
        s, t = torch.chunk(affine_params, 2, dim=-1)

        s = 0.3 * torch.tanh(s)  # div9: match forward
        gen_mask = (structure_generate_flag > 0).float().unsqueeze(-1)
        s = s * (1.0 + self.gen_scale * gen_mask)
        t = t * (1.0 + self.gen_shift * gen_mask)

        eps = 1e-6
        f2_inv = (f2 - 0.3 * t) / (1.0 + s + eps)  # div9: match forward
        f_inv = torch.cat([f1, f2_inv], dim=-1)

        f_inv = f_inv * valid_mask.unsqueeze(-1)

        return f_inv



class RegionConditionerDecoder(nn.Module):
    """
    区域调节器解码器：将区域级特征 R 解码为残基级引导特征。

    功能:
    1. 区域交互: FinedRegionModel 进行区域间信息交换
    2. 残基引导: SelfGuide + CrossGuide 生成 per-residue 特征
    3. 特征融合: 门控融合生成最终引导

    Args:
        D_res: int, 残基特征维度
        D_region: int, 区域特征维度
        N_total_regions: int, 区域数量 (默认16)
    """

    def __init__(self, D_res: int, D_region: int, N_total_regions: int = REGION_NUM, relation_opt: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.D_res = D_res
        self.D_region = D_region
        self.N_total_regions = N_total_regions
        self.relation_opt = relation_opt or {}

        self.fined_region_extractor = FinedRegionModel(
            D_res, D_region, N_total_regions, num_layers=1, relation_opt=self.relation_opt
        )

    def forward(self,
                f_res: torch.Tensor,
                R_all: torch.Tensor,
                region_type: torch.Tensor,
                structure_generate_flag: torch.Tensor,
                valid_mask: torch.Tensor,
                t_mod: torch.Tensor,
                region_valid_mask: torch.Tensor = None,
                graph_time_beta: torch.Tensor = None
                ) -> torch.Tensor:
        """
        Args:
            f_res: (B, L, D_res) 残基特征
            R_all: (B, N_regions, D_region) 区域级特征 (来自 Encoder)
            region_type: (B, L) 区域类型 1-16
            structure_generate_flag: (B, L) 生成标志 0/1-16
            valid_mask: (B, L) 有效掩码
            t_mod: (B, 2, D_region) 时间调制参数 (来自 Encoder)
            region_valid_mask: (B, N_regions) 区域有效掩码

        Returns:
            fined_region_feat: (B, L, D_region) 残基级区域引导特征
        """
        fined_region_feat = self.fined_region_extractor(
            f_res, R_all, region_type, structure_generate_flag, valid_mask, t_mod,
            region_valid_mask, graph_time_beta
        )

        return fined_region_feat



class RegionConditioner(nn.Module):
    """
    完整的区域调节器：组合 Encoder + Decoder。
    在 core/rim 主线基础上，加入更强的 contiguous-fake logit ranking 辅助。
    """

    def __init__(self, D_res: int, D_region: int, lambda_cond: float = 0.3, N_total_regions=REGION_NUM,
                 v3_config: Optional[Dict] = None, relation_opt: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.encoder = RegionConditionerEncoder(
            D_res, D_region, lambda_cond, N_total_regions, v3_config
        )
        self.decoder = RegionConditionerDecoder(
            D_res, D_region, N_total_regions, relation_opt=relation_opt
        )
        relation_opt = relation_opt or {}
        self.aux_losses_cfg = relation_opt.get('aux_losses', {})
        self.aux_enabled = bool(self.aux_losses_cfg.get('enabled', False))
        self.contig_cfg = self.aux_losses_cfg.get('contig_logit_rank', self.aux_losses_cfg.get('contig_hn_attn', {}))
        self.contig_enabled = bool(self.contig_cfg.get('enabled', False))
        self.contig_margin = float(self.contig_cfg.get('margin_logit', self.contig_cfg.get('margin_contig', 0.1)))
        self.contig_start_step = int(self.contig_cfg.get('start_step', 250))
        self.contig_t_max = int(self.contig_cfg.get('t_cdr_max', 40))
        self.token_div_cfg = self.aux_losses_cfg.get('token_diversity', {})
        self.token_div_enabled = bool(self.token_div_cfg.get('enabled', False))
        self.token_div_start_step = int(self.token_div_cfg.get('start_step', 100))
        self.token_div_cos_margin = float(self.token_div_cfg.get('cos_margin', 0.5))
        self.token_div_var_target = float(self.token_div_cfg.get('var_target', 0.2))
        self.token_div_cos_weight = float(self.token_div_cfg.get('cos_weight', 1.0))
        self.token_div_var_weight = float(self.token_div_cfg.get('var_weight', 1.0))
        self.pair_contact_cfg = self.aux_losses_cfg.get('pair_contact', {})
        self.pair_contact_enabled = bool(self.pair_contact_cfg.get('enabled', False))
        self.pair_contact_start_step = int(self.pair_contact_cfg.get('start_step', 100))
        self.pair_contact_cutoff = float(self.pair_contact_cfg.get('cutoff', 8.0))
        self.pair_contact_cdr_epi_weight = float(self.pair_contact_cfg.get('cdr_epi_weight', 2.0))
        self.region_pair_head = nn.Sequential(
            nn.Linear(D_region * 4, D_region),
            nn.ReLU(),
            nn.Linear(D_region, 1),
        )
    def _prepare_f_res_for_decoder(self, f_res: torch.Tensor, b_factor: torch.Tensor, region_type: torch.Tensor, valid_mask: torch.Tensor):
        f_res_for_decoder = f_res
        if hasattr(self.encoder, 'v3_config') and self.encoder.v3_config.bfactor_enabled:
            if b_factor is not None:
                b_factor_norm = self.encoder.bfactor_preprocessor(b_factor, region_type, valid_mask)
                b_embed = self.encoder.bfactor_encoder(b_factor_norm)
                b_embed_proj = self.encoder.bfactor_proj(b_embed)
                f_res_for_decoder = f_res + b_embed_proj
        return f_res_for_decoder

    def _compute_t_mod(self, time_conditioning: torch.Tensor, region_valid_mask: torch.Tensor, batch_size: int):
        if time_conditioning.dim() == 2:
            return time_conditioning.unsqueeze(1)
        if region_valid_mask is not None:
            return time_conditioning * region_valid_mask.unsqueeze(-1).float()
        return time_conditioning

    def _compute_region_graph_time_beta(self, t: torch.Tensor, region_type: torch.Tensor, valid_mask: torch.Tensor, region_valid_mask: torch.Tensor):
        if t is None:
            return None
        if t.dim() == 3:
            beta = t[..., 0]
        elif t.dim() == 2 and t.shape == region_type.shape:
            beta = t
        elif t.dim() == 2 and t.size(-1) == 3:
            beta = t[:, 0].unsqueeze(1).expand_as(region_type)
        elif t.dim() == 1:
            beta = t.unsqueeze(1).expand_as(region_type)
        else:
            return None
        region_type_clamped = region_type.to(torch.long).clamp(min=0, max=self.encoder.N_total_high)
        region_onehot = F.one_hot(region_type_clamped, num_classes=self.encoder.N_total_high + 1)[:, :, 1:].float()
        mask = region_onehot * valid_mask.float().unsqueeze(-1)
        beta_sum = (beta.unsqueeze(-1).float() * mask).sum(dim=1)
        beta_count = mask.sum(dim=1).clamp_min(1.0)
        region_beta = beta_sum / beta_count
        if region_valid_mask is not None:
            region_beta = region_beta * region_valid_mask.float()
        return region_beta

    def _run_region_pass(self, f_res, region_type, structure_generate_flag, valid_mask, t, region_valid_mask, b_factor, deterministic=False):
        R_all, R_sample_info, reg_kl_loss, _, time_conditioning, region_valid_mask_out = self.encoder(
            f_res, region_type, structure_generate_flag, valid_mask, t, region_valid_mask, b_factor
        )
        if deterministic:
            region_embed = self.encoder.E_embed.weight[1:].unsqueeze(0).expand_as(R_all)
            R_all = region_embed + R_sample_info['mu'] * region_valid_mask_out.unsqueeze(-1).float()
            R_all = R_all * region_valid_mask_out.unsqueeze(-1).float()
        f_res_for_decoder = self._prepare_f_res_for_decoder(f_res, b_factor, region_type, valid_mask)
        t_mod = self._compute_t_mod(time_conditioning, region_valid_mask_out, f_res.size(0))
        graph_time_beta = self._compute_region_graph_time_beta(t, region_type, valid_mask, region_valid_mask_out)
        fined_region_feat = self.decoder(
            f_res_for_decoder, R_all, region_type, structure_generate_flag, valid_mask,
            t_mod, region_valid_mask_out, graph_time_beta
        )
        region_extractor = self.decoder.fined_region_extractor
        attn_logits = getattr(region_extractor.cross_guide, '_last_attn_logits', None)
        interaction_state = {
            'region_post': getattr(region_extractor, '_last_R_prime', None),
            'self_guide': getattr(region_extractor, '_last_self_guide', None),
            'cross_guide': getattr(region_extractor, '_last_cross_guide', None),
            'gate_weights': getattr(region_extractor, '_last_gate_weights', None),
            'final_guide': getattr(region_extractor, '_last_final_guide', None),
        }
        return fined_region_feat, R_sample_info, reg_kl_loss, region_valid_mask_out, attn_logits, interaction_state

    def _compute_logit_score(self, attn_logits, cdr_mask, region_valid_mask):
        if attn_logits is None:
            zero = cdr_mask.float().sum() * 0.0
            scores = zero.new_zeros(cdr_mask.shape[0])
            valid = cdr_mask.new_zeros(cdr_mask.shape[0], dtype=torch.bool)
            return scores, valid, scores.clone(), scores.clone()

        scores = []
        valid = []
        core_scores = []
        rim_scores = []
        core_idx_global = int(AG.EPI_CORE) - 1
        rim_idx_global = int(AG.EPI_RIM) - 1
        for b in range(attn_logits.shape[0]):
            cdr_idx = cdr_mask[b].nonzero(as_tuple=True)[0]
            has_core = bool(region_valid_mask[b, core_idx_global]) if core_idx_global < region_valid_mask.shape[1] else False
            has_rim = bool(region_valid_mask[b, rim_idx_global]) if rim_idx_global < region_valid_mask.shape[1] else False
            epi_idx = []
            if has_core:
                epi_idx.append(core_idx_global)
            if has_rim:
                epi_idx.append(rim_idx_global)
            if cdr_idx.numel() == 0 or not epi_idx:
                zero = attn_logits.new_zeros(())
                scores.append(zero)
                core_scores.append(zero)
                rim_scores.append(zero)
                valid.append(False)
                continue
            logits_b = attn_logits[b, cdr_idx]
            core_score = logits_b[:, [core_idx_global]].mean() if has_core else attn_logits.new_zeros(())
            rim_score = logits_b[:, [rim_idx_global]].mean() if has_rim else attn_logits.new_zeros(())
            epi_score = logits_b[:, epi_idx].mean()
            scores.append(epi_score)
            core_scores.append(core_score)
            rim_scores.append(rim_score)
            valid.append(True)
        return (
            torch.stack(scores),
            torch.tensor(valid, device=attn_logits.device, dtype=torch.bool),
            torch.stack(core_scores),
            torch.stack(rim_scores),
        )

    def _compute_contig_logit_rank(self, f_res, region_type, structure_generate_flag, valid_mask, t, region_valid_mask, b_factor, region_aux_inputs):
        zero = f_res.sum() * 0.0
        if not self.contig_enabled or region_aux_inputs is None:
            return {}
        global_step = int(region_aux_inputs.get('global_step', 0))
        t_cdr = region_aux_inputs.get('t_cdr')
        if global_step < self.contig_start_step or t_cdr is None:
            return {'region_contig_logit_rank': zero}
        active_batch = t_cdr.long() <= self.contig_t_max
        if not active_batch.any():
            return {'region_contig_logit_rank': zero}

        antigen_mask = region_aux_inputs.get('antigen_mask_raw')
        true_soft_mask = region_aux_inputs.get('antigen_soft_mask_raw')
        true_core_mask = region_aux_inputs.get('antigen_core_mask_raw')
        chain_nb = region_aux_inputs.get('chain_nb')
        res_nb = region_aux_inputs.get('res_nb')
        cdr_mask = region_aux_inputs.get('generate_flag')
        pos_heavyatom = region_aux_inputs.get('pos_heavyatom')
        if antigen_mask is None or true_soft_mask is None or chain_nb is None or res_nb is None or cdr_mask is None or pos_heavyatom is None:
            return {'region_contig_logit_rank': zero}

        antigen_mask = antigen_mask.to(torch.bool)
        true_soft_mask = true_soft_mask.to(torch.bool)
        true_core_mask = true_core_mask.to(torch.bool) if true_core_mask is not None else None
        cdr_mask = cdr_mask.to(torch.bool)

        _, _, _, region_valid_mask_true, attn_logits_true, _ = self._run_region_pass(
            f_res, region_type, structure_generate_flag, valid_mask, t, None, b_factor, deterministic=True
        )

        contig_soft_mask = build_fake_epitope_mask(true_soft_mask, antigen_mask, chain_nb, res_nb, mode='contiguous_fake')
        region_type_contig = build_alt_region_type(
            region_type,
            antigen_mask,
            contig_soft_mask,
            pos_heavyatom=pos_heavyatom,
            true_core_mask=true_core_mask,
        )
        _, _, _, region_valid_mask_contig, attn_logits_contig, _ = self._run_region_pass(
            f_res, region_type_contig, structure_generate_flag, valid_mask, t, None, b_factor, deterministic=True
        )

        s_true, valid_true, s_true_core, s_true_rim = self._compute_logit_score(attn_logits_true, cdr_mask, region_valid_mask_true)
        s_contig, valid_contig, s_contig_core, s_contig_rim = self._compute_logit_score(attn_logits_contig, cdr_mask, region_valid_mask_contig)
        valid = valid_true & valid_contig & active_batch.to(valid_true.device)
        if valid.any():
            loss = F.relu(self.contig_margin - s_true + s_contig)
            loss = loss[valid].mean()
            s_true_mean = s_true[valid].mean().detach()
            s_contig_mean = s_contig[valid].mean().detach()
            gap_mean = (s_true - s_contig)[valid].mean().detach()
            s_true_core_mean = s_true_core[valid].mean().detach()
            s_true_rim_mean = s_true_rim[valid].mean().detach()
            s_contig_core_mean = s_contig_core[valid].mean().detach()
            s_contig_rim_mean = s_contig_rim[valid].mean().detach()
            active_count = valid.sum().detach().float()
        else:
            loss = zero
            s_true_mean = zero.detach()
            s_contig_mean = zero.detach()
            gap_mean = zero.detach()
            s_true_core_mean = zero.detach()
            s_true_rim_mean = zero.detach()
            s_contig_core_mean = zero.detach()
            s_contig_rim_mean = zero.detach()
            active_count = zero.detach()

        return {
            'region_contig_logit_rank': loss,
            'monitor_region_logit_true': s_true_mean,
            'monitor_region_logit_contig': s_contig_mean,
            'monitor_region_logit_gap': gap_mean,
            'monitor_region_logit_true_core': s_true_core_mean,
            'monitor_region_logit_true_rim': s_true_rim_mean,
            'monitor_region_logit_contig_core': s_contig_core_mean,
            'monitor_region_logit_contig_rim': s_contig_rim_mean,
            'monitor_region_active_count': active_count,
        }


    def _compute_token_diversity(self, R_sample_info, region_valid_mask, region_aux_inputs=None, region_post=None):
        region_features = region_post if region_post is not None else R_sample_info.get('mu')
        if region_features is None:
            zero = region_valid_mask.float().sum() * 0.0
            return {}
        zero = region_features.sum() * 0.0
        if not self.token_div_enabled:
            return {}
        global_step = int((region_aux_inputs or {}).get('global_step', 0))
        if global_step < self.token_div_start_step:
            return {'region_token_diversity': zero}

        losses = []
        cos_means = []
        std_means = []
        cdr_epi_means = []
        cdr_ids = {0, 1, 2, 3, 4, 5}
        epi_ids = {14, 15}

        for b in range(region_features.shape[0]):
            valid_idx = region_valid_mask[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() < 2:
                continue
            feat_b = region_features[b, valid_idx]
            feat_norm = F.normalize(feat_b, dim=-1)
            sim = torch.matmul(feat_norm, feat_norm.transpose(0, 1))
            eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            offdiag = sim[~eye]
            cos_loss = F.relu(offdiag - self.token_div_cos_margin).mean() if offdiag.numel() > 0 else zero
            std = feat_b.std(dim=0, unbiased=False)
            var_loss = F.relu(self.token_div_var_target - std).mean()
            losses.append(self.token_div_cos_weight * cos_loss + self.token_div_var_weight * var_loss)
            cos_means.append(offdiag.mean() if offdiag.numel() > 0 else zero.detach())
            std_means.append(std.mean().detach())

            cdr_local = [i for i, rid in enumerate(valid_idx.tolist()) if rid in cdr_ids]
            epi_local = [i for i, rid in enumerate(valid_idx.tolist()) if rid in epi_ids]
            if cdr_local and epi_local:
                cdr_feat = feat_norm[cdr_local]
                epi_feat = feat_norm[epi_local]
                cdr_epi_means.append(torch.matmul(cdr_feat, epi_feat.transpose(0, 1)).mean().detach())

        if losses:
            loss = torch.stack(losses).mean()
            cos_mean = torch.stack(cos_means).mean()
            std_mean = torch.stack(std_means).mean()
            cdr_epi_mean = torch.stack(cdr_epi_means).mean() if cdr_epi_means else zero.detach()
        else:
            loss = zero
            cos_mean = zero.detach()
            std_mean = zero.detach()
            cdr_epi_mean = zero.detach()

        return {
            'region_token_diversity': loss,
            'monitor_region_token_offdiag_cos': cos_mean,
            'monitor_region_token_std': std_mean,
            'monitor_region_token_cdr_epi_cos': cdr_epi_mean,
        }

    def _compute_region_pair_contact(self, R_sample_info, region_type, valid_mask, region_valid_mask, region_aux_inputs=None, region_post=None):
        region_features = region_post if region_post is not None else R_sample_info.get('mu')
        if region_features is None:
            zero = valid_mask.float().sum() * 0.0
            return {}
        zero = region_features.sum() * 0.0
        if not self.pair_contact_enabled:
            return {}
        global_step = int((region_aux_inputs or {}).get('global_step', 0))
        if global_step < self.pair_contact_start_step:
            return {'region_pair_contact': zero}

        pos_heavyatom = None if region_aux_inputs is None else region_aux_inputs.get('pos_heavyatom')
        if pos_heavyatom is None:
            return {'region_pair_contact': zero}

        losses = []
        pred_cdr_epi = []
        true_cdr_epi = []
        cdr_ids = set(range(0, 6))
        epi_ids = {14, 15}

        for b in range(region_features.shape[0]):
            valid_idx = region_valid_mask[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() < 2:
                continue
            feat_b = region_features[b, valid_idx]
            n = feat_b.shape[0]
            pair_feat = torch.cat([
                feat_b.unsqueeze(1).expand(n, n, -1),
                feat_b.unsqueeze(0).expand(n, n, -1),
                (feat_b.unsqueeze(1) - feat_b.unsqueeze(0)).abs(),
                feat_b.unsqueeze(1) * feat_b.unsqueeze(0),
            ], dim=-1)
            logits = self.region_pair_head(pair_feat).squeeze(-1)
            logits = 0.5 * (logits + logits.transpose(0, 1))

            labels = torch.zeros((n, n), device=region_features.device, dtype=logits.dtype)
            weights = torch.ones((n, n), device=region_features.device, dtype=logits.dtype)
            pos_ca = pos_heavyatom[b, :, BBHeavyAtom.CA]
            valid_res = valid_mask[b].to(torch.bool)

            for ii, rid_i in enumerate(valid_idx.tolist()):
                res_i = ((region_type[b] == (rid_i + 1)) & valid_res).nonzero(as_tuple=True)[0]
                if res_i.numel() == 0:
                    continue
                for jj, rid_j in enumerate(valid_idx.tolist()):
                    if ii >= jj:
                        continue
                    res_j = ((region_type[b] == (rid_j + 1)) & valid_res).nonzero(as_tuple=True)[0]
                    if res_j.numel() == 0:
                        continue
                    dist = torch.cdist(pos_ca[res_i], pos_ca[res_j]).min()
                    label = (dist < self.pair_contact_cutoff).to(logits.dtype)
                    labels[ii, jj] = labels[jj, ii] = label
                    if (rid_i in cdr_ids and rid_j in epi_ids) or (rid_j in cdr_ids and rid_i in epi_ids):
                        weights[ii, jj] = weights[jj, ii] = self.pair_contact_cdr_epi_weight
                        pred_cdr_epi.append(torch.sigmoid(logits[ii, jj]).detach())
                        true_cdr_epi.append(label.detach())

            offdiag = ~torch.eye(n, dtype=torch.bool, device=region_features.device)
            if offdiag.any():
                loss = F.binary_cross_entropy_with_logits(logits[offdiag], labels[offdiag], weight=weights[offdiag])
                losses.append(loss)

        if losses:
            loss = torch.stack(losses).mean()
            pred_mean = torch.stack(pred_cdr_epi).mean() if pred_cdr_epi else zero.detach()
            true_mean = torch.stack(true_cdr_epi).mean() if true_cdr_epi else zero.detach()
        else:
            loss = zero
            pred_mean = zero.detach()
            true_mean = zero.detach()

        return {
            'region_pair_contact': loss,
            'monitor_region_pair_cdr_epi_pred': pred_mean,
            'monitor_region_pair_cdr_epi_true': true_mean,
        }

    def forward(self,
                f_res: torch.Tensor,
                region_type: torch.Tensor,
                structure_generate_flag: torch.Tensor,
                valid_mask: torch.Tensor,
                t: torch.Tensor,
                region_valid_mask: torch.Tensor = None,
                b_factor: torch.Tensor = None,
                region_aux_inputs: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        fined_region_feat, R_sample_info, reg_kl_loss, region_valid_mask_out, _, interaction_state = self._run_region_pass(
            f_res, region_type, structure_generate_flag, valid_mask, t, region_valid_mask, b_factor, deterministic=False
        )
        aux_dict = None
        if self.training and self.aux_enabled:
            aux_dict = {}
            aux_dict.update(
                self._compute_contig_logit_rank(
                    f_res=f_res,
                    region_type=region_type,
                    structure_generate_flag=structure_generate_flag,
                    valid_mask=valid_mask,
                    t=t,
                    region_valid_mask=region_valid_mask_out,
                    b_factor=b_factor,
                    region_aux_inputs=region_aux_inputs,
                )
            )
            aux_dict.update(
                self._compute_token_diversity(
                    R_sample_info=R_sample_info,
                    region_valid_mask=region_valid_mask_out,
                    region_aux_inputs=region_aux_inputs,
                    region_post=interaction_state.get('region_post'),
                )
            )
            aux_dict.update(
                self._compute_region_pair_contact(
                    R_sample_info=R_sample_info,
                    region_type=region_type,
                    valid_mask=valid_mask,
                    region_valid_mask=region_valid_mask_out,
                    region_aux_inputs=region_aux_inputs,
                    region_post=interaction_state.get('region_post'),
                )
            )
        return fined_region_feat, R_sample_info, reg_kl_loss, aux_dict
