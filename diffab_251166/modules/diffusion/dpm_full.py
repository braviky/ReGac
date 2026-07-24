import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
from tqdm.auto import tqdm

DEBUG_MODE = False  # 打开调试
SKIP_SAMPLING_ASSERTIONS = True  # 跳过采样过程中的断言
DEBUG_TIME_STEP = None  # None=随机采样, max=t=num_steps, min=t=0
DEBUG_BFACTOR = False  # 关闭B因子相关调试
DEBUG_LOSS_DETAILS = False  # 调试loss详情
DEBUG_INFERENCE = False  # 推理调试开关
DEBUG_GRADIENT = False  # 梯度范围监控开关

from diffab_251166.modules.encoders.ga import GAEncoder
from diffab_251166.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix, \
    reconstruct_backbone, get_backbone_dihedral_angles
from diffab_251166.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3

from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition
from diffab_251166.modules.common.adaln import TimeMLP
from diffab_251166.modules.common.topology import get_consecutive_flag
from diffab_251166.modules.common.bfactor_utils import (
    GeomLossScaler,
    GeometryLossCalculator, DynamicLossWeightScheduler, apply_dynamic_weights
)
from diffab_251166.utils.protein.constants import FLAGS, BBHeavyAtom, CORE_LOSSES, MONITOR_LOSSES, CDR
from diffab_251166.utils.protein import constants
from diffab_251166.utils.noise_mode import is_cdr_noise_mode, is_epitope_noise_mode
from diffab_251166.utils.cfg_utils import build_cfg_unconditional_inputs
from diffab_251166.modules.common.bfactor_utils import debug_log, DEBUG_WEIGHTS
_loss_compute_counter = 0


def rotation_matrix_cosine_loss(R_next, R_true):
    size = list(R_next.shape[:-2])
    ncol = R_next.numel() // 3
    RT_pred = R_next.transpose(-2, -1).reshape(ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)
    ones = torch.ones([ncol, ], dtype=torch.long, device=R_next.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')
    loss = loss.reshape(size + [3]).sum(dim=-1)
    return loss


def rotation_matrix_geodesic_loss(R_pred, R_true):
    """
    测地距离损失: 使用旋转角度
    R_error = R_pred^T @ R_true
    angle = arccos((trace(R_error) - 1) / 2)

    优势：在小误差下梯度更强，有利于旋转预测学习
    """
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_true)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos_angle)
    return angle ** 2


def _get_runtime_modal_config(batch, default_noise_mode='cdr_only'):
    modal_config = batch.get('val_modal_training')
    if modal_config is None:
        modal_config = batch.get('modal_training')
    if modal_config is None:
        modal_config = {'noise_sampling': {'noise_mode': default_noise_mode}}
    return modal_config


def _get_runtime_noise_sampling(batch, default_noise_mode='cdr_only'):
    modal_config = _get_runtime_modal_config(batch, default_noise_mode=default_noise_mode)
    if isinstance(modal_config, dict):
        noise_sampling = modal_config.get('noise_sampling', None)
    else:
        noise_sampling = getattr(modal_config, 'noise_sampling', None)
    if noise_sampling is None:
        noise_sampling = {'noise_mode': default_noise_mode}
    return noise_sampling


def _resolve_sync_warmup_steps(async_cfg, num_steps):
    if not isinstance(async_cfg, dict):
        return 0
    if 'sync_warmup_steps' in async_cfg:
        value = async_cfg.get('sync_warmup_steps')
        if isinstance(value, str):
            expr = value.strip().lower().replace(' ', '')
            if expr in {'num_steps', 'nsteps', 'steps', 't'}:
                warmup = int(num_steps)
            elif '/' in expr and expr.split('/', 1)[0] in {'num_steps', 'nsteps', 'steps', 't'}:
                denom = float(expr.split('/', 1)[1])
                if denom == 0.0:
                    raise ValueError(f'sync_warmup_steps has zero denominator: {value}')
                warmup = int(round(float(num_steps) / denom))
            elif '*' in expr:
                left, right = expr.split('*', 1)
                if left in {'num_steps', 'nsteps', 'steps', 't'}:
                    warmup = int(round(float(num_steps) * float(right)))
                elif right in {'num_steps', 'nsteps', 'steps', 't'}:
                    warmup = int(round(float(left) * float(num_steps)))
                else:
                    warmup = int(round(float(expr)))
            else:
                warmup = int(round(float(expr)))
        else:
            warmup = int(round(float(value)))
    else:
        ratio = float(async_cfg.get('sync_warmup_ratio', async_cfg.get('sync_ratio', 0.0)))
        ratio = max(0.0, min(ratio, 1.0))
        warmup = int(round(float(num_steps) * ratio))
    return max(0, min(int(warmup), int(num_steps)))


def _get_runtime_noise_mode(batch, default_noise_mode='cdr_only'):
    noise_sampling = _get_runtime_noise_sampling(batch, default_noise_mode=default_noise_mode)
    if isinstance(noise_sampling, dict):
        return noise_sampling.get('noise_mode', default_noise_mode)
    return getattr(noise_sampling, 'noise_mode', default_noise_mode)


def _resolve_encoder_runtime_context(batch, noise_mode, mask_res):
    batch_flag = {f: batch[f] for f in FLAGS if f in batch}
    mask_cdr = batch.get('mask_cdr', batch_flag.get('generate_flag', None))
    if mask_cdr is None:
        mask_cdr = batch['structure_generate_flag'] > 0
    else:
        mask_cdr = mask_cdr.to(torch.bool)

    mask_soft_antigen = batch.get('antigen_soft_mask', batch_flag.get('antigen_soft_mask', None))
    if mask_soft_antigen is None:
        mask_soft_antigen = torch.zeros_like(mask_res, dtype=torch.bool)
    else:
        mask_soft_antigen = mask_soft_antigen.to(torch.bool)

    antigen_rigid_mask = batch.get('antigen_rigid_mask', batch_flag.get('antigen_rigid_mask', None))
    if antigen_rigid_mask is None:
        antigen_rigid_mask = torch.zeros_like(mask_res, dtype=torch.bool)
    else:
        antigen_rigid_mask = antigen_rigid_mask.to(torch.bool)

    region_type = batch['region_type']
    structure_generate_flag = batch_flag['structure_generate_flag']
    mask_generate = batch['structure_generate_flag'] > 0
    antigen_mask_raw = batch.get('antigen_mask_raw', batch.get('antigen_mask'))

    if noise_mode == 'cdr_only':
        mask_generate = mask_cdr
        structure_generate_flag = mask_cdr

    batch_flag['region_type'] = region_type
    return {
        'batch_flag': batch_flag,
        'mask_cdr': mask_cdr,
        'mask_soft_antigen': mask_soft_antigen,
        'antigen_rigid_mask': antigen_rigid_mask,
        'region_type': region_type,
        'structure_generate_flag': structure_generate_flag,
        'mask_generate': mask_generate,
        'antigen_mask_raw': antigen_mask_raw,
        'mask_res': mask_res,
    }


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, region_dim, pair_feat_dim, cond_cfg, denoise_net, num_layers,
                 encoder_opt=None, beta_embed_dim=3, dropout=0.15, cfg_dropout=0.0):
        super().__init__()
        self.res_feat_dim = res_feat_dim
        if region_dim is None:
            region_dim = res_feat_dim // 2
        self.region_dim = region_dim
        encoder_opt = encoder_opt or {}
        self.cond_cfg = cond_cfg

        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )

        self.denoise_net = denoise_net if denoise_net else "ga"
        self.beta_embed_dim = beta_embed_dim

        self.time_embed_dim = 64
        self.time_encoder = TimeMLP(input_dim=beta_embed_dim, time_dim=self.time_embed_dim, use_silu=True)

        self.dropout = dropout
        self.cfg_dropout = cfg_dropout

        if self.denoise_net != "ga":
            raise ValueError(f"unsupported denoise_net={self.denoise_net!r}; this build keeps only ga")
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, self.region_dim, num_layers, **encoder_opt)
        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim + beta_embed_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )
        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim + beta_embed_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )
        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim + beta_embed_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20)  # 输出 logits
        )

    def forward(self, batch, v_t, p_t, s_t, res_feat_ctx, pair_feat_ctx, beta):
        """
            Args:
                v_t:    (N, L, 3).
                p_t:    (N, L, 3).
                s_t:    (N, L).
                res_feat:   (N, L, res_dim).
                pair_feat:  (N, L, L, pair_dim).
                beta:   (N,) 或 dict {'cdr': t_cdr, 'epitope': t_epi} 时间步，支持浮点
                mask_generate:    (N, L).
                bfactor: (N, L, n_atom) or (N, L) B-factor

            Returns:
                v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
                eps_pos: (N, L, 3).
                representations: dict with 'seq' and 'pair' keys for self-conditioning
        """
        noise_mode = _get_runtime_noise_mode(batch)
        mask_res = batch['mask'].to(torch.bool)
        R_t = so3vec_to_rotation(v_t)
        B, L = mask_res.size()

        context = _resolve_encoder_runtime_context(batch, noise_mode, mask_res)
        batch_flag = context['batch_flag']
        mask_cdr = context['mask_cdr']
        mask_soft_antigen = context['mask_soft_antigen']
        antigen_rigid_mask = context['antigen_rigid_mask']
        region_type = context['region_type']
        structure_generate_flag = context['structure_generate_flag']
        mask_generate = context['mask_generate']
        antigen_mask_raw = context['antigen_mask_raw']
        mask_res = context['mask_res']


        seq_emb = self.current_sequence_embedding(s_t)  # (B, L, res_dim)
        res_feat = self.res_feat_mixer(torch.cat([res_feat_ctx, seq_emb], dim=-1))

        seq_generate_flag = batch_flag['generate_flag']
        structure_generate_flag = structure_generate_flag
        beta_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, L, 3)

        time_embed = self.time_encoder(beta_embed)

        extra_configs = {
            'structure_generate_flag': structure_generate_flag,
            'beta_embed': beta_embed,
            'region_aux_inputs': {
                'antigen_mask_raw': antigen_mask_raw,
                'antigen_soft_mask_raw': batch.get('antigen_soft_mask_raw', batch.get('antigen_soft_mask')),
                'antigen_rigid_mask_raw': batch.get('antigen_rigid_mask_raw', batch.get('antigen_rigid_mask')),
                'antigen_core_mask_raw': batch.get('antigen_core_mask_raw', batch.get('antigen_core_mask')),
                'chain_nb': batch.get('chain_nb'),
                'res_nb': batch.get('res_nb'),
                'pos_heavyatom': batch.get('pos_heavyatom'),
                'mask_heavyatom': batch.get('mask_heavyatom', batch.get('atom_mask')),
                'generate_flag': batch.get('generate_flag'),
                'fix_cdr_flag': batch.get('fix_cdr_flag'),
                't_cdr': batch.get('_t_cdr_aux'),
                'global_step': batch.get('global_step', 0),
            },
        }
        if self.training and self.cfg_dropout > 0.0:
            mask_epitope = ((region_type == constants.AG.EPI_CORE) | (region_type == constants.AG.EPI_RIM))  # (B, L)
            cfg_mask = (torch.rand(B, device=mask_epitope.device) >= self.cfg_dropout)  # (B,)
            cfg_keep = cfg_mask.unsqueeze(1) | (~mask_epitope)  # non-epitope always kept
            res_feat = res_feat * cfg_keep.unsqueeze(-1).float()
            cfg_keep_2d = (cfg_keep.unsqueeze(1) & cfg_keep.unsqueeze(2)).unsqueeze(-1)
            pair_feat_ctx = pair_feat_ctx * cfg_keep_2d.float()

        res_feat, aux = self.encoder(
            R_t,  # Pass rotation matrices for spatial attention
            p_t,  # Normalized coordinates for GA spatial attention
            res_feat,
            pair_feat_ctx,
            mask_res,
            region_type,
            p_contact=batch.get('_hgacd_p_contact', p_t),
            **extra_configs
        )
        if isinstance(aux, dict):
            self._kl_loss_dict = {
                k: v for k, v in aux.items()
                if k.startswith('ib_kl_l') and isinstance(v, torch.Tensor)
            }
            self._region_aux_loss_dict = {
                k: v for k, v in aux.items()
                if k.startswith('region_') and isinstance(v, torch.Tensor)
            }
            self._region_aux_monitor_dict = {
                k: v for k, v in aux.items()
                if k.startswith('monitor_') and isinstance(v, torch.Tensor)
            }
        else:
            self._kl_loss_dict = {}
            self._region_aux_loss_dict = {}
            self._region_aux_monitor_dict = {}

        in_feat = torch.cat([res_feat, beta_embed], dim=-1)  # (B, L, res_dim + beta_embed_dim)
        eps_crd = self.eps_crd_net(in_feat)
        eps_rot = self.eps_rot_net(in_feat)
        s_0_pred_logits = self.eps_seq_net(in_feat)
        eps_pos = apply_rotation_to_vector(R_t, eps_crd)  # (B,L,3)
        eps_pos = torch.where(mask_generate.to(torch.bool)[:, :, None].expand_as(eps_pos), eps_pos,
                              torch.zeros_like(eps_pos))

        U = quaternion_1ijk_to_rotation_matrix(eps_rot)  # (B,L,3,3)
        R_next = R_t @ U

        if self.training and hasattr(self, '_debug_rot_iter'):
            self._debug_rot_iter += 1
            if self._debug_rot_iter <= 10 or self._debug_rot_iter % 50 == 0:
                import numpy as np
                eps_rot_mean = eps_rot.mean().item()
                eps_rot_std = eps_rot.std().item()
                eps_rot_max = eps_rot.abs().max().item()
                U_trace = U.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().item()
                R_diff = rotation_matrix_geodesic_loss(R_next, R_t).mean().item()
                print(f"[ROT调试 iter={self._debug_rot_iter}] eps_rot: mean={eps_rot_mean:.4f}, std={eps_rot_std:.4f}, max={eps_rot_max:.4f}")
                print(f"[ROT调试] U_trace={U_trace:.4f} (应为~1.0), R_next-R_t geodesic={R_diff:.4f} rad²")
                if eps_rot_max > 5.0:
                    print(f"[ROT警告] eps_rot值过大! max={eps_rot_max:.4f}, 可能导致quaternion不稳定")
                if abs(U_trace - 1.0) > 0.5:
                    print(f"[ROT警告] U不是有效旋转矩阵! trace={U_trace:.4f}")
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        representations = None
        if isinstance(aux, dict):
            representations = {
                'seq': res_feat,
                'pair': aux.get('pair', None),
                'kl_loss_dict': self._kl_loss_dict,
            }

        return v_next, R_next, eps_pos, s_0_pred_logits, representations


class FullDPM(nn.Module):
    def __init__(self, res_feat_dim, pair_feat_dim, region_dim, cond_cfg, num_steps, loss_weights=None,
                 eps_net_opt=None, trans_rot_opt=None, trans_pos_opt=None, trans_seq_opt=None,
                 position_mean=[0.0, 0.0, 0.0], position_scale=[10.0], guidance_scale=1.0):
        """
        Args:
            res_feat_dim: residue feature dimension
            pair_feat_dim: pair feature dimension
            region_dim: region feature dimension
            cond_cfg: conditioner configuration
            num_steps: number of diffusion steps (for discrete diffusion)
            eps_net_opt: epsilon network options
            trans_rot_opt: rotation transition options
            trans_pos_opt: position transition options
            trans_seq_opt: sequence transition options
            position_mean: position normalization mean
            position_scale: position normalization scale
            loss_weights已移除，使用DynamicLossWeightScheduler动态计算
        """
        super().__init__()
        eps_net_opt = eps_net_opt or {}
        trans_rot_opt = trans_rot_opt or {}
        trans_pos_opt = trans_pos_opt or {}
        trans_seq_opt = trans_seq_opt or {}

        self.loss_weights = loss_weights or {}
        self.guidance_scale = guidance_scale

        if self.loss_weights:
            try:
                if DEBUG_WEIGHTS:
                    debug_log(f"[FullDPM] Received loss_weights: {self.loss_weights}")
            except:
                pass

        cfg_d = float(eps_net_opt.pop('cfg_dropout', 0.0))
        self.eps_net = EpsilonNet(res_feat_dim, region_dim, pair_feat_dim, cond_cfg, cfg_dropout=cfg_d, **eps_net_opt)
        self.num_steps = num_steps

        if DEBUG_MODE:
            print(f"[FullDPM-Debug] EpsilonNet encoder type: {type(self.eps_net.encoder).__name__}")

        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        if DEBUG_MODE:
            print(f"[FullDPM-Debug] Using discrete-time diffusion (num_steps={num_steps})")

        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))
        self._debug_counter = 0

        self.use_geom_scaler = False
        if self.use_geom_scaler:
            self.geom_scaler = GeomLossScaler(
                init_bone_weight=0.05,
                init_omega_weight=0.01,
            )
            print(f"[FullDPM __init__] GeomLossScaler已创建: s_bone初始≈3.0, s_omega初始≈4.6")
        else:
            self.geom_scaler = None

    def _normalize_position(self, p):
        return (p - self.position_mean) / self.position_scale

    def _unnormalize_position(self, p_norm):
        return p_norm * self.position_scale + self.position_mean

    def _set_hgacd_contact_positions(self, batch, p_t):
        batch['_hgacd_p_contact'] = self._unnormalize_position(p_t)

    def _expand_timestep_map(self, t, mask_like):
        if t.dim() == 1:
            return t.long().view(-1, 1).expand_as(mask_like)
        if t.dim() == 2:
            if t.shape != mask_like.shape:
                raise ValueError(f"Expected timestep shape {tuple(mask_like.shape)}, got {tuple(t.shape)}")
            return t.long()
        raise ValueError(f"Unsupported timestep rank: {t.dim()}")

    def _batch_timestep_from_map(self, t, mask):
        if t.dim() == 1:
            return t.long()
        t_map = self._expand_timestep_map(t, mask)
        mask = mask.to(torch.bool)
        denom = mask.float().sum(dim=1).clamp_min(1.0)
        avg_t = (t_map.float() * mask.float()).sum(dim=1) / denom
        return avg_t.round().long().clamp(min=0, max=self.num_steps)

    def _masked_process_scalar(self, t, mask):
        if t.dim() == 1:
            return t.float()
        t_map = self._expand_timestep_map(t, mask)
        mask = mask.to(torch.bool)
        denom = mask.float().sum(dim=1).clamp_min(1.0)
        return (t_map.float() * mask.float()).sum(dim=1) / denom

    def _compute_async_epitope_t(self, t_c, noise_sampling):
        strategy = noise_sampling.get('epitope_time_strategy', 'paired_ratio')
        t_c = t_c.long().clamp(min=0, max=self.num_steps)

        if strategy in (None, 'same_index', 'legacy'):
            return t_c

        if strategy == 'induced_fit':
            max_t = int(noise_sampling.get('induced_fit_max_t', 10))
            return torch.minimum(t_c, torch.full_like(t_c, max_t)).clamp(min=0, max=self.num_steps)

        if strategy == 'induced_fit_late':
            max_t = int(noise_sampling.get('induced_fit_max_t', 10))
            unfreeze_t_max = int(noise_sampling.get('induced_fit_unfreeze_t_max', 25))
            t_e_small = torch.minimum(t_c, torch.full_like(t_c, max_t))
            t_e = torch.where(t_c <= unfreeze_t_max, t_e_small, torch.zeros_like(t_c))
            return t_e.clamp(min=0, max=self.num_steps)

        if strategy != 'paired_ratio':
            raise ValueError(f'Unsupported epitope_time_strategy: {strategy}')

        min_ratio = float(noise_sampling.get('epitope_time_min_ratio', 0.35))
        max_ratio = float(noise_sampling.get('epitope_time_max_ratio', 0.80))
        power = float(noise_sampling.get('epitope_time_power', noise_sampling.get('power', 1.4)))
        min_gap = int(noise_sampling.get('epitope_time_min_gap', 1))

        min_ratio = max(0.0, min(min_ratio, 1.0))
        max_ratio = max(min_ratio, min(max_ratio, 1.0))
        power = max(power, 1.0)

        t_norm = t_c.float() / float(max(self.num_steps, 1))
        ratio = min_ratio + (max_ratio - min_ratio) * torch.pow(t_norm, power)
        t_e = torch.floor(t_c.float() * ratio).long()

        if min_gap > 0:
            t_e = torch.where(
                t_c > min_gap,
                torch.minimum(t_e, t_c - min_gap),
                torch.zeros_like(t_e),
            )

        return t_e.clamp(min=0, max=self.num_steps)

    def _build_epitope_loss_mask(self, mask_soft_antigen, t_epi):
        if mask_soft_antigen is None or t_epi is None:
            return mask_soft_antigen
        active_samples = self._batch_timestep_from_map(t_epi, mask_soft_antigen) > 0 if t_epi.dim() == 2 else t_epi > 0
        return mask_soft_antigen & active_samples.unsqueeze(1)

    def _sample_region_async_training_t(self, base_t, mask_cdr, region_type, noise_sampling):
        cfg = noise_sampling.get('region_async_training', {}) if isinstance(noise_sampling, dict) else {}
        if not bool(cfg.get('enabled', False)) or region_type is None:
            return base_t
        prob = float(cfg.get('prob', 1.0))
        max_lag = int(cfg.get('region_max_lag', cfg.get('max_lag', 8)))
        max_lag = max(0, max_lag)
        if max_lag <= 0:
            return base_t
        half_window = max(1, max_lag // 2)
        min_t = int(cfg.get('min_t', 1))
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        t_map = base_t.view(-1, 1).expand_as(mask_cdr).clone()
        for b in range(mask_cdr.size(0)):
            if prob < 1.0 and torch.rand((), device=base_t.device) > prob:
                continue
            present = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b].to(mask_cdr.device) == rid)
                if bool(rmask.any().item()):
                    present.append((rid, rmask))
            if len(present) <= 1:
                continue
            for _, rmask in present:
                offset = torch.randint(
                    low=-half_window,
                    high=half_window + 1,
                    size=(),
                    device=base_t.device,
                )
                t_region = torch.clamp(base_t[b] + offset, min=min_t, max=self.num_steps)
                t_map[b, rmask] = t_region
        return t_map

    def _compute_clean_position_from_eps(self, p_t, eps_p, mask_cdr, t_cdr, epitope_mask=None, t_epitope=None):
        t_cdr_map = self._expand_timestep_map(t_cdr, mask_cdr)
        alpha_cdr = self.trans_pos.var_sched.alphas[t_cdr_map].clamp_min(self.trans_pos.var_sched.alphas[-2])
        alpha_bar_cdr = self.trans_pos.var_sched.alpha_bars[t_cdr_map]
        c0_cdr = (1.0 / torch.sqrt(alpha_cdr + 1e-8)).unsqueeze(-1)
        c1_cdr = ((1 - alpha_cdr) / torch.sqrt(1 - alpha_bar_cdr + 1e-8)).unsqueeze(-1)
        p_clean_cdr = c0_cdr * (p_t - c1_cdr * eps_p)
        p_clean_cdr = torch.where(mask_cdr[..., None].expand_as(p_clean_cdr), p_clean_cdr, p_t)

        if epitope_mask is None or t_epitope is None:
            return p_clean_cdr
        if not epitope_mask.any():
            return p_clean_cdr

        t_epi_map = self._expand_timestep_map(t_epitope, mask_cdr)
        alpha_epi = self.trans_pos.var_sched.alphas_epi[t_epi_map].clamp_min(self.trans_pos.var_sched.alphas_epi[-2])
        alpha_bar_epi = self.trans_pos.var_sched.alpha_bars_epi[t_epi_map]
        c0_epi = (1.0 / torch.sqrt(alpha_epi + 1e-8)).unsqueeze(-1)
        c1_epi = ((1 - alpha_epi) / torch.sqrt(1 - alpha_bar_epi + 1e-8)).unsqueeze(-1)
        p_clean_epi = c0_epi * (p_t - c1_epi * eps_p)
        p_clean_epi = torch.where(epitope_mask[..., None].expand_as(p_clean_epi), p_clean_epi, p_t)
        return torch.where(epitope_mask[..., None], p_clean_epi, p_clean_cdr)

    def _compute_true_position_eps(self, p_t, p_0, mask_cdr, t_cdr, epitope_mask=None, t_epitope=None):
        t_cdr_map = self._expand_timestep_map(t_cdr, mask_cdr)
        alpha_bar_cdr = self.trans_pos.var_sched.alpha_bars[t_cdr_map]
        c0_cdr = torch.sqrt(alpha_bar_cdr).unsqueeze(-1)
        c1_cdr = torch.sqrt(1 - alpha_bar_cdr + 1e-8).unsqueeze(-1)
        eps_cdr = (p_t - c0_cdr * p_0) / c1_cdr
        eps_cdr = torch.where(mask_cdr[..., None].expand_as(eps_cdr), eps_cdr, torch.zeros_like(eps_cdr))

        if epitope_mask is None or t_epitope is None:
            return eps_cdr
        if not epitope_mask.any():
            return eps_cdr

        t_epi_map = self._expand_timestep_map(t_epitope, mask_cdr)
        alpha_bar_epi = self.trans_pos.var_sched.alpha_bars_epi[t_epi_map]
        c0_epi = torch.sqrt(alpha_bar_epi).unsqueeze(-1)
        c1_epi = torch.sqrt(1 - alpha_bar_epi + 1e-8).unsqueeze(-1)
        eps_epi = (p_t - c0_epi * p_0) / c1_epi
        eps_epi = torch.where(epitope_mask[..., None].expand_as(eps_epi), eps_epi, torch.zeros_like(eps_epi))
        return torch.where(epitope_mask[..., None], eps_epi, eps_cdr)

    def build_forward_t_schedule(self, N, L, noise_sampling, mask_dict, batch=None):
        t_dict = {}
        t = torch.zeros(N, L, device=self._dummy.device)

        beta_dict = {}
        beta = torch.zeros(N, L, device=self._dummy.device)

        noise_mode = noise_sampling['noise_mode']
        t_dict['noise_mode'] = noise_mode

        assert 'mask_cdr' in mask_dict, f"mask_cdr not found in {mask_dict.keys()}"
        if DEBUG_MODE:
            print(f"[DPM-Debug] Using fallback training mode (modal_noise_enabled=False)")
        mask_cdr = mask_dict['mask_cdr']
        region_type = batch.get('region_type') if batch is not None and isinstance(batch, dict) else None

        t_c_base = torch.randint(1, self.num_steps + 1, (N,), dtype=torch.long, device=self._dummy.device)
        t_c = self._sample_region_async_training_t(t_c_base, mask_cdr, region_type, noise_sampling)
        t_dict['cdr'] = t_c
        t_dict['cdr_base'] = t_c_base
        beta_c = self.trans_pos.var_sched.betas[t_c]
        beta_dict['cdr'] = beta_c

        if t_c.dim() == 1:
            t = torch.where(mask_cdr, t_c.unsqueeze(1).expand(N, L), t)
            beta = torch.where(mask_cdr, beta_c.unsqueeze(1).expand(N, L), beta)
            t_c_for_epitope = t_c
        else:
            t = torch.where(mask_cdr, t_c, t)
            beta = torch.where(mask_cdr, beta_c, beta)
            t_c_for_epitope = self._batch_timestep_from_map(t_c, mask_cdr)

        if is_epitope_noise_mode(noise_mode):
            t_e = self._compute_async_epitope_t(t_c_for_epitope, noise_sampling)
            t_dict['epitope'] = t_e
            beta_e = self.trans_pos.var_sched.betas_epi[t_e]
            beta_dict['epitope'] = beta_e

            assert 'mask_soft_antigen' in mask_dict, f"mask_soft_antigen not found in {mask_dict.keys()}"
            assert 'mask_full_antigen' in mask_dict, f"mask_full_antigen not found in {mask_dict.keys()}"
            mask_soft_antigen = mask_dict['mask_soft_antigen']
            t = torch.where(mask_soft_antigen, t_e.unsqueeze(1).expand(N, L), t)
            beta = torch.where(mask_soft_antigen, beta_e.unsqueeze(1).expand(N, L), beta)

        self.t_dict = t_dict
        self.t = t
        self.beta_dict = beta_dict
        self.beta = beta
        return t_dict, t, beta_dict, beta

    def build_sample_t_schedule(self, N, L, t, noise_sampling, mask_dict):
        t_dict = {}
        noise_mode = noise_sampling['noise_mode']
        t_dict['noise_mode'] = noise_mode

        t_c = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)
        t_dict['cdr'] = t_c

        beta_dict = {}
        beta = torch.zeros(N, L, device=self._dummy.device)
        beta_c = self.trans_pos.var_sched.betas[t_c]
        beta_dict['cdr'] = beta_c

        assert 'mask_cdr' in mask_dict, f"mask_cdr not found in {mask_dict.keys()}"
        if DEBUG_MODE:
            print(f"[DPM-Debug] Using fallback training mode (modal_noise_enabled=False)")
        mask_cdr = mask_dict['mask_cdr']
        beta = torch.where(mask_cdr, beta_c.unsqueeze(1).expand(N, L), beta)

        if is_epitope_noise_mode(noise_mode):
            t_e = self._compute_async_epitope_t(t_c, noise_sampling)
            t_dict['epitope'] = t_e
            beta_e = self.trans_pos.var_sched.betas_epi[t_e]
            beta_dict['epitope'] = beta_e

            assert 'mask_soft_antigen' in mask_dict, f"mask_soft_antigen not found in {mask_dict.keys()}"
            assert 'mask_full_antigen' in mask_dict, f"mask_full_antigen not found in {mask_dict.keys()}"
            mask_soft_antigen = mask_dict['mask_soft_antigen']
            beta = torch.where(mask_soft_antigen, beta_e.unsqueeze(1).expand(N, L), beta)
        return t_dict, t, beta_dict, beta

    def build_forward_t_schedule0(self, N, L, noise_sampling, mask_dict):
        raise NotImplementedError(
            'build_forward_t_schedule0() is deprecated and inconsistent with the active diffusion '
            'implementation. Use build_forward_t_schedule() instead.'
        )

    def add_noise_modal(self, v, p, s, mask_dict,
                        t_dict, bfactor=None):
        """
        对 CDR/Epitope 独立加噪，每个模态可有不同噪声水平 t。

        Args:
            v, p, s:            干净的真值 (SO3-vec, position, sequence)。
            mask_dict: 各模态的布尔 mask, (N, L)。
            t_dict:           对应各模态的噪声水平 (N,)，支持 int 或 float tensor。
            bfactor:           (N, L, n_atom) 或 (N, L)，B-factor 值，用于噪声缩放。
        Returns:
            v_noised, p_noised, s_noised: 加噪后的状态。
        """
        v_out = v.clone()
        p_out = p.clone()
        s_out = s.clone()
        eps_p_out = torch.zeros_like(p)
        noise_mode = t_dict['noise_mode']
        if DEBUG_MODE:
            print(f"[add_noise_modal-Debug] === Mask Debug Info ===")
            print(f"[add_noise_modal-Debug] noise_mode: {noise_mode}")
            if 'mask_cdr' in mask_dict:
                cdr_sum = mask_dict['mask_cdr'].sum().item()
                print(
                    f"[add_noise_modal-Debug] mask_cdr: shape={mask_dict['mask_cdr'].shape}, t.shape={t_dict['cdr'].shape}, sum={cdr_sum}, source=parameter")
            else:
                print(f"[add_noise_modal-Debug] mask_cdr: None (ERROR!)")
            if 'mask_soft_antigen' in mask_dict:
                epi_sum = mask_dict['mask_soft_antigen'].sum().item()
                print(
                    f"[add_noise_modal-Debug] mask_soft_antigen: shape={mask_dict['mask_soft_antigen'].shape},t.shape={t_dict['epitope'].shape}, sum={epi_sum}, source=parameter")
            else:
                print(f"[add_noise_modal-Debug] mask_soft_antigen: None (ERROR!)")
            print(f"[add_noise_modal-Debug] ========================")

        noise_regions = []
        if noise_mode == 'cdr_only':
            assert 'mask_cdr' in mask_dict, f"mask_cdr not in mask_dict({mask_dict.keys()})"
            noise_regions = [('cdr', mask_dict['mask_cdr'], t_dict['cdr'])]
            if DEBUG_MODE:
                print(f"[add_noise_modal-Debug] noise_mode=cdr_only -> noise_regions=[(mask_cdr, t_c)]")
        elif noise_mode == 'cdr_epitope':
            noise_regions = [('cdr', mask_dict['mask_cdr'], t_dict['cdr']),
                             ('epitope', mask_dict['mask_soft_antigen'], t_dict['epitope'])]
            if DEBUG_MODE:
                print(
                    f"[add_noise_modal-Debug] noise_mode={noise_mode} -> noise_regions=[(mask_soft_antigen, t_e), (mask_cdr, t_c)]")
        elif noise_mode == 'cdr_fr':
            noise_regions = [('cdr', mask_dict['mask_cdr'], t_dict['cdr']), ('fr', mask_dict['mask_fr'], t_dict['t_f'])]
            if DEBUG_MODE:
                print(
                    f"[add_noise_modal-Debug] noise_mode={noise_mode} -> noise_regions=[(mask_cdr, t_c), (mask_fr, t_f)]")

        elif noise_mode == 'cdr_fr_epitope':
            noise_regions = [('cdr', mask_dict['mask_cdr'], t_dict['cdr']), ('fr', mask_dict['mask_fr'], t_dict['t_f']),
                             ('epitope', mask_dict['mask_soft_antigen'], t_dict['epitope'])]
            if DEBUG_MODE:
                print(
                    f"[add_noise_modal-Debug] noise_mode={noise_mode} -> noise_regions=[(mask_cdr, t_c), (mask_fr, t_f), (mask_soft_antigen, t_e)]")

        if is_cdr_noise_mode(noise_mode):
            _, s_noised_cdr = self.trans_seq.add_noise(s, mask_dict['mask_cdr'], t_dict['cdr'], region_nm='cdr')
            s_out = torch.where(mask_dict['mask_cdr'], s_noised_cdr, s_out)

        for i, (region_nm, mask, t_curr) in enumerate(noise_regions):
            scale = None
            is_epitope_region = region_nm == 'epitope'
            if is_epitope_region:
                scale = None

            v_noised, _ = self.trans_rot.add_noise(v, mask, t_curr, region_nm, scale=scale)

            p_noised, eps_p = self.trans_pos.add_noise(
                p, mask, t_curr, region_nm,
                scale=scale
            )

            v_out = torch.where(mask[:, :, None].expand_as(v_out), v_noised, v_out)
            p_out = torch.where(mask[:, :, None].expand_as(p_out), p_noised, p_out)
            eps_p_out = torch.where(mask[:, :, None].expand_as(eps_p_out), eps_p, eps_p_out)

        return v_out, p_out, s_out, eps_p_out

    def forward(self, batch, v_0, p_0, s_0, res_feat_ctx, pair_feat_ctx, bfactor, mask_dict,
                t_dict=None):
        """
        前向传播：加噪 -> 网络预测 -> 损失计算
        """
        N, L = res_feat_ctx.shape[:2]
        device = res_feat_ctx.device

        noise_sampling = _get_runtime_noise_sampling(batch)

        if t_dict is None:
            t_dict, t, beta_dict, beta = self.build_forward_t_schedule(N, L, noise_sampling, mask_dict, batch=batch)
            batch['_t_cdr_aux'] = self._batch_timestep_from_map(t_dict['cdr'], mask_dict['mask_cdr'])
        else:
            beta_dict = {}
            beta = torch.zeros(N, L, device=device)

            t_c = t_dict['cdr']
            beta_c = self.trans_pos.var_sched.betas[t_c]
            beta_dict['cdr'] = beta_c
            mask_cdr = mask_dict['mask_cdr']
            if t_c.dim() == 1:
                beta = torch.where(mask_cdr, beta_c.unsqueeze(1).expand(N, L), beta)
            else:
                beta = torch.where(mask_cdr, beta_c, beta)
            batch['_t_cdr_aux'] = self._batch_timestep_from_map(t_c, mask_cdr)

            if is_epitope_noise_mode(t_dict.get('noise_mode', '')):
                t_e = t_dict['epitope']
                beta_e = self.trans_pos.var_sched.betas_epi[t_e]  # (N,)
                beta_dict['epitope'] = beta_e
                mask_soft_antigen = mask_dict['mask_soft_antigen']
                beta = torch.where(mask_soft_antigen, beta_e.unsqueeze(1).expand(N, L), beta)
        p_0 = self._normalize_position(p_0)
        R_0 = so3vec_to_rotation(v_0)
        v_noisy, p_noisy, s_noisy, eps_p = self.add_noise_modal(
            v_0, p_0, s_0,
            mask_dict=mask_dict,
            t_dict=t_dict,
            bfactor=bfactor
        )

        if DEBUG_MODE:
            B = v_0.shape[0]
            mask_struct_generate_local = mask_dict['mask_cdr'].clone()
            if is_epitope_noise_mode(t_dict.get('noise_mode', '')) and 'mask_soft_antigen' in mask_dict and mask_dict['mask_soft_antigen'] is not None:
                mask_struct_generate_local = mask_struct_generate_local | mask_dict['mask_soft_antigen']
            mask_seq_generate_local = mask_dict['mask_cdr'].clone()
            
            print("\n[训练-加噪断言] Batch size={}".format(B))
            
            for b in range(B):
                v_diff = (v_noisy[b] - v_0[b]).abs().sum(dim=-1)
                p_diff = (p_noisy[b] - p_0[b]).abs().sum(dim=-1)
                struct_gen_mask = mask_struct_generate_local[b]
                
                cdr_t = t_dict.get("cdr", None)
                if cdr_t is not None:
                    cdr_t_ref = self._batch_timestep_from_map(cdr_t, mask_dict['mask_cdr'])
                    cdr_t_val = cdr_t_ref[b].item()
                else:
                    cdr_t_val = 0
                
                if struct_gen_mask.any():
                    v_changed_count = (v_diff[struct_gen_mask] > 1e-5).sum().item()
                    p_changed_count = (p_diff[struct_gen_mask] > 1e-5).sum().item()
                    struct_total = struct_gen_mask.sum().item()
                    
                    if cdr_t_val > 5:
                        assert v_changed_count > 0, "[训练] 样本{}: t={}, mask区域v_noisy完全没有变化".format(b, cdr_t_val)
                        assert p_changed_count > 0, "[训练] 样本{}: t={}, mask区域p_noisy完全没有变化".format(b, cdr_t_val)
                    elif cdr_t_val <= 5 and v_changed_count == 0:
                        print(f"  [样本{b}信息] t={cdr_t_val}低噪声, v变化0/{struct_total}, p变化0/{struct_total}（正常）")
                
                if (~struct_gen_mask).any():
                    v_unchanged = (v_diff[~struct_gen_mask] < 1e-5).all()
                    p_unchanged = (p_diff[~struct_gen_mask] < 1e-5).all()
                    assert v_unchanged, "[训练] 样本{}: 非mask区域v_noisy被修改, max={}".format(b, v_diff[~struct_gen_mask].max().item())
                    assert p_unchanged, "[训练] 样本{}: 非mask区域p_noisy被修改, max={}".format(b, p_diff[~struct_gen_mask].max().item())
                
                print("  样本{}: t={}, 结构(v变{}/{},p变{}/{}) ".format(
                    b, cdr_t_val,
                    v_changed_count if struct_gen_mask.any() else 0, struct_total if struct_gen_mask.any() else 0,
                    p_changed_count if struct_gen_mask.any() else 0, struct_total if struct_gen_mask.any() else 0))
                
                s_diff = (s_noisy[b] != s_0[b])
                seq_gen_mask = mask_seq_generate_local[b]
                if seq_gen_mask.any():
                    s_changed_count = s_diff[seq_gen_mask].sum().item()
                    s_total_count = seq_gen_mask.sum().item()
                    s_change_ratio = s_changed_count / (s_total_count + 1e-8)
                    if s_change_ratio < 0.1 and t_dict.get("cdr", None) is not None:
                        cdr_t = self._batch_timestep_from_map(t_dict["cdr"], mask_dict['mask_cdr'])[b].item()
                        cdr_alpha = self.trans_seq.var_sched.alpha_bars[cdr_t].item()
                        if cdr_alpha < 0.99 and s_change_ratio < 0.05:
                            print(f"  [样本{b}警告] alpha_bar={cdr_alpha:.3f}, 但变化率={s_change_ratio:.3f}很低")
                
                if (~seq_gen_mask).any():
                    s_unchanged = (~s_diff[~seq_gen_mask]).all()
                    assert s_unchanged, "[训练] 样本{}: 非mask_cdr区域s_noisy被修改, changed={}".format(b, s_diff[~seq_gen_mask].sum().item())
                
                print("  样本{}: 结构OK(v变{},不变{}), 序列(非mask不变, mask变{}/{})".format(
                    b, struct_gen_mask.sum().item(), (~struct_gen_mask).sum().item(),
                    s_changed_count if seq_gen_mask.any() else 0, seq_gen_mask.sum().item() if seq_gen_mask.any() else 0))
        print("[训练-加噪断言] OK\n")

        self._set_hgacd_contact_positions(batch, p_noisy)
        v_next, R_next, eps_p_pred, s_0_pred_logits, _ = self.eps_net(
            batch, v_noisy, p_noisy, s_noisy,
            res_feat_ctx, pair_feat_ctx, beta,
        )

        if DEBUG_MODE:
            B = v_0.shape[0]
            print("\n[训练-网络输出断言] Batch size={}".format(B))
            
            for b in range(B):
                assert v_next[b].shape == v_0[b].shape, "v_next shape mismatch"
                assert R_next[b].shape == (v_0[b].shape[0], 3, 3), "R_next shape mismatch"
                assert eps_p_pred[b].shape == p_0[b].shape, "eps_p_pred shape mismatch"
                assert s_0_pred_logits[b].shape[:-1] == s_0[b].shape, "s_0_pred_logits shape mismatch"
                
                assert not torch.isnan(v_next[b]).any(), "v_next contains NaN"
                assert not torch.isnan(R_next[b]).any(), "R_next contains NaN"
                assert not torch.isnan(eps_p_pred[b]).any(), "eps_p_pred contains NaN"
                assert not torch.isnan(s_0_pred_logits[b]).any(), "s_0_pred_logits contains NaN"
                
                R_check = torch.matmul(R_next[b].transpose(-2, -1), R_next[b])
                identity = torch.eye(3, device=R_check.device).unsqueeze(0)
                ortho_error = (R_check - identity).abs().max().item()
                assert ortho_error < 1e-3, "R_next not orthogonal, error={}".format(ortho_error)
                
                c_prob = torch.softmax(s_0_pred_logits[b], dim=-1)
                prob_sum = c_prob.sum(dim=-1)
                assert (prob_sum - 1.0).abs().max() < 1e-5, "s_0_pred_logits softmax sum != 1"
                
                print("  样本{}: 形状OK, 数值OK, 旋转正交OK(err={:.6f}), 序列概率OK".format(b, ortho_error))
            
            print("[训练-网络输出断言] OK\n")

        noise_mode = t_dict['noise_mode']

        compute_cdr_loss = False
        compute_epitope_loss = False
        if is_cdr_noise_mode(noise_mode):
            compute_cdr_loss = True

        if is_epitope_noise_mode(noise_mode):
            compute_epitope_loss = True

        mask_soft_antigen = mask_dict.get('mask_soft_antigen', None)
        mask_soft_antigen_loss = mask_soft_antigen
        if is_epitope_noise_mode(noise_mode):
            mask_soft_antigen_loss = self._build_epitope_loss_mask(mask_soft_antigen, t_dict.get('epitope'))
            compute_epitope_loss = mask_soft_antigen_loss is not None and mask_soft_antigen_loss.any()
        if compute_epitope_loss:
            assert mask_soft_antigen_loss is not None and mask_soft_antigen_loss.any(), \
                f"[LossMode] noise_mode={noise_mode} requires non-empty mask_soft_antigen"

        atom_mask = batch.get('mask_heavyatom')  # (N, L, n_atom)
        geometry_calc = GeometryLossCalculator(
            cdr_mask=mask_dict['mask_cdr'],
            mask_soft_antigen=mask_soft_antigen_loss,
            mask_antigen=mask_dict['mask_full_antigen'],
            atom_mask=atom_mask,
        )


        sample_v_next, sample_p_next, sample_s_next = self.denoise_to_get_next_state(v_noisy, v_next, p_noisy, eps_p_pred, noise_mode, mask_dict, t_dict, s_noisy, s_0_pred_logits)


        t_cdr = t_dict['cdr']
        p_clean = self._compute_clean_position_from_eps(
            p_noisy,
            eps_p_pred,
            mask_dict['mask_cdr'],
            t_cdr,
            epitope_mask=mask_soft_antigen_loss if 'epitope' in noise_mode else None,
            t_epitope=t_dict.get('epitope') if 'epitope' in noise_mode else None,
        )

        bb_pos_pred_for_geometry = reconstruct_backbone(
            R=so3vec_to_rotation(v_next),  # 网络输出（不含噪声E）
            t=self._unnormalize_position(p_clean),  # 干净的去噪位置（不含随机噪声z）
            aa=s_0,
            chain_nb=batch['chain_nb'],
            res_nb=batch['res_nb'],
            mask=batch['mask']  # (N, L) 标记哪些AA是padding的
        )
        bb_pos_pred = reconstruct_backbone(
            R=so3vec_to_rotation(sample_v_next),
            t=self._unnormalize_position(sample_p_next),
            aa=s_0,
            chain_nb=batch['chain_nb'],
            res_nb=batch['res_nb'],
            mask=batch['mask']  # (N, L) 标记哪些AA是padding的
        )
        t_seq = t_dict['cdr']
        post_true = self.trans_seq.posterior(s_noisy, s_0, t_seq, region_nm='cdr')  # (N, L, 20)

        s_0_pred_prob = F.softmax(s_0_pred_logits, dim=-1)  # (N, L, 20)
        post_pred = self.trans_seq.posterior(s_noisy, s_0_pred_prob, t_seq, region_nm='cdr')  # (N, L, 20)
        if DEBUG_WEIGHTS and torch.rand(1).item() < 0.01:
            debug_log(
                f"[CDR Seq] t_seq range: max={t_seq.max().item()}, max={t_seq.max().item()}, mean={t_seq.float().mean().item():.1f}")

        seq_adjacent_mask_dict = {
            'cdr': mask_dict['mask_cdr'][:, :-1] & mask_dict['mask_cdr'][:, 1:],
            'epitope': mask_soft_antigen_loss[:, :-1] & mask_soft_antigen_loss[:, 1:] if mask_soft_antigen_loss is not None else torch.zeros_like(mask_dict['mask_cdr'][:, :-1]),
        }

        bb_pos_true_for_analysis = reconstruct_backbone(
            R=R_0,  # 真实旋转矩阵
            t=self._unnormalize_position(p_0),  # 真实位置（物理空间）
            aa=s_0,  # 真实序列
            chain_nb=batch['chain_nb'],
            res_nb=batch['res_nb'],
            mask=batch['mask']
        )

        loss_dict_raw = geometry_calc.compute_losses(
            R_next=R_next, R_0=R_0,
            eps_p_pred=eps_p_pred, eps_p=eps_p,
            post_pred=post_pred, post_true=post_true,
            s_0_pred_logits=s_0_pred_logits, s_0=s_0,
            bb_pos_pred=bb_pos_pred_for_geometry,  # 使用不含噪声E的骨架
            seq_adjacent_mask_dict=seq_adjacent_mask_dict,
            chain_nb=batch['chain_nb'],
            res_nb=batch['res_nb'],
            mask=batch['mask'],
            compute_cdr_loss=compute_cdr_loss,
            compute_epitope_loss=compute_epitope_loss,  # 是否计算epitope损失
            bfactor=bfactor,  # 传递 B-factor 用于表位损失加权
            bb_pos_true=bb_pos_true_for_analysis,  # 新增：真实骨架位置用于详细分析
        )

        if DEBUG_MODE:
            print("\n[训练-Loss区域断言] 验证loss在生成区域内")
            if compute_cdr_loss:
                cdr_res_count = mask_dict['mask_cdr'].sum().item()
                cdr_rot_raw = loss_dict_raw.get('cdr_rot_mean', loss_dict_raw.get('cdr_rot'))
                if isinstance(cdr_rot_raw, torch.Tensor) and cdr_rot_raw.item() > 0:
                    assert cdr_res_count > 0, "[Loss断言] cdr_rot > 0 但 mask_cdr 为空"
                cdr_pos_raw = loss_dict_raw.get('cdr_pos_mean', loss_dict_raw.get('cdr_pos'))
                if isinstance(cdr_pos_raw, torch.Tensor) and cdr_pos_raw.item() > 0:
                    assert cdr_res_count > 0, "[Loss断言] cdr_pos > 0 但 mask_cdr 为空"
                print(f"  CDR: res_count={cdr_res_count}, rot_loss={cdr_rot_raw.item() if isinstance(cdr_rot_raw, torch.Tensor) else 0:.4f}")

            if compute_epitope_loss and mask_dict.get('mask_soft_antigen') is not None:
                epi_res_count = mask_dict['mask_soft_antigen'].sum().item()
                epi_pos_raw = loss_dict_raw.get('epitope_pos_mean', loss_dict_raw.get('epitope_pos'))
                if isinstance(epi_pos_raw, torch.Tensor) and epi_pos_raw.item() > 0:
                    assert epi_res_count > 0, "[Loss断言] epitope_pos > 0 但 mask_soft_antigen 为空"
                print(f"  Epitope: res_count={epi_res_count}, pos_loss={epi_pos_raw.item() if isinstance(epi_pos_raw, torch.Tensor) else 0:.4f}")

            cdr_bone_raw = loss_dict_raw.get('cdr_bone_mean', loss_dict_raw.get('cdr_bone'))
            if isinstance(cdr_bone_raw, torch.Tensor) and cdr_bone_raw.item() > 0:
                print(f"  Bond: loss={cdr_bone_raw.item():.4f}")

            contact_raw = loss_dict_raw.get('contact_mean', loss_dict_raw.get('contact'))
            if isinstance(contact_raw, torch.Tensor) and contact_raw.item() > 0:
                assert cdr_res_count > 0, "[Loss断言] contact > 0 但 CDR 为空"
                if compute_epitope_loss:
                    assert epi_res_count > 0, "[Loss断言] contact > 0 但 Epitope 为空"
                print(f"  Contact: loss={contact_raw.item():.4f}, requires both CDR and Epitope")

            print("[训练-Loss区域断言] OK\n")
        if hasattr(self.eps_net, '_region_aux_loss_dict') and self.eps_net._region_aux_loss_dict:
            for key, value in self.eps_net._region_aux_loss_dict.items():
                loss_dict_raw[key] = value
                loss_dict_raw[f'{key}_mean'] = value.mean() if isinstance(value, torch.Tensor) and value.dim() > 0 else value
            for key, value in getattr(self.eps_net, '_region_aux_monitor_dict', {}).items():
                loss_dict_raw[f'{key}_mean'] = value.mean() if isinstance(value, torch.Tensor) and value.dim() > 0 else value


        process = self._masked_process_scalar(t_dict['cdr'], mask_dict['mask_cdr'])

        process_norm = process.float() / self.num_steps


        process_norm = process.float() / self.num_steps

        scheduler = DynamicLossWeightScheduler(eta=0.5, override_weights=self.loss_weights if self.loss_weights else None)

        loss_weights = scheduler.compute_weights(process_norm)

        core_losses = CORE_LOSSES + [key for key in getattr(self.eps_net, '_region_aux_loss_dict', {}).keys() if key not in CORE_LOSSES]  # 使用constants.py中的定义
        loss_dict, weight_info, loss_dict_raw_returned = apply_dynamic_weights(
            loss_dict_raw=loss_dict_raw,
            weights_dict=loss_weights,  # 使用配置权重
            core_losses=core_losses,
            geom_scaler=self.geom_scaler if hasattr(self, 'geom_scaler') else None,
            t=process_norm,  # 传递timestep用于门控
            use_geom_scaler=False  # cdr_bone/cdr_omega 仅监控，不参与训练目标
        )

        if hasattr(self.eps_net, '_region_aux_loss_dict') and self.eps_net._region_aux_loss_dict:
            for key, value in self.eps_net._region_aux_loss_dict.items():
                if key in loss_dict:
                    continue
                raw_mean = value.mean() if isinstance(value, torch.Tensor) and value.dim() > 0 else value
                base_weight = 1.0
                if self.loss_weights and key in self.loss_weights:
                    base_weight = float(self.loss_weights[key])
                weighted = raw_mean * base_weight
                loss_dict[key] = weighted
                if isinstance(weighted, torch.Tensor):
                    weight_info[key] = torch.tensor(base_weight, device=weighted.device, dtype=weighted.dtype)
                else:
                    weight_info[key] = base_weight
                loss_dict['overall'] = loss_dict['overall'] + weighted
                loss_dict_raw_returned[key] = value
                loss_dict_raw_returned[f'{key}_mean'] = raw_mean
            if 'overall' in loss_dict_raw_returned:
                for key, value in self.eps_net._region_aux_loss_dict.items():
                    raw_mean = value.mean() if isinstance(value, torch.Tensor) and value.dim() > 0 else value
                    loss_dict_raw_returned['overall'] = loss_dict_raw_returned['overall'] + raw_mean

        if self.training and DEBUG_GRADIENT:
            for key in core_losses:
                if key in loss_dict and isinstance(loss_dict[key], torch.Tensor):
                    val = loss_dict[key]
                    if val.item() > 0.001:
                        assert val.requires_grad, f"[梯度断言] {key} 值={val.item():.4f} 但无梯度"
                        print(f"[梯度验证] {key}: val={val.item():.4f}, requires_grad={val.requires_grad}")
                    elif not val.requires_grad:
                        raw_val = loss_dict_raw_returned.get(key + '_mean')
                        if raw_val is not None and isinstance(raw_val, torch.Tensor):
                            print(f"[WARNING] {key}: val={val.item():.4f}, raw.requires_grad={raw_val.requires_grad}")

            global _loss_compute_counter
            _loss_compute_counter += 1
            if _loss_compute_counter % 100 == 0:
                print("\n[梯度范围监控] 检查关键模块梯度范围")
                for name, param in self.eps_net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 100:
                            print(f"[WARNING] {name}: grad_norm={grad_norm:.4f} > 100 (可能梯度爆炸)")
                        elif grad_norm < 1e-7 and param.requires_grad:
                            print(f"[WARNING] {name}: grad_norm={grad_norm:.8f} < 1e-7 (可能梯度消失)")
                        else:
                            print(f"[梯度监控] {name}: grad_norm={grad_norm:.4f}")

        if hasattr(self.eps_net, '_kl_loss_dict') and self.eps_net._kl_loss_dict:
            kl_dict = self.eps_net._kl_loss_dict
            kl_weight = 0.01
            total_kl = torch.stack([v.mean() for v in kl_dict.values()]).sum()
            kl_loss = kl_weight * total_kl
            loss_dict['kl_loss'] = kl_loss
            loss_dict['overall'] = loss_dict['overall'] + kl_loss if loss_dict.get('overall') is not None else kl_loss
            loss_dict_raw_returned['kl_loss_mean'] = total_kl
            if 'overall' in loss_dict_raw_returned:
                loss_dict_raw_returned['overall'] = loss_dict_raw_returned['overall'] + total_kl
            weight_info['kl_loss'] = kl_weight

        return loss_dict, weight_info, loss_dict_raw_returned

    def denoise_to_get_next_state(self, v_t, v_next, p_t, eps_p, noise_mode, mask_dict, t_dict, s_t=None, s_0_pred_logits=None):
        mask_cdr = mask_dict['mask_cdr']
        v_pred = v_next
        v_next = self.trans_rot.denoise(v_t, v_pred, mask_cdr, t_dict['cdr'], region_nm='cdr')
        p_next = self.trans_pos.denoise(p_t, eps_p, mask_cdr, t_dict['cdr'], region_nm='cdr')

        if 'epitope' in noise_mode and mask_dict.get('mask_soft_antigen') is not None:
            mask_soft_antigen = mask_dict['mask_soft_antigen']
            if mask_soft_antigen.any():
                v_epi = self.trans_rot.denoise(v_t, v_pred, mask_soft_antigen, t_dict['epitope'],
                                               region_nm='epitope')
                p_epi = self.trans_pos.denoise(p_t, eps_p, mask_soft_antigen, t_dict['epitope'],
                                               region_nm='epitope')

                v_next = torch.where(mask_soft_antigen[..., None], v_epi, v_next)
                p_next = torch.where(mask_soft_antigen[..., None], p_epi, p_next)

        s_next = None
        if s_0_pred_logits is not None and s_t is not None:
            s_0_pred_logits_prob = F.softmax(s_0_pred_logits, dim=-1)

            _, s_next = self.trans_seq.denoise(s_t, s_0_pred_logits_prob, mask_cdr, t_dict['cdr'],
                                               region_nm='cdr')
        return v_next, p_next, s_next


    def _build_anchor_support_prior(self, batch, mask_cdr, mask_res):
        """Geometry-free condition support prior for generated CDR slots."""
        device = mask_cdr.device
        B, L = mask_cdr.shape
        support = torch.zeros(B, L, device=device, dtype=torch.float32)
        if 'res_nb' not in batch or 'chain_nb' not in batch:
            return torch.where(mask_cdr, torch.full_like(support, 0.5), support)
        res_nb = batch['res_nb'].to(device)
        chain_nb = batch['chain_nb'].to(device)
        valid_context = mask_res.to(torch.bool) & (~mask_cdr.to(torch.bool))
        for b in range(B):
            cdr_idx = torch.nonzero(mask_cdr[b], as_tuple=False).flatten()
            if cdr_idx.numel() == 0:
                continue
            ctx_idx = torch.nonzero(valid_context[b], as_tuple=False).flatten()
            if ctx_idx.numel() == 0:
                support[b, cdr_idx] = 0.5
                continue
            same_chain = chain_nb[b, cdr_idx][:, None] == chain_nb[b, ctx_idx][None, :]
            dist = torch.abs(res_nb[b, cdr_idx][:, None].float() - res_nb[b, ctx_idx][None, :].float())
            dist = torch.where(same_chain, dist, torch.full_like(dist, 1e4))
            min_dist = dist.min(dim=1).values
            prior = torch.exp(-min_dist / 4.0).clamp(0.0, 1.0)
            support[b, cdr_idx] = prior
        return support

    def _select_asyndm_slow_mask(self, s_0_pred_logits, mask_cdr, anchor_support, cfg):
        """Select condition-sensitive residues for the slow AsynDM branch."""
        probs = F.softmax(s_0_pred_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))
        support_weight = float(cfg.get('support_weight', 0.35))
        score = entropy + support_weight * (1.0 - anchor_support)
        score = torch.where(mask_cdr, score, torch.full_like(score, -1e4))
        slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
        frac = float(cfg.get('slow_fraction', 0.50))
        frac = max(0.05, min(frac, 0.95))
        for b in range(mask_cdr.size(0)):
            idx = torch.nonzero(mask_cdr[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            k = max(1, int(round(float(idx.numel()) * frac)))
            vals = score[b, idx]
            selected = torch.topk(vals, k=k, largest=True).indices
            slow[b, idx[selected]] = True
        return slow, entropy

    def _get_hgacd_async_state(self):
        encoder = getattr(self.eps_net, 'encoder', None)
        getter = getattr(encoder, 'get_hierarchical_async_state', None)
        if getter is None:
            return None
        return getter()

    def _select_hgacd_region_slow_mask(self, batch, s_0_pred_logits, mask_cdr, anchor_support, cfg):
        """Region-governed slow mask with residue-level uncertainty as a secondary signal."""
        probs = F.softmax(s_0_pred_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))

        region_type = batch['region_type'].to(mask_cdr.device)
        state = self._get_hgacd_async_state()
        contact_mass = None if state is None else state.get('region_contact_mass')
        contact_hit = None if state is None else state.get('region_contact_hit')
        contact_prob = None if state is None else state.get('region_contact_prob')
        if contact_mass is not None:
            contact_mass = contact_mass.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_hit is not None:
            contact_hit = contact_hit.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_prob is not None:
            contact_prob = contact_prob.to(mask_cdr.device).float().clamp(0.0, 1.0)

        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        support_weight = float(cfg.get('support_weight', 0.50))
        contact_weight = float(cfg.get('contact_weight', 0.50))
        slow_fraction = max(0.05, min(float(cfg.get('slow_region_fraction', cfg.get('slow_fraction', 0.50))), 0.95))
        slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
        region_support_res = torch.zeros_like(anchor_support)

        for b in range(mask_cdr.size(0)):
            region_scores = []
            region_masks = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid)
                if not bool(rmask.any().item()):
                    continue
                region_idx = rid - 1
                anchor_score = anchor_support[b, rmask].mean()
                contact_score = anchor_score.new_tensor(0.0)
                if contact_prob is not None and region_idx < contact_prob.size(1):
                    contact_score = contact_prob[b, region_idx]
                elif contact_mass is not None and region_idx < contact_mass.size(1):
                    contact_score = contact_mass[b, region_idx]
                    if contact_hit is not None and region_idx < contact_hit.size(1):
                        contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx]
                elif contact_hit is not None and region_idx < contact_hit.size(1):
                    contact_score = contact_hit[b, region_idx]
                if bool(cfg.get('blend_contact_hit_with_prediction', False)) and contact_prob is not None and contact_hit is not None and region_idx < contact_hit.size(1):
                    contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx]
                support = (1.0 - contact_weight) * anchor_score + contact_weight * contact_score
                support = support.clamp(0.0, 1.0)
                region_support_res[b, rmask] = support
                uncertainty = entropy[b, rmask].mean()
                region_scores.append(uncertainty + support_weight * (1.0 - support))
                region_masks.append(rmask)
            if not region_scores:
                continue
            scores = torch.stack(region_scores)
            k = max(1, int(round(float(len(region_scores)) * slow_fraction)))
            selected = torch.topk(scores, k=k, largest=True).indices
            for idx in selected.tolist():
                slow[b] = slow[b] | region_masks[idx]
        return slow, entropy, region_support_res

    def _select_hgacd_rule_based_slow_mask(self, batch, s_0_pred_logits, mask_cdr, anchor_support, cfg, t_res=None):
        """Select slow regions by relative rule instead of fixed top-k fraction."""
        probs = F.softmax(s_0_pred_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))

        region_type = batch['region_type'].to(mask_cdr.device)
        state = self._get_hgacd_async_state()
        contact_mass = None if state is None else state.get('region_contact_mass')
        contact_hit = None if state is None else state.get('region_contact_hit')
        contact_prob = None if state is None else state.get('region_contact_prob')
        if contact_mass is not None:
            contact_mass = contact_mass.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_hit is not None:
            contact_hit = contact_hit.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_prob is not None:
            contact_prob = contact_prob.to(mask_cdr.device).float().clamp(0.0, 1.0)

        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
        region_support_res = torch.zeros_like(anchor_support)

        for b in range(mask_cdr.size(0)):
            region_scores = []
            region_masks = []
            region_clocks = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid)
                if not bool(rmask.any().item()):
                    continue
                region_idx = rid - 1
                anchor_score = anchor_support[b, rmask].mean()
                contact_score = None
                if contact_prob is not None and region_idx < contact_prob.size(1):
                    contact_score = contact_prob[b, region_idx]
                elif contact_mass is not None and region_idx < contact_mass.size(1):
                    contact_score = contact_mass[b, region_idx]
                    if contact_hit is not None and region_idx < contact_hit.size(1):
                        contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx]
                elif contact_hit is not None and region_idx < contact_hit.size(1):
                    contact_score = contact_hit[b, region_idx]

                support_terms = [anchor_score]
                if contact_score is not None:
                    support_terms.append(contact_score)
                support = torch.stack(support_terms).mean().clamp(0.0, 1.0)
                region_support_res[b, rmask] = support

                uncertainty = entropy[b, rmask].mean()
                region_scores.append(uncertainty + (1.0 - support))
                region_masks.append(rmask)
                if t_res is not None:
                    region_clocks.append(int(t_res[b, rmask].max().item()))

            if not region_scores:
                continue
            scores = torch.stack(region_scores)
            selected = scores >= scores.median()
            if not bool(selected.any().item()):
                selected = torch.zeros_like(scores, dtype=torch.bool)
                selected[scores.argmax()] = True
            elif int(selected.sum().item()) == scores.numel() and region_clocks:
                max_clock = max(region_clocks)
                selected = torch.tensor([clock == max_clock for clock in region_clocks], device=scores.device, dtype=torch.bool)

            for idx in torch.nonzero(selected, as_tuple=False).flatten().tolist():
                slow[b] = slow[b] | region_masks[idx]
        return slow, entropy, region_support_res

    def _sample_hgacd_trainmatch_region_offsets(self, batch, mask_cdr, noise_sampling):
        """Sample per-region reverse delays from training-time region offsets."""
        cfg = noise_sampling.get('region_async_training', {}) if isinstance(noise_sampling, dict) else {}
        offsets = torch.zeros(mask_cdr.shape, dtype=torch.long, device=mask_cdr.device)
        region_type = batch['region_type'].to(mask_cdr.device)
        max_lag = int(cfg.get('region_max_lag', cfg.get('max_lag', 8)))
        max_lag = max(0, max_lag)
        if max_lag <= 0 or region_type is None:
            return offsets
        prob = float(cfg.get('prob', 1.0))
        half_window = max(1, max_lag // 2)
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        for b in range(mask_cdr.size(0)):
            if prob < 1.0 and torch.rand((), device=mask_cdr.device) > prob:
                continue
            present = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid)
                if bool(rmask.any().item()):
                    present.append(rmask)
            if len(present) <= 1:
                continue
            sampled = []
            for rmask in present:
                raw_offset = torch.randint(
                    low=-half_window,
                    high=half_window + 1,
                    size=(),
                    device=mask_cdr.device,
                    dtype=torch.long,
                )
                sampled.append((rmask, raw_offset))
            min_offset = torch.stack([x[1] for x in sampled]).min()
            for rmask, raw_offset in sampled:
                offsets[b, rmask] = raw_offset - min_offset
        return offsets

    def _build_hgacd_graph_guided_trainmatch_offsets(self, batch, mask_cdr, region_type, base_offsets,
                                                      entropy, region_support, cfg, t_res=None):
        """Permute training-sampled delays using current region-graph scores.

        This keeps the per-sample delay multiset exactly equal to the
        trainmatch sample. The graph only decides which unfinished CDR receives
        which sampled delay; it never creates a new timestep combination.
        """
        if base_offsets is None:
            return torch.zeros_like(mask_cdr, dtype=torch.long)
        support_weight = float(cfg.get('support_weight', 0.50))
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        effective_offsets = torch.zeros_like(base_offsets, dtype=torch.long)

        for b in range(mask_cdr.size(0)):
            region_items = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid)
                if t_res is not None:
                    rmask = rmask & (t_res[b] > 0)
                if not bool(rmask.any().item()):
                    continue
                base_delay = int(base_offsets[b, rmask].max().item())
                region_score = float(entropy[b, rmask].mean().detach().cpu().item())
                if region_support is not None:
                    support = float(region_support[b, rmask].mean().detach().cpu().item())
                    region_score += support_weight * (1.0 - support)
                region_items.append((rmask, base_delay, region_score))

            if not region_items:
                continue
            if max(item[1] for item in region_items) <= 0:
                continue

            delay_values = sorted([item[1] for item in region_items])
            ranked_regions = sorted(region_items, key=lambda item: item[2])
            for idx, (rmask, _, _) in enumerate(ranked_regions):
                effective_offsets[b, rmask] = delay_values[idx]

        return effective_offsets

    def _select_hgacd_window_adaptive_active_mask(self, batch, s_0_pred_logits, mask_cdr,
                                                  anchor_support, cfg, t_res):
        """Pick active CDR regions adaptively while staying inside the training lag window.

        The controller starts from all CDR clocks at T and only allows t -> t-1
        moves whose post-update unfinished clocks have pairwise lag <=
        region_max_lag. Region-graph/contact/entropy scores decide which valid
        regions are held back at each step.
        """
        probs = F.softmax(s_0_pred_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))

        region_type = batch['region_type'].to(mask_cdr.device)
        state = self._get_hgacd_async_state()
        contact_mass = None if state is None else state.get('region_contact_mass')
        contact_hit = None if state is None else state.get('region_contact_hit')
        contact_prob = None if state is None else state.get('region_contact_prob')
        if contact_mass is not None:
            contact_mass = contact_mass.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_hit is not None:
            contact_hit = contact_hit.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_prob is not None:
            contact_prob = contact_prob.to(mask_cdr.device).float().clamp(0.0, 1.0)

        max_lag = int(cfg.get('region_max_lag', cfg.get('max_lag', 8)))
        max_lag = max(0, max_lag)
        min_active_fraction = float(cfg.get('window_min_active_fraction', 0.50))
        min_active_fraction = max(0.0, min(min_active_fraction, 1.0))
        support_weight = float(cfg.get('support_weight', 0.50))
        contact_weight = float(cfg.get('contact_weight', 0.50))
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]

        active = torch.zeros_like(mask_cdr, dtype=torch.bool)
        slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
        region_support_res = torch.zeros_like(anchor_support)

        for b in range(mask_cdr.size(0)):
            regions = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid) & (t_res[b] > 0)
                if not bool(rmask.any().item()):
                    continue
                region_idx = rid - 1
                anchor_score = anchor_support[b, rmask].mean()
                contact_score = anchor_score.new_tensor(0.0)
                has_contact = False
                if contact_prob is not None and region_idx < contact_prob.size(1):
                    contact_score = contact_prob[b, region_idx]
                    has_contact = True
                elif contact_mass is not None and region_idx < contact_mass.size(1):
                    contact_score = contact_mass[b, region_idx]
                    if contact_hit is not None and region_idx < contact_hit.size(1):
                        contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx]
                    has_contact = True
                elif contact_hit is not None and region_idx < contact_hit.size(1):
                    contact_score = contact_hit[b, region_idx]
                    has_contact = True

                if has_contact:
                    support = ((1.0 - contact_weight) * anchor_score + contact_weight * contact_score).clamp(0.0, 1.0)
                else:
                    support = anchor_score.clamp(0.0, 1.0)
                region_support_res[b, rmask] = support

                score = entropy[b, rmask].mean() + support_weight * (1.0 - support)
                clock = int(t_res[b, rmask].max().item())
                regions.append({'mask': rmask, 'score': float(score.detach().cpu().item()), 'clock': clock})

            if not regions:
                continue
            if len(regions) == 1:
                active[b] = active[b] | regions[0]['mask']
                continue

            target_active = max(1, int(round(len(regions) * min_active_fraction)))
            selected = [False for _ in regions]

            min_clock = min(item['clock'] for item in regions)
            for idx, item in enumerate(regions):
                if item['clock'] > min_clock + max_lag:
                    selected[idx] = True

            order = sorted(range(len(regions)), key=lambda idx: (regions[idx]['score'], -regions[idx]['clock']))
            for idx in order:
                if selected[idx]:
                    continue
                trial = selected[:]
                trial[idx] = True
                trial_clocks = [max(0, item['clock'] - (1 if trial[j] else 0)) for j, item in enumerate(regions)]
                unfinished = [clock for clock in trial_clocks if clock > 0]
                if unfinished and (max(unfinished) - min(unfinished) > max_lag):
                    continue
                selected[idx] = True
                if sum(selected) >= target_active:
                    break

            if not any(selected):
                max_clock = max(item['clock'] for item in regions)
                for idx, item in enumerate(regions):
                    if item['clock'] == max_clock:
                        selected[idx] = True
                        break

            for idx, item in enumerate(regions):
                if selected[idx]:
                    active[b] = active[b] | item['mask']
                else:
                    slow[b] = slow[b] | item['mask']

        return active, slow, entropy, region_support_res

    def _select_hgacd_window_adaptive_next_active(self, batch, s_0_pred_logits, mask_cdr,
                                                   anchor_support, cfg, t_res):
        active, slow, entropy, region_support_res = self._select_hgacd_window_adaptive_active_mask(
            batch, s_0_pred_logits, mask_cdr, anchor_support, cfg, t_res
        )
        return active, slow, entropy, region_support_res

    def _select_hgacd_bioprior_window_active_mask(self, batch, s_0_pred_logits, mask_cdr,
                                                  anchor_support, cfg, t_res):
        """Biological-prior window controller for interface-preserving CDR sampling.

        Uses the same lag-window constraint, then ranks valid updates with a
        soft CDR3/interface prior.
        """
        probs = F.softmax(s_0_pred_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))

        region_type = batch['region_type'].to(mask_cdr.device)
        state = self._get_hgacd_async_state()
        contact_mass = None if state is None else state.get('region_contact_mass')
        contact_hit = None if state is None else state.get('region_contact_hit')
        contact_prob = None if state is None else state.get('region_contact_prob')
        if contact_mass is not None:
            contact_mass = contact_mass.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_hit is not None:
            contact_hit = contact_hit.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_prob is not None:
            contact_prob = contact_prob.to(mask_cdr.device).float().clamp(0.0, 1.0)

        max_lag = int(cfg.get('region_max_lag', cfg.get('max_lag', 8)))
        max_lag = max(0, max_lag)
        min_active_fraction = float(cfg.get('window_min_active_fraction', 0.50))
        min_active_fraction = max(0.0, min(min_active_fraction, 1.0))
        support_weight = float(cfg.get('support_weight', 0.50))
        contact_weight = float(cfg.get('contact_weight', 0.50))
        bio_prior_weight = float(cfg.get('bio_prior_weight', cfg.get('cdr3_prior_weight', 0.20)))
        contact_priority_weight = float(cfg.get('contact_priority_weight', 0.10))
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        bio_prior = {
            int(CDR.H1): 0.35,
            int(CDR.H2): 0.65,
            int(CDR.H3): 1.00,
            int(CDR.L1): 0.35,
            int(CDR.L2): 0.20,
            int(CDR.L3): 0.70,
        }

        active = torch.zeros_like(mask_cdr, dtype=torch.bool)
        slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
        region_support_res = torch.zeros_like(anchor_support)

        for b in range(mask_cdr.size(0)):
            regions = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid) & (t_res[b] > 0)
                if not bool(rmask.any().item()):
                    continue
                region_idx = rid - 1
                anchor_score = anchor_support[b, rmask].mean()
                contact_score = anchor_score.new_tensor(0.0)
                has_contact = False
                if contact_prob is not None and region_idx < contact_prob.size(1):
                    contact_score = contact_prob[b, region_idx]
                    has_contact = True
                elif contact_mass is not None and region_idx < contact_mass.size(1):
                    contact_score = contact_mass[b, region_idx]
                    if contact_hit is not None and region_idx < contact_hit.size(1):
                        contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx]
                    has_contact = True
                elif contact_hit is not None and region_idx < contact_hit.size(1):
                    contact_score = contact_hit[b, region_idx]
                    has_contact = True

                if has_contact:
                    support = ((1.0 - contact_weight) * anchor_score + contact_weight * contact_score).clamp(0.0, 1.0)
                else:
                    support = anchor_score.clamp(0.0, 1.0)
                region_support_res[b, rmask] = support

                base_score = entropy[b, rmask].mean() + support_weight * (1.0 - support)
                prior = anchor_score.new_tensor(float(bio_prior.get(rid, 0.0)))
                if has_contact:
                    prior_gate = 0.5 + 0.5 * contact_score
                    contact_bonus = contact_priority_weight * contact_score
                else:
                    prior_gate = anchor_score.new_tensor(0.5)
                    contact_bonus = anchor_score.new_tensor(0.0)
                score = base_score - bio_prior_weight * prior * prior_gate - contact_bonus
                clock = int(t_res[b, rmask].max().item())
                regions.append({'mask': rmask, 'score': float(score.detach().cpu().item()), 'clock': clock})

            if not regions:
                continue
            if len(regions) == 1:
                active[b] = active[b] | regions[0]['mask']
                continue

            target_active = max(1, int(round(len(regions) * min_active_fraction)))
            selected = [False for _ in regions]

            min_clock = min(item['clock'] for item in regions)
            for idx, item in enumerate(regions):
                if item['clock'] > min_clock + max_lag:
                    selected[idx] = True

            order = sorted(range(len(regions)), key=lambda idx: (regions[idx]['score'], -regions[idx]['clock']))
            for idx in order:
                if selected[idx]:
                    continue
                trial = selected[:]
                trial[idx] = True
                trial_clocks = [max(0, item['clock'] - (1 if trial[j] else 0)) for j, item in enumerate(regions)]
                unfinished = [clock for clock in trial_clocks if clock > 0]
                if unfinished and (max(unfinished) - min(unfinished) > max_lag):
                    continue
                selected[idx] = True
                if sum(selected) >= target_active:
                    break

            if not any(selected):
                max_clock = max(item['clock'] for item in regions)
                for idx, item in enumerate(regions):
                    if item['clock'] == max_clock:
                        selected[idx] = True
                        break

            for idx, item in enumerate(regions):
                if selected[idx]:
                    active[b] = active[b] | item['mask']
                else:
                    slow[b] = slow[b] | item['mask']

        return active, slow, entropy, region_support_res

    def _select_hgacd_bioprior_window_next_active(self, batch, s_0_pred_logits, mask_cdr,
                                                   anchor_support, cfg, t_res):
        active, slow, entropy, region_support_res = self._select_hgacd_bioprior_window_active_mask(
            batch, s_0_pred_logits, mask_cdr, anchor_support, cfg, t_res
        )
        return active, slow, entropy, region_support_res

    def _select_hgacd_dynamic_schedule_active_mask(self, batch, s_0_pred_logits, mask_cdr,
                                                   anchor_support, cfg, t_res, step_idx):
        """Allocate an AsynDM-style denoising schedule to each CDR.

        This controller keeps the checkpoint fixed and changes only inference
        clocks. Each unfinished CDR receives a power-law target schedule
        progress_i(u)=u^gamma_i. Functionally important CDRs get gamma<1 and
        therefore advance earlier; less supported CDRs get gamma>1. Updates
        are still filtered by the training-time region_max_lag window.
        """
        probs = F.softmax(s_0_pred_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))

        region_type = batch['region_type'].to(mask_cdr.device)
        state = self._get_hgacd_async_state()
        contact_mass = None if state is None else state.get('region_contact_mass')
        contact_hit = None if state is None else state.get('region_contact_hit')
        contact_prob = None if state is None else state.get('region_contact_prob')
        if contact_mass is not None:
            contact_mass = contact_mass.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_hit is not None:
            contact_hit = contact_hit.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_prob is not None:
            contact_prob = contact_prob.to(mask_cdr.device).float().clamp(0.0, 1.0)

        max_lag = max(0, int(cfg.get('region_max_lag', cfg.get('max_lag', 8))))
        schedule_strength = max(0.0, float(cfg.get('dynamic_schedule_strength', cfg.get('schedule_strength', 0.75))))
        prior_weight = max(0.0, float(cfg.get('bio_prior_weight', cfg.get('cdr3_prior_weight', 0.35))))
        contact_weight = max(0.0, float(cfg.get('contact_weight', 0.45)))
        anchor_weight = max(0.0, float(cfg.get('anchor_weight', cfg.get('support_weight', 0.20))))
        entropy_weight = max(0.0, float(cfg.get('entropy_weight', 0.0)))
        min_gamma = max(0.05, float(cfg.get('dynamic_schedule_min_gamma', 0.35)))
        max_gamma = max(min_gamma, float(cfg.get('dynamic_schedule_max_gamma', 2.25)))
        final_sync_clock = max(1, int(cfg.get('dynamic_schedule_final_sync_clock', max(1, max_lag))))
        priority_mode = str(cfg.get('dynamic_schedule_priority_mode', 'function_first')).lower()
        u = float(step_idx + 1) / float(max(1, self.num_steps))
        u = min(1.0, max(1e-4, u))

        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        cdr_names = {
            int(CDR.H1): 'HCDR1', int(CDR.H2): 'HCDR2', int(CDR.H3): 'HCDR3',
            int(CDR.L1): 'LCDR1', int(CDR.L2): 'LCDR2', int(CDR.L3): 'LCDR3',
        }
        bio_prior = {
            int(CDR.H1): 0.35,
            int(CDR.H2): 0.65,
            int(CDR.H3): 1.00,
            int(CDR.L1): 0.35,
            int(CDR.L2): 0.20,
            int(CDR.L3): 0.70,
        }

        active = torch.zeros_like(mask_cdr, dtype=torch.bool)
        slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
        region_support_res = torch.zeros_like(anchor_support)
        region_meta = [[] for _ in range(mask_cdr.size(0))]

        for b in range(mask_cdr.size(0)):
            regions = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid) & (t_res[b] > 0)
                if not bool(rmask.any().item()):
                    continue
                region_idx = rid - 1
                anchor_score = anchor_support[b, rmask].mean().clamp(0.0, 1.0)
                contact_score = anchor_score.new_tensor(0.0)
                has_contact = False
                if contact_prob is not None and region_idx < contact_prob.size(1):
                    contact_score = contact_prob[b, region_idx].clamp(0.0, 1.0)
                    has_contact = True
                elif contact_mass is not None and region_idx < contact_mass.size(1):
                    contact_score = contact_mass[b, region_idx].clamp(0.0, 1.0)
                    if contact_hit is not None and region_idx < contact_hit.size(1):
                        contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx].clamp(0.0, 1.0)
                    has_contact = True
                elif contact_hit is not None and region_idx < contact_hit.size(1):
                    contact_score = contact_hit[b, region_idx].clamp(0.0, 1.0)
                    has_contact = True

                prior = anchor_score.new_tensor(float(bio_prior.get(rid, 0.0)))
                support = ((1.0 - contact_weight) * anchor_score + contact_weight * contact_score).clamp(0.0, 1.0) if has_contact else anchor_score
                region_support_res[b, rmask] = support
                ent = entropy[b, rmask].mean().clamp(0.0, 1.0)
                raw_score = prior_weight * prior + contact_weight * contact_score + anchor_weight * anchor_score - entropy_weight * ent
                clock = int(t_res[b, rmask].max().item())
                regions.append({
                    'mask': rmask,
                    'rid': rid,
                    'region': cdr_names.get(rid, str(rid)),
                    'clock': clock,
                    'entropy': float(ent.detach().cpu().item()),
                    'support': float(support.detach().cpu().item()),
                    'contact': float(contact_score.detach().cpu().item()),
                    'prior': float(prior.detach().cpu().item()),
                    'score': float(raw_score.detach().cpu().item()),
                })

            if not regions:
                continue
            if len(regions) == 1:
                active[b] = active[b] | regions[0]['mask']
                regions[0].update({
                    'policy': 'dynamic_schedule',
                    'selection_reason': 'single_region',
                    'schedule_progress': u,
                    'schedule_gamma': 1.0,
                    'schedule_target_t': max(0, regions[0]['clock'] - 1),
                    'schedule_score': regions[0]['score'],
                    'schedule_lead': 0.0,
                    'max_lag': max_lag,
                })
                region_meta[b] = regions
                continue

            max_clock = max(item['clock'] for item in regions)
            min_clock = min(item['clock'] for item in regions)
            if max_clock <= final_sync_clock:
                for item in regions:
                    active[b] = active[b] | item['mask']
                    item.update({
                        'policy': 'dynamic_schedule',
                        'selection_reason': 'final_sync',
                        'schedule_progress': u,
                        'schedule_gamma': 1.0,
                        'schedule_target_t': 0,
                        'schedule_score': item['score'],
                        'schedule_lead': 0.0,
                        'max_lag': max_lag,
                    })
                region_meta[b] = regions
                continue

            scores = [item['score'] for item in regions]
            score_center = sum(scores) / float(len(scores))
            score_span = max(scores) - min(scores)
            if score_span < 1e-6:
                leads = [0.0 for _ in regions]
            else:
                leads = [max(-1.0, min(1.0, (score - score_center) / score_span)) for score in scores]
            if priority_mode in {'slow_high', 'asyn_slow_high', 'important_slow'}:
                leads = [-lead for lead in leads]

            selected = [False for _ in regions]
            reasons = ['hold' for _ in regions]
            due = []
            for idx, item in enumerate(regions):
                gamma = float(torch.exp(torch.tensor(-schedule_strength * leads[idx], device=mask_cdr.device)).detach().cpu().item())
                gamma = max(min_gamma, min(max_gamma, gamma))
                progress = min(1.0, max(0.0, u ** gamma))
                target_t = int(round(float(self.num_steps) * (1.0 - progress)))
                target_t = max(0, min(self.num_steps, target_t))
                item.update({
                    'policy': 'dynamic_schedule',
                    'schedule_progress': progress,
                    'schedule_gamma': gamma,
                    'schedule_target_t': target_t,
                    'schedule_score': item['score'],
                    'schedule_lead': leads[idx],
                    'score_center': score_center,
                    'score_spread': score_span,
                    'max_lag': max_lag,
                    'effective_policy': priority_mode,
                })
                due.append(item['clock'] > target_t)

            for idx, item in enumerate(regions):
                if item['clock'] > min_clock + max_lag:
                    selected[idx] = True
                    reasons[idx] = 'lag_window'

            order = sorted(
                range(len(regions)),
                key=lambda idx: (
                    -(regions[idx]['clock'] - int(regions[idx]['schedule_target_t'])),
                    -regions[idx]['schedule_score'],
                    -regions[idx]['clock'],
                )
            )
            for idx in order:
                if selected[idx] or not due[idx]:
                    continue
                trial = selected[:]
                trial[idx] = True
                trial_clocks = [max(0, item['clock'] - (1 if trial[j] else 0)) for j, item in enumerate(regions)]
                unfinished = [clock for clock in trial_clocks if clock > 0]
                if unfinished and (max(unfinished) - min(unfinished) > max_lag):
                    continue
                selected[idx] = True
                reasons[idx] = 'schedule_due'

            if not any(selected):
                fallback = max(order, key=lambda idx: (regions[idx]['clock'], regions[idx]['schedule_score']))
                selected[fallback] = True
                reasons[fallback] = 'fallback_max_clock'

            selected_count = sum(1 for value in selected if value)
            for idx, item in enumerate(regions):
                item['selection_reason'] = reasons[idx]
                item['selected_count'] = selected_count
                item['target_active'] = sum(1 for value in due if value)
                item['max_clock'] = max_clock
                item['min_clock'] = min_clock
                if selected[idx]:
                    active[b] = active[b] | item['mask']
                else:
                    slow[b] = slow[b] | item['mask']
            region_meta[b] = regions

        batch['_hgacd_async_region_meta'] = region_meta
        return active, slow, entropy, region_support_res

    def _select_hgacd_dynamic_schedule_next_active(self, batch, s_0_pred_logits, mask_cdr,
                                                    anchor_support, cfg, t_res, step_idx):
        active, slow, entropy, region_support_res = self._select_hgacd_dynamic_schedule_active_mask(
            batch, s_0_pred_logits, mask_cdr, anchor_support, cfg, t_res, step_idx
        )
        return active, slow, entropy, region_support_res


    def _select_hgacd_dockq_window_active_mask(self, batch, s_0_pred_logits, mask_cdr,
                                               anchor_support, cfg, t_res):
        """DockQ-oriented CDR scheduler under the same training lag window.

        The previous bioprior controller made interface-prior CDRs update early,
        which can freeze them before other CDRs have settled. DockQ depends on
        interface contacts, so this controller keeps high-contact/CDR3 loops
        available for later refinement while still enforcing region_max_lag.
        """
        probs = F.softmax(s_0_pred_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))

        region_type = batch['region_type'].to(mask_cdr.device)
        state = self._get_hgacd_async_state()
        contact_mass = None if state is None else state.get('region_contact_mass')
        contact_hit = None if state is None else state.get('region_contact_hit')
        contact_prob = None if state is None else state.get('region_contact_prob')
        if contact_mass is not None:
            contact_mass = contact_mass.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_hit is not None:
            contact_hit = contact_hit.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_prob is not None:
            contact_prob = contact_prob.to(mask_cdr.device).float().clamp(0.0, 1.0)

        max_lag = int(cfg.get('region_max_lag', cfg.get('max_lag', 8)))
        max_lag = max(0, max_lag)
        min_active_fraction = float(cfg.get('window_min_active_fraction', 0.50))
        min_active_fraction = max(0.0, min(min_active_fraction, 1.0))
        support_weight = float(cfg.get('support_weight', 0.50))
        contact_weight = float(cfg.get('contact_weight', 0.50))
        late_refine_clock = max(3 * max_lag, self.num_steps // 3)
        final_sync_clock = max(1, max_lag)
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        bio_prior = {
            int(CDR.H1): 0.35,
            int(CDR.H2): 0.65,
            int(CDR.H3): 1.00,
            int(CDR.L1): 0.35,
            int(CDR.L2): 0.20,
            int(CDR.L3): 0.70,
        }

        active = torch.zeros_like(mask_cdr, dtype=torch.bool)
        slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
        region_support_res = torch.zeros_like(anchor_support)

        for b in range(mask_cdr.size(0)):
            regions = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid) & (t_res[b] > 0)
                if not bool(rmask.any().item()):
                    continue
                region_idx = rid - 1
                anchor_score = anchor_support[b, rmask].mean()
                contact_score = anchor_score.new_tensor(0.0)
                has_contact = False
                if contact_prob is not None and region_idx < contact_prob.size(1):
                    contact_score = contact_prob[b, region_idx]
                    has_contact = True
                elif contact_mass is not None and region_idx < contact_mass.size(1):
                    contact_score = contact_mass[b, region_idx]
                    if contact_hit is not None and region_idx < contact_hit.size(1):
                        contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx]
                    has_contact = True
                elif contact_hit is not None and region_idx < contact_hit.size(1):
                    contact_score = contact_hit[b, region_idx]
                    has_contact = True

                if has_contact:
                    support = ((1.0 - contact_weight) * anchor_score + contact_weight * contact_score).clamp(0.0, 1.0)
                else:
                    support = anchor_score.clamp(0.0, 1.0)
                region_support_res[b, rmask] = support

                prior = anchor_score.new_tensor(float(bio_prior.get(rid, 0.0)))
                interface_priority = (0.75 * contact_score + 0.25 * prior).clamp(0.0, 1.0)
                base_risk = entropy[b, rmask].mean() + support_weight * (1.0 - support)
                clock = int(t_res[b, rmask].max().item())
                regions.append({
                    'mask': rmask,
                    'clock': clock,
                    'risk': float(base_risk.detach().cpu().item()),
                    'interface_priority': float(interface_priority.detach().cpu().item()),
                })

            if not regions:
                continue
            if len(regions) == 1:
                active[b] = active[b] | regions[0]['mask']
                continue

            max_clock = max(item['clock'] for item in regions)
            min_clock = min(item['clock'] for item in regions)
            if max_clock <= final_sync_clock:
                for item in regions:
                    active[b] = active[b] | item['mask']
                continue

            late_refine = max_clock <= late_refine_clock
            target_active = max(1, int(round(len(regions) * min_active_fraction)))
            selected = [False for _ in regions]

            for idx, item in enumerate(regions):
                if item['clock'] > min_clock + max_lag:
                    selected[idx] = True

            if late_refine:
                order = sorted(
                    range(len(regions)),
                    key=lambda idx: (-regions[idx]['interface_priority'], regions[idx]['risk'], -regions[idx]['clock'])
                )
            else:
                order = sorted(
                    range(len(regions)),
                    key=lambda idx: (regions[idx]['interface_priority'], regions[idx]['risk'], -regions[idx]['clock'])
                )

            for idx in order:
                if selected[idx]:
                    continue
                trial = selected[:]
                trial[idx] = True
                trial_clocks = [max(0, item['clock'] - (1 if trial[j] else 0)) for j, item in enumerate(regions)]
                unfinished = [clock for clock in trial_clocks if clock > 0]
                if unfinished and (max(unfinished) - min(unfinished) > max_lag):
                    continue
                selected[idx] = True
                if sum(selected) >= target_active:
                    break

            if not any(selected):
                max_clock = max(item['clock'] for item in regions)
                for idx, item in enumerate(regions):
                    if item['clock'] == max_clock:
                        selected[idx] = True
                        break

            for idx, item in enumerate(regions):
                if selected[idx]:
                    active[b] = active[b] | item['mask']
                else:
                    slow[b] = slow[b] | item['mask']

        return active, slow, entropy, region_support_res

    def _select_hgacd_dockq_window_next_active(self, batch, s_0_pred_logits, mask_cdr,
                                                anchor_support, cfg, t_res):
        active, slow, entropy, region_support_res = self._select_hgacd_dockq_window_active_mask(
            batch, s_0_pred_logits, mask_cdr, anchor_support, cfg, t_res
        )
        return active, slow, entropy, region_support_res


    def _select_hgacd_contact_refine_window_active_mask(self, batch, s_0_pred_logits, mask_cdr,
                                                        anchor_support, cfg, t_res):
        """Adaptive contact-refinement scheduler for DockQ-oriented sampling.

        The controller remains adaptive: current graph/contact/support signals
        dominate the decision, and the biological CDR prior only breaks weak or
        ambiguous contact-score ties. The method writes per-region scheduling
        metadata into ``batch`` so async traces record the exact decision basis.
        """
        probs = F.softmax(s_0_pred_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))

        region_type = batch['region_type'].to(mask_cdr.device)
        state = self._get_hgacd_async_state()
        contact_mass = None if state is None else state.get('region_contact_mass')
        contact_hit = None if state is None else state.get('region_contact_hit')
        contact_prob = None if state is None else state.get('region_contact_prob')
        if contact_mass is not None:
            contact_mass = contact_mass.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_hit is not None:
            contact_hit = contact_hit.to(mask_cdr.device).float().clamp(0.0, 1.0)
        if contact_prob is not None:
            contact_prob = contact_prob.to(mask_cdr.device).float().clamp(0.0, 1.0)

        policy = str(cfg.get('contact_refine_policy', 'adaptive')).lower()
        max_lag = int(cfg.get('region_max_lag', cfg.get('max_lag', 8)))
        max_lag = max(0, max_lag)
        min_active_fraction = float(cfg.get('window_min_active_fraction', 0.50))
        min_active_fraction = max(0.0, min(min_active_fraction, 1.0))
        support_weight = float(cfg.get('support_weight', 0.50))
        contact_weight = float(cfg.get('contact_weight', 0.50))
        default_prior_weight = 0.0 if policy == 'prior_adaptive_sync_async_gated' else 0.20
        prior_weight = float(cfg.get('bio_prior_weight', cfg.get('cdr3_prior_weight', default_prior_weight)))
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        cdr_names = {
            int(CDR.H1): 'HCDR1', int(CDR.H2): 'HCDR2', int(CDR.H3): 'HCDR3',
            int(CDR.L1): 'LCDR1', int(CDR.L2): 'LCDR2', int(CDR.L3): 'LCDR3',
        }
        bio_prior = {
            int(CDR.H1): 0.35,
            int(CDR.H2): 0.65,
            int(CDR.H3): 1.00,
            int(CDR.L1): 0.35,
            int(CDR.L2): 0.20,
            int(CDR.L3): 0.70,
        }

        active = torch.zeros_like(mask_cdr, dtype=torch.bool)
        slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
        region_support_res = torch.zeros_like(anchor_support)
        region_meta = [[] for _ in range(mask_cdr.size(0))]
        gated_sync_state_key = '_hgacd_prior_adaptive_gated_last_global_sync_step'
        gated_last_sync_steps = batch.get(gated_sync_state_key)
        if not isinstance(gated_last_sync_steps, list) or len(gated_last_sync_steps) != mask_cdr.size(0):
            gated_last_sync_steps = [-10 ** 9 for _ in range(mask_cdr.size(0))]
            batch[gated_sync_state_key] = gated_last_sync_steps

        for b in range(mask_cdr.size(0)):
            regions = []
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid) & (t_res[b] > 0)
                if not bool(rmask.any().item()):
                    continue
                region_idx = rid - 1
                anchor_score = anchor_support[b, rmask].mean()
                contact_score = anchor_score.new_tensor(0.0)
                has_contact = False
                if contact_prob is not None and region_idx < contact_prob.size(1):
                    contact_score = contact_prob[b, region_idx]
                    has_contact = True
                elif contact_mass is not None and region_idx < contact_mass.size(1):
                    contact_score = contact_mass[b, region_idx]
                    if contact_hit is not None and region_idx < contact_hit.size(1):
                        contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx]
                    has_contact = True
                elif contact_hit is not None and region_idx < contact_hit.size(1):
                    contact_score = contact_hit[b, region_idx]
                    has_contact = True

                if has_contact:
                    support = ((1.0 - contact_weight) * anchor_score + contact_weight * contact_score).clamp(0.0, 1.0)
                else:
                    support = anchor_score.clamp(0.0, 1.0)
                region_support_res[b, rmask] = support

                ent = entropy[b, rmask].mean()
                risk = ent + support_weight * (1.0 - support)
                prior = anchor_score.new_tensor(float(bio_prior.get(rid, 0.0)))
                interface_score = ((1.0 - prior_weight) * contact_score + prior_weight * prior).clamp(0.0, 1.0)
                clock = int(t_res[b, rmask].max().item())
                regions.append({
                    'mask': rmask,
                    'rid': rid,
                    'region': cdr_names.get(rid, str(rid)),
                    'clock': clock,
                    'entropy': float(ent.detach().cpu().item()),
                    'support': float(support.detach().cpu().item()),
                    'contact': float(contact_score.detach().cpu().item()),
                    'prior': float(prior.detach().cpu().item()),
                    'interface_score': float(interface_score.detach().cpu().item()),
                    'risk': float(risk.detach().cpu().item()),
                })

            if not regions:
                continue

            max_clock = max(item['clock'] for item in regions)
            min_clock = min(item['clock'] for item in regions)
            selected = [False for _ in regions]
            score_center = None
            score_spread = None
            contact_spread = None
            gate_enabled = None
            effective_policy = policy
            mixture_mode = None
            mixture_high_interface_spread = None
            mixture_high_contact_spread = None
            mixture_low_interface_spread = None
            mixture_very_low_interface_spread = None
            mixture_very_low_contact_spread = None
            tristate_mode = None
            tristate_low_contact_spread = None
            tristate_high_contact_spread = None
            dual_guard_mode = None
            dual_guard_contact_floor = None
            dual_guard_risk_floor = None
            gated_sync_enabled = None
            gated_need_mean = None
            gated_need_std = None
            gated_need_top_gap = None
            gated_coupling = None
            gated_global_need_high = None
            gated_dispersion_low = None
            gated_coupling_high = None
            gated_cooldown_ok = None
            gated_current_step = None
            gated_sync_cooldown = None
            gated_local_active_fraction = None
            refine_phase = False
            dynamic_fraction = None
            selection_reason = 'policy'

            if len(regions) == 1:
                selected[0] = True
                target_active = 1
                selection_reason = 'single_region'
            elif max_clock <= max(1, max_lag):
                selected = [True for _ in regions]
                target_active = len(regions)
                selection_reason = 'within_lag_sync'
            else:
                target_active = max(1, int(round(len(regions) * min_active_fraction)))
                progress = 1.0 - (float(max_clock) / float(max(1, self.num_steps)))
                if policy in {'balanced', 'prior_adaptive_balanced', 'balanced_refine', 'early_interface_balanced', 'gated_early_interface_balanced'}:
                    dynamic_fraction = min(
                        1.0,
                        max(min_active_fraction, min_active_fraction + (1.0 - min_active_fraction) * progress),
                    )
                    target_active = max(target_active, int(math.ceil(len(regions) * dynamic_fraction)))

                for idx, item in enumerate(regions):
                    if item['clock'] > min_clock + max_lag:
                        selected[idx] = True

                interface_scores = torch.tensor(
                    [item['interface_score'] for item in regions],
                    device=mask_cdr.device,
                    dtype=torch.float32,
                )
                score_center = float(interface_scores.mean().detach().cpu().item())
                score_spread = float(interface_scores.std(unbiased=False).detach().cpu().item())
                contact_scores = torch.tensor(
                    [item['contact'] for item in regions],
                    device=mask_cdr.device,
                    dtype=torch.float32,
                )
                contact_spread = float((contact_scores.max() - contact_scores.min()).detach().cpu().item())
                effective_policy = policy
                if policy == 'gated_early_interface_balanced':
                    gate_interface_spread = float(cfg.get('gated_early_interface_min_spread', 0.50))
                    gate_contact_spread = float(cfg.get('gated_early_contact_min_spread', 0.43))
                    gate_enabled = (score_spread >= gate_interface_spread) or (contact_spread >= gate_contact_spread)
                    effective_policy = 'early_interface_balanced' if gate_enabled else 'balanced'
                    selection_reason = 'gated_early_on' if gate_enabled else 'gated_early_off'
                elif policy == 'dockq_adaptive_mixture':
                    mixture_high_interface_spread = float(cfg.get('dockq_mixture_high_interface_spread', 0.54))
                    mixture_high_contact_spread = float(cfg.get('dockq_mixture_high_contact_spread', 0.55))
                    mixture_low_interface_spread = float(cfg.get('dockq_mixture_low_interface_spread', 0.42))
                    mixture_very_low_interface_spread = float(cfg.get('dockq_mixture_very_low_interface_spread', 0.34))
                    mixture_very_low_contact_spread = float(cfg.get('dockq_mixture_very_low_contact_spread', 0.28))
                    high_spread = (score_spread >= mixture_high_interface_spread) or (contact_spread >= mixture_high_contact_spread)
                    very_low_spread = (score_spread <= mixture_very_low_interface_spread) or (contact_spread <= mixture_very_low_contact_spread)
                    low_spread = score_spread <= mixture_low_interface_spread
                    if high_spread:
                        effective_policy = 'early_interface_balanced'
                        mixture_mode = 'high_spread_early_balanced'
                    elif very_low_spread or low_spread:
                        effective_policy = 'prior_adaptive_balanced'
                        mixture_mode = 'low_spread_prior_balanced'
                    else:
                        effective_policy = 'adaptive'
                        mixture_mode = 'mid_spread_contact_adaptive'
                    selection_reason = 'dockq_mixture_' + mixture_mode
                elif policy == 'dockq_spread_tristate':
                    tristate_low_contact_spread = float(cfg.get('dockq_tristate_low_contact_spread', 0.3271))
                    tristate_high_contact_spread = float(cfg.get('dockq_tristate_high_contact_spread', 0.3946))
                    if contact_spread <= tristate_low_contact_spread:
                        effective_policy = 'adaptive'
                        tristate_mode = 'low_contact_refine'
                    elif contact_spread < tristate_high_contact_spread:
                        effective_policy = 'interface_first'
                        tristate_mode = 'mid_interface_first'
                    else:
                        effective_policy = 'early_interface_balanced'
                        tristate_mode = 'high_early_balanced'
                    selection_reason = 'dockq_tristate_' + tristate_mode
                elif policy == 'dockq_dual_guard':
                    effective_policy = 'dual_guard'
                    dual_guard_mode = 'contact_and_risk_guard'
                    dual_guard_contact_floor = float(cfg.get('dockq_dual_guard_contact_floor', 0.35))
                    dual_guard_risk_floor = float(cfg.get('dockq_dual_guard_risk_floor', 0.0))
                    dynamic_fraction = min(
                        1.0,
                        max(min_active_fraction, min_active_fraction + (1.0 - min_active_fraction) * progress),
                    )
                    target_active = max(target_active, min(len(regions), int(math.ceil(len(regions) * dynamic_fraction))))
                    selection_reason = 'dockq_dual_guard'
                elif policy == 'prior_adaptive_sync_async_gated':
                    effective_policy = 'gated_prior_adaptive'
                    gated_need_weight_risk = float(cfg.get('gated_need_weight_risk', 0.50))
                    gated_need_weight_interface = float(cfg.get('gated_need_weight_interface_inconsistency', 0.30))
                    gated_need_weight_clock = float(cfg.get('gated_need_weight_clock', 0.20))
                    gated_overcontact_margin = float(cfg.get('gated_overcontact_margin', 0.20))
                    gated_overcontact_weight = float(cfg.get('gated_overcontact_weight', 0.50))
                    gated_sync_need_threshold = float(cfg.get('gated_sync_need_threshold', 0.62))
                    gated_sync_std_threshold = float(cfg.get('gated_sync_std_threshold', 0.10))
                    gated_sync_gap_threshold = float(cfg.get('gated_sync_gap_threshold', 0.08))
                    gated_sync_coupling_threshold = float(cfg.get('gated_sync_coupling_threshold', 0.45))
                    gated_sync_cooldown = int(cfg.get('gated_sync_cooldown_steps', 6))
                    gated_local_active_fraction = float(cfg.get('gated_local_active_fraction', 0.34))
                    gated_local_active_fraction = max(0.05, min(gated_local_active_fraction, 0.95))

                    need_values = []
                    for item in regions:
                        contact_prior = float(item['interface_score'])
                        current_contact = float(item['contact'])
                        contact_deficit = max(contact_prior - current_contact, 0.0)
                        over_contact = max(current_contact - contact_prior - gated_overcontact_margin, 0.0)
                        interface_inconsistency = contact_deficit + gated_overcontact_weight * over_contact
                        risk_norm = min(float(item['risk']) / max(1e-6, 1.0 + support_weight), 1.0)
                        clock_frac = min(max(float(item['clock']) / float(max(1, self.num_steps)), 0.0), 1.0)
                        need = (
                            gated_need_weight_risk * risk_norm
                            + gated_need_weight_interface * interface_inconsistency
                            + gated_need_weight_clock * clock_frac
                        )
                        item['need_score'] = float(max(0.0, min(need, 1.0)))
                        item['interface_inconsistency'] = float(interface_inconsistency)
                        need_values.append(item['need_score'])

                    need_tensor = torch.tensor(need_values, device=mask_cdr.device, dtype=torch.float32)
                    gated_need_mean = float(need_tensor.mean().detach().cpu().item())
                    gated_need_std = float(need_tensor.std(unbiased=False).detach().cpu().item())
                    if need_tensor.numel() >= 2:
                        top2 = torch.topk(need_tensor, k=2, largest=True).values
                        gated_need_top_gap = float((top2[0] - top2[1]).detach().cpu().item())
                    else:
                        gated_need_top_gap = 1.0
                    support_mean = sum(float(item['support']) for item in regions) / float(len(regions))
                    contact_mean = sum(float(item['contact']) for item in regions) / float(len(regions))
                    gated_coupling = float(0.5 * support_mean + 0.5 * contact_mean)
                    gated_current_step = int(round(self.num_steps - float(max_clock)))
                    gated_cooldown_ok = (gated_current_step - int(gated_last_sync_steps[b])) >= gated_sync_cooldown
                    gated_global_need_high = gated_need_mean >= gated_sync_need_threshold
                    gated_dispersion_low = (gated_need_std <= gated_sync_std_threshold) and (gated_need_top_gap <= gated_sync_gap_threshold)
                    gated_coupling_high = gated_coupling >= gated_sync_coupling_threshold
                    gated_sync_enabled = bool(
                        gated_global_need_high
                        and gated_dispersion_low
                        and gated_coupling_high
                        and gated_cooldown_ok
                    )
                    if gated_sync_enabled:
                        selected = [True for _ in regions]
                        target_active = len(regions)
                        gated_last_sync_steps[b] = gated_current_step
                        selection_reason = 'gated_global_sync'
                    else:
                        local_target = max(1, int(math.ceil(len(regions) * gated_local_active_fraction)))
                        local_target = min(local_target, max(1, len(regions) - 1))
                        target_active = max(target_active, local_target)
                        target_active = min(target_active, max(1, len(regions) - 1))
                        selection_reason = 'gated_adaptive_local_sync_async'
                if dynamic_fraction is None and effective_policy in {'balanced', 'prior_adaptive_balanced', 'balanced_refine', 'early_interface_balanced'}:
                    dynamic_fraction = min(
                        1.0,
                        max(min_active_fraction, min_active_fraction + (1.0 - min_active_fraction) * progress),
                    )
                    target_active = max(target_active, int(math.ceil(len(regions) * dynamic_fraction)))
                early_refine_policy = effective_policy in {
                    'early_interface_refine', 'early_interface_balanced',
                    'dockq_refine', 'dockq_adaptive_refine',
                }
                if early_refine_policy:
                    refine_start_fraction = float(cfg.get('early_refine_start_fraction', 0.65))
                    refine_clock_threshold = max(2 * max_lag, int(round(self.num_steps * refine_start_fraction)))
                else:
                    refine_clock_threshold = max(2 * max_lag, self.num_steps // 4)
                late_by_clock = max_clock <= refine_clock_threshold
                interface_separable = score_spread > 1e-3
                refine_phase = late_by_clock and interface_separable

                if effective_policy == 'dual_guard':
                    unfinished_indices = [idx for idx, item in enumerate(regions) if item['clock'] > 0]
                    if unfinished_indices:
                        contact_ranked = sorted(
                            unfinished_indices,
                            key=lambda idx: (
                                -(regions[idx]['interface_score'] + 0.25 * regions[idx]['contact']),
                                -regions[idx]['risk'],
                                -regions[idx]['clock'],
                            ),
                        )
                        risk_ranked = sorted(
                            unfinished_indices,
                            key=lambda idx: (
                                -regions[idx]['risk'],
                                regions[idx]['interface_score'],
                                -regions[idx]['clock'],
                            ),
                        )
                        for guard_idx in (contact_ranked[0], risk_ranked[0]):
                            if regions[guard_idx]['contact'] >= dual_guard_contact_floor or regions[guard_idx]['risk'] >= dual_guard_risk_floor:
                                selected[guard_idx] = True
                        target_active = max(target_active, min(len(regions), sum(selected)))

                if effective_policy in {'context_first', 'background_first', 'low_contact_first'}:
                    def key(idx):
                        item = regions[idx]
                        return (item['interface_score'], item['risk'], -item['clock'])
                elif effective_policy in {'balanced', 'prior_adaptive_balanced', 'balanced_refine'}:
                    if refine_phase:
                        def key(idx):
                            item = regions[idx]
                            score_excess = item['interface_score'] - score_center
                            refine_score = score_excess + 0.50 * item['risk'] + 0.10 * (item['clock'] / float(self.num_steps))
                            return (-refine_score, -item['clock'])
                    else:
                        def key(idx):
                            item = regions[idx]
                            return (item['interface_score'], item['risk'], -item['clock'])
                elif effective_policy in {'interface_first', 'key_first', 'high_contact_first'}:
                    def key(idx):
                        item = regions[idx]
                        key_score = item['interface_score'] + 0.50 * item['risk'] + 0.10 * (item['clock'] / float(self.num_steps))
                        return (-key_score, -item['clock'])
                elif effective_policy == 'dual_guard':
                    def key(idx):
                        item = regions[idx]
                        contact_term = item['interface_score'] + 0.25 * item['contact']
                        weak_term = item['risk'] + 0.50 * (1.0 - item['support'])
                        guard_score = contact_term + weak_term + 0.10 * (item['clock'] / float(self.num_steps))
                        return (-guard_score, -item['clock'])
                elif effective_policy == 'gated_prior_adaptive':
                    def key(idx):
                        item = regions[idx]
                        return (-float(item.get('need_score', item['risk'])), -item['clock'])
                elif refine_phase:
                    def key(idx):
                        item = regions[idx]
                        score_excess = item['interface_score'] - score_center
                        refine_score = score_excess + 0.50 * item['risk'] + 0.10 * (item['clock'] / float(self.num_steps))
                        return (-refine_score, -item['clock'])
                else:
                    def key(idx):
                        item = regions[idx]
                        return (item['interface_score'], item['risk'], -item['clock'])
                order = sorted(range(len(regions)), key=key)

                for idx in order:
                    if selected[idx]:
                        continue
                    trial = selected[:]
                    trial[idx] = True
                    trial_clocks = [max(0, item['clock'] - (1 if trial[j] else 0)) for j, item in enumerate(regions)]
                    unfinished = [clock for clock in trial_clocks if clock > 0]
                    if unfinished and (max(unfinished) - min(unfinished) > max_lag):
                        continue
                    selected[idx] = True
                    if sum(selected) >= target_active:
                        break

                if not any(selected):
                    max_clock = max(item['clock'] for item in regions)
                    for idx, item in enumerate(regions):
                        if item['clock'] == max_clock:
                            selected[idx] = True
                            break
                    selection_reason = 'fallback_max_clock'

            selected_count = int(sum(selected))
            for idx, item in enumerate(regions):
                is_active = bool(selected[idx])
                if is_active:
                    active[b] = active[b] | item['mask']
                else:
                    slow[b] = slow[b] | item['mask']
                meta = {k: v for k, v in item.items() if k != 'mask'}
                meta.update({
                    'active': is_active,
                    'slow': not is_active,
                    'policy': policy,
                    'selection_reason': selection_reason,
                    'target_active': int(target_active),
                    'selected_count': selected_count,
                    'max_clock': int(max_clock),
                    'min_clock': int(min_clock),
                    'score_center': score_center,
                    'score_spread': score_spread,
                    'contact_spread': contact_spread,
                    'effective_policy': effective_policy,
                    'gate_enabled': gate_enabled,
                    'gate_interface_min_spread': float(cfg.get('gated_early_interface_min_spread', 0.50)) if policy == 'gated_early_interface_balanced' else None,
                    'gate_contact_min_spread': float(cfg.get('gated_early_contact_min_spread', 0.43)) if policy == 'gated_early_interface_balanced' else None,
                    'mixture_mode': mixture_mode,
                    'mixture_high_interface_spread': mixture_high_interface_spread,
                    'mixture_high_contact_spread': mixture_high_contact_spread,
                    'mixture_low_interface_spread': mixture_low_interface_spread,
                    'mixture_very_low_interface_spread': mixture_very_low_interface_spread,
                    'mixture_very_low_contact_spread': mixture_very_low_contact_spread,
                    'tristate_mode': tristate_mode,
                    'tristate_low_contact_spread': tristate_low_contact_spread,
                    'tristate_high_contact_spread': tristate_high_contact_spread,
                    'dual_guard_mode': dual_guard_mode,
                    'dual_guard_contact_floor': dual_guard_contact_floor,
                    'dual_guard_risk_floor': dual_guard_risk_floor,
                    'gated_sync_enabled': gated_sync_enabled,
                    'gated_need_mean': gated_need_mean,
                    'gated_need_std': gated_need_std,
                    'gated_need_top_gap': gated_need_top_gap,
                    'gated_coupling': gated_coupling,
                    'gated_global_need_high': gated_global_need_high,
                    'gated_dispersion_low': gated_dispersion_low,
                    'gated_coupling_high': gated_coupling_high,
                    'gated_cooldown_ok': gated_cooldown_ok,
                    'gated_current_step': gated_current_step,
                    'gated_sync_cooldown': gated_sync_cooldown,
                    'gated_local_active_fraction': gated_local_active_fraction,
                    'refine_phase': bool(refine_phase),
                    'refine_clock_threshold': int(refine_clock_threshold) if 'refine_clock_threshold' in locals() else None,
                    'dynamic_fraction': dynamic_fraction,
                    'max_lag': int(max_lag),
                    'min_active_fraction': float(min_active_fraction),
                    'prior_weight': float(prior_weight),
                })
                region_meta[b].append(meta)

        batch['_hgacd_async_region_meta'] = region_meta
        return active, slow, entropy, region_support_res

    def _select_hgacd_contact_refine_window_next_active(self, batch, s_0_pred_logits, mask_cdr,
                                                         anchor_support, cfg, t_res):
        active, slow, entropy, region_support_res = self._select_hgacd_contact_refine_window_active_mask(
            batch, s_0_pred_logits, mask_cdr, anchor_support, cfg, t_res
        )
        return active, slow, entropy, region_support_res

    def _apply_hgacd_region_lag(self, active, t_res, region_type, mask_cdr, cfg):
        """Keep related CDR clocks within a bounded lag while preserving convergence."""
        max_lag = int(cfg.get('region_max_lag', 8))
        if max_lag <= 0:
            return active
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        active = active.clone()
        for b in range(mask_cdr.size(0)):
            present = []
            unfinished = t_res[b] > 0
            for rid in cdr_ids:
                rmask = mask_cdr[b] & unfinished & (region_type[b] == rid)
                if bool(rmask.any().item()):
                    present.append((rid, rmask, int(t_res[b, rmask].max().item())))
            if len(present) <= 1:
                continue
            times = torch.tensor([x[2] for x in present], device=t_res.device, dtype=torch.long)
            for idx, (_, rmask, r_t) in enumerate(present):
                other = torch.cat([times[:idx], times[idx + 1:]]) if len(present) > 1 else times[:0]
                if other.numel() == 0:
                    continue
                if r_t > int(other.min().item()) + max_lag:
                    active[b] = active[b] | rmask
                if r_t < int(other.max().item()) - max_lag:
                    active[b, rmask] = False
        return active

    def _compute_hgacd_structure_risk(self, batch, v_t, p_t, s_t, mask_cdr, cfg):
        """Aggregate residue-level peptide geometry violations into CDR-region risk."""
        enabled = bool(cfg.get('enabled', cfg.get('structure_risk_feedback', False)))
        if not enabled or 'region_type' not in batch or 'chain_nb' not in batch or 'res_nb' not in batch:
            return None

        region_type = batch['region_type'].to(mask_cdr.device)
        chain_nb = batch['chain_nb'].to(mask_cdr.device)
        res_nb = batch['res_nb'].to(mask_cdr.device)
        mask_res = batch.get('mask', mask_cdr).to(mask_cdr.device).to(torch.bool)
        R_t = so3vec_to_rotation(v_t)
        bb_pos = reconstruct_backbone(
            R=R_t,
            t=self._unnormalize_position(p_t),
            aa=s_t,
            chain_nb=chain_nb,
            res_nb=res_nb,
            mask=mask_res,
            include_oxygen=False,
        )

        bond_target = float(cfg.get('bond_target', 1.33))
        bond_tol = float(cfg.get('bond_tolerance', 0.10))
        omega_tol = float(cfg.get('omega_cos_tolerance', 0.015))
        w_intra = float(cfg.get('intra_weight', 1.0))
        w_seam = float(cfg.get('seam_weight', 2.0))
        w_omega = float(cfg.get('omega_weight', 0.5))

        n_pos = bb_pos[:, :, BBHeavyAtom.N]
        c_pos = bb_pos[:, :, BBHeavyAtom.C]
        bond_len = torch.sqrt(((c_pos[:, :-1] - n_pos[:, 1:]) ** 2).sum(dim=-1) + 1e-8)
        bond_err = torch.clamp(torch.abs(bond_len - bond_target) - bond_tol, min=0.0)
        seq_consec = get_consecutive_flag(chain_nb, res_nb, mask_res)
        bond_err = bond_err * seq_consec.float()

        bb_dihedral, mask_bb = get_backbone_dihedral_angles(bb_pos, chain_nb, res_nb, mask_res)
        omega_err = torch.clamp(torch.abs(torch.cos(bb_dihedral[..., 0]) + 1.0) - omega_tol, min=0.0)
        omega_err = omega_err * mask_bb[..., 0].float()

        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        risk = torch.zeros_like(mask_cdr, dtype=p_t.dtype)
        risk_meta = [[] for _ in range(mask_cdr.size(0))]
        for b in range(mask_cdr.size(0)):
            for rid in cdr_ids:
                rmask = mask_cdr[b] & (region_type[b] == rid)
                if not bool(rmask.any().item()):
                    continue
                left = rmask[:-1]
                right = rmask[1:]
                adjacent = (left | right) & seq_consec[b]
                internal = (left & right) & seq_consec[b]
                seam = adjacent & (~internal)
                intra_loss = (
                    bond_err[b, internal].mean()
                    if bool(internal.any().item())
                    else bond_err.new_tensor(0.0)
                )
                seam_loss = (
                    bond_err[b, seam].mean()
                    if bool(seam.any().item())
                    else bond_err.new_tensor(0.0)
                )
                omega_mask = rmask & mask_bb[b, :, 0]
                omega_loss = (
                    omega_err[b, omega_mask].mean()
                    if bool(omega_mask.any().item())
                    else omega_err.new_tensor(0.0)
                )
                score = w_intra * intra_loss + w_seam * seam_loss + w_omega * omega_loss
                risk[b, rmask] = score
                risk_meta[b].append({
                    'rid': int(rid),
                    'risk': float(score.detach().cpu().item()),
                    'geom_risk': float(score.detach().cpu().item()),
                    'geom_intra_bond': float(intra_loss.detach().cpu().item()),
                    'geom_seam_bond': float(seam_loss.detach().cpu().item()),
                    'geom_omega': float(omega_loss.detach().cpu().item()),
                })

        return {'risk_res': risk, 'region_meta': risk_meta}

    def _apply_hgacd_structure_risk_feedback(self, batch, active, slow_mask, t_res, mask_cdr, struct_risk, cfg):
        """Use geometry-risk feedback to adjust region-level asynchronous updates."""
        if struct_risk is None or not bool(cfg.get('enabled', cfg.get('structure_risk_feedback', False))):
            return active, slow_mask
        if 'region_type' not in batch:
            return active, slow_mask

        region_type = batch['region_type'].to(mask_cdr.device)
        risk_res = struct_risk.get('risk_res')
        if risk_res is None:
            return active, slow_mask

        active = active.clone()
        slow_mask = slow_mask.clone()
        cdr_ids = [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]
        high_threshold = float(cfg.get('high_threshold', 0.05))
        hold_low_risk = bool(cfg.get('hold_low_risk', False))
        low_threshold = float(cfg.get('low_threshold', 0.01))
        local_sync_seam_threshold = float(cfg.get('local_sync_seam_threshold', high_threshold))
        local_sync_neighbors = bool(cfg.get('local_sync_neighbors', True))
        max_extra_fraction = float(cfg.get('max_extra_region_fraction', 1.0))
        max_extra_fraction = max(0.0, min(max_extra_fraction, 1.0))
        max_extra_regions = int(cfg.get('max_extra_regions', 0))

        region_meta = struct_risk.get('region_meta') or [[] for _ in range(mask_cdr.size(0))]
        for b in range(mask_cdr.size(0)):
            candidates = []
            for meta in region_meta[b]:
                rid = int(meta.get('rid'))
                rmask = mask_cdr[b] & (region_type[b] == rid) & (t_res[b] > 0)
                if not bool(rmask.any().item()):
                    continue
                candidates.append((float(meta.get('risk', 0.0)), rid, rmask, meta))
            if not candidates:
                continue
            if max_extra_regions <= 0:
                extra_cap = max(1, int(math.ceil(len(candidates) * max_extra_fraction)))
            else:
                extra_cap = max(1, min(max_extra_regions, len(candidates)))
            added = 0
            for score, rid, rmask, meta in sorted(candidates, key=lambda x: x[0], reverse=True):
                if score < high_threshold:
                    continue
                if added >= extra_cap:
                    break
                if not bool((active[b] & rmask).any().item()):
                    active[b] = active[b] | rmask
                    slow_mask[b, rmask] = False
                    added += 1
                    meta['geom_feedback'] = 'force_active'
                if local_sync_neighbors and float(meta.get('geom_seam_bond', 0.0)) >= local_sync_seam_threshold:
                    idx = torch.nonzero(rmask, as_tuple=False).flatten()
                    if idx.numel() > 0:
                        left_i = int(idx.min().item()) - 1
                        right_i = int(idx.max().item()) + 1
                        if left_i >= 0 and bool(mask_cdr[b, left_i].item()) and int(t_res[b, left_i].item()) > 0:
                            active[b, left_i] = True
                            slow_mask[b, left_i] = False
                        if right_i < mask_cdr.size(1) and bool(mask_cdr[b, right_i].item()) and int(t_res[b, right_i].item()) > 0:
                            active[b, right_i] = True
                            slow_mask[b, right_i] = False
                    meta['geom_feedback'] = 'local_sync'
            if hold_low_risk:
                for score, _, rmask, meta in candidates:
                    if score <= low_threshold:
                        active[b, rmask] = False
                        slow_mask[b, rmask] = True
                        meta['geom_feedback'] = 'hold_low_risk'

        batch['_hgacd_structure_risk_res'] = risk_res.detach()
        batch['_hgacd_structure_risk_meta'] = region_meta
        return active, slow_mask

    def _merge_hgacd_structure_risk_meta(self, batch):
        base_meta = batch.get('_hgacd_async_region_meta', None)
        risk_meta = batch.get('_hgacd_structure_risk_meta', None)
        if risk_meta is None:
            return
        if base_meta is None:
            batch['_hgacd_async_region_meta'] = risk_meta
            return
        for b in range(min(len(base_meta), len(risk_meta))):
            by_rid = {int(item.get('rid')): item for item in risk_meta[b] if item is not None and 'rid' in item}
            for item in base_meta[b]:
                rid = int(item.get('rid')) if item is not None and 'rid' in item else None
                if rid in by_rid:
                    old_risk = item.get('risk', None)
                    item.update(by_rid[rid])
                    if old_risk is not None:
                        item['controller_risk'] = old_risk
                        item['risk'] = old_risk

    def _residue_beta_from_t(self, t_res, mask_cdr):
        t_idx = t_res.long().clamp(min=0, max=self.num_steps)
        beta = self.trans_pos.var_sched.betas[t_idx]
        return torch.where(mask_cdr, beta, torch.zeros_like(beta))

    def _apply_residue_group_denoise(self, v_t, p_t, s_t, v_pred, eps_p, s_0_pred_logits,
                                     active_mask, t_res, mask_seq_generate, mask_struct_generate):
        if not active_mask.any():
            return v_t, p_t, s_t
        s_prob = F.softmax(s_0_pred_logits, dim=-1)
        v_out, p_out, s_out = v_t, p_t, s_t
        active_ts = torch.unique(t_res[active_mask]).long()
        for t_val in active_ts.tolist():
            if t_val <= 0:
                continue
            group = active_mask & (t_res == int(t_val))
            t_batch = torch.full((v_t.size(0),), int(t_val), dtype=torch.long, device=v_t.device)
            struct_group = group & mask_struct_generate
            seq_group = group & mask_seq_generate
            if struct_group.any():
                v_candidate = self.trans_rot.denoise(v_out, v_pred, struct_group, t_batch, region_nm='cdr')
                p_candidate = self.trans_pos.denoise(p_out, eps_p, struct_group, t_batch, region_nm='cdr')
                v_out = torch.where(struct_group[..., None], v_candidate, v_out)
                p_out = torch.where(struct_group[..., None], p_candidate, p_out)
            if seq_group.any():
                _, s_candidate = self.trans_seq.denoise(s_out, s_prob, seq_group, t_batch, region_nm='cdr')
                s_out = torch.where(seq_group, s_candidate, s_out)
        return v_out, p_out, s_out

    @torch.no_grad()
    def sample_asyndm_residue(self, batch, v, p, s, res_feat, pair_feat,
                              mask_seq_generate, mask_struct_generate, mask_dict, bfactor,
                              pbar=False, compute_loss=False, async_cfg=None):
        """AsynDM-style condition-aligned residue-asynchronous sampler."""
        async_cfg = async_cfg or {}
        controller = str(async_cfg.get('controller', 'rule_based')).lower()
        legacy_controller = controller in {'legacy', 'legacy_topk_power', 'topk_power', 'fraction_power'}
        trainmatch_controller = controller in {'trainmatch', 'trainmatch_random_lag', 'train_lag_match', 'training_lag_match'}
        bioprior_window_controller = controller in {
            'graph_bioprior_window_async', 'bioprior_window_async', 'cdr3_bioprior_window_async',
        }
        dynamic_schedule_controller = controller in {
            'graph_dynamic_schedule_async', 'dynamic_schedule_async', 'asyn_schedule_async',
            'cdr_dynamic_schedule_async', 'graph_asyn_schedule_async',
        }
        dockq_window_controller = controller in {
            'graph_dockq_window_async', 'dockq_window_async', 'interface_dockq_window_async',
        }
        contact_refine_window_controller = controller in {
            'graph_contact_refine_window_async', 'contact_refine_window_async', 'adaptive_contact_refine_window_async',
            'graph_early_interface_refine_window_async', 'early_interface_refine_window_async',
            'graph_dockq_refine_window_async', 'dockq_refine_window_async',
        }
        context_first_window_controller = controller in {
            'graph_context_first_window_async', 'context_first_window_async', 'background_first_window_async',
        }
        interface_first_window_controller = controller in {
            'graph_interface_first_window_async', 'interface_first_window_async', 'key_first_window_async',
        }
        balanced_window_controller = controller in {
            'graph_prior_adaptive_balanced_window_async', 'prior_adaptive_balanced_window_async', 'balanced_contact_refine_window_async',
            'graph_early_interface_balanced_window_async', 'early_interface_balanced_window_async',
            'graph_gated_early_balanced_window_async', 'gated_early_balanced_window_async',
            'graph_dockq_adaptive_mixture_window_async', 'dockq_adaptive_mixture_window_async',
            'graph_dockq_spread_tristate_window_async', 'dockq_spread_tristate_window_async',
            'graph_dockq_dual_guard_window_async', 'dockq_dual_guard_window_async',
            'graph_prior_adaptive_sync_async_gated_window_async', 'prior_adaptive_sync_async_gated_window_async',
        }
        graph_window_controller = controller in {
            'graph_adaptive_window_async', 'graph_window_async', 'adaptive_graph_window_async',
            'graph_guided_trainmatch', 'graph_trainmatch', 'adaptive_graph_trainmatch',
            'graph_guided_trainmatch_random_lag',
        } or bioprior_window_controller or dynamic_schedule_controller or dockq_window_controller or contact_refine_window_controller or context_first_window_controller or interface_first_window_controller or balanced_window_controller
        N, L = v.shape[:2]
        device = v.device
        mask_cdr = mask_dict['mask_cdr'].to(torch.bool)
        mask_res = mask_dict.get('mask_res', batch['mask']).to(torch.bool)
        mask_seq_generate = mask_seq_generate.to(torch.bool) & mask_cdr
        mask_struct_generate = mask_struct_generate.to(torch.bool) & mask_cdr

        if mask_struct_generate.any():
            v_rand = random_uniform_so3([N, L], device=device)
            p_rand = torch.randn_like(p)
            v_t = torch.where(mask_struct_generate[..., None], v_rand, v)
            p_t = torch.where(mask_struct_generate[..., None], p_rand, p)
        else:
            v_t, p_t = v, p
        if mask_seq_generate.any():
            s_rand = torch.randint_like(s, low=0, high=20)
            s_t = torch.where(mask_seq_generate, s_rand, s)
        else:
            s_t = s

        t_res = torch.zeros(N, L, dtype=torch.long, device=device)
        t_res = torch.where(mask_cdr, torch.full_like(t_res, self.num_steps), t_res)
        runtime_noise_sampling = _get_runtime_noise_sampling(batch)
        trainmatch_offsets = None
        if trainmatch_controller:
            trainmatch_offsets = self._sample_hgacd_trainmatch_region_offsets(
                batch, mask_cdr, runtime_noise_sampling
            )
        anchor_support = self._build_anchor_support_prior(batch, mask_cdr, mask_res)
        traj = {self.num_steps: (v_t, self._unnormalize_position(p_t), s_t)}
        loss_history = {} if compute_loss else None
        trace_enabled = bool(async_cfg.get('trace', False))
        trace_stride = max(1, int(async_cfg.get('trace_stride', 1)))
        trace_region_states = bool(async_cfg.get('trace_region_states', False))
        trace_state_stride = max(1, int(async_cfg.get('trace_state_stride', 25)))
        trace_selector = batch.get('_async_trace_selector', None)
        if 'pos_heavyatom_native' in batch:
            trace_p_ref = self._normalize_position(batch['pos_heavyatom_native'][:, :, BBHeavyAtom.CA, :])
        else:
            trace_p_ref = p.clone()
        if trace_enabled:
            if trace_selector is None:
                trace_selector = torch.ones(N, dtype=torch.bool, device=device)
            else:
                trace_selector = trace_selector.to(device=device, dtype=torch.bool)
            trace_records = [[] for _ in range(N)]
        else:
            trace_records = None

        def _record_async_trace_step(label, phase, step_idx_value, active_mask, slow_mask, t_before, t_after, entropy_tensor, support_tensor, p_prev=None, p_cur=None):
            if trace_records is None:
                return
            if isinstance(step_idx_value, int) and step_idx_value % trace_stride != 0:
                return
            region_type_trace = batch['region_type'].to(device)
            cdr_items = [
                (int(CDR.H1), 'HCDR1'), (int(CDR.H2), 'HCDR2'), (int(CDR.H3), 'HCDR3'),
                (int(CDR.L1), 'LCDR1'), (int(CDR.L2), 'LCDR2'), (int(CDR.L3), 'LCDR3'),
            ]
            region_meta_all = batch.get('_hgacd_async_region_meta', None) if phase == 'main' else None
            state = self._get_hgacd_async_state()
            contact_mass = None if state is None else state.get('region_contact_mass')
            contact_hit = None if state is None else state.get('region_contact_hit')
            contact_prob = None if state is None else state.get('region_contact_prob')
            if contact_mass is not None:
                contact_mass = contact_mass.to(device).float().clamp(0.0, 1.0)
            if contact_hit is not None:
                contact_hit = contact_hit.to(device).float().clamp(0.0, 1.0)
            if contact_prob is not None:
                contact_prob = contact_prob.to(device).float().clamp(0.0, 1.0)
            trace_policy = str(async_cfg.get('contact_refine_policy', async_cfg.get('controller', controller))).lower()
            trace_prior_weight = float(async_cfg.get('bio_prior_weight', async_cfg.get('cdr3_prior_weight', 0.20)))
            trace_bio_prior = {
                int(CDR.H1): 0.35, int(CDR.H2): 0.65, int(CDR.H3): 1.00,
                int(CDR.L1): 0.35, int(CDR.L2): 0.20, int(CDR.L3): 0.70,
            }
            for b in range(N):
                if not bool(trace_selector[b].item()):
                    continue
                regions = []
                meta_by_rid = {}
                if region_meta_all is not None and b < len(region_meta_all) and region_meta_all[b] is not None:
                    meta_by_rid = {
                        int(meta['rid']): meta
                        for meta in region_meta_all[b]
                        if meta is not None and 'rid' in meta
                    }
                for rid, rname in cdr_items:
                    rmask = mask_cdr[b] & (region_type_trace[b] == rid)
                    if not bool(rmask.any().item()):
                        continue
                    entry = {
                        'region': rname,
                        'rid': rid,
                        't_before': int(t_before[b, rmask].max().detach().cpu().item()) if t_before is not None else None,
                        't_after': int(t_after[b, rmask].max().detach().cpu().item()) if t_after is not None else None,
                        'active': bool((active_mask[b] & rmask).any().detach().cpu().item()),
                        'slow': bool((slow_mask[b] & rmask).any().detach().cpu().item()),
                    }
                    if entropy_tensor is not None and entropy_tensor.shape[:2] == t_before.shape[:2]:
                        entry['entropy'] = float(entropy_tensor[b, rmask].mean().detach().cpu().item())
                    if support_tensor is not None and support_tensor.shape[:2] == t_before.shape[:2]:
                        entry['support'] = float(support_tensor[b, rmask].mean().detach().cpu().item())
                    if p_cur is not None:
                        p_cur_ang = self._unnormalize_position(p_cur)
                        p_ref_ang = self._unnormalize_position(trace_p_ref)
                        residual = (p_cur_ang[b, rmask] - p_ref_ang[b, rmask]).float().norm(dim=-1)
                        entry['ca_residual'] = float(residual.mean().detach().cpu().item())
                        entry['ca_residual_unit'] = 'angstrom'
                        if p_prev is not None:
                            p_prev_ang = self._unnormalize_position(p_prev)
                            update = (p_cur_ang[b, rmask] - p_prev_ang[b, rmask]).float().norm(dim=-1)
                            entry['update_norm'] = float(update.mean().detach().cpu().item())
                            entry['update_norm_unit'] = 'angstrom'
                    if trace_region_states and isinstance(step_idx_value, int) and step_idx_value % trace_state_stride == 0:
                        encoder = getattr(self.eps_net, 'encoder', None)
                        conditioners = getattr(encoder, 'region_conditioners', []) if encoder is not None else []
                        state_vectors = {}
                        for layer_idx, cond in enumerate(conditioners):
                            decoder = getattr(cond, 'decoder', None)
                            extractor = getattr(decoder, 'fined_region_extractor', None) if decoder is not None else None
                            if extractor is None:
                                continue
                            region_idx = int(rid) - 1
                            tensors = {
                                'region_post': getattr(extractor, '_last_R_prime', None),
                                'self_guide': getattr(extractor, '_last_self_guide', None),
                                'cross_guide': getattr(extractor, '_last_cross_guide', None),
                                'final_guide': getattr(extractor, '_last_final_guide', None),
                                'gate_weights': getattr(extractor, '_last_gate_weights', None),
                            }
                            for state_name, state_tensor in tensors.items():
                                if state_tensor is None or b >= state_tensor.size(0):
                                    continue
                                if state_name == 'region_post':
                                    if region_idx < 0 or region_idx >= state_tensor.size(1):
                                        continue
                                    vec = state_tensor[b, region_idx].detach().float().cpu().tolist()
                                else:
                                    if state_tensor.dim() < 3:
                                        continue
                                    vec = state_tensor[b, rmask].detach().float().mean(dim=0).cpu().tolist()
                                state_vectors[f'l{layer_idx}_{state_name}'] = [round(float(x), 6) for x in vec]
                        if state_vectors:
                            entry['state_vectors'] = state_vectors
                    meta = meta_by_rid.get(rid)
                    if meta is not None:
                        for key in (
                            'contact', 'prior', 'interface_score', 'risk', 'policy',
                            'controller_risk', 'geom_risk', 'geom_intra_bond', 'geom_seam_bond',
                            'geom_omega', 'geom_feedback',
                            'selection_reason', 'target_active', 'selected_count', 'max_clock',
                            'min_clock', 'score_center', 'score_spread', 'refine_phase',
                            'refine_clock_threshold', 'contact_spread', 'effective_policy', 'gate_enabled',
                            'gate_interface_min_spread', 'gate_contact_min_spread',
                            'mixture_mode', 'mixture_high_interface_spread', 'mixture_high_contact_spread',
                            'mixture_low_interface_spread', 'mixture_very_low_interface_spread',
                            'mixture_very_low_contact_spread',
                            'tristate_mode', 'tristate_low_contact_spread', 'tristate_high_contact_spread',
                            'dual_guard_mode', 'dual_guard_contact_floor', 'dual_guard_risk_floor',
                            'dynamic_fraction', 'max_lag',
                            'min_active_fraction', 'prior_weight',
                            'schedule_progress', 'schedule_gamma', 'schedule_target_t',
                            'schedule_score', 'schedule_lead',
                        ):
                            if key in meta and meta[key] is not None:
                                entry[key] = meta[key]
                    else:
                        region_idx = rid - 1
                        contact_score = None
                        if contact_prob is not None and region_idx < contact_prob.size(1):
                            contact_score = contact_prob[b, region_idx]
                        elif contact_mass is not None and region_idx < contact_mass.size(1):
                            contact_score = contact_mass[b, region_idx]
                            if contact_hit is not None and region_idx < contact_hit.size(1):
                                contact_score = 0.5 * contact_score + 0.5 * contact_hit[b, region_idx]
                        elif contact_hit is not None and region_idx < contact_hit.size(1):
                            contact_score = contact_hit[b, region_idx]
                        if contact_score is not None:
                            contact_value = float(contact_score.detach().cpu().item())
                            prior_value = float(trace_bio_prior.get(rid, 0.0))
                            entry['contact'] = contact_value
                            entry['prior'] = prior_value
                            entry['interface_score'] = max(
                                0.0,
                                min(1.0, (1.0 - trace_prior_weight) * contact_value + trace_prior_weight * prior_value),
                            )
                    entry.setdefault('policy', trace_policy)
                    regions.append(entry)
                trace_records[b].append({
                    'label': str(label),
                    'phase': str(phase),
                    'step_idx': int(step_idx_value) if isinstance(step_idx_value, int) else None,
                    'controller': str(controller),
                    'regions': regions,
                })

        if compute_loss:
            if 'pos_heavyatom_native' in batch:
                p_native_ca = batch['pos_heavyatom_native'][:, :, BBHeavyAtom.CA, :]
                p_ref = self._normalize_position(p_native_ca)
            else:
                p_ref = p.clone()
            v_ref = v.clone()
            s_ref = s.clone()
            R_ref = so3vec_to_rotation(v_ref)
            loss_weights = self.loss_weights if self.loss_weights else {}
            cdr_rot_w = float(loss_weights.get('cdr_rot', 1.0))
            cdr_pos_w = float(loss_weights.get('cdr_pos', 1.0))
            cdr_seq_w = float(loss_weights.get('cdr_seq', 1.0))

            def _masked_mean(value, mask):
                mask_f = mask.to(value.dtype)
                while mask_f.dim() < value.dim():
                    mask_f = mask_f.unsqueeze(-1)
                denom = mask_f.sum().clamp_min(1.0)
                return (value * mask_f).sum() / denom

            def _record_async_path_loss(label, phase, active_mask, slow_mask, R_pred, eps_p_pred, logits_pred):
                eval_mask = mask_cdr & mask_res & (t_res > 0)
                if not bool(eval_mask.any().item()):
                    return
                rot_map = rotation_matrix_cosine_loss(R_pred, R_ref)
                eps_true = self._compute_true_position_eps(p_t, p_ref, mask_cdr, t_res)
                pos_map = F.mse_loss(eps_p_pred, eps_true, reduction='none').sum(dim=-1)
                if mask_seq_generate.any():
                    post_true = self.trans_seq.posterior(s_t, s_ref, t_res, region_nm='cdr')
                    s0_prob = F.softmax(logits_pred, dim=-1)
                    post_pred = self.trans_seq.posterior(s_t, s0_prob, t_res, region_nm='cdr')
                    seq_map = F.kl_div(torch.log(post_pred + 1e-8), post_true, reduction='none', log_target=False).sum(dim=-1)
                    cdr_seq = _masked_mean(seq_map, eval_mask)
                else:
                    cdr_seq = rot_map.new_tensor(0.0)
                cdr_rot = _masked_mean(rot_map, eval_mask)
                cdr_pos = _masked_mean(pos_map, eval_mask)
                overall = cdr_rot_w * cdr_rot + cdr_pos_w * cdr_pos + cdr_seq_w * cdr_seq
                region_mask = mask_cdr & mask_res
                t_values = t_res[region_mask].float()
                denom = region_mask.float().sum().clamp_min(1.0)
                active_rate = (active_mask & region_mask).float().sum() / denom
                hold_rate = 1.0 - active_rate
                slow_rate = (slow_mask & region_mask).float().sum() / denom
                loss_history[str(label)] = {
                    'phase': phase,
                    'cdr_rot': float(cdr_rot.detach().cpu().item()),
                    'cdr_pos': float(cdr_pos.detach().cpu().item()),
                    'cdr_seq': float(cdr_seq.detach().cpu().item()),
                    'overall': float(overall.detach().cpu().item()),
                    'active_rate': float(active_rate.detach().cpu().item()),
                    'hold_rate': float(hold_rate.detach().cpu().item()),
                    'slow_rate': float(slow_rate.detach().cpu().item()),
                    't_mean': float(t_values.mean().detach().cpu().item()) if t_values.numel() else 0.0,
                    't_std': float(t_values.std(unbiased=False).detach().cpu().item()) if t_values.numel() else 0.0,
                    't_min': float(t_values.min().detach().cpu().item()) if t_values.numel() else 0.0,
                    't_max': float(t_values.max().detach().cpu().item()) if t_values.numel() else 0.0,
                }

        region_governed = bool(async_cfg.get('region_governed', False))
        if region_governed and trainmatch_controller:
            desc = 'HGACD-TrainMatchAsync'
        elif region_governed and context_first_window_controller:
            desc = 'HGACD-ContextFirstWindowAsync'
        elif region_governed and interface_first_window_controller:
            desc = 'HGACD-InterfaceFirstWindowAsync'
        elif region_governed and balanced_window_controller:
            if controller in {'graph_early_interface_balanced_window_async', 'early_interface_balanced_window_async'}:
                desc = 'HGACD-EarlyInterfaceBalancedWindowAsync'
            elif controller in {'graph_gated_early_balanced_window_async', 'gated_early_balanced_window_async'}:
                desc = 'HGACD-GatedEarlyBalancedWindowAsync'
            elif controller in {'graph_dockq_adaptive_mixture_window_async', 'dockq_adaptive_mixture_window_async'}:
                desc = 'HGACD-DockQAdaptiveMixtureWindowAsync'
            elif controller in {'graph_dockq_spread_tristate_window_async', 'dockq_spread_tristate_window_async'}:
                desc = 'HGACD-DockQSpreadTristateWindowAsync'
            elif controller in {'graph_dockq_dual_guard_window_async', 'dockq_dual_guard_window_async'}:
                desc = 'HGACD-DockQDualGuardWindowAsync'
            elif controller in {'graph_prior_adaptive_sync_async_gated_window_async', 'prior_adaptive_sync_async_gated_window_async'}:
                desc = 'HGACD-PriorAdaptiveSyncAsyncGatedWindowAsync'
            else:
                desc = 'HGACD-PriorAdaptiveBalancedWindowAsync'
        elif region_governed and contact_refine_window_controller:
            if controller in {'graph_early_interface_refine_window_async', 'early_interface_refine_window_async', 'graph_dockq_refine_window_async', 'dockq_refine_window_async'}:
                desc = 'HGACD-EarlyInterfaceRefineWindowAsync'
            else:
                desc = 'HGACD-ContactRefineWindowAsync'
        elif region_governed and dockq_window_controller:
            desc = 'HGACD-DockQWindowAsync'
        elif region_governed and dynamic_schedule_controller:
            desc = 'HGACD-DynamicScheduleAsync'
        elif region_governed and graph_window_controller:
            desc = 'HGACD-GraphWindowAsync'
        elif region_governed and legacy_controller:
            desc = 'HGACD-RegionAsync'
        elif region_governed:
            desc = 'HGACD-RuleAsync'
        else:
            desc = 'AsynDM-CDR'
        sync_warmup_steps = _resolve_sync_warmup_steps(async_cfg, self.num_steps)
        if sync_warmup_steps > 0 and pbar:
            desc = f'{desc}-Warmup{sync_warmup_steps}'
        piter = functools.partial(tqdm, total=self.num_steps, desc=desc) if pbar else (lambda x: x)
        last_slow = mask_cdr
        for step_idx in piter(range(self.num_steps)):
            beta = self._residue_beta_from_t(t_res, mask_cdr)
            self._set_hgacd_contact_positions(batch, p_t)
            if self.guidance_scale != 1.0:
                v_next_c, R_next_c, eps_p_c, logits_c, _ = self.eps_net(batch, v_t, p_t, s_t, res_feat, pair_feat, beta)
                res_feat_uncond, pair_feat_uncond, _, _ = build_cfg_unconditional_inputs(
                    res_feat, pair_feat, batch['region_type'],
                    epitope_region_ids=(constants.AG.EPI_CORE, constants.AG.EPI_RIM),
                )
                v_next_u, R_next_u, eps_p_u, logits_u, _ = self.eps_net(batch, v_t, p_t, s_t, res_feat_uncond, pair_feat_uncond, beta)
                gs = self.guidance_scale
                v_pred = v_next_u + gs * (v_next_c - v_next_u)
                R_pred = R_next_u + gs * (R_next_c - R_next_u)
                eps_p = eps_p_u + gs * (eps_p_c - eps_p_u)
                logits = logits_u + gs * (logits_c - logits_u)
            else:
                v_pred, R_pred, eps_p, logits, _ = self.eps_net(batch, v_t, p_t, s_t, res_feat, pair_feat, beta)
            linear_desired = max(self.num_steps - step_idx - 1, 0)
            in_sync_warmup = step_idx < sync_warmup_steps
            batch['_hgacd_async_region_meta'] = None
            if in_sync_warmup:
                slow_mask = torch.zeros_like(mask_cdr, dtype=torch.bool)
                entropy = torch.zeros_like(t_res, dtype=logits.dtype)
                region_support = anchor_support
                if region_governed:
                    batch['_hgacd_region_support_res'] = region_support.detach()
            elif region_governed:
                if trainmatch_controller:
                    slow_mask = mask_cdr & (t_res > 0) & (trainmatch_offsets > 0)
                    entropy = torch.zeros_like(t_res, dtype=logits.dtype)
                    region_support = torch.zeros_like(anchor_support)
                elif context_first_window_controller or interface_first_window_controller or balanced_window_controller or contact_refine_window_controller:
                    contact_cfg = dict(async_cfg)
                    if context_first_window_controller:
                        contact_cfg['contact_refine_policy'] = 'context_first'
                    elif interface_first_window_controller:
                        contact_cfg['contact_refine_policy'] = 'interface_first'
                    elif balanced_window_controller:
                        if controller in {'graph_early_interface_balanced_window_async', 'early_interface_balanced_window_async'}:
                            contact_cfg['contact_refine_policy'] = 'early_interface_balanced'
                        elif controller in {'graph_gated_early_balanced_window_async', 'gated_early_balanced_window_async'}:
                            contact_cfg['contact_refine_policy'] = 'gated_early_interface_balanced'
                        elif controller in {'graph_dockq_adaptive_mixture_window_async', 'dockq_adaptive_mixture_window_async'}:
                            contact_cfg['contact_refine_policy'] = 'dockq_adaptive_mixture'
                        elif controller in {'graph_dockq_spread_tristate_window_async', 'dockq_spread_tristate_window_async'}:
                            contact_cfg['contact_refine_policy'] = 'dockq_spread_tristate'
                        elif controller in {'graph_dockq_dual_guard_window_async', 'dockq_dual_guard_window_async'}:
                            contact_cfg['contact_refine_policy'] = 'dockq_dual_guard'
                        elif controller in {'graph_prior_adaptive_sync_async_gated_window_async', 'prior_adaptive_sync_async_gated_window_async'}:
                            contact_cfg['contact_refine_policy'] = 'prior_adaptive_sync_async_gated'
                        else:
                            contact_cfg['contact_refine_policy'] = 'balanced'
                    else:
                        if controller in {'graph_early_interface_refine_window_async', 'early_interface_refine_window_async', 'graph_dockq_refine_window_async', 'dockq_refine_window_async'}:
                            contact_cfg['contact_refine_policy'] = 'early_interface_refine'
                        else:
                            contact_cfg['contact_refine_policy'] = contact_cfg.get('contact_refine_policy', 'adaptive')
                    active_mask, slow_mask, entropy, region_support = self._select_hgacd_contact_refine_window_next_active(
                        batch, logits, mask_cdr, anchor_support, contact_cfg, t_res
                    )
                elif dockq_window_controller:
                    active_mask, slow_mask, entropy, region_support = self._select_hgacd_dockq_window_next_active(
                        batch, logits, mask_cdr, anchor_support, async_cfg, t_res
                    )
                elif dynamic_schedule_controller:
                    active_mask, slow_mask, entropy, region_support = self._select_hgacd_dynamic_schedule_next_active(
                        batch, logits, mask_cdr, anchor_support, async_cfg, t_res, step_idx
                    )
                elif bioprior_window_controller:
                    active_mask, slow_mask, entropy, region_support = self._select_hgacd_bioprior_window_next_active(
                        batch, logits, mask_cdr, anchor_support, async_cfg, t_res
                    )
                elif graph_window_controller:
                    active_mask, slow_mask, entropy, region_support = self._select_hgacd_window_adaptive_next_active(
                        batch, logits, mask_cdr, anchor_support, async_cfg, t_res
                    )
                elif legacy_controller:
                    slow_mask, entropy, region_support = self._select_hgacd_region_slow_mask(
                        batch, logits, mask_cdr & (t_res > 0), anchor_support, async_cfg
                    )
                else:
                    slow_mask, entropy, region_support = self._select_hgacd_rule_based_slow_mask(
                        batch, logits, mask_cdr & (t_res > 0), anchor_support, async_cfg, t_res=t_res
                    )
                batch['_hgacd_region_support_res'] = region_support.detach()
            else:
                slow_mask, entropy = self._select_asyndm_slow_mask(
                    logits, mask_cdr & (t_res > 0), anchor_support, async_cfg
                )
            last_slow = slow_mask
            if in_sync_warmup:
                active = mask_cdr & (t_res > linear_desired) & (t_res > 0)
            elif graph_window_controller:
                active = active_mask
            elif trainmatch_controller:
                desired = torch.clamp(
                    torch.full_like(t_res, linear_desired) + trainmatch_offsets,
                    min=0,
                    max=self.num_steps,
                )
            elif legacy_controller:
                slow_power = float(async_cfg.get('slow_power', 2.0))
                slow_desired = int(round(self.num_steps - (float(step_idx + 1) ** slow_power) / (float(self.num_steps) ** (slow_power - 1.0))))
                slow_desired = max(min(slow_desired, self.num_steps), 0)
                desired = torch.where(
                    slow_mask,
                    torch.full_like(t_res, slow_desired),
                    torch.full_like(t_res, linear_desired),
                )
            else:
                lag_bias = max(1, int(async_cfg.get('region_max_lag', 8)))     
                slow_desired = min(self.num_steps, linear_desired + lag_bias)
                slow_desired = max(min(slow_desired, self.num_steps), 0)
                desired = torch.where(
                    slow_mask,
                    torch.full_like(t_res, slow_desired),
                    torch.full_like(t_res, linear_desired),
                )
            if not graph_window_controller and not in_sync_warmup:
                active = mask_cdr & (t_res > desired) & (t_res > 0)
            if region_governed and not graph_window_controller and not in_sync_warmup:
                active = self._apply_hgacd_region_lag(active, t_res, batch['region_type'].to(device), mask_cdr, async_cfg)
            if not active.any() and (t_res[mask_cdr] > 0).any():
                active = mask_cdr & (t_res == t_res[mask_cdr].max())
            structure_risk_cfg = async_cfg.get('structure_risk_feedback', {})
            if not isinstance(structure_risk_cfg, dict):
                structure_risk_cfg = {'enabled': bool(structure_risk_cfg)}
            if (not in_sync_warmup) and bool(structure_risk_cfg.get('enabled', False)):
                struct_risk = self._compute_hgacd_structure_risk(
                    batch, v_t, p_t, s_t, mask_cdr, structure_risk_cfg
                )
                active, slow_mask = self._apply_hgacd_structure_risk_feedback(
                    batch, active, slow_mask, t_res, mask_cdr, struct_risk, structure_risk_cfg
                )
                self._merge_hgacd_structure_risk_meta(batch)
                if not active.any() and (t_res[mask_cdr] > 0).any():
                    active = mask_cdr & (t_res == t_res[mask_cdr].max())
            t_before_trace = t_res.detach().clone() if trace_records is not None else None
            p_before_trace = p_t.detach().clone() if trace_records is not None else None
            if compute_loss:
                _record_async_path_loss(self.num_steps - step_idx, 'main', active, slow_mask, R_pred, eps_p, logits)
            v_t, p_t, s_t = self._apply_residue_group_denoise(
                v_t, p_t, s_t, v_pred, eps_p, logits,
                active, t_res, mask_seq_generate, mask_struct_generate,
            )
            t_next = torch.where(active, torch.clamp(t_res - 1, min=0), t_res)
            if trace_records is not None:
                phase = 'sync_warmup' if in_sync_warmup else 'main'
                _record_async_trace_step(
                    self.num_steps - step_idx, phase, step_idx, active, slow_mask,
                    t_before_trace, t_next.detach(), entropy, region_support,
                    p_prev=p_before_trace, p_cur=p_t.detach(),
                )
            t_res = t_next

        drain = 0
        max_drain = int(async_cfg.get('max_drain_steps', self.num_steps))
        while (t_res[mask_cdr] > 0).any() and drain < max_drain:
            beta = self._residue_beta_from_t(t_res, mask_cdr)
            self._set_hgacd_contact_positions(batch, p_t)
            v_pred, R_pred, eps_p, logits, _ = self.eps_net(batch, v_t, p_t, s_t, res_feat, pair_feat, beta)
            active = mask_cdr & (t_res > 0)
            t_before_trace = t_res.detach().clone() if trace_records is not None else None
            p_before_trace = p_t.detach().clone() if trace_records is not None else None
            drain_slow = torch.zeros_like(mask_cdr, dtype=torch.bool)
            if compute_loss:
                _record_async_path_loss(f'drain_{drain + 1}', 'drain', active, drain_slow, R_pred, eps_p, logits)
            v_t, p_t, s_t = self._apply_residue_group_denoise(
                v_t, p_t, s_t, v_pred, eps_p, logits,
                active, t_res, mask_seq_generate, mask_struct_generate,
            )
            t_next = torch.where(active, torch.clamp(t_res - 1, min=0), t_res)
            if trace_records is not None:
                _record_async_trace_step(
                    f'drain_{drain + 1}', 'drain', self.num_steps + drain, active, drain_slow,
                    t_before_trace, t_next.detach(), None, None,
                    p_prev=p_before_trace, p_cur=p_t.detach(),
                )
            t_res = t_next
            drain += 1

        if not mask_struct_generate.any():
            v_t, p_t = v, p
        if not mask_seq_generate.any():
            s_t = s
        traj[0] = (v_t, self._unnormalize_position(p_t), s_t)
        batch['_asyndm_last_slow_mask'] = last_slow.detach()
        batch['_asyndm_final_t_res'] = t_res.detach()
        if trace_records is not None:
            batch['_asyndm_trace_records'] = trace_records
        if compute_loss:
            batch['_asyndm_loss_history'] = loss_history
            batch['_asyndm_drain_steps'] = drain
            return traj, loss_history
        return traj

    @torch.no_grad()
    def sample(self, batch, v, p, s, res_feat, pair_feat, mask_seq_generate, mask_struct_generate, mask_dict, bfactor,
               pbar=False, compute_loss=False):
        """
        采样函数

        Args:
            compute_loss: 是否在采样过程中计算loss（用于验证）
        """
        N, L = v.shape[:2]
        device = v.device
        p = self._normalize_position(p)

        if compute_loss:
            if 'pos_heavyatom_native' in batch:
                p_native_ca = batch['pos_heavyatom_native'][:, :, BBHeavyAtom.CA, :]  # (N, L, 3) CA位置
                p_0 = self._normalize_position(p_native_ca)
                print(f"[推理Loss修正] 使用native位置: p_0 shape={p_0.shape}")
            else:
                p_0 = p.clone()

            v_0 = v.clone()
            s_0 = s.clone()
            R_0 = so3vec_to_rotation(v_0)
        else:
            v_0, p_0, s_0 = v.clone(), p.clone(), s.clone()
            R_0 = so3vec_to_rotation(v_0)

        noise_sampling = _get_runtime_noise_sampling(batch)
        noise_mode = noise_sampling.get('noise_mode', 'cdr_only')
        mask_cdr = mask_dict['mask_cdr']
        mask_seq_generate = mask_cdr.clone()
        mask_struct_generate = mask_cdr.clone()
        if is_epitope_noise_mode(noise_mode) and mask_dict.get('mask_soft_antigen') is not None:
            mask_struct_generate = mask_struct_generate | mask_dict['mask_soft_antigen']

        loss_history = {t: {} for t in range(self.num_steps + 1)} if compute_loss else None

        async_cfg = noise_sampling.get('asyndm_residue', {}) if isinstance(noise_sampling, dict) else {}
        if async_cfg and bool(async_cfg.get('enabled', False)):
            if noise_mode != 'cdr_only':
                raise ValueError('AsynDM-CDR residue async sampler currently supports noise_mode=cdr_only only.')
            return self.sample_asyndm_residue(
                batch, v, p, s, res_feat, pair_feat,
                mask_seq_generate, mask_struct_generate, mask_dict, bfactor,
                pbar=pbar, compute_loss=compute_loss, async_cfg=async_cfg,
            )

        if compute_loss:
            mask_soft_antigen = mask_dict.get('mask_soft_antigen', None)
            mask_antigen = mask_dict.get('mask_full_antigen', mask_dict.get('mask_antigen', None))
            atom_mask = batch.get('atom_mask', None)
            geometry_calc = GeometryLossCalculator(
                cdr_mask=mask_cdr,
                mask_soft_antigen=mask_soft_antigen,
                mask_antigen=mask_antigen,
                atom_mask=atom_mask
            )
            weight_scheduler = DynamicLossWeightScheduler(eta=0.5, override_weights=self.loss_weights if self.loss_weights else None)

        t_dict, t, beta_dict, beta = self.build_sample_t_schedule(N, L, self.num_steps, noise_sampling, mask_dict)
        if mask_struct_generate.any():
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)  # 全噪音
            p_rand = torch.randn_like(p)

            mask_cdr_only = mask_struct_generate
            if 'epitope' in t_dict.get('noise_mode', '') and mask_dict.get('mask_soft_antigen') is not None:
                mask_soft_antigen = mask_dict['mask_soft_antigen']
                mask_cdr_only = mask_struct_generate & (~mask_soft_antigen)

            v_t = torch.where(mask_cdr_only[:, :, None].expand_as(v), v_rand, v)
            p_t = torch.where(mask_cdr_only[:, :, None].expand_as(p), p_rand, p)

            if is_epitope_noise_mode(t_dict.get('noise_mode', '')):
                t_e = t_dict['epitope']
                beta_e = self.trans_pos.var_sched.betas_epi[t_e]  # (N,)
                beta_dict['epitope'] = beta_e
                mask_soft_antigen = mask_dict['mask_soft_antigen']
                beta = torch.where(mask_soft_antigen, beta_e.unsqueeze(1).expand(N, L), beta)
                scale = None

                v_epi, _ = self.trans_rot.add_noise(v, mask_soft_antigen, t_dict['epitope'],
                                                    region_nm='epitope', scale=scale)

                p_epi, _ = self.trans_pos.add_noise(
                    p, mask_soft_antigen, t_dict['epitope'], region_nm='epitope',
                    scale=scale
                )

                v_t = torch.where(mask_soft_antigen[:, :, None].expand_as(v_t), v_epi, v_t)
                p_t = torch.where(mask_soft_antigen[:, :, None].expand_as(p_t), p_epi, p_t)
        else:
            v_t, p_t = v, p

        if mask_seq_generate.any():
            s_rand = torch.randint_like(s, low=0, high=20)
            s_t = torch.where(mask_seq_generate, s_rand, s)
        else:
            s_t = s

        traj = {self.num_steps: (v_t, self._unnormalize_position(p_t), s_t)}


        piter = functools.partial(tqdm, total=self.num_steps, desc='Sampling') if pbar else (lambda x: x)

        for t in piter(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)

            t_current = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)
            t_dict['cdr'] = t_current
            if is_epitope_noise_mode(t_dict.get('noise_mode', '')):
                t_epitope_current = self._compute_async_epitope_t(t_current, noise_sampling)
                t_dict['epitope'] = t_epitope_current

            beta_c = self.trans_pos.var_sched.betas[t_current]
            beta = torch.where(mask_cdr, beta_c.unsqueeze(1).expand(N, L),
                               torch.zeros(N, L, device=self._dummy.device))
            if is_epitope_noise_mode(t_dict.get('noise_mode', '')):
                beta_e = self.trans_pos.var_sched.betas_epi[t_dict['epitope']]
                beta = torch.where(mask_soft_antigen, beta_e.unsqueeze(1).expand(N, L), beta)
                assert beta_e.max() <= beta_c.max(), f"beta_e > beta_c; {beta_e} > {beta_c}"
            if DEBUG_MODE:
                if is_epitope_noise_mode(t_dict.get('noise_mode', '')):
                    print(
                        f"[sample debug]: t={t_current[:3]} === beta_cdr={beta_c[:3]} === beta_e={beta_e[:3]} \n beta[mask_cdr]={beta[:3][mask_cdr[:3]]}] \n bata[mask_epi]={beta[:3][mask_cdr[:3]]}")

            self._set_hgacd_contact_positions(batch, p_t)
            if self.guidance_scale != 1.0:
                v_next_c, R_next_c, eps_p_c, s_0_logits_c, _ = self.eps_net(
                    batch, v_t, p_t, s_t,
                    res_feat, pair_feat, beta,
                )
                res_feat_uncond, pair_feat_uncond, _, _ = build_cfg_unconditional_inputs(
                    res_feat,
                    pair_feat,
                    batch['region_type'],
                    epitope_region_ids=(constants.AG.EPI_CORE, constants.AG.EPI_RIM),
                )
                v_next_u, R_next_u, eps_p_u, s_0_logits_u, _ = self.eps_net(
                    batch, v_t, p_t, s_t,
                    res_feat_uncond, pair_feat_uncond, beta,
                )
                gs = self.guidance_scale
                eps_p = eps_p_u + gs * (eps_p_c - eps_p_u)
                v_next = v_next_u + gs * (v_next_c - v_next_u)
                s_0_pred_logits = s_0_logits_u + gs * (s_0_logits_c - s_0_logits_u)
                R_next = R_next_u + gs * (R_next_c - R_next_u)
            else:
                v_next, R_next, eps_p, s_0_pred_logits, _ = self.eps_net(
                    batch, v_t, p_t, s_t,
                    res_feat, pair_feat, beta,
                )

            v_next, p_next, s_next = self.denoise_to_get_next_state(v_t, v_next, p_t, eps_p, noise_mode, mask_dict, t_dict, s_t, s_0_pred_logits)

            if compute_loss:
                mask_soft_antigen = mask_dict.get('mask_soft_antigen', None)

                if t == 100:
                    cdr_sum = mask_cdr.sum().item()
                    epi_sum = mask_soft_antigen.sum().item() if mask_soft_antigen is not None else 0
                    print("[Debug] mask_cdr sum={}, mask_soft_antigen sum={}".format(cdr_sum, epi_sum))


                R_gt = so3vec_to_rotation(v_0)
                R_pred_no_noise = R_next  # 网络直接输出
                R_denoised = so3vec_to_rotation(v_next)  # 含噪声E的去噪结果（用于生成）
                p_pred_physical = self._unnormalize_position(p_next)
                p_gt_physical = self._unnormalize_position(p_0)  # p_0现在是native位置（如果推理时保存了）
                if compute_loss and torch.rand(1).item() < 0.1:
                    pos_diff = (p_pred_physical - p_gt_physical).abs()
                    cdr_diff = pos_diff[mask_cdr].mean()
                    print(f"[推理Loss验证] cdr_pos物理空间差异: {cdr_diff.item():.2f} Å (应该<5Å)")
                

                epitope_eval_mask = self._build_epitope_loss_mask(mask_soft_antigen, t_dict.get('epitope'))
                compute_epitope_loss_step = epitope_eval_mask is not None and epitope_eval_mask.any()
                p_clean = self._compute_clean_position_from_eps(
                    p_t,
                    eps_p,
                    mask_cdr,
                    t_current,
                    epitope_mask=epitope_eval_mask,
                    t_epitope=t_dict.get('epitope') if compute_epitope_loss_step else None,
                )
                eps_p_true = self._compute_true_position_eps(
                    p_t,
                    p_0,
                    mask_cdr,
                    t_current,
                    epitope_mask=epitope_eval_mask,
                    t_epitope=t_dict.get('epitope') if compute_epitope_loss_step else None,
                )
                p_clean_physical = self._unnormalize_position(p_clean)

                bb_pos_pred_for_geometry = reconstruct_backbone(
                    R=R_pred_no_noise,  # 网络输出（不含噪声E）
                    t=p_clean_physical,  # 干净的去噪位置（不含随机噪声z）
                    aa=s_next,  # 使用去噪后的序列
                    chain_nb=batch['chain_nb'],
                    res_nb=batch['res_nb'],
                    mask=batch['mask']
                )
                bb_pos_pred_reconstructed = reconstruct_backbone(
                    R=R_denoised,
                    t=p_pred_physical,
                    aa=s_next,
                    chain_nb=batch['chain_nb'],
                    res_nb=batch['res_nb'],
                    mask=batch['mask']
                )

                if mask_seq_generate.any():
                    post_true = self.trans_seq.posterior(s_t, s, t_current, region_nm='cdr')  # (N, L, 20)
                    s_0_pred_prob = F.softmax(s_0_pred_logits, dim=-1)  # (N, L, 20)
                    post_pred = self.trans_seq.posterior(s_t, s_0_pred_prob, t_current, region_nm='cdr')  # (N, L, 20)
                else:
                    post_pred, post_true = None, None

                bb_pos_true_for_validation = reconstruct_backbone(
                    R=R_gt,  # 验证时的真实旋转矩阵
                    t=p_gt_physical,  # 验证时的真实位置（物理空间）
                    aa=s,  # 验证时的真实序列
                    chain_nb=batch['chain_nb'],
                    res_nb=batch['res_nb'],
                    mask=batch['mask']
                )

                geometry_calc.antigen_soft_mask = epitope_eval_mask

                loss_dict_raw = geometry_calc.compute_losses(
                    R_next=R_pred_no_noise, R_0=R_gt,  # 使用网络输出（不含噪声E）
                    eps_p_pred=eps_p, eps_p=eps_p_true,
                    post_pred=post_pred, post_true=post_true,  # FIX: 验证时也要计算seq后验
                    s_0_pred_logits=s_0_pred_logits, s_0=s,
                    bb_pos_pred=bb_pos_pred_for_geometry,  # 使用不含噪声E的骨架重建
                    chain_nb=batch['chain_nb'],
                    res_nb=batch['res_nb'],
                    mask=batch['mask'],
                    compute_cdr_loss=True,
                    compute_epitope_loss=compute_epitope_loss_step,
                    bfactor=bfactor,
                    bb_pos_true=bb_pos_true_for_validation,  # 新增：验证时也传入真实位置
                )

                process_norm = t_dict['cdr'].float() / self.num_steps
                weights_dict = weight_scheduler.compute_weights(process_norm)

                core_losses = CORE_LOSSES  # 使用constants.py中的定义
                loss_dict, weight_info, loss_dict_raw_returned = apply_dynamic_weights(
                    loss_dict_raw=loss_dict_raw,
                    geom_scaler=self.geom_scaler if hasattr(self, 'geom_scaler') else None,
                    t=torch.zeros_like(process_norm),  # 验证时t=0（结构已清晰）
                    use_geom_scaler=self.use_geom_scaler if hasattr(self, 'use_geom_scaler') else False,
                    weights_dict=weights_dict,
                    core_losses=core_losses
                )

                for key in loss_dict:
                    if key == 'overall':
                        loss_history[t]['overall'] = loss_dict[key].item()
                    else:
                        loss_history[t][key] = loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
                        if key in weight_info:
                            w = weight_info[key]
                            loss_history[t][key + '_weight'] = w.item() if isinstance(w, torch.Tensor) else w
                        if key + '_mean' in loss_dict_raw:
                            raw = loss_dict_raw[key + '_mean']
                            loss_history[t][key + '_raw'] = raw.item() if isinstance(raw, torch.Tensor) else raw

                if t % 20 == 0 or t == 1:
                    core_str = f"[t={t}] "
                    for k in ['cdr_rot', 'cdr_pos', 'cdr_seq', 'cdr_seq_nll', 'epitope_rot', 'epitope_pos', 'cdr_bone', 'epitope_bone', 'contact', 'clash']:
                        if k in loss_history[t]:
                            w_str = ""
                            if k + '_weight' in loss_history[t]:
                                w_str = f"(w={loss_history[t][k+'_weight']:.2f})"
                            core_str += f"{k}={loss_history[t][k]:.4f}{w_str} "
                    print(core_str)
            if not mask_struct_generate.any():
                v_next, p_next = v_t, p_t
            if not mask_seq_generate.any():
                s_next = s_t

            traj[t - 1] = (v_next, self._unnormalize_position(p_next), s_next)

        if compute_loss:
            return traj, loss_history
        return traj

    @torch.no_grad()
    def optimize(self, batch, v, p, s, opt_step, res_feat, pair_feat, mask_generate, mask_res,
                 sample_structure=True, sample_sequence=True, pbar=False, static_region_feats_info=None):
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        if sample_structure:
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t, region_nm='cdr')
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t, region_nm='cdr')
            v_t = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_t = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_t, p_t = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t, region_nm='cdr')
            s_t = torch.where(mask_generate, s_noisy, s)
        else:
            s_t = s

        traj = {opt_step: (v_t, self._unnormalize_position(p_t), s_t)}

        for t in range(opt_step, 0, -1):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)
            beta_t = self.trans_pos.var_sched.betas[t_tensor]
            beta = torch.where(
                mask_generate,
                beta_t.unsqueeze(1).expand(N, L),
                torch.zeros(N, L, device=self._dummy.device),
            )

            self._set_hgacd_contact_positions(batch, p_t)
            if self.guidance_scale != 1.0:
                v_next_c, R_next_c, eps_p_c, s_0_logits_c, _ = self.eps_net(
                    batch, v_t, p_t, s_t, res_feat, pair_feat, beta,
                )
                res_feat_uncond, pair_feat_uncond, _, _ = build_cfg_unconditional_inputs(
                    res_feat,
                    pair_feat,
                    batch['region_type'],
                    epitope_region_ids=(constants.AG.EPI_CORE, constants.AG.EPI_RIM),
                )
                v_next_u, R_next_u, eps_p_u, s_0_logits_u, _ = self.eps_net(
                    batch, v_t, p_t, s_t, res_feat_uncond, pair_feat_uncond, beta,
                )
                gs = self.guidance_scale
                v_next = v_next_u + gs * (v_next_c - v_next_u)
                R_next = R_next_u + gs * (R_next_c - R_next_u)
                eps_p = eps_p_u + gs * (eps_p_c - eps_p_u)
                s_0_pred_logits = s_0_logits_u + gs * (s_0_logits_c - s_0_logits_u)
            else:
                v_next, R_next, eps_p, s_0_pred_logits, _ = self.eps_net(
                    batch, v_t, p_t, s_t, res_feat, pair_feat, beta,
                )

            t_dict = {'cdr': t_tensor, 'noise_mode': 'cdr_only'}
            mask_dict = {'mask_cdr': mask_generate}
            v_next, p_next, s_next = self.denoise_to_get_next_state(
                v_t, v_next, p_t, eps_p, 'cdr_only', mask_dict, t_dict,
                s_t=s_t, s_0_pred_logits=s_0_pred_logits,
            )

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t - 1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])

        return traj
