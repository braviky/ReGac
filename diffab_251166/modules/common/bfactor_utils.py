"""
V3: B-factor utilities for adaptive epitope conditioning
Includes B-factor preprocessing, encoding, and coupled noise scheduling

=== 代码版本标记 (2026-04-08) ===
修复内容:
1. 旋转损失: 使用 Geodesic Loss (非 Cosine Embedding)
2. 序列预测: 移除 Softmax (修复双重 Softmax bug)
3. 权重配置: cdr_rot=5.0, cdr_seq=3.0, cdr_pos=2.0
=================================
"""

CODE_VERSION = "2026-04-08-fix-v2"
LOSS_TYPE_ROT = "Cosine"  # 实验测试  # 或 "Cosine"
LOSS_TYPE_SEQ = "Logits"    # 或 "Softmax" (旧版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from diffab_251166.utils.protein.constants import *
from diffab_251166.modules.common.geometry import get_backbone_dihedral_angles
from diffab_251166.modules.common.topology import get_consecutive_flag
import numpy as np
import os
import logging
from datetime import datetime


_debug_logger = None

def get_debug_logger():
    """获取调试日志器，输出到单独文件"""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = logging.getLogger('diffab_debug')
        _debug_logger.setLevel(logging.DEBUG)
        if not _debug_logger.handlers:
            log_dir = os.path.join(os.getcwd(), 'archive', 'debug_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            fh = logging.FileHandler(log_file, mode='a')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
            fh.setFormatter(formatter)
            _debug_logger.addHandler(fh)
    return _debug_logger

def debug_log(msg):
    """写入调试日志"""
    logger = get_debug_logger()
    logger.info(msg); print(msg)


def _assert_zero_loss_tensor(name, value, atol=1e-7):
    if torch.is_tensor(value):
        zero = torch.zeros_like(value.float())
        assert torch.allclose(value.float(), zero, atol=atol), \
            f"[Loss断言] {name} should be zero when epitope loss is disabled, got max={value.float().abs().max().item():.6f}"

DEBUG_MODE = False  # 主调试开关
DEBUG_LOSS_DETAILS = False  # 专门用于loss调试（只打印简要信息）
DEBUG_BFACTOR = False
DEBUG_WEIGHTS = False  # 权重调试日志开关（打开以监控loss计算）
GLOBAL_B_FACTOR_MEAN = 40.0   # 典型蛋白质B-factor均值 (Å²)
GLOBAL_B_FACTOR_STD = 20.0    # 典型蛋白质B-factor标准差 (Å²)
GLOBAL_B_FACTOR_MIN = 10.0    # 最低合理值（高质量晶体）
GLOBAL_B_FACTOR_MAX = 150.0   # 最高合理值（柔性区域）


def rotation_matrix_cosine_loss(R_next, R_true):
    """计算旋转矩阵的余弦损失"""
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

    if torch.rand(1).item() < 0.02:  # 2%概率打印
        angle_deg = angle.mean().item() * 180 / 3.14159
        cos_mean = cos_angle.mean().item()
        trace_mean = trace.mean().item()
        print(f"[GEODESIC] angle={angle_deg:.1f}°, cos={cos_mean:.4f}, trace={trace_mean:.4f}")
        if angle_deg > 150:
            print(f"[GEODESIC警告] 角度过大! R_pred和R_true几乎相反")
        if trace_mean < -0.5:
            print(f"[GEODESIC警告] trace异常! R_rel不是有效旋转")

    return angle ** 2


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Cosine embedding loss for rotation matrices (参考Diffab_add_region_feat)
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3
    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)
    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')
    loss = loss.reshape(size + [3]).sum(dim=-1)
    return loss


import torch
import torch.nn as nn



class GeomLossScaler(nn.Module):
    """
    自动学习的Bone/Omega权重缩放器

    公式: L_geom = 0.5 * exp(-s_b) * α_b(t) * L_bone + 0.5 * s_b
                   + 0.5 * exp(-s_ω) * α_ω(t) * L_ω + 0.5 * s_ω

    其中:
    - s_b, s_ω 是可学习参数 (nn.Parameter)
    - exp(-s_b), exp(-s_ω) 是自动学习的全局缩放
    - α_b(t), α_ω(t) 是diffusion timestep门控
    - +0.5*s 防止权重无脑推到极端 (类似uncertainty weighting)

    参考论文: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    def __init__(
        self,
        init_bone_weight=0.05,  # 初始bone权重 (s_b = -log(0.05) ≈ 3.0)
        init_omega_weight=0.01, # 初始omega权重 (s_ω = -log(0.01) ≈ 4.6)
        clamp_min=-2.0,         # s最小值 (exp(-s)最大≈7.4)
        clamp_max=8.0,          # s最大值 (exp(-s)最小≈0.0003)
    ):
        super().__init__()
        import math

        self.s_bone = nn.Parameter(
            torch.tensor(-math.log(init_bone_weight), dtype=torch.float32)
        )
        self.s_omega = nn.Parameter(
            torch.tensor(-math.log(init_omega_weight), dtype=torch.float32)
        )

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self._init_s_bone = -math.log(init_bone_weight)
        self._init_s_omega = -math.log(init_omega_weight)

        debug_log(f"[GeomLossScaler] 初始化完成:")
        debug_log(f"  init_bone_weight={init_bone_weight} → s_bone={self._init_s_bone:.2f}")
        debug_log(f"  init_omega_weight={init_omega_weight} → s_omega={self._init_s_omega:.2f}")
        debug_log(f"  clamp范围: [{clamp_min}, {clamp_max}]")

    def get_bone_alpha(self, t):
        """
        Bone的timestep门控: (1-t)^2
        t=1(高噪) → α=0 (不约束)
        t=0(清晰) → α=1 (强约束)
        """
        return torch.pow(1.0 - t, 2.0)

    def get_omega_alpha(self, t):
        """
        Omega的timestep门控: (1-t)^4, 更晚更保守
        t=1(高噪) → α=0
        t=0.5 → α=0.0625
        t=0(清晰) → α=1
        """
        return torch.pow(1.0 - t, 4.0)

    def forward(self, bone_loss, omega_loss, t):
        """
        Args:
            bone_loss: bone loss, 可以是标量或per-sample向量 (N,)
            omega_loss: omega loss, 可以是标量或per-sample向量 (N,)
            t: diffusion timestep tensor (N,) 或 scalar, 范围[0,1], t=1高噪, t=0低噪

        Returns:
            geom_loss: 加权后的几何损失
            aux_info: 辅助信息字典 (用于logging)
        """
        s_bone = torch.clamp(self.s_bone, self.clamp_min, self.clamp_max)
        s_omega = torch.clamp(self.s_omega, self.clamp_min, self.clamp_max)

        bone_scale = torch.exp(-s_bone)
        omega_scale = torch.exp(-s_omega)

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=bone_loss.device if isinstance(bone_loss, torch.Tensor) else 'cpu')
        t = t.float()

        alpha_b = self.get_bone_alpha(t)  # 可能是(N,)或scalar
        alpha_o = self.get_omega_alpha(t)  # 可能是(N,)或scalar

        if not isinstance(bone_loss, torch.Tensor):
            bone_loss = torch.tensor(bone_loss, device=t.device)
        if not isinstance(omega_loss, torch.Tensor):
            omega_loss = torch.tensor(omega_loss, device=t.device)

        is_per_sample = (t.dim() > 0 and t.shape[0] > 1) and                         (bone_loss.dim() > 0 and bone_loss.shape[0] == t.shape[0])

        if is_per_sample:
            bone_data_vec = 0.5 * bone_scale * alpha_b * bone_loss  # (N,)
            omega_data_vec = 0.5 * omega_scale * alpha_o * omega_loss  # (N,)
            geom_data_loss = bone_data_vec.mean() + omega_data_vec.mean()  # scalar - 用mean！

            bone_data = bone_data_vec.mean().item() if isinstance(bone_data_vec, torch.Tensor) else bone_data_vec
            omega_data = omega_data_vec.mean().item() if isinstance(omega_data_vec, torch.Tensor) else omega_data_vec
            alpha_b_mean = alpha_b.mean().item()
            alpha_o_mean = alpha_o.mean().item()
        else:
            alpha_b_mean = alpha_b.mean().item() if isinstance(alpha_b, torch.Tensor) else alpha_b
            alpha_o_mean = alpha_o.mean().item() if isinstance(alpha_o, torch.Tensor) else alpha_o

            bone_data = 0.5 * bone_scale * alpha_b_mean * bone_loss.item()
            omega_data = 0.5 * omega_scale * alpha_o_mean * omega_loss.item()
            geom_data_loss = torch.tensor(bone_data + omega_data, device=t.device)

        bone_reg = 0.5 * s_bone
        omega_reg = 0.5 * s_omega
        geom_reg_loss = bone_reg + omega_reg

        geom_loss = geom_data_loss + geom_reg_loss

        aux_info = {
            "s_bone": s_bone.detach().item(),
            "s_omega": s_omega.detach().item(),
            "bone_scale": bone_scale.detach().item(),
            "omega_scale": omega_scale.detach().item(),
            "alpha_b": alpha_b_mean,  # 使用平均alpha
            "alpha_o": alpha_o_mean,  # 使用平均alpha
            "geom_data_loss": geom_data_loss.detach().item() if isinstance(geom_data_loss, torch.Tensor) else geom_data_loss,
            "geom_reg_loss": geom_reg_loss.detach().item() if isinstance(geom_reg_loss, torch.Tensor) else geom_reg_loss,
            "bone_data": bone_data,
            "omega_data": omega_data,
            "bone_loss_raw": bone_loss.detach().mean().item() if (isinstance(bone_loss, torch.Tensor) and bone_loss.dim() > 0) else (bone_loss.item() if isinstance(bone_loss, torch.Tensor) else bone_loss),
            "omega_loss_raw": omega_loss.detach().mean().item() if (isinstance(omega_loss, torch.Tensor) and omega_loss.dim() > 0) else (omega_loss.item() if isinstance(omega_loss, torch.Tensor) else omega_loss),
            "is_per_sample": is_per_sample,
        }

        return geom_loss, aux_info



class DynamicLossWeightScheduler(nn.Module):
    def __init__(self, eta=1.0, override_weights=None):
        super().__init__()
        default_weights = {
            'cdr_rot': 1.0,
            'cdr_pos': 1.0,  # 导航基准，高强度
            'cdr_seq': 1.0,
            'cdr_seq_nll': 1.0,
            'epitope_rot': 1.0,
            'epitope_pos': 1.0,  # 基础权重 1.0
            'contact': 1.0,  # 早期引导
            'clash': 2.0,  # 后期精修，高权重
            'bsa': 1.0,  # 埋藏面积损失
            'shape': 0.5,  # 形状互补损失
            'cdr_bone': 1.5,  # CDR肽键长度约束
            'cdr_omega': 0.75,  # CDR omega角度约束
            'epitope_bone': 1.5,  # Epitope肽键长度约束
            'epitope_omega': 0.75,  # Epitope omega角度约束
        }
        if override_weights:
            for k, v in override_weights.items():
                if k in default_weights:
                    default_weights[k] = v
        self.base_weights = default_weights
        if DEBUG_WEIGHTS:
            debug_log(f"[DynamicLossWeightScheduler] Final base_weights: {self.base_weights}")


        self.eta = eta

    def compute_weights(self, process):
        """
        Four-layer dynamic weight design:
        Layer 1: Base generation (cdr_rot/pos/seq) - always active
        Layer 2: Conditional guidance (epitope) - late enhancement
        Layer 3: Physical hard constraints (bond/omega/clash) - very late
        Layer 4: Interface optimization (contact/bsa/shape) - mid window

        Time axis: t=1.0 (early/noisy) -> t=0.0 (late/clear)

        Return configured base weights directly.
        This project uses explicit config weights instead of the temporary
        fixed-all-ones diagnostic mode.
        """
        if DEBUG_WEIGHTS:
            debug_log(f"[compute_weights] Using configured base weights: {self.base_weights}")
        return dict(self.base_weights)

        w_cdr_rot = 3.0 * (1.0 - 0.2 * t)  # t=1.0 -> 2.4, t=0.0 -> 3.0

        w_cdr_pos = 2.0

        w_cdr_seq = 0.5 + 2.5 * torch.pow(1.0 - t, 2.0)  # t=1.0 -> 0.5, t=0.0 -> 3.0

        w_epitope_pos = 1.0 + 2.0 * torch.pow(1.0 - t, 2.0)  # t=1.0 -> 1.0, t=0.0 -> 3.0
        w_epitope_rot = w_epitope_pos

        w_bond = 3.0 * torch.pow(1.0 - t, 4.0)  # t=0.5 -> 0.19, t=0.0 -> 3.0
        w_omega = w_bond

        w_clash = 5.0 * torch.pow(1.0 - t, 6.0)  # t=0.5 -> 0.08, t=0.0 -> 5.0

        w_contact = 2.0 * torch.exp(-8.0 * (t - 0.5)**2)  # Peak at t=0.5

        w_bsa = 5.0 * torch.pow(1.0 - t, 1.5)  # t=1.0 -> ~0, t=0.0 -> 5.0
        w_shape = w_bsa

        weights_dict = dict(self.base_weights)
        dynamic_weights = {
            'cdr_rot': w_cdr_rot,
            'cdr_pos': w_cdr_pos,
            'cdr_seq': w_cdr_seq,
            'epitope_pos': w_epitope_pos,
            'epitope_rot': w_epitope_rot,
            'cdr_bone': w_bond,
            'cdr_omega': w_omega,
            'epitope_bone': w_bond,
            'epitope_omega': w_omega,
            'clash': w_clash,
            'contact': w_contact,
            'bsa': w_bsa,
            'shape': w_shape,
        }
        for key in dynamic_weights:
            if key in weights_dict:
                base_weight = weights_dict[key]
                if isinstance(base_weight, (int, float)):
                    weights_dict[key] = dynamic_weights[key] * base_weight
                else:
                    weights_dict[key] = dynamic_weights[key] * base_weight
            else:
                weights_dict[key] = dynamic_weights[key]

        if DEBUG_WEIGHTS and np.random.random() < 0.01:
            core_weights = {k: v.mean().item() if isinstance(v, torch.Tensor) else v
                          for k, v in weights_dict.items()
                          if k in ['cdr_rot', 'cdr_pos', 'cdr_seq', 'cdr_seq_nll', 'epitope_rot', 'epitope_pos', 'cdr_bone', 'clash']}
            debug_log(f"[compute_weights] t={t.mean().item():.3f}, weights={core_weights}")
        return weights_dict

class GeometryLossCalculator:
    """几何损失配置"""
    def __init__(self, cdr_mask, mask_soft_antigen, mask_antigen, atom_mask):
        self.bsa_target = 15.0  # Å² - 调整为实际值的1.5倍
        self.bsa_sigma = 4.5   # Å，高斯核宽度
        self.bsa_cutoff = 10.0  # Å，界面截断距离

        self.shape_temperature = 1.0
        self.shape_cutoff = 10.0  # Å，界面截断距离


        self.rmsd_bfactor_gamma = 0.2
        self.bond_target_length = 1.33
        self.bond_tolerance = 0.1
        self.cdr_boundary_geom_weight = 2.0
        self.epitope_boundary_geom_weight = 5.0
        self.omega_min = 90 * (3.14159 / 180.0)
        self.omega_max = 180 * (3.14159 / 180.0)
        self.contact_cutoff = 5.0
        self.clash_cutoff = 3.0
        self.cdr_mask = cdr_mask
        self.antigen_soft_mask = mask_soft_antigen
        self.antigen_mask = mask_antigen
        self.atom_mask = atom_mask
        self.loss_dict_raw = {}

    def compute_base_loss(self, R_next, R_0, eps_p_pred, eps_p, post_pred, post_true, bfactor, compute_cdr_loss,
                          compute_epitope_loss, s_0_pred_logits=None, s_0=None):
        if compute_cdr_loss:
            loss_rot = rotation_matrix_cosine_loss(R_next, R_0)

            if DEBUG_WEIGHTS and True:  # 临时100%触发
                self._print_rot_bone_detail(R_next, R_0, loss_rot)

            loss_cdr_rot = (loss_rot * self.cdr_mask).sum() / (self.cdr_mask.sum().float() + 1e-8)
            self.loss_dict_raw['cdr_rot'] = loss_cdr_rot
            self.loss_dict_raw['cdr_rot_mean'] = loss_cdr_rot

            loss_pos_raw = F.mse_loss(eps_p_pred, eps_p, reduction='none')  # (N, L, 3)
            loss_pos = loss_pos_raw.sum(dim=-1)  # (N, L) 每个残基的loss

            loss_cdr_pos = (loss_pos * self.cdr_mask).sum() / (self.cdr_mask.sum().float() + 1e-8)
            self.loss_dict_raw['cdr_pos'] = loss_cdr_pos
            self.loss_dict_raw['cdr_pos_mean'] = loss_cdr_pos

            if post_pred is not None and post_true is not None:
                log_post_pred = torch.log(post_pred + 1e-8)  # (N, L, 20)
                kldiv = F.kl_div(
                    input=log_post_pred,
                    target=post_true,
                    reduction='none',
                    log_target=False
                ).sum(dim=-1)  # (N, L)

                loss_cdr_seq = (kldiv * self.cdr_mask).sum() / (self.cdr_mask.sum().float() + 1e-8)
                self.loss_dict_raw['cdr_seq'] = loss_cdr_seq
                self.loss_dict_raw['cdr_seq_mean'] = loss_cdr_seq
                if DEBUG_WEIGHTS and True:
                    entropy_pred = -(post_pred * torch.log(post_pred + 1e-8)).sum(dim=-1).mean().item()
                    entropy_true = -(post_true * torch.log(post_true + 1e-8)).sum(dim=-1).mean().item()
                    max_entropy = np.log(20)
                    debug_log(f"[序列分布] pred熵={entropy_pred:.3f} (均匀={max_entropy:.3f}), true熵={entropy_true:.3f}, KL={loss_cdr_seq.item():.4f}")
            else:
                self.loss_dict_raw['cdr_seq'] = torch.zeros(R_next.shape[0], device=R_next.device)
                self.loss_dict_raw["cdr_seq_mean"] = torch.tensor(0.0, device=R_next.device)

            if s_0_pred_logits is not None and s_0 is not None:
                vocab_size = s_0_pred_logits.size(-1)
                target_seq = s_0.long()
                valid_nll_mask = self.cdr_mask & (target_seq >= 0) & (target_seq < vocab_size)
                if valid_nll_mask.any():
                    per_res_nll = F.cross_entropy(
                        s_0_pred_logits.reshape(-1, vocab_size),
                        target_seq.clamp(min=0, max=vocab_size - 1).reshape(-1),
                        reduction='none',
                    ).view_as(target_seq)
                    loss_cdr_seq_nll = (per_res_nll * valid_nll_mask.float()).sum() / (valid_nll_mask.float().sum() + 1e-8)
                else:
                    loss_cdr_seq_nll = R_next.sum() * 0.0
                self.loss_dict_raw['cdr_seq_nll'] = loss_cdr_seq_nll
                self.loss_dict_raw['cdr_seq_nll_mean'] = loss_cdr_seq_nll
            else:
                self.loss_dict_raw['cdr_seq_nll'] = torch.zeros(R_next.shape[0], device=R_next.device)
                self.loss_dict_raw['cdr_seq_nll_mean'] = torch.tensor(0.0, device=R_next.device)

            if DEBUG_WEIGHTS:
                global_cdr_rot = (loss_rot * self.cdr_mask).sum() / (self.cdr_mask.sum().float() + 1e-8)
                global_cdr_pos = (loss_pos * self.cdr_mask).sum() / (self.cdr_mask.sum().float() + 1e-8)
                if post_pred is not None and post_true is not None:
                    global_cdr_seq = (kldiv * self.cdr_mask).sum() / (self.cdr_mask.sum().float() + 1e-8)
                    our_cdr_seq = loss_cdr_seq
                else:
                    global_cdr_seq = 0.0
                    our_cdr_seq = 0.0

                debug_log(f"[Loss计算对比] ===== 全局平均模式 =====")
                debug_log(f"[Loss计算对比] cdr_mask总residues: {self.cdr_mask.sum().item()}")
                debug_log(f"[Loss计算对比] cdr_rot: {global_cdr_rot.item():.4f}")
                debug_log(f"[Loss计算对比] cdr_pos: {global_cdr_pos.item():.4f}")
                debug_log(f"[Loss计算对比] cdr_seq: {global_cdr_seq:.4f}")
                if 'cdr_seq_nll_mean' in self.loss_dict_raw:
                    debug_log(f"[Loss计算对比] cdr_seq_nll: {self.loss_dict_raw['cdr_seq_nll_mean'].item():.4f}")
        else:
            ref_zero = R_next.sum() * 0.0
            self.loss_dict_raw['cdr_rot'] = ref_zero
            self.loss_dict_raw['cdr_rot_mean'] = ref_zero
            self.loss_dict_raw['cdr_pos'] = ref_zero
            self.loss_dict_raw['cdr_pos_mean'] = ref_zero
            self.loss_dict_raw['cdr_seq'] = ref_zero
            self.loss_dict_raw['cdr_seq_mean'] = ref_zero
            self.loss_dict_raw['cdr_seq_nll'] = ref_zero
            self.loss_dict_raw['cdr_seq_nll_mean'] = ref_zero

        loss_epitope_rot = R_next.sum() * 0.0
        loss_epitope_pos = R_next.sum() * 0.0

        if compute_epitope_loss and self.antigen_soft_mask is not None and self.antigen_soft_mask.any():
            loss_rot_epi = rotation_matrix_cosine_loss(R_next, R_0)
            loss_pos_epi = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)

            if DEBUG_WEIGHTS:
                epi_residues_per_sample = self.antigen_soft_mask.sum(dim=-1)
                debug_log(
                    f"[Epitope] antigen_soft_mask residues per sample: min={epi_residues_per_sample.min().item()}, max={epi_residues_per_sample.max().item()}, mean={epi_residues_per_sample.float().mean().item():.1f}")

            loss_epitope_rot = (loss_rot_epi * self.antigen_soft_mask.float()).sum() / (self.antigen_soft_mask.float().sum() + 1e-8)
            loss_epitope_pos = (loss_pos_epi * self.antigen_soft_mask.float()).sum() / (self.antigen_soft_mask.float().sum() + 1e-8)

        self.loss_dict_raw['epitope_rot'] = loss_epitope_rot
        self.loss_dict_raw['epitope_rot_mean'] = loss_epitope_rot
        self.loss_dict_raw['epitope_pos'] = loss_epitope_pos
        self.loss_dict_raw['epitope_pos_mean'] = loss_epitope_pos

        if DEBUG_WEIGHTS and compute_epitope_loss and True:
            debug_log(
                f"[Epitope Loss Raw] epitope_rot: {loss_epitope_rot.item():.4f}, epitope_pos: {loss_epitope_pos.item():.4f}")

    def compute_geometry_losses(self, pos, pos_noisy=None, seq_adjacent_mask_dict=None, chain_nb=None, res_nb=None, mask=None, pos_true=None, compute_cdr=True, compute_epitope=True):
        if chain_nb is not None and res_nb is not None and mask is not None:
            seq_consec = get_consecutive_flag(chain_nb, res_nb, mask)
        else:
            seq_consec = None

        self.bb_pos_pred_for_analysis = pos  # (N, L, 4, 3)
        if pos_true is not None:
            self.bb_pos_true_for_analysis = pos_true

        if pos.dim() == 4 and pos.shape[-1] == 3:
            self.pos_pred_for_analysis = pos[:, :, BBHeavyAtom.CA, :]  # (N, L, 3) CA位置
        else:
            self.pos_pred_for_analysis = pos

        if pos_true is not None and pos_true.dim() == 4 and pos_true.shape[-1] == 3:
            self.pos_true_for_analysis = pos_true[:, :, BBHeavyAtom.CA, :]  # (N, L, 3) CA位置
        elif pos_true is not None:
            self.pos_true_for_analysis = pos_true

        geometry_pos = pos_noisy if pos_noisy is not None else pos
        self._compute_bond_length_loss(geometry_pos, seq_consec, compute_cdr=compute_cdr, compute_epitope=compute_epitope)
        self._compute_omega_angle_loss(geometry_pos, seq_consec, chain_nb, res_nb, mask, compute_cdr=compute_cdr, compute_epitope=compute_epitope)



    def _print_rot_bone_detail(self, R_next, R_0, loss_rot):
        """打印每个残基的rot loss和对应的bone loss"""
        N, L = loss_rot.shape

        print("\n" + "=" * 80)
        print("[Rot vs Bone调试] 每个残基的详细loss")
        print("=" * 80)

        b = 0

        cdr_m = self.cdr_mask[b]  # (L,)
        cdr_m_prev = cdr_m[:-1]  # (L-1)
        cdr_m_next = cdr_m[1:]   # (L-1)

        entry_boundary = (~cdr_m_prev & cdr_m_next)  # CDR入口（前一残基是FR，当前是CDR）
        exit_boundary = (cdr_m_prev & ~cdr_m_next)   # CDR出口（前一残基是CDR，当前是FR）

        print(f"样本{b}: CDR残基数={cdr_m.sum().item()}")
        print(f"  CDR入口边界数: {entry_boundary.sum().item()}")
        print(f"  CDR出口边界数: {exit_boundary.sum().item()}")

        cdr_indices_raw = cdr_m.nonzero(as_tuple=True)[0]
        cdr_indices = [int(x.item() if hasattr(x, 'item') else x) for x in cdr_indices_raw]

        print("\n[CDR内部残基的Rot Loss]:")
        print("-" * 80)

        internal_rot_total = 0.0
        for idx in cdr_indices[:10]:  # 只打印前10个
            rot_val = loss_rot[b, idx].item()
            internal_rot_total += rot_val
            print(f"  残基{idx}: rot_loss={rot_val:.4f}")

        if len(cdr_indices) > 10:
            remaining_rot = loss_rot[b, cdr_indices[10:]].sum().item()
            internal_rot_total += remaining_rot
            print(f"  ... 还有{len(cdr_indices)-10}个残基, 总rot={internal_rot_total:.4f}")

        avg_internal_rot = internal_rot_total / len(cdr_indices) if len(cdr_indices) > 0 else 0
        print(f"  CDR内部平均rot_loss: {avg_internal_rot:.4f}")

        print("\n[边界残基的Rot Loss]:")
        print("-" * 80)

        entry_indices = entry_boundary.nonzero(as_tuple=True)[0].tolist()
        exit_indices = exit_boundary.nonzero(as_tuple=True)[0].tolist()

        boundary_rot_total = 0.0

        for idx in entry_indices[:5]:
            rot_val = loss_rot[b, idx].item() if idx < L else 0
            boundary_rot_total += rot_val
            print(f"  入口残基{idx}: rot_loss={rot_val:.4f} (FR残基{idx-1}→CDR残基{idx})")

        for idx in exit_indices[:5]:
            rot_val = loss_rot[b, idx].item() if idx < L else 0
            boundary_rot_total += rot_val
            print(f"  出口残基{idx}: rot_loss={rot_val:.4f} (CDR残基{idx-1}→FR残基{idx})")

        print("\n" + "=" * 80)
        print("[关键分析]")
        print("=" * 80)
        print(f"CDR内部rot_loss总和: {internal_rot_total:.4f}")
        print(f"边界rot_loss总和: {boundary_rot_total:.4f}")

        if boundary_rot_total > internal_rot_total * 0.1:
            print(f"⚠️ 边界贡献较大！但边界rot不被cdr_rot loss计算")
        else:
            print(f"边界贡献较小，rot loss主要来自CDR内部")

        print("\n[Bone Loss来源分析]:")
        print("-" * 80)

        pos = self.pos_heavyatom if hasattr(self, 'pos_heavyatom') else None
        if pos is not None and pos.dim() == 4:
            n_coords = pos[b, :, BBHeavyAtom.N, :]
            c_coords = pos[b, :, BBHeavyAtom.C, :]
            c_i = c_coords[:-1, :]
            n_ip1 = n_coords[1:, :]
            bond_lengths = torch.sqrt(((c_i - n_ip1) ** 2).sum(dim=-1) + 1e-8)

            target_length = 1.33
            tolerance = 0.1

            internal_bonds = (cdr_m_prev & cdr_m_next)
            internal_bond_indices = internal_bonds.nonzero(as_tuple=True)[0].tolist()

            boundary_bonds = (cdr_m_prev | cdr_m_next) & ~internal_bonds
            boundary_bond_indices = boundary_bonds.nonzero(as_tuple=True)[0].tolist()

            internal_bone = 0.0
            boundary_bone = 0.0

            for idx in internal_bond_indices[:5]:
                bl = bond_lengths[idx].item()
                dev = max(0, abs(bl - target_length) - tolerance)
                internal_bone += dev
                status = "正常" if bl > 1.2 and bl < 1.5 else "异常"
                print(f"  内部键{idx}→{idx+1}: {bl:.2f}Å, 偏差={dev:.2f}Å [{status}]")

            for idx in boundary_bond_indices[:5]:
                bl = bond_lengths[idx].item()
                dev = max(0, abs(bl - target_length) - tolerance)
                boundary_bone += dev
                bond_type = "入口" if entry_boundary[idx] else "出口"
                status = "正常" if bl > 1.2 and bl < 1.5 else "异常"
                print(f"  边界键{idx}→{idx+1}: {bl:.2f}Å, 偏差={dev:.2f}Å [{bond_type}] [{status}]")

            print(f"\n  CDR内部bone贡献: {internal_bone:.4f}Å")
            print(f"  边界bone贡献: {boundary_bone:.4f}Å")

            if boundary_bone > internal_bone:
                print(f"  ⚠️ Bone loss主要来自边界键断裂！")
            else:
                print(f"  ⚠️ Bone loss主要来自CDR内部键断裂！")

        self._detailed_pos_error_analysis(b)

        print("=" * 80)

    def _detailed_pos_error_analysis(self, sample_idx=0):
        """
        Bone Loss来源分析 - 使用与Bone计算相同的数据源

        分析目标：
        1. N/C原子坐标是否正确
        2. 键长偏差的来源（Pos误差 vs Rot误差）
        3. Rot误差如何影响键长
        """
        print("\n" + "=" * 80)
        print("[Bone Loss来源分析]")
        print("=" * 80)

        b = sample_idx

        print("\n[数据源检查]")
        print("-" * 80)

        if hasattr(self, 'bb_pos_pred_for_analysis') and self.bb_pos_pred_for_analysis is not None:
            print(f"bb_pos_pred shape: {self.bb_pos_pred_for_analysis.shape}")  # (N, L, 4, 3)
            n_coords = self.bb_pos_pred_for_analysis[:, :, 0, :]   # (N, L, 3) 预测的N原子
            c_coords = self.bb_pos_pred_for_analysis[:, :, BBHeavyAtom.C, :]   # (N, L, 3) 预测的C原子
            print(f"  N原子坐标范围: [{n_coords.min().item():.2f}, {n_coords.max().item():.2f}] Å")
            print(f"  C原子坐标范围: [{c_coords.min().item():.2f}, {c_coords.max().item():.2f}] Å")
        else:
            print("  ⚠️ 无bb_pos_pred数据，无法直接分析Bone")

        if hasattr(self, 'pos_pred_for_analysis') and self.pos_pred_for_analysis is not None:
            print(f"\npos_pred_for_analysis shape: {self.pos_pred_for_analysis.shape}")
            pos_pred = self.pos_pred_for_analysis[b]
            print(f"  batch {b} pred shape: {pos_pred.shape}")
            if pos_pred.dim() == 3 and pos_pred.shape[-1] == 3:
                ca_pred = pos_pred[:, 1, :]  # CA是index 1
                print(f"  CA预测坐标范围: [{ca_pred.min().item():.2f}, {ca_pred.max().item():.2f}] Å")
            elif pos_pred.dim() == 2:
                print(f"  已是(L, 3)格式，假设为CA坐标")
                print(f"  CA预测坐标范围: [{pos_pred.min().item():.2f}, {pos_pred.max().item():.2f}] Å")

        print("\n[键长偏差分析] - 使用与Bone相同的方法")
        print("-" * 80)

        if hasattr(self, 'pos_heavyatom') and self.pos_heavyatom is not None:
            pos = self.pos_heavyatom  # (N, L, N_heavy, 3)
            n_coords = pos[:, :, BBHeavyAtom.N, :]  # (N, L, 3)
            c_coords = pos[:, :, BBHeavyAtom.C, :]  # (N, L, 3)

            c_i = c_coords[:, :-1, :]    # (N, L-1, 3)
            n_ip1 = n_coords[:, 1:, :]   # (N, L-1, 3)
            bond_lengths = torch.sqrt(((c_i - n_ip1) ** 2).sum(dim=-1) + 1e-8)  # (N, L-1)

            bond_target = 1.33  # 理想键长
            bond_deviation = torch.abs(bond_lengths - bond_target)  # (N, L-1)

            print(f"键长统计 (所有残基):")
            print(f"  均值: {bond_lengths.mean().item():.3f} Å (理想: {bond_target} Å)")
            print(f"  偏差均值: {bond_deviation.mean().item():.3f} Å")
            print(f"  偏差最大: {bond_deviation.max().item():.3f} Å")

            if hasattr(self, 'cdr_mask') and self.cdr_mask is not None:
                cdr_m = self.cdr_mask  # (N, L)
                cdr_bond_mask = (cdr_m[:, :-1] | cdr_m[:, 1:])  # (N, L-1)
                cdr_bond_dev = bond_deviation[cdr_bond_mask]
                print(f"\nCDR区域键偏差:")
                cdr_bond_dev_mean = cdr_bond_dev.mean().item()
                print(f"  均值: {cdr_bond_dev_mean:.3f} Å")
                self._cdr_bond_dev_cached = cdr_bond_dev_mean  # 保存供后续使用
                print(f"  最大: {cdr_bond_dev.max().item():.3f} Å")
                print(f"  键数: {cdr_bond_mask.sum().item()}")

                interior_mask = cdr_m[:, :-1] & cdr_m[:, 1:]  # 两端都是CDR
                boundary_mask = cdr_bond_mask & ~interior_mask  # 至少一端不是CDR
                if interior_mask.sum() > 0:
                    print(f"\nCDR内部键偏差: {bond_deviation[interior_mask].mean().item():.3f} Å (键数: {interior_mask.sum().item()})")
                if boundary_mask.sum() > 0:
                    print(f"CDR边界键偏差: {bond_deviation[boundary_mask].mean().item():.3f} Å (键数: {boundary_mask.sum().item()})")

        print("\n[CA位置误差 vs 键偏差]")
        print("-" * 80)

        if hasattr(self, 'pos_pred_for_analysis') and hasattr(self, 'pos_true_for_analysis'):
            pos_pred = self.pos_pred_for_analysis[b]
            pos_true = self.pos_true_for_analysis[b]

            if pos_pred.dim() == 3 and pos_pred.shape[-1] == 3:
                ca_pred = pos_pred[:, 1, :]
            else:
                ca_pred = pos_pred
            if pos_true.dim() == 3 and pos_true.shape[-1] == 3:
                ca_true = pos_true[:, 1, :]
            else:
                ca_true = pos_true

            ca_error = torch.norm(ca_pred - ca_true, dim=-1)  # (L,)

            if hasattr(self, 'cdr_mask'):
                cdr_m = self.cdr_mask[b]
                cdr_ca_error = ca_error[cdr_m]
                print(f"CA位置误差 (CDR区域):")
                print(f"  均值: {cdr_ca_error.mean().item():.3f} Å")
                print(f"  最大: {cdr_ca_error.max().item():.3f} Å")

                print(f"\n理论分析:")
                print(f"  CA误差均值 ~{cdr_ca_error.mean().item():.3f} Å")
                print(f"  若两CA误差完全反向叠加，键偏差上限 ~{2 * cdr_ca_error.mean().item():.3f} Å")
                if hasattr(self, '_cdr_bond_dev_cached'):
                    print(f"  实际键偏差 ~{self._cdr_bond_dev_cached:.3f} Å (来自上面计算)")

                cdr_indices = cdr_m.nonzero(as_tuple=True)[0].tolist()
                cos_angles = []
                for i, idx in enumerate(cdr_indices[:-1]):
                    next_idx = cdr_indices[i+1]
                    if next_idx != idx + 1:
                        continue
                    err_i = (ca_pred[idx] - ca_true[idx]).detach().cpu().numpy()
                    err_j = (ca_pred[next_idx] - ca_true[next_idx]).detach().cpu().numpy()
                    norm_i = np.linalg.norm(err_i)
                    norm_j = np.linalg.norm(err_j)
                    if norm_i > 0.01 and norm_j > 0.01:
                        cos_a = np.dot(err_i, err_j) / (norm_i * norm_j)
                        cos_angles.append(cos_a)

                if len(cos_angles) > 0:
                    print(f"\n相邻CA误差向量夹角:")
                    print(f"  均值cos: {np.mean(cos_angles):.2f}")
                    print(f"  解读: cos≈0表示随机, cos<0表示反向, cos>0表示同向")

        print("\n[分析总结]")
        print("-" * 80)
        print("数据流验证:")
        print("  1. Bone Loss使用bb_pos_pred中的预测N和C原子")
        print("  2. CA位置误差来自pos_pred_for_analysis")
        print("  3. 需要验证pos_heavyatom的来源（是预测还是真实？）")
        print("\n关键问题:")
        print("  - 如果pos_heavyatom是预测结构，则Bone Loss直接反映预测质量")
        print("  - Rot误差通过影响N/C相对CA的位置，间接影响键长")
        print("  - 需要进一步分析Rot Loss与键偏差的关系")
    def _compute_bond_length_loss(self, pos, seq_consec=None, compute_cdr=True, compute_epitope=True):
        if pos.dim() != 4:
            return

        mask_dict = {}
        if compute_cdr:
            mask_dict["cdr"] = self.cdr_mask
        if compute_epitope:
            mask_dict["epitope"] = self.antigen_soft_mask

        n_coords = pos[:, :, BBHeavyAtom.N, :]
        c_coords = pos[:, :, BBHeavyAtom.C, :]
        c_i = c_coords[:, :-1, :]
        n_ip1 = n_coords[:, 1:, :]
        bond_lengths = torch.sqrt(((c_i - n_ip1) ** 2).sum(dim=-1) + 1e-8)
        deviation = torch.abs(bond_lengths - self.bond_target_length)
        deviation = torch.clamp(deviation - self.bond_tolerance, min=0.0)

        for obj, mask in mask_dict.items():
            if mask is None or not mask.any():
                continue
            mask_bond = (mask[:, :-1] | mask[:, 1:]).float()
            if seq_consec is not None:
                mask_bond = mask_bond * seq_consec.float()
            if self.atom_mask is not None:
                mask_bond = mask_bond * self.atom_mask[:, :-1, BBHeavyAtom.C].float() * self.atom_mask[:, 1:, BBHeavyAtom.N].float()

            boundary_pair = (mask[:, :-1] ^ mask[:, 1:]).float()
            if obj == 'epitope':
                mask_bond = mask_bond * (1.0 + (self.epitope_boundary_geom_weight - 1.0) * boundary_pair)
            elif obj == 'cdr':
                mask_bond = mask_bond * (1.0 + (self.cdr_boundary_geom_weight - 1.0) * boundary_pair)

            if mask_bond.sum() > 0:
                loss_per_sample = (deviation * mask_bond).sum(dim=-1) / (mask_bond.sum(dim=-1).float() + 1e-8)
                loss = loss_per_sample.mean()
            else:
                loss = pos.sum() * 0.0
                loss_per_sample = torch.zeros(pos.shape[0], device=pos.device)

            self.loss_dict_raw[f"{obj}_bone"] = loss_per_sample
            self.loss_dict_raw[f"{obj}_bone_mean"] = loss

            if DEBUG_WEIGHTS:
                global_bone = (deviation * mask_bond).sum() / (mask_bond.sum() + 1e-8)
                debug_log(f"[Loss计算对比] {obj}_bone: 全局={global_bone.item():.4f}, per-sample平均={loss.item():.4f}")
                bonds_per_sample = mask_bond.sum(dim=-1).tolist()
                debug_log(f"[Loss计算对比] {obj}_bone键数分布: min={min(bonds_per_sample)}, max={max(bonds_per_sample)}, mean={sum(bonds_per_sample)/len(bonds_per_sample):.1f}")

        ref_zero = pos.sum() * 0.0
        if not compute_cdr:
            self.loss_dict_raw["cdr_bone"] = ref_zero
            self.loss_dict_raw["cdr_bone_mean"] = ref_zero
        if not compute_epitope:
            self.loss_dict_raw["epitope_bone"] = ref_zero
            self.loss_dict_raw["epitope_bone_mean"] = ref_zero

        if DEBUG_WEIGHTS and torch.rand(1).item() < 0.05:
            self._print_bond_details(pos, seq_consec, compute_cdr, compute_epitope)

    def _print_bond_details(self, pos, seq_consec=None, compute_cdr=True, compute_epitope=True):
        """详细打印每个残基对的键长（仅用于调试）"""
        N, L = pos.shape[:2]

        n_coords = pos[:, :, BBHeavyAtom.N, :]
        c_coords = pos[:, :, BBHeavyAtom.C, :]
        c_i = c_coords[:, :-1, :]
        n_ip1 = n_coords[:, 1:, :]
        bond_lengths = torch.sqrt(((c_i - n_ip1) ** 2).sum(dim=-1) + 1e-8)

        print("\n" + "=" * 80)
        print("[Bone Loss调试] 每个残基对的键长分析 (目标1.33Å)")
        print("=" * 80)

        mask_dict = {}
        if compute_cdr:
            mask_dict["cdr"] = self.cdr_mask
        if compute_epitope:
            mask_dict["epitope"] = self.antigen_soft_mask

        for obj, mask in mask_dict.items():
            if mask is None or not mask.any():
                continue

            print(f"\n[{obj.upper()}] 样本0的残基对键长:")
            print("-" * 80)

            b = 0
            m = mask[b]  # (L,)

            mask_bond = (mask[:, :-1] | mask[:, 1:]).float()
            mask_bond_and = (mask[:, :-1] & mask[:, 1:]).float()

            bond_indices = (mask_bond[b] > 0).nonzero(as_tuple=True)[0].tolist()

            for idx in bond_indices:
                res_i = idx
                res_j = idx + 1
                bond_len = bond_lengths[b, idx].item()
                dev = abs(bond_len - self.bond_target_length)
                is_internal = mask_bond_and[b, idx].item() > 0

                if bond_len > 2.0 or bond_len < 1.0 or not is_internal:
                    status = "断裂" if bond_len > 2.0 or bond_len < 1.0 else "偏高"
                    region = "CDR内部" if is_internal else "CDR边界"
                    print(f"  残基{res_i}→{res_j}: 键长={bond_len:.2f}Å, 偏差={dev:.2f}Å [{region}] {status}")

        print("=" * 80)

    def _compute_omega_angle_loss(self, pos, seq_consec=None, chain_nb=None, res_nb=None, mask=None, compute_cdr=True,
                                  compute_epitope=True):
        if pos.dim() != 4:
            return
        mask_dict = {}
        if compute_cdr:
            mask_dict["cdr"] = self.cdr_mask
        if compute_epitope:
            mask_dict["epitope"] = self.antigen_soft_mask

        omega_mask_global = None
        if chain_nb is not None and res_nb is not None and mask is not None:
            bb_dihedral, mask_bb = get_backbone_dihedral_angles(pos, chain_nb, res_nb, mask)
            if DEBUG_WEIGHTS:
                omega_raw = bb_dihedral[..., 0]
                omega_deg = omega_raw * 180.0 / 3.14159
                debug_log(f"[Omega Raw] omega_deg均值={omega_deg.mean().item():.1f}, mask_bb有效数={mask_bb[..., 0].sum().item()}")
                debug_log(f"[Omega Raw] chain_nb shape={chain_nb.shape}, res_nb shape={res_nb.shape}, mask shape={mask.shape}")
                debug_log(f"[Omega Raw] mask有效数={mask.sum().item()}, omega_mask有效数={mask_bb[..., 0].sum().item()}")
            omega_raw = bb_dihedral[..., 0]  # 原始omega值（可能为负）
            cos_omega = torch.cos(omega_raw)  # cos(omega), trans时≈-1

            omega_mask = mask_bb[..., 0]  # True for valid omega
            omega_valid = omega_raw[omega_mask]
            cos_omega_valid = cos_omega[omega_mask]

            if omega_valid.numel() > 0:
                omega_deg_mean = (omega_valid * 180.0 / 3.14159).mean().item()
                cos_omega_mean = cos_omega_valid.mean().item()
                if DEBUG_WEIGHTS:
                    debug_log(f"[Omega Raw] omega_deg有效均值={omega_deg_mean:.1f}°, cos均值={cos_omega_mean:.3f}, 有效数={omega_valid.numel()}")

            tolerance_cos = 0.015
            omega_loss = torch.clamp(torch.abs(cos_omega + 1.0) - tolerance_cos, min=0.0)

            omega_mask_global = mask_bb[..., 0]  # (N, L), True for residues with valid omega
        else:
            if DEBUG_WEIGHTS:
                debug_log("[Omega警告] else分支被触发！omega计算可能不正确，请检查chain_nb/res_nb/mask参数")

            N, L = pos.shape[:2]
            omega_loss = torch.zeros(N, L, device=pos.device)
            omega_mask_global = torch.zeros(N, L, dtype=torch.bool, device=pos.device)

        for obj, mask in mask_dict.items():
            if mask is None or not mask.any():
                continue
            for b in range(mask.shape[0]):
                m = mask[b]
                n_res = m.sum().item()
                n_and = (m[:-1] & m[1:]).sum().item()
                n_or = (m[:-1] | m[1:]).sum().item()
                n_boundary = n_or - n_and
                transitions = (~m[:-1] & m[1:]).sum().item()
                n_segments = transitions if m[0] else transitions + 1
                expected_boundary = n_segments * 2 if n_segments > 0 else 0
                if n_res > 0:
                    pass  # Debug assertions disabled
            if omega_mask_global is not None:
                mask_omega = mask.float() * omega_mask_global.float()
                left_pad = torch.zeros_like(mask[:, :1]).float()
                right_pad = torch.zeros_like(mask[:, :1]).float()
                prev_mask = torch.cat([left_pad, mask[:, :-1].float()], dim=1)
                next_mask = torch.cat([mask[:, 1:].float(), right_pad], dim=1)
                boundary_residue = mask.float() * ((prev_mask != mask.float()) | (next_mask != mask.float())).float()
                if obj == 'epitope':
                    mask_omega = mask_omega * (1.0 + (self.epitope_boundary_geom_weight - 1.0) * boundary_residue)
                elif obj == 'cdr':
                    mask_omega = mask_omega * (1.0 + (self.cdr_boundary_geom_weight - 1.0) * boundary_residue)
            else:
                mask_omega = (mask[:, :-1] | mask[:, 1:]).float()
                if seq_consec is not None:
                    mask_omega = mask_omega * seq_consec.float()
            if mask_omega.sum() > 0:
                loss_per_sample = (omega_loss * mask_omega).sum(dim=-1) / (mask_omega.sum(dim=-1).float() + 1e-8)  # (N,)
                loss = loss_per_sample.mean()
            else:
                loss = pos.sum() * 0.0
                loss_per_sample = torch.zeros(pos.shape[0], device=pos.device)
            self.loss_dict_raw[f"{obj}_omega"] = loss_per_sample  # (N,) vector
            self.loss_dict_raw[f"{obj}_omega_mean"] = loss  # scalar mean
            if DEBUG_WEIGHTS and mask_omega.sum() > 0:
                global_omega = (omega_loss * mask_omega).sum() / (mask_omega.sum() + 1e-8)
                debug_log(f"[Loss计算对比] {obj}_omega: 全局={global_omega.item():.4f}, per-sample平均={loss.item():.4f}, 差异={abs(global_omega.item()-loss.item()):.6f}")

    def compute_losses(self, R_next, R_0, eps_p_pred, eps_p, post_pred, post_true, bb_pos_pred, bb_pos_noisy=None, seq_adjacent_mask_dict=None, chain_nb=None, res_nb=None, mask=None, compute_epitope_loss=True, compute_cdr_loss=True,
                            bfactor=None, bb_pos_true=None, s_0_pred_logits=None, s_0=None):
        """
        计算所有原始损失（不加权）

        Args:
            R_next: (N, L, 3, 3) - 预测的旋转矩阵
            R_0: (N, L, 3, 3) - 真实的旋转矩阵
            eps_p_pred: (N, L, 3) - 预测的位置噪声
            eps_p: (N, L, 3) - 真实的位置噪声
            s_0_pred_logits: (N, L, 20) - 预测的 clean-token logits
            s_noisy: (N, L) - 加噪后的序列
            s_0: (N, L) - 真实序列
            seq_adjacent_mask_dict: {'cdr': (N, L-1), 'epitope': (N, L-1)} - 序列邻接mask
            compute_epitope_loss: 是否计算Epitope损失
            trans_seq: 序列转移对象，用于计算KL散度
            compute_cdr_loss: 是否计算CDR损失（用于单区域加噪模式）
            bfactor: (N, L, n_atom) 或 (N, L) - B-factor值，用于表位损失加权

        Returns:
            loss_dict_raw: 原始损失字典（未加权）
        """

        self.pos_pred_for_analysis = bb_pos_pred  # (N, L, 3, 3) backbone positions
        if bb_pos_true is not None:
            self.pos_true_for_analysis = bb_pos_true
        elif hasattr(self, 'pos_heavyatom'):
            self.pos_true_for_analysis = self.pos_heavyatom

        self.compute_base_loss(
            R_next, R_0, eps_p_pred, eps_p, post_pred, post_true, bfactor,
            compute_cdr_loss, compute_epitope_loss,
            s_0_pred_logits=s_0_pred_logits, s_0=s_0,
        )
        if 'cdr_bone' not in self.loss_dict_raw:
            self.loss_dict_raw['cdr_bone'] = torch.zeros(R_next.shape[0], device=R_next.device)
            self.loss_dict_raw['cdr_bone_mean'] = torch.tensor(0.0, device=R_next.device)
        if 'cdr_omega' not in self.loss_dict_raw:
            self.loss_dict_raw['cdr_omega'] = torch.zeros(R_next.shape[0], device=R_next.device)
            self.loss_dict_raw['cdr_omega_mean'] = torch.tensor(0.0, device=R_next.device)
        if 'epitope_bone' not in self.loss_dict_raw:
            self.loss_dict_raw['epitope_bone'] = torch.zeros(R_next.shape[0], device=R_next.device)
            self.loss_dict_raw['epitope_bone_mean'] = torch.tensor(0.0, device=R_next.device)
        if 'epitope_omega' not in self.loss_dict_raw:
            self.loss_dict_raw['epitope_omega'] = torch.zeros(R_next.shape[0], device=R_next.device)
            self.loss_dict_raw['epitope_omega_mean'] = torch.tensor(0.0, device=R_next.device)

        self.compute_geometry_losses(
            bb_pos_pred,
            None,
            seq_adjacent_mask_dict,
            chain_nb,
            res_nb,
            mask,
            pos_true=bb_pos_true,
            compute_cdr=compute_cdr_loss,
            compute_epitope=compute_epitope_loss,
        )

        soft_mask_present = self.antigen_soft_mask is not None and self.antigen_soft_mask.any()
        if compute_epitope_loss:
            assert soft_mask_present, "[Loss断言] compute_epitope_loss=True but antigen_soft_mask is empty"
        else:
            for key in (
                'epitope_rot', 'epitope_rot_mean', 'epitope_pos', 'epitope_pos_mean',
                'epitope_bone', 'epitope_bone_mean', 'epitope_omega', 'epitope_omega_mean',
                'epitope_rmsd', 'epitope_rmsd_mean',
            ):
                if key in self.loss_dict_raw:
                    _assert_zero_loss_tensor(key, self.loss_dict_raw[key])

        if DEBUG_WEIGHTS and torch.rand(1).item() < 0.05:
            antigen_soft_count = 0.0
            if self.antigen_soft_mask is not None:
                antigen_soft_count = self.antigen_soft_mask.sum(dim=-1).float().mean().item()
            debug_log(f"[Loss Raw] cdr_mask residues: {self.cdr_mask.sum(dim=-1).float().mean().item():.1f}, antigen_mask residues: {antigen_soft_count:.1f}")
            loss_parts = []
            for k in ['cdr_rot', 'cdr_pos', 'cdr_seq', 'cdr_seq_nll', 'epitope_rot', 'epitope_pos']:
                if k in self.loss_dict_raw:
                    v = self.loss_dict_raw[k]
                    if isinstance(v, torch.Tensor):
                        loss_parts.append(f"{k}={v.mean().item():.4f}")
            debug_log(f"[Loss Raw] {', '.join(loss_parts)}")

        return self.loss_dict_raw


    def _compute_rmsd_loss(self, pos):
        """计算CDR和Epitope的RMSD损失（相对于自身参考）"""
        if self.cdr_mask is not None and self.cdr_mask.any():
            loss_cdr_rmsd = self._compute_region_rmsd(pos, self.cdr_mask)
            self.loss_dict_raw['cdr_rmsd'] = loss_cdr_rmsd
            self.loss_dict_raw['cdr_rmsd_mean'] = loss_cdr_rmsd

        if self.antigen_soft_mask is not None and self.antigen_soft_mask.any():
            loss_epitope_rmsd = self._compute_region_rmsd(pos, self.antigen_soft_mask)
            self.loss_dict_raw['epitope_rmsd'] = loss_epitope_rmsd
            self.loss_dict_raw['epitope_rmsd_mean'] = loss_epitope_rmsd

    def _compute_region_rmsd(self, pos, mask):
        """
        计算单个区域的RMSD
        由于需要参考坐标，这里计算的是区域内CA坐标的方差（作为结构紧密性指标）
        """
        if pos.dim() == 4:
            ca_pos = pos[:, :, BBHeavyAtom.CA, :]  # (N, L, 3)
        else:
            ca_pos = pos

        masked_pos = ca_pos * mask.unsqueeze(-1).float()
        center = masked_pos.sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)

        dist_to_center = ((ca_pos - center) ** 2).sum(dim=-1)  # (N, L)

        masked_dist = dist_to_center * mask.float()
        rmsd = torch.sqrt(masked_dist.sum(dim=1) / (mask.sum(dim=1).float() + 1e-8))

        return rmsd.mean()

    def _compute_contact_loss(self, pos):
        """计算接触损失 - 完全避免索引操作"""
        import torch
        ca_pos = pos[:, :, BBHeavyAtom.CA, :] if pos.dim() == 4 else pos
        B, L, _ = ca_pos.shape

        diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)

        cdr_m = self.cdr_mask.float().unsqueeze(-1)
        ag_m = self.antigen_mask.float().unsqueeze(1)
        interface = cdr_m * ag_m  # (B, L, L)

        contact_penalty = torch.clamp(dist - 5.0, min=0.0)  # (B, L, L)
        weighted_penalty = contact_penalty * interface  # 只对界面内的残基对计算
        total_penalty = weighted_penalty.sum()
        num_pairs = interface.sum() + 1e-8

        contact_loss = total_penalty / num_pairs
        self.loss_dict_raw['contact'] = contact_loss
        self.loss_dict_raw['contact_mean'] = contact_loss

    def _compute_clash_loss(self, pos):
        """计算位阻损失 - 完全避免索引操作"""
        import torch
        ca_pos = pos[:, :, BBHeavyAtom.CA, :] if pos.dim() == 4 else pos
        B, L, _ = ca_pos.shape

        diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)

        cdr_m = self.cdr_mask.float().unsqueeze(-1)
        ag_m = self.antigen_mask.float().unsqueeze(1)
        interface = cdr_m * ag_m

        close_mask = interface * (dist < 5.0).float()  # 只考虑5Å以内的对
        penalty = torch.clamp(3.0 - dist, min=0.0) ** 2

        weighted_penalty = penalty * close_mask
        total_penalty = weighted_penalty.sum()
        num_pairs = close_mask.sum() + 1e-8

        clash_loss = total_penalty / num_pairs
        self.loss_dict_raw['clash'] = clash_loss
        self.loss_dict_raw['clash_mean'] = clash_loss


    def _compute_bsa_loss(self, pos):
        """计算BSA损失 - 完全避免索引操作"""
        import torch
        import torch.nn.functional as F
        ca_pos = pos[:, :, BBHeavyAtom.CA, :] if pos.dim() == 4 else pos

        diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        cdr_m = self.cdr_mask.float().unsqueeze(-1)
        ag_m = self.antigen_mask.float().unsqueeze(1)
        interface = cdr_m * ag_m

        sigma = 4.5
        gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))

        contact = (gaussian * interface).sum()
        target = 15.0

        bsa_loss = F.relu(target - contact) / target
        self.loss_dict_raw['bsa'] = bsa_loss
        self.loss_dict_raw['bsa_mean'] = bsa_loss


    def _compute_shape_loss(self, R, pos=None):
        """
        形状互补损失 (Shape Complementarity Loss) - 单独调用

        ===================================================================
        改进版本: 引入方向一致性因子
        ===================================================================

        原始公式：
        L_shape = Σ_{(i,j)∈interface} (1 + cos(θ_ij))

        改进公式（加入方向一致性判断）：
        L_shape = Σ_{(i,j)∈interface} (1 + cos(θ_ij)) · 1(n_i · n_j < 0)

        其中：
        - v_i, v_j 是残基的法向量 (取旋转矩阵的第3列)
        - n_i 是法向量 v_i 相对于参考向量的方向指示符
        - 1(n_i · n_j < 0) 只计算"面对面"的法向量对

        参考向量计算：
        - v_ref_i = geometric_center - CA_i (或 CB_i - CA_i)
        - sign_i = sign(n_i · v_ref_i), 如果为负说明法向量指内了

        目标: 鼓励CDR表面与表位凹槽在几何上严丝合缝 (法向量反向对齐)
        当法向量反向对齐时 (夹角180°), cos(180°) = -1, loss = 0
        当法向量同向对齐时 (夹角0°), cos(0°) = 1, loss = 2

        Args:
            R: (B, L, 3, 3) - 旋转矩阵
            pos: (B, L, 3) or (B, L, 4, 3) - 坐标位置，用于计算参考向量
                 如果是 (B, L, 4, 3)，则使用 CA 和 CB 原子计算参考向量
                 如果是 (B, L, 3)，则使用 CA 和几何中心计算参考向量

        修复: 避免对零向量进行归一化，这会导致NaN
        """
        if self.cdr_mask is None or self.antigen_mask is None:
            self.loss_dict_raw['shape'] = R.sum() * 0.0
            self.loss_dict_raw['shape_mean'] = R.sum() * 0.0
            return

        if R is None:
            self.loss_dict_raw['shape'] = R.sum() * 0.0
            self.loss_dict_raw['shape_mean'] = R.sum() * 0.0
            return

        B, L, _, _ = R.shape

        v_all = R[:, :, :, 2]  # 所有残基的法向量

        v_all_norm = F.normalize(v_all, dim=-1)  # (B, L, 3)

        v_cdr_norm = v_all_norm * self.cdr_mask.unsqueeze(-1).float()  # (B, L, 3)
        v_epi_norm = v_all_norm * self.antigen_mask.unsqueeze(-1).float()  # (B, L, 3)


        mask_cdr = self.cdr_mask.unsqueeze(-1)  # (B, L, 1)
        mask_epi = self.antigen_mask.unsqueeze(1)  # (B, 1, L)
        interface_mask = mask_cdr * mask_epi  # (B, L, L)

        direction_consistency_mask = interface_mask.float()

        if pos is not None:
            if pos.dim() == 4 and pos.shape[2] >= 5:
                ca_pos = pos[:, :, BBHeavyAtom.CA, :]  # (B, L, 3)
                cb_pos = pos[:, :, BBHeavyAtom.CB, :]  # (B, L, 3)
                v_ref = cb_pos - ca_pos  # (B, L, 3)
            elif pos.dim() == 3:
                ca_pos = pos  # (B, L, 3)
                cdr_ca = ca_pos * self.cdr_mask.unsqueeze(-1).float()
                cdr_center = cdr_ca.sum(dim=1, keepdim=True) / (self.cdr_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
                epi_ca = ca_pos * self.antigen_mask.unsqueeze(-1).float()
                epi_center = epi_ca.sum(dim=1, keepdim=True) / (self.antigen_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)

                v_ref = ca_pos - torch.where(
                    self.cdr_mask.unsqueeze(-1).float() > 0,
                    cdr_center,
                    epi_center
                )  # (B, L, 3)
            else:
                v_ref = None

            if v_ref is not None:
                v_ref_norm = F.normalize(v_ref + 1e-8, dim=-1)  # (B, L, 3)

                ref_dot = (v_all * v_ref_norm).sum(dim=-1)  # (B, L)

                direction_sign = torch.sign(ref_dot)  # (B, L)

                direction_i = direction_sign.unsqueeze(2)  # (B, L, 1)
                direction_j = direction_sign.unsqueeze(1)  # (B, 1, L)
                direction_opposite = (direction_i * direction_j) < 0  # (B, L, L), bool

                direction_consistency_mask = (direction_opposite & interface_mask).float()

        v_cdr_exp = v_cdr_norm.unsqueeze(2)  # (B, L, 1, 3)
        v_epi_exp = v_epi_norm.unsqueeze(1)  # (B, 1, L, 3)

        cos_theta = (v_cdr_exp * v_epi_exp).sum(dim=-1)  # (B, L, L)

        shape_loss_per_pair = 1.0 + cos_theta

        shape_loss_per_pair = shape_loss_per_pair * direction_consistency_mask

        shape_loss_sum = shape_loss_per_pair.sum(dim=[1, 2])  # (B,)
        num_pairs = direction_consistency_mask.sum(dim=[1, 2]) + 1e-8

        L_shape = shape_loss_sum / num_pairs

        L_shape = L_shape * self.shape_temperature

        self.loss_dict_raw['shape'] = L_shape
        self.loss_dict_raw['shape_mean'] = L_shape

def compute_bfactor_sigma(bfactor, mask, mask_norm, gamma=0.2, use_global_norm=True):
        """
        计算 B-factor 缩放因子。

        Args:
            bfactor: (N, L, n_atom) 或 (N, L) - B-factor 值
            mask: (N, L) - 有效区域 mask (用于应用缩放)
            gamma: 缩放系数
            mask_norm: (N, L) - 用于计算归一化统计量的 mask (默认使用 mask)
            use_global_norm: 是否使用全局标准化（跨样本可比），否则使用局部标准化

        Returns:
            sigma: (N, L, 3) - 位置噪声缩放因子
        """
        if mask_norm is None:
            mask_norm = mask

        if bfactor.dim() == 3:
            b_ca = bfactor[:, :, BBHeavyAtom.CA]  # (N, L)
        else:
            b_ca = bfactor

        if use_global_norm:
            b_mean = torch.tensor(GLOBAL_B_FACTOR_MEAN, device=b_ca.device, dtype=b_ca.dtype)
            b_std = torch.tensor(GLOBAL_B_FACTOR_STD, device=b_ca.device, dtype=b_ca.dtype)
            b_norm = (b_ca - b_mean) / (b_std + 1e-8)
            b_norm_for_debug = b_norm
        else:
            b_masked = torch.where(mask_norm, b_ca, torch.zeros_like(b_ca))
            b_mean = b_masked.sum(dim=1, keepdim=True) / (mask_norm.sum(dim=1, keepdim=True) + 1e-8)
            b_std = torch.sqrt(((b_masked - b_mean) ** 2 * mask_norm).sum(dim=1, keepdim=True) / (mask_norm.sum(dim=1, keepdim=True) + 1e-8))
            b_norm = (b_ca - b_mean) / (b_std + 1e-8)
            b_norm_for_debug = b_norm

        sigma = torch.exp(gamma * b_norm)  # (N, L)
        sigma_before_clamp = sigma.clone()
        sigma = torch.clamp(sigma, 0.5, 2.0)  # 限制范围 [0.5, 2.0]

        if DEBUG_BFACTOR:
            epi_indices = mask_norm[0].nonzero(as_tuple=False).squeeze(-1)
            n_epi_show = min(5, epi_indices.numel()) if epi_indices.numel() > 0 else 0

            norm_type = "GLOBAL" if use_global_norm else "LOCAL"
            print(f"[sigma_b-Debug] ====================== Start ({norm_type}) ======================")
            if use_global_norm:
                print(f"[sigma_b-Debug] Global B-factor stats: mean={GLOBAL_B_FACTOR_MEAN:.1f}, std={GLOBAL_B_FACTOR_STD:.1f}")
            else:
                print(f"[sigma_b-Debug] Local B-factor stats: mean={b_mean[0].item():.3f}, std={b_std[0].item():.3f}")
            if n_epi_show > 0:
                print(f"[sigma_b-Debug] Batch 0 B-factor (epitope only): min={b_ca[0, epi_indices].min().item():.3f}, max={b_ca[0, epi_indices].max().item():.3f}, mean={b_ca[0, epi_indices].mean().item():.3f}")
                print(f"[sigma_b-Debug] Batch 0 b_norm (epitope only): min={b_norm_for_debug[0, epi_indices].min().item():.3f}, max={b_norm_for_debug[0, epi_indices].max().item():.3f}, mean={b_norm_for_debug[0, epi_indices].mean().item():.3f}")
                print(f"[sigma_b-Debug] Batch 0 sigma BEFORE clamp: min={sigma_before_clamp[0, epi_indices].min().item():.3f}, max={sigma_before_clamp[0, epi_indices].max().item():.3f}, mean={sigma_before_clamp[0, epi_indices].mean().item():.3f}")
            print(f"[sigma_b-Debug] sigma stats (after clamp [0.5, 2.0]): min={sigma.min():.3f}, max={sigma.max():.3f}, mean={sigma.mean():.3f}")

            if n_epi_show > 0:
                print(f"[sigma_b-Debug] Batch 0 - First {n_epi_show} epitope residues:")
                for i, idx in enumerate(epi_indices[:n_epi_show]):
                    print(f"  EpiRes {idx:3d}: b_ca={b_ca[0, idx].item():6.2f}, b_norm={b_norm[0, idx].item():6.2f}, sigma_before={sigma_before_clamp[0, idx].item():6.3f}, sigma_after={sigma[0, idx].item():6.3f}")
            print(f"[sigma_b-Debug] ======================= End =======================")

        sigma = sigma.unsqueeze(-1).expand(-1, -1, 3)  # (N, L, 3)
        sigma = sigma * mask.unsqueeze(-1).float()

        return sigma

def apply_dynamic_weights(
    loss_dict_raw,
    weights_dict,
    core_losses,
    geom_scaler=None,
    t=None,
    use_geom_scaler=False
):
    """
    Args:
        loss_dict_raw: 原始损失字典（未加权）
        weights: 权重字典
        eta: 清晰度补偿指数

    Returns:
        loss_dict: 加权后的损失字典
        weight_dict: 权重信息
    """
    if DEBUG_LOSS_DETAILS and True:
        print(f"[DEBUG] loss_dict_raw keys: {list(loss_dict_raw.keys())}")
        for k in ['bsa', 'shape', 'contact', 'clash']:
            if k in loss_dict_raw:
                val = loss_dict_raw[k]
                if isinstance(val, torch.Tensor):
                    print(f"[DEBUG] {k}: {val.item():.4f}, requires_grad={val.requires_grad}")
                else:
                    print(f"[DEBUG] {k}: {val}")

    loss_dict = {}
    weight_info = {}

    if DEBUG_LOSS_DETAILS:
        print("\n[Loss区域断言] 验证非零loss在生成区域内")
        print("[Loss区域断言] OK\n")

    total_loss = None
    for key, weight in weights_dict.items():
        if key in loss_dict_raw and (key+'_mean') in loss_dict_raw:
            raw_loss = loss_dict_raw[key+'_mean']

            if key in core_losses and isinstance(raw_loss, torch.Tensor) and torch.is_grad_enabled():
                assert raw_loss.requires_grad or raw_loss.item() == 0.0,                     f"[Loss断言] {key} 无梯度但非零值={raw_loss.item():.4f}"
            if isinstance(weight, (int, float)):
                weight = torch.tensor(weight, device=raw_loss.device, dtype=raw_loss.dtype)
            elif isinstance(weight, torch.Tensor):
                weight = weight.to(raw_loss.device).squeeze()
            else:
                weight = torch.tensor(weight, device=raw_loss.device, dtype=raw_loss.dtype).squeeze()

            if raw_loss.dim() == 0 and weight.dim() > 0:
                weight = weight.mean()
            elif weight.dim() == 0 and raw_loss.dim() > 0:
                pass
            elif weight.shape != raw_loss.shape:
                if weight.numel() > 1 and raw_loss.numel() == 1:
                    weight = weight.mean()
                else:
                    assert False, f"{key} weight.shape({weight.shape})!=raw_loss.shape({raw_loss.shape})"

            weighted_loss = raw_loss * weight
            weighted_loss = weighted_loss.mean()
            assert len(weighted_loss.shape) <= 1, f"{key} weighted_loss.shape({weighted_loss.shape}) should be scalar or 1D"
            weight_info[key] = weight
            if key in core_losses and DEBUG_LOSS_DETAILS:
                print(f"[DEBUG] core_loss={key}, raw_loss.requires_grad={raw_loss.requires_grad}, weighted_loss.requires_grad={weighted_loss.requires_grad}")

            loss_dict[key] = weighted_loss
            if raw_loss.requires_grad:
                assert weighted_loss.requires_grad, f"{key} is no_grad during training. Need to check."
            if key in core_losses:

                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

    if use_geom_scaler and geom_scaler is not None and t is not None:
        cdr_bone_per_sample = loss_dict_raw.get('cdr_bone', None)
        epitope_bone_per_sample = loss_dict_raw.get('epitope_bone', None)
        cdr_omega_per_sample = loss_dict_raw.get('cdr_omega', None)
        epitope_omega_per_sample = loss_dict_raw.get('epitope_omega', None)

        if cdr_bone_per_sample is not None and epitope_bone_per_sample is not None and isinstance(cdr_bone_per_sample, torch.Tensor):
            bone_loss = cdr_bone_per_sample + epitope_bone_per_sample  # (N,)
        else:
            bone_loss = loss_dict_raw.get('cdr_bone_mean', 0) + loss_dict_raw.get('epitope_bone_mean', 0)  # scalar

        if cdr_omega_per_sample is not None and epitope_omega_per_sample is not None and isinstance(cdr_omega_per_sample, torch.Tensor):
            omega_loss = cdr_omega_per_sample + epitope_omega_per_sample  # (N,)
        else:
            omega_loss = loss_dict_raw.get('cdr_omega_mean', 0) + loss_dict_raw.get('epitope_omega_mean', 0)  # scalar

        if isinstance(bone_loss, torch.Tensor) and isinstance(omega_loss, torch.Tensor):
            geom_loss, geom_aux = geom_scaler(bone_loss, omega_loss, t)
            loss_dict['geom_loss'] = geom_loss
            loss_dict['geom_aux'] = geom_aux

            loss_dict_raw['geom_loss_mean'] = geom_loss  # 让log_losses识别有梯度
            loss_dict_raw['geom_loss_data'] = geom_aux.get('geom_data_loss', 0)
            loss_dict_raw['geom_loss_reg'] = geom_aux.get('geom_reg_loss', 0)

            if total_loss is not None:
                total_loss = total_loss + geom_loss
            else:
                total_loss = geom_loss

            if DEBUG_LOSS_DETAILS and torch.rand(1).item() < 0.05:
                print(f"[GeomLossScaler] bone_scale={geom_aux['bone_scale']:.4f}, omega_scale={geom_aux['omega_scale']:.4f}")
                print(f"[GeomLossScaler] alpha_b={geom_aux['alpha_b']:.4f}, alpha_o={geom_aux['alpha_o']:.4f}")
                print(f"[GeomLossScaler] weighted_bone={geom_aux['weighted_bone']:.4f}, weighted_omega={geom_aux['weighted_omega']:.4f}")

    loss_dict['overall'] = total_loss

    raw_overall_terms = []
    for key in core_losses:
        mean_key = key + '_mean'
        if mean_key in loss_dict_raw:
            raw_val = loss_dict_raw[mean_key]
            if isinstance(raw_val, torch.Tensor) and raw_val.dim() > 0:
                raw_val = raw_val.mean()
            raw_overall_terms.append(raw_val)
    if raw_overall_terms:
        loss_dict_raw['overall'] = sum(raw_overall_terms)
    else:
        ref = next((v for v in loss_dict_raw.values() if isinstance(v, torch.Tensor)), None)
        loss_dict_raw['overall'] = ref.new_zeros(()) if ref is not None else 0.0

    if DEBUG_LOSS_DETAILS and torch.rand(1).item() < 0.05:
        print(f"[DEBUG OVERALL] total_loss.requires_grad={total_loss.requires_grad}, grad_fn={total_loss.grad_fn is not None}")
        if total_loss.grad_fn is not None:
            print(f"[DEBUG OVERALL] grad_fn: {total_loss.grad_fn}")

    if 'geom_aux' in loss_dict:
        geom_aux = loss_dict['geom_aux']
        weight_info['geom_bone_scale'] = geom_aux.get('bone_scale', 0)
        weight_info['geom_omega_scale'] = geom_aux.get('omega_scale', 0)
        weight_info['geom_alpha_b'] = geom_aux.get('alpha_b', 0)
        weight_info['geom_alpha_o'] = geom_aux.get('alpha_o', 0)

    return loss_dict, weight_info, loss_dict_raw

class EpitopeConstraintLoss(nn.Module):
    """
    表位物理约束损失
    用于约束表位区域的结构合理性
    """
    def __init__(self, rmsd_weight=1.0, bone_weight=1.0, omega_weight=0.5,
                 contact_weight=1.0, clash_weight=2.0, rmsd_max=2.0,
                 contact_cutoff=5.0, clash_cutoff=3.0,
                 time_dependent_weighting=True, T=100):
        super().__init__()
        self.rmsd_weight = rmsd_weight
        self.bone_weight = bone_weight
        self.omega_weight = omega_weight
        self.contact_weight = contact_weight
        self.clash_weight = clash_weight
        self.rmsd_max = rmsd_max
        self.contact_cutoff = contact_cutoff
        self.clash_cutoff = clash_cutoff
        self.time_dependent_weighting = time_dependent_weighting
        self.T = T

    def forward(self, pos, mask_epitope, t=None):
        """
        Args:
            pos: (N, L, 3) or (N, L, num_atoms, 3) - 原子坐标
            mask_epitope: (N, L) - 表位mask
            t: (N,) optional - 时间步，用于时间相关加权
        Returns:
            loss: 标量损失值
        """
        losses = {}

        if pos.dim() == 4:
            ca_pos = pos[:, :, BBHeavyAtom.CA, :]  # 提取CA原子
        else:
            ca_pos = pos

        if mask_epitope.any():
            center = (ca_pos * mask_epitope.unsqueeze(-1)).sum(dim=1, keepdim=True) / \
                     (mask_epitope.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
            dist_to_center = ((ca_pos - center) ** 2).sum(dim=-1)
            rmsd = torch.sqrt((dist_to_center * mask_epitope).sum(dim=1) / (mask_epitope.sum(dim=1) + 1e-8))
            losses['rmsd'] = rmsd.mean()

            if self.time_dependent_weighting and t is not None:
                time_weight = (t / self.T).clamp(0.1, 1.0)
                losses['rmsd'] = losses['rmsd'] * time_weight.mean()
        else:
            losses['rmsd'] = pos.sum() * 0.0

        total_loss = self.rmsd_weight * losses['rmsd']

        return total_loss, losses
