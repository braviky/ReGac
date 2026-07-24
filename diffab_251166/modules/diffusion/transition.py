import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from diffab_251166.modules.common.layers import clampped_one_hot
from diffab_251166.modules.common.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, \
    rotation_to_so3vec

DEBUG_MODE = False


def _as_residue_t(t, N, L):
    """Return timestep tensor as (N, L), accepting legacy (N,) inputs."""
    if t.dim() == 0:
        return t.view(1, 1).expand(N, L).long()
    if t.dim() == 1:
        return t.long().view(N, 1).expand(N, L)
    if t.dim() == 2:
        if t.shape != (N, L):
            raise ValueError(f"Expected timestep shape {(N, L)} or {(N,)}, got {tuple(t.shape)}")
        return t.long()
    raise ValueError(f"Unsupported timestep rank: {t.dim()}")


class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间编码：支持浮点输入的时间步编码
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        """
        Args:
            t: (N,) Tensor, 支持 float 和 int
        Returns:
            embedding: (N, dim) 正弦编码
        """
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, device=device).float() / (half_dim - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class VarianceSchedule(nn.Module):
    def __init__(self, num_steps=100, s=0.01, epitope_offset=40, epitope_power=1.5, epitope_gamma=None):  # 余弦加噪 加噪需要的参数
        super().__init__()
        self.T = num_steps
        self.s = s
        if epitope_gamma is not None and epitope_power == 1.5:
            epitope_power = epitope_gamma
        self.epitope_offset = float(epitope_offset)
        self.epitope_power = float(epitope_power)

        T = num_steps
        t = torch.arange(0, num_steps + 1, dtype=torch.float)
        f_t = torch.cos((np.pi / 2) * ((t / T) + s) / (1 + s)) ** 2
        alpha_bars = f_t / f_t[0]  # 归一化，保证t=0时，alpha_bar=1 累积信号保留比例

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            numer = torch.clamp(1 - alpha_bars[i - 1], min=0.0)
            denom = torch.clamp(1 - alpha_bars[i], min=1e-8)
            sigmas[i] = (numer / denom) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('t', t)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)

        self.register_buffer('f_0', f_t[0:1])

        offset = max(0.0, min(self.epitope_offset, float(num_steps)))
        power = max(0.0, self.epitope_power)
        t_epi_mapped = ((t / num_steps) ** power) * (num_steps - offset)
        f_t_epi = torch.cos((np.pi / 2) * ((t_epi_mapped / num_steps) + s) / (1 + s)) ** 2
        alpha_bars_epi = f_t_epi / f_t_epi[0]

        betas_epi = 1 - (alpha_bars_epi[1:] / alpha_bars_epi[:-1])
        betas_epi = torch.cat([torch.zeros([1]), betas_epi], dim=0).clamp_max(0.999)

        sigmas_epi = torch.zeros_like(betas_epi)
        for i in range(1, betas_epi.size(0)):
            numer = torch.clamp(1 - alpha_bars_epi[i - 1], min=0.0)
            denom = torch.clamp(1 - alpha_bars_epi[i], min=1e-8)
            sigmas_epi[i] = (numer / denom) * betas_epi[i]
        sigmas_epi = torch.sqrt(sigmas_epi)

        self.register_buffer('alpha_bars_epi', alpha_bars_epi)
        self.register_buffer('t_epi', t_epi_mapped)
        self.register_buffer('betas_epi', betas_epi)
        self.register_buffer('sigmas_epi', sigmas_epi)
        self.register_buffer('alphas_epi', 1 - betas_epi)


class PositionTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t, region_nm, scale=None):
        """
        Args:
            p_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,) or dict - 如果是dict，应包含 't_fr', 't_c', 't_e' 键。
                支持浮点和整数时间步
            scale: (N, L, 3) or None - B-factor 缩放因子，对噪声进行逐元素缩放
        """
        if mask_generate.dtype is not torch.bool:
            mask_generate = mask_generate.to(torch.bool)
        N, L = mask_generate.size()
        t_res = _as_residue_t(t, N, L)

        if region_nm == 'epitope':
            alpha_bar = self.var_sched.alpha_bars_epi[t_res]
        else:
            alpha_bar = self.var_sched.alpha_bars[t_res]
        c0 = torch.sqrt(alpha_bar).unsqueeze(-1)
        c1 = torch.sqrt(1 - alpha_bar).unsqueeze(-1)

        e_rand = torch.randn_like(p_0)  # 采样高斯噪声

        if scale is not None:
            e_rand = e_rand * scale

        p_noisy = c0 * p_0 + c1 * e_rand  # 噪声加权
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)  # 只对需要生成的AA对象加噪，其余地方保持不变
        e_rand = torch.where(mask_generate[..., None].expand_as(e_rand), e_rand, torch.zeros_like(e_rand))  # 只对需要生成的AA对象加噪，其余地方保持不变
        if DEBUG_MODE:
            t_print = t[:3] if t.numel() > 3 else t
            ab_print = alpha_bar[:3] if alpha_bar.numel() > 3 else alpha_bar
            print(f"[add_noise_modal-Debug-{region_nm}] t={t_print}, alpha_bar={ab_print}")
            diff_mask = (p_noisy[:3] != p_0[:3]).any(dim=-1)  # (3, L) -> (3, L)
            for i in range(3):
                diff_idx = diff_mask[i].nonzero(as_tuple=True)[0].tolist()
                mask_idx = mask_generate[:3][i].nonzero(as_tuple=True)[0].tolist()
                diff_idx_show = diff_idx[:5] if len(diff_idx) > 5 else diff_idx
                if diff_idx_show:
                    for j in diff_idx_show:
                        p_n = p_noisy[:3][i, j]
                        p_0v = p_0[:3][i, j]
                        diff_norm = torch.norm(p_n - p_0v).item()
                        print(f"  [sample {i}, pos {j}] p_noisy={p_n.tolist()}, p_0={p_0v.tolist()}, |diff|={diff_norm:.4f}, in_mask={j in mask_idx}")
                if diff_idx:
                    all_in_mask = all(idx in mask_idx for idx in diff_idx)
                    print(f"  [sample {i}] all_diff_in_mask={all_in_mask}, diff_count={len(diff_idx)}, mask_count={len(mask_idx)}")

        return p_noisy, e_rand  # 加噪后的复合物数据，整个复合物上的噪声

    def denoise(self, p_t, eps_p, mask_generate, t, region_nm):
        N, L = mask_generate.size()
        t_res = _as_residue_t(t, N, L)
        if region_nm == 'epitope':
            alpha = self.var_sched.alphas_epi[t_res].clamp_min(
                self.var_sched.alphas_epi[-2]
            )
            alpha_bar = self.var_sched.alpha_bars_epi[t_res]
            sigma = self.var_sched.sigmas_epi[t_res].unsqueeze(-1)
        else:
            alpha = self.var_sched.alphas[t_res].clamp_min(
                self.var_sched.alphas[-2]
            )
            alpha_bar = self.var_sched.alpha_bars[t_res]
            sigma = self.var_sched.sigmas[t_res].unsqueeze(-1)

        c0 = (1.0 / torch.sqrt(alpha + 1e-8)).unsqueeze(-1)
        c1 = ((1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8)).unsqueeze(-1)

        z = torch.where(
            (t_res > 1).unsqueeze(-1).expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)
        return p_next


class RotationTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)  # 加噪scedule

        c1 = torch.sqrt(1 - self.var_sched.alpha_bars)  # (T,).
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        c1_epi = torch.sqrt(1 - self.var_sched.alpha_bars_epi)  # (T,).
        self.angular_distrib_fwd_epi = ApproxAngularDistribution(c1_epi.tolist(), **angular_distrib_fwd_opt)

        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)
        sigma_epi = self.var_sched.sigmas_epi
        self.angular_distrib_inv_epi = ApproxAngularDistribution(sigma_epi.tolist(), **angular_distrib_inv_opt)  # 保持原bug以与训练一致

        self.register_buffer('_dummy', torch.empty([0, ]))

    def add_noise(self, v_0, mask_generate, t, region_nm, scale=None):
        """
        Args:
            v_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,)，支持浮点和整数
            scale:  (N, L, 3) 或 (N, L, 1) 或 (N, L)，可选的噪声缩放因子
        """
        N, L = mask_generate.size()
        t_res = _as_residue_t(t, N, L)
        data_type = v_0.dtype

        if region_nm == 'epitope':
            alpha_bar = self.var_sched.alpha_bars_epi[t_res]
            e_scaled = random_normal_so3(t_res, self.angular_distrib_fwd_epi, device=self._dummy.device,
                                         dtype=data_type)  # (N, L, 3) 得到so3噪声向量

        else:
            alpha_bar = self.var_sched.alpha_bars[t_res]
            e_scaled = random_normal_so3(t_res, self.angular_distrib_fwd, device=self._dummy.device,
                                         dtype=data_type)  # (N, L, 3) 得到so3噪声向量


        if scale is not None:
            if scale.dim() == 2:
                scale = scale.unsqueeze(-1)  # (N, L) -> (N, L, 1)
            e_scaled = e_scaled * scale

        c0 = torch.sqrt(alpha_bar).unsqueeze(-1)
        c1 = torch.sqrt(1 - alpha_bar).unsqueeze(-1)

        e_normal = e_scaled / (c1 + 1e-8)

        E_scaled = so3vec_to_rotation(e_scaled)

        R0_scaled = so3vec_to_rotation(c0 * v_0)
        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)
        e_scaled = torch.where(mask_generate[..., None].expand_as(e_scaled), e_scaled, torch.zeros_like(e_scaled))

        if DEBUG_MODE:
            t_print = t[:3] if t.numel() > 3 else t
            ab_print = alpha_bar[:3] if alpha_bar.numel() > 3 else alpha_bar
            print(f"[add_noise_modal-Debug-{region_nm}] t={t_print}, alpha_bar={ab_print}")
            diff_mask = (v_noisy[:3] != v_0[:3]).any(dim=-1)  # (3, L, 3) -> (3, L)
            for i in range(3):
                diff_idx = diff_mask[i].nonzero(as_tuple=True)[0].tolist()
                mask_idx = mask_generate[:3][i].nonzero(as_tuple=True)[0].tolist()
                diff_idx_show = diff_idx[:5] if len(diff_idx) > 5 else diff_idx
                if diff_idx_show:
                    for j in diff_idx_show:
                        v_n = v_noisy[:3][i, j]
                        v_0v = v_0[:3][i, j]
                        diff_norm = torch.norm(v_n - v_0v).item()
                        print(f"  [sample {i}, pos {j}] v_noisy={v_n.tolist()}, v_0={v_0v.tolist()}, |diff|={diff_norm:.4f}, in_mask={j in mask_idx}")
                if diff_idx:
                    all_in_mask = all(idx in mask_idx for idx in diff_idx)
                    print(f"  [sample {i}] all_diff_in_mask={all_in_mask}, diff_count={len(diff_idx)}, mask_count={len(mask_idx)}")


        return v_noisy, e_scaled

    def denoise(self, v_t, v_next, mask_generate, t, region_nm):
        N, L = mask_generate.size()
        t_res = _as_residue_t(t, N, L)
        data_type = v_t.dtype
        if region_nm == 'epitope':
            e = random_normal_so3(t_res, self.angular_distrib_inv_epi, device=self._dummy.device,
                                  dtype=data_type)  # (N, L, 3)
        else:
            e = random_normal_so3(t_res, self.angular_distrib_inv, device=self._dummy.device,
                              dtype=data_type)  # (N, L, 3)

        e = torch.where(
            (t_res > 1).unsqueeze(-1).expand(N, L, 3),
            e,
            torch.zeros_like(e)  # Simply denoise and don't add noise at the last step
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[..., None].expand_as(v_next), v_next, v_t)

        return v_next


class AminoacidCategoricalTransition(nn.Module):

    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (N, L, K).
        Returns:
            x:    (N, L).
        """
        N, L, K = c.size()
        c = c.view(N * L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)  # 每一行为改AA在20个AA上的概率分布，每一行按照分布抽1个样本对应的做因
        return x  # 抽采样构成的AA序列

    def add_noise(self, x_0, mask_generate, t, region_nm):
        """
        Args:
            x_0:    (N, L)
            mask_generate:    (N, L).
            t:  (N,)，支持浮点和整数
        Returns:
            c_t:    Probability, (N, L, K).
            x_t:    Sample, LongTensor, (N, L).
        """
        if mask_generate.dtype is not torch.bool:
            mask_generate = mask_generate.to(torch.bool)
        N, L = x_0.size()
        t_res = _as_residue_t(t, N, L)
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float()
        if region_nm == 'epitope':
            alpha_bar = self.var_sched.alpha_bars_epi[t_res].unsqueeze(-1)
        else:
            alpha_bar = self.var_sched.alpha_bars[t_res].unsqueeze(-1)
        c_noisy = (alpha_bar * c_0) + ((1 - alpha_bar) / K)
        c_t = torch.where(mask_generate[..., None].expand(N, L, K), c_noisy, c_0)
        x_t = self._sample(c_t)
        if DEBUG_MODE:
            t_print = t[:3] if t.numel() > 3 else t
            ab_print = alpha_bar[:3, 0, 0] if alpha_bar.dim() == 3 else alpha_bar[:3]
            print(f"[add_noise_modal-Debug-{region_nm}] t={t_print}, alpha_bar={ab_print}")
            diff_mask = x_t[:3] != x_0[:3]  # (3, L)
            for i in range(3):
                diff_idx = diff_mask[i].nonzero(as_tuple=True)[0].tolist()
                mask_idx = mask_generate[:3][i].nonzero(as_tuple=True)[0].tolist()
                diff_idx_show = diff_idx[:5] if len(diff_idx) > 5 else diff_idx
                if diff_idx_show:
                    for j in diff_idx_show:
                        x_n = x_t[:3][i, j].item()
                        x_0v = x_0[:3][i, j].item()
                        diff = abs(x_n - x_0v)
                        print(f"  [sample {i}, pos {j}] x_t={x_n}, x_0={x_0v}, diff={diff}, in_mask={j in mask_idx}")
                if diff_idx:
                    all_in_mask = all(idx in mask_idx for idx in diff_idx)
                    print(f"  [sample {i}] all_diff_in_mask={all_in_mask}, diff_count={len(diff_idx)}, mask_count={len(mask_idx)}")

        return c_t, x_t

    def posterior(self, x_t, x_0, t, region_nm=None): #$$p(x_{t-1} | x_t, x_0) = \frac{p(x_t | x_{t-1}) \cdot p(x_{t-1} | x_0)}{p(x_t | x_0)}$$
        """
        Args:
            x_t:    Category LongTensor (N, L) or Probability FloatTensor (N, L, K).
            x_0:    Category LongTensor (N, L) or Probability FloatTensor (N, L, K).
            t:  (N,)，支持浮点和整数
            region_nm:  'cdr' or 'epitope', 用于选择正确的噪声调度表
        Returns:
            theta:  Posterior probability at (t-1)-th step, (N, L, K).
        """
        K = self.num_classes
        if x_t.dim() == 3:
            c_t = x_t
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float()

        if x_0.dim() == 3:
            c_0 = x_0
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float()

        assert region_nm == 'cdr', f"only cdr region naming is supported for Seq denoise"
        N, L = x_t.shape[:2]
        t_index = _as_residue_t(t, N, L).clamp(min=0, max=self.var_sched.alpha_bars.size(0) - 1)
        alpha = self.var_sched.alpha_bars[t_index].unsqueeze(-1)
        alpha_bar = self.var_sched.alpha_bars[t_index].unsqueeze(-1)

        theta = ((alpha * c_t) + (1 - alpha) / K) * (
            (alpha_bar * c_0) + (1 - alpha_bar) / K
        )
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def denoise(self, x_t, c_0_pred, mask_generate, t, region_nm):  # $$p(x_{t-1} | x_t, x_0) = \frac{p(x_t | x_{t-1}) \cdot p(x_{t-1} | x_0)}{p(x_t | x_0)}$$
        """
        Args:
            x_t:        (N, L).
            c_0_pred:   Normalized probability predicted by networks, (N, L, K).
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            post:   Posterior probability at (t-1)-th step, (N, L, K).
            x_next: Sample at (t-1)-th step, LongTensor, (N, L).
        """
        N, L = x_t.shape
        t_res = _as_residue_t(t, N, L)
        t_is_zero = t_res <= 0.5

        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()  # (N, L, K)
        post = self.posterior(c_t, c_0_pred, t=t_res, region_nm=region_nm)  # (N, L, K)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        x_next = self._sample(post)

        if t_is_zero.any():
            x_next = torch.where(t_is_zero, x_t, x_next)

        return post, x_next

