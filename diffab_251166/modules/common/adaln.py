"""
Adaptive Layer Normalization (AdaLN) 模块
用于扩散模型中的时间条件控制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeMLP(nn.Module):
    """
    Time-MLP: 将正弦时间编码转换为语义向量
    """
    def __init__(self, input_dim, time_dim=64, use_silu=True):
        super().__init__()
        self.input_dim = input_dim

        if use_silu:
            self.time_embed = nn.Sequential(
                nn.Linear(input_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            self.time_embed = nn.Sequential(
                nn.Linear(input_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

    def forward(self, t):
        """
        Args:
            t: (N,) 或 (N, L) 或 (N, N_high) 时间步，支持浮点
        Returns:
            time_embed: (N, d) 或 (N, L, d) 或 (N, N_high, d) 时间语义向量
        """
        time_embed = self.time_embed(t)

        return time_embed


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization

    根据时间编码动态预测 scale 和 shift 参数:
    y = scale(t) * (x - mean) / std + shift(t)

    可选实现 adaLN-Zero 初始化（预测值初始为0）
    """
    def __init__(self, feature_dim, time_embed_dim, zero_init=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.time_embed_dim = time_embed_dim

        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)

        self.scale_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, feature_dim)
        )

        self.shift_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, feature_dim)
        )

        if zero_init:
            nn.init.zeros_(self.scale_mlp[-1].weight)
            nn.init.zeros_(self.scale_mlp[-1].bias)
            nn.init.zeros_(self.shift_mlp[-1].weight)
            nn.init.zeros_(self.shift_mlp[-1].bias)

    def forward(self, x, time_embed):
        """
        Args:
            x: (N, L, feature_dim) 输入特征
            time_embed: (N, time_embed_dim) 全局时间编码
                       或 (N, L, time_embed_dim) per-residue 时间编码
        Returns:
            y: (N, L, feature_dim) 自适应归一化后的特征
        """
        x_norm = self.norm(x)  # (N, L, feature_dim)

        if time_embed.dim() == 3:
            scale = self.scale_mlp(time_embed)  # (N, L, feature_dim)
            shift = self.shift_mlp(time_embed)  # (N, L, feature_dim)
        else:
            scale = self.scale_mlp(time_embed)  # (N, feature_dim)
            shift = self.shift_mlp(time_embed)  # (N, feature_dim)
            scale = scale[:, None, :]  # (N, 1, feature_dim)
            shift = shift[:, None, :]  # (N, 1, feature_dim)

        y = x_norm * (1 + scale) + shift  # (N, L, feature_dim)

        return y


class DualTimeAdaLN(nn.Module):
    """
    双时间步 AdaLN: 融合 CDR 和 Epitope 时间编码

    用于异步推理，融合 t_cdr 和 t_epi 的信息
    """
    def __init__(self, feature_dim, time_embed_dim=64, zero_init=True,
                 fusion_mode='mlp'):
        super().__init__()
        self.feature_dim = feature_dim
        self.time_embed_dim = time_embed_dim
        self.fusion_mode = fusion_mode

        self.time_mlp_cdr = TimeMLP(feature_dim, time_embed_dim)
        self.time_mlp_epi = TimeMLP(feature_dim, time_embed_dim)

        if fusion_mode == 'mlp':
            self.time_fusion = nn.Sequential(
                nn.Linear(time_embed_dim * 2, time_embed_dim * 2),
                nn.SiLU(),
                nn.Linear(time_embed_dim * 2, time_embed_dim)
            )
        elif fusion_mode == 'attention':
            self.query_proj = nn.Linear(time_embed_dim, time_embed_dim)
            self.key_proj = nn.Linear(time_embed_dim, time_embed_dim)
            self.value_proj = nn.Linear(time_embed_dim, time_embed_dim)
            self.attention = nn.MultiheadAttention(time_embed_dim, num_heads=4, batch_first=True)
            self.fusion_proj = nn.Linear(time_embed_dim * 2, time_embed_dim)

        self.adaln = AdaLN(feature_dim, time_embed_dim, zero_init=zero_init)

    def forward(self, x, t_cdr, t_epi):
        """
        Args:
            x: (N, L, feature_dim) 输入特征
            t_cdr: (N,) CDR 时间步
            t_epi: (N,) Epitope 时间步
        Returns:
            y: (N, L, feature_dim) 自适应归一化后的特征
        """
        print(f"t_cdr {t_cdr.shape}; t_epi {t_epi.shape}")
        embed_cdr = self.time_mlp_cdr(t_cdr)  # (N, time_embed_dim)
        embed_epi = self.time_mlp_epi(t_epi)  # (N, time_embed_dim)

        if self.fusion_mode == 'mlp':
            time_embed = self.time_fusion(torch.cat([embed_cdr, embed_epi], dim=-1))
        elif self.fusion_mode == 'attention':
            query = self.query_proj(embed_cdr)[:, None, :]  # (N, 1, d)
            key = self.key_proj(embed_epi)[:, None, :]  # (N, 1, d)
            value = self.value_proj(embed_epi)[:, None, :]  # (N, 1, d)
            attn_out, _ = self.attention(query, key, value)  # (N, 1, d)
            attn_out = attn_out.squeeze(1)  # (N, d)
            time_embed = self.fusion_proj(torch.cat([embed_cdr, attn_out], dim=-1))
        else:  # simple_sum
            time_embed = embed_cdr + embed_epi

        y = self.adaln(x, time_embed)

        return y, time_embed


class TimeConditionedBlock(nn.Module):
    """
    时间条件化的网络块
    结合 Time-MLP 和特征变换
    """
    def __init__(self, in_dim, out_dim, time_embed_dim, use_adaln=True, zero_init=True):
        super().__init__()
        self.use_adaln = use_adaln

        if use_adaln:
            self.adaln = AdaLN(in_dim, time_embed_dim, zero_init=zero_init)
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.linear = nn.Linear(in_dim + time_embed_dim, out_dim)

    def forward(self, x, time_embed):
        """
        Args:
            x: (N, L, in_dim)
            time_embed: (N, time_embed_dim)
        Returns:
            y: (N, L, out_dim)
        """
        if self.use_adaln:
            x_cond = self.adaln(x, time_embed)
            y = self.linear(x_cond)
        else:
            time_embed_expanded = time_embed[:, None, :].expand(x.size(0), x.size(1), -1)
            x_concat = torch.cat([x, time_embed_expanded], dim=-1)
            y = self.linear(x_concat)

        return y