import torch
import torch.nn.functional as F

from diffab_251166.utils.protein.constants import (
    BBHeavyAtom,
    backbone_atom_coordinates_tensor,
    bb_oxygen_coordinate_tensor,
)
from .topology import get_terminus_flag


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def pairwise_distances(x, y=None, return_v=False):
    """
    Args:
        x:  (B, N, d)
        y:  (B, M, d)
    """
    if y is None: y = x
    v = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, d)
    d = safe_norm(v, dim=-1)
    if return_v:
        return d, v
    else:
        return d


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):  # 刻画残基AA主链（N C Ca）在整个structure中的扭转方向根据每个AA的全局坐标，得到每个AA的局部坐标系，原点在每个AA的CA处
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center  # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)  # 单位向量

    v2 = p2 - center  # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)  # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)坐标系按列看，第一列e1,第二列e2,第三列e3（.., .., 3, 3_index） 3指的时坐标时3维的，3_index指的是3个坐标轴
    return mat


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).  # CA坐标
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)  # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)  # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def global_to_local(R, t, q):  # q 编码时，是全原子坐标（15个）；转化为 以t(Ca)为原点、R为坐标系的坐标p
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3  # q全局坐标
    q_size = q.size()  # (N, L, 15, 3)
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)  # 每一列是原子坐标
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(
        -1)))  # (N, L, 3, *) R.transpose(-1, -2) 3行 每行表示一个单位坐标轴向量；(q - t.unsqueeze(-1)所有坐标-CA得到以CA为中心的坐标，此时对人坐标系中心是CA，但是坐标系和目标坐标系不一样，左乘心坐标系进行转化
    p = p.transpose(-1, -2).reshape(
        q_size)  # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)  # 反转一下 每一行是atom的坐标，而且是局部坐标
    return p


def apply_rotation_to_vector(R, p):
    return local_to_global(R, torch.zeros_like(p), p)


def compose_rotation_and_translation(R1, t1, R2, t2):
    """
    Args:
        R1,t1:  Frame basis and coordinate, (N, L, 3, 3), (N, L, 3).
        R2,t2:  Rotation and translation to be applied to (R1, t1), (N, L, 3, 3), (N, L, 3).
    Returns
        R_new <- R1R2
        t_new <- R1t2 + t1
    """
    R_new = torch.matmul(R1, R2)  # (N, L, 3, 3)
    t_new = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_new, t_new


def compose_chain(Ts):
    while len(Ts) >= 2:
        R1, t1 = Ts[-2]
        R2, t2 = Ts[-1]
        T_next = compose_rotation_and_translation(R1, t1, R2, t2)
        Ts = Ts[:-2] + [T_next]
    return Ts[0]


def quaternion_to_rotation_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


"""
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def quaternion_1ijk_to_rotation_matrix(q):
    """
    (1 + ai + bj + ck) -> R
    Args:
        q:  (..., 3)
    """
    b, c, d = torch.unbind(q, dim=-1)
    s = torch.sqrt(1 + b ** 2 + c ** 2 + d ** 2)
    a, b, c, d = 1 / s, b / s, c / s, d / s

    o = torch.stack(
        (
            a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c,
            2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b,
            2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2,
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def repr_6d_to_rotation_matrix(x):
    """
    Args:
        x:  6D representations, (..., 6).
    Returns:
        Rotation matrices, (..., 3, 3_index).
    """
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = normalize_vector(a1, dim=-1)
    b2 = normalize_vector(a2 - project_v2v(a2, b1, dim=-1), dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    mat = torch.cat([
        b1.unsqueeze(-1), b2.unsqueeze(-1), b3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign((torch.cross(v1, v2, dim=-1) * v0).sum(-1))
    dihed = sgn * torch.acos((n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    dihed = torch.nan_to_num(dihed)
    return dihed


def cos_dihedral_from_four_points(p0, p1, p2, p3, eps=1e-6):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Cosine of dihedral angles, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    u2 = torch.cross(v0, v2, dim=-1)
    denom = safe_norm(u1, dim=-1, eps=eps) * safe_norm(u2, dim=-1, eps=eps)
    cos_dihed = (u1 * u2).sum(-1) / denom
    return cos_dihed.clamp(min=-1.0, max=1.0)


def knn_gather(idx, value):
    """
    Args:
        idx:    (B, N, K)
        value:  (B, M, d)
    Returns:
        (B, N, K, d)
    """
    N, d = idx.size(1), value.size(-1)
    idx = idx.unsqueeze(-1).repeat(1, 1, 1, d)  # (B, N, K, d)
    value = value.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, M, d)
    return torch.gather(value, dim=2, index=idx)


def knn_points(q, p, K):
    """
    Args:
        q: (B, M, d)
        p: (B, N, d)
    Returns:
        (B, M, K), (B, M, K), (B, M, K, d)
    """
    _, L, _ = p.size()
    d = pairwise_distances(q, p)  # (B, N, M)
    dist, idx = d.topk(min(L, K), dim=-1, largest=False)  # (B, M, K), (B, M, K)
    return dist, idx, knn_gather(idx, p)


def angstrom_to_nm(x):  # 将距离单位从埃（Å）转换为纳米
    return x / 10


def nm_to_angstrom(x):
    return x * 10


def get_backbone_dihedral_angles(pos_atoms, chain_nb, res_nb, mask):  # 利用全局/绝对坐标计算每个AA内部的moega phi psi值
    """
    Args:
        pos_atoms:  (N, L, A, 3).
        chain_nb:   (N, L).
        res_nb:     (N, L).
        mask:       (N, L).
    Returns:
        bb_dihedral:    Omega, Phi, and Psi angles in radian, (N, L, 3).
        mask_bb_dihed:  Masks of dihedral angles, (N, L, 3).
    """
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]  # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_flag, C_term_flag = get_terminus_flag(chain_nb, res_nb, mask)  # (N, L)  标记出N端的AA、不连续处的AA、padding的AA
    omega_mask = torch.logical_not(N_term_flag)  # 0, 1, 1, ..., 1, 0, 0
    phi_mask = torch.logical_not(N_term_flag)  # 0, 1, 1, ..., 1, 0, 0
    psi_mask = torch.logical_not(C_term_flag)  # 1, 1, 1, ..., 1, 1, 0

    omega = F.pad(
        dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]),
        pad=(1, 0), value=0,
    )
    phi = F.pad(
        dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]),
        pad=(1, 0), value=0,
    )

    psi = F.pad(
        dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]),
        pad=(0, 1), value=0,
    )

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihedral = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed  # 保证没有omega phi psi角度的AA 角度值对应的位置为0 (N, L, 3)
    return bb_dihedral, mask_bb_dihed


def get_backbone_omega_cos(pos_atoms, chain_nb, res_nb, mask):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
        chain_nb:   (N, L).
        res_nb:     (N, L).
        mask:       (N, L).
    Returns:
        cos_omega:      (N, L), stored on residue i+1 for bond i->i+1.
        omega_mask:     (N, L), valid omega positions.
    """
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_flag, _ = get_terminus_flag(chain_nb, res_nb, mask)
    omega_mask = torch.logical_not(N_term_flag)
    cos_omega = F.pad(
        cos_dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]),
        pad=(1, 0), value=0,
    )
    return cos_omega, omega_mask


def pairwise_dihedrals(pos_atoms):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
    Returns:
        Inter-residue Phi and Psi angles, (N, L, L, 2).
    """
    N, L = pos_atoms.shape[:2]
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]  # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    ir_phi = dihedral_from_four_points(
        pos_C[:, :, None].expand(N, L, L, 3),
        pos_N[:, None, :].expand(N, L, L, 3),
        pos_CA[:, None, :].expand(N, L, L, 3),
        pos_C[:, None, :].expand(N, L, L, 3)
    )
    ir_psi = dihedral_from_four_points(
        pos_N[:, :, None].expand(N, L, L, 3),
        pos_CA[:, :, None].expand(N, L, L, 3),
        pos_C[:, :, None].expand(N, L, L, 3),
        pos_N[:, None, :].expand(N, L, L, 3)
    )
    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)
    return ir_dihed


def apply_rotation_matrix_to_rot6d(R, O):
    """
    Args:
        R:  (..., 3, 3)
        O:  (..., 6)
    Returns:
        Rotated 6D representation, (..., 6).
    """
    u1, u2 = O[..., :3, None], O[..., 3:, None]  # (..., 3, 1)
    v1 = torch.matmul(R, u1).squeeze(-1)  # (..., 3)
    v2 = torch.matmul(R, u2).squeeze(-1)
    return torch.cat([v1, v2], dim=-1)


def normalize_rot6d(O):
    """
    Args:
        O:  (..., 6)
    """
    u1, u2 = O[..., :3], O[..., 3:]  # (..., 3)
    v1 = F.normalize(u1, p=2, dim=-1)  # (..., 3)
    v2 = F.normalize(u2 - project_v2v(u2, v1), p=2, dim=-1)
    return torch.cat([v1, v2], dim=-1)


def reconstruct_backbone(R, t, aa, chain_nb, res_nb, mask, include_oxygen=True):
    """
    Args:
        R:  (N, L, 3, 3)
        t:  (N, L, 3)
        aa: (N, L)
        chain_nb:   (N, L)
        res_nb:     (N, L)
        mask:       (N, L)
    Returns:
        Reconstructed backbone atoms, (N, L, 4, 3) by default.
        When include_oxygen is False, only N/CA/C are returned: (N, L, 3, 3).
    """
    N, L = aa.size()
    bb_coords = backbone_atom_coordinates_tensor.clone().to(t)  # (21, 3, 3)
    oxygen_coord = bb_oxygen_coordinate_tensor.clone().to(t)  # (21, 3)
    aa = aa.clamp(min=0, max=20)  # 20 for UNK

    bb_coords = bb_coords[aa.flatten()].reshape(N, L, -1, 3)  # (N, L, 3, 3)
    oxygen_coord = oxygen_coord[aa.flatten()].reshape(N, L, -1)  # (N, L, 3)
    bb_pos = local_to_global(R, t, bb_coords)  # Global coordinates of N, CA, C. (N, L, 3, 3).

    if not include_oxygen:
        return bb_pos

    bb_dihedral, _ = get_backbone_dihedral_angles(bb_pos, chain_nb, res_nb, mask)
    psi = bb_dihedral[..., 2]  # (N, L)
    sin_psi = torch.sin(psi).reshape(N, L, 1, 1)
    cos_psi = torch.cos(psi).reshape(N, L, 1, 1)
    zero = torch.zeros_like(sin_psi)
    one = torch.ones_like(sin_psi)
    row1 = torch.cat([one, zero, zero], dim=-1)  # (N, L, 1, 3)
    row2 = torch.cat([zero, cos_psi, -sin_psi], dim=-1)  # (N, L, 1, 3)
    row3 = torch.cat([zero, sin_psi, cos_psi], dim=-1)  # (N, L, 1, 3)
    R_psi = torch.cat([row1, row2, row3], dim=-2)  # (N, L, 3, 3)

    R_psi, t_psi = compose_chain([
        (R, t),  # Backbone
        (R_psi, torch.zeros_like(t)),  # PSI angle
    ])
    O_pos = local_to_global(R_psi, t_psi, oxygen_coord.reshape(N, L, 1, 3))

    bb_pos = torch.cat([bb_pos, O_pos], dim=2)  # (N, L, 4, 3)
    return bb_pos


def reconstruct_backbone_partially(pos_ctx, R_new, t_new, aa, chain_nb, res_nb, mask_atoms,
                                   mask_recons):  # 为生产区域AA构造全局原子坐标，其中前4个原子（骨干原子）有值，剩下的为0（因为不估计侧链）
    """
    Args:
        pos:    (N, L, A, 3).
        R_new:  (N, L, 3, 3).
        t_new:  (N, L, 3).
        mask_atoms: (N, L, A).
        mask_recons:(N, L).
    Returns:
        pos_new:    (N, L, A, 3).
        mask_new:   (N, L, A).
    """
    N, L, A = mask_atoms.size()
    mask_recons = mask_recons.to(torch.bool)
    mask_res = mask_atoms[:, :, BBHeavyAtom.CA]  # 标记哪些AA是padding的
    pos_recons = reconstruct_backbone(R_new, t_new, aa, chain_nb, res_nb, mask_res)  # (N, L, 4, 3) 全局骨干骨干坐标
    pos_recons = F.pad(pos_recons, pad=(0, 0, 0, A - 4),
                       value=0)  # (N, L, A, 3) 把 4 原子对齐到完整原子15 因为完整坐标可能有侧链、氢原子等，需要补零对齐维度。

    pos_new = torch.where(  # 根据 mask_recons 决定用新坐标还是旧坐标。将生成区域坐标替换旧坐标。
        mask_recons[:, :, None, None].expand_as(pos_ctx),
        pos_recons, pos_ctx
    )  # (N, L, A, 3)
    mask_bb_atoms = torch.zeros_like(mask_atoms)
    mask_bb_atoms[:, :, :4] = True
    mask_new = torch.where(
        mask_recons[:, :, None].expand_as(mask_atoms),
        mask_bb_atoms, mask_atoms
    )

    return pos_new, mask_new



def shape_spectrum_augmentation(coords, bfactor=None, amplitude=2.0, device=None):
    """
    构象谱数据增强（Shape Spectrum Augmentation）。

    在训练前对 epitope 坐标执行随机扰动，模拟 apo→holo 的多样性。
    柔性表位（高 bfactor）允许大偏移，刚性表位（低 bfactor）只能小偏移。

    Args:
        coords:   (N, L, 3) 或 (N, L, A, 3) 全原子/CA 坐标。
        bfactor:  (N, L) B-factor（可选），越高越柔性。
        amplitude: float，最大扰动幅度（Å）。
        device:   计算设备。

    Returns:
        coords_aug:  扰动后的坐标。
        lambda_e:    int，采样到的噪声水平（0=holo, K=apo）。
    """
    if coords.dim() == 4:
        coords_ca = coords[:, :, BBHeavyAtom.CA, :]
    else:
        coords_ca = coords

    N, L = coords_ca.shape[:2]
    dev = device if device is not None else coords.device

    t = torch.rand(N, 1, device=dev)   # 0=apo(大偏移), 1=holo(小偏移)

    if bfactor is not None:
        bf = bfactor.to(dev)
        if bf.dim() == 3:
            bf = bf[:, :, 1]
        if bf.dim() == 1:
            bf = bf.unsqueeze(-1)
        bf_norm = torch.where(bf > 0, bf / (bf.max() + 1e-8),
                              torch.full_like(bf, 0.5))
    else:
        bf_norm = torch.full((N, L), 0.5, device=dev)

    scale = amplitude * (1 - t) * (bf_norm * 0.8 + 0.2)
    noise = torch.randn(N, L, 3, device=dev)
    delta = noise * scale.unsqueeze(-1)
    mask_res = (coords_ca.abs().sum(dim=-1) > 1e-6).float()
    delta = delta * mask_res.unsqueeze(-1)

    if coords.dim() == 4:
        A = coords.shape[2]
        delta_full = delta.unsqueeze(2).expand(N, L, A, 3)
        coords_aug = coords + delta_full
    else:
        coords_aug = coords + delta

    K = 500
    lambda_e = (t.squeeze(-1) * K).long()
    return coords_aug, lambda_e


def contact_loss(p_cdr, p_epi, mask_cdr=None, mask_epi=None, threshold=8.0):
    """
    接触感知损失（Contact-Aware Loss）。

    鼓励 CDR 和 epitope 残基对之间形成物理接触（距离 < threshold Å）。

    Args:
        p_cdr:   (N, L_cdr, 3) CDR CA 坐标。
        p_epi:   (N, L_epi, 3) Epitope CA 坐标。
        mask_cdr: (N, L_cdr) 可选，CDR mask。
        mask_epi: (N, L_epi) 可选，Epitope mask。
        threshold: float，接触判断阈值（Å）。

    Returns:
        loss: scalar，希望 CDR-epi 接触数量最大化。
    """
    dist = pairwise_distances(p_cdr, p_epi)
    contact = torch.sigmoid((threshold - dist) / 2.0).clamp(1e-6, 1 - 1e-6)

    if mask_cdr is not None:
        contact = contact * mask_cdr.unsqueeze(-1)
    if mask_epi is not None:
        contact = contact * mask_epi.unsqueeze(-2)

    loss = -contact.log().mean()
    return loss


def penetration_loss(p_cdr, p_epi, mask_cdr=None, mask_epi=None,
                    radius_c=1.5, radius_e=1.5):
    """
    穿透惩罚（Penetration Loss）。

    惩罚 CDR 和 epitope 原子之间的重叠（范德华硬球排斥）。

    Args:
        p_cdr:    (N, L_cdr, 3) CDR CA 坐标。
        p_epi:    (N, L_epi, 3) Epitope CA 坐标。
        mask_cdr: (N, L_cdr) 可选，CDR mask。
        mask_epi: (N, L_epi) 可选，Epitope mask。
        radius_c:  float，CDR 原子半径（Å）。
        radius_e:  float，Epitope 原子半径（Å）。

    Returns:
        loss: scalar，惩罚原子重叠。
    """
    dist = pairwise_distances(p_cdr, p_epi)
    min_dist = radius_c + radius_e
    penalty = F.relu(min_dist - dist) ** 2

    if mask_cdr is not None:
        penalty = penalty * mask_cdr.unsqueeze(-1)
    if mask_epi is not None:
        penalty = penalty * mask_epi.unsqueeze(-2)

    loss = penalty.sum() / (penalty.numel() + 1e-8)
    return loss
