import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab_251166.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals
from diffab_251166.modules.common.layers import AngularEncoding
from diffab_251166.utils.protein.constants import BBHeavyAtom, AA, REGION_NUM


class PairEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms  # 15
        self.max_aa_types = max_aa_types  # 22
        self.max_relpos = max_relpos  # 32 相对位置
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)  # 每对AA的特征向量
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim // 2)  # AA的序列相对位置
        self.relregion_embed = nn.Embedding((REGION_NUM+1) * (REGION_NUM+1), feat_dim // 2)  # AA的相对区域位置

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types * self.max_aa_types,
                                               max_num_atoms * max_num_atoms)  # 每对AA的原子特征
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(  # 每对AA的原子距离特征编码
            nn.Linear(max_num_atoms * max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()  # 纯数学计算。角度值+sin高频编码+cos高频编码
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)  # Phi and Psi  26维度

        infeat_dim = feat_dim + feat_dim // 2 + feat_dim + feat_dihed_dim + feat_dim // 2
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, region_type, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()

        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L) 标记每个复合物中真实存在的残基，0标识是padding的
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None,
                                               :]  # (N, L, L) 标记每个复合物中真实存在的残基对，如果两个残基都存在，则为1；至少一个不存在，则为0
        pair_structure_mask = structure_mask[:, :, None] * structure_mask[:, None,
                                                           :] if structure_mask is not None else None  # (N, L, L) 标记需要掩码的残基对。如果两个残基中只是一个是生成区域残基，则需要标记，之后删除对应信息，避免信息泄露；否则不用标记

        if sequence_mask is not None:
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))  # aa序列中生成区域残基用AA.UNK(20)替代
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None,
                                                       :]  # (N, L, L) AA类型编号的混合。乘 max_aa_types 是为了把二维氨基酸对 (i,j) 展平成唯一的一维索引，从而让 nn.Embedding 能用一个查表方式高效编码所有可能的氨基酸对特征
        feat_aapair = self.aa_pair_embed(aa_pair)  # (N, L, L, F(64)) Embedding层

        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])  # 标记残基对是否来自相同的chain(heavy light antigen)
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :],
            min=-self.max_relpos, max=self.max_relpos,
        )  # (N, L, L) 在复合物的残基序列上的序列距离。被限制为[-32, 32]，而且是对称矩阵
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :,
                                                                    None]  # (N, L, L, 64) 只保留相同chain上的特征。由于Eembedding层类似一个参数矩阵，通过索引确定对应输出特征。+self.max_relpos是为了将索引转化为非负，否则会产生越界错误

        d = angstrom_to_nm(torch.linalg.norm(  # 最终得到一个 (N, L, L, 15, 15) 的张量，表示任意两个原子之间的欧几里得距离。
            pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],
            dim=-1, ord=2,
        )).reshape(N, L, L, -1)  # (N, L, L, A*A)  拉平，标识两个AA之间的欧式距离
        c = F.softplus(self.aapair_to_distcoef(
            aa_pair))  # (N, L, L, A*A)  为每个原子对计算一个距离权重系数。【因为不是所有距离对于任务而言都同等重要】F.softplus平滑版 ReLU 函数，作用是把输入 压缩成严格正数
        d_gauss = torch.exp(-1 * c * d ** 2)  # 加权并平滑距离。使得所有原子对的距离在(0, 1]之间
        mask_atom_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(N, L, L,
                                                                                                     -1)  # 因为每个AA中15个原始并不是全都是真实存在的，不存在的原子坐标被置为0，但是在原子对的计算中也算了。所以要“删除”这些不合理的原子对距离信息，将其置为0【因为之前用e平滑后，最小的距离也不会到0】
        feat_dist = self.distance_embed(
            d_gauss * mask_atom_pair)  # (N, L, L, 255(每个原子对之间的距离))->(N, L, L, feat(64) 得到AA之间的空间距离。经过两个线性层编码。d_gauss * mask_atom_pair保留正常原子
        if pair_structure_mask is not None:
            feat_dist = feat_dist * pair_structure_mask[:, :, :,
                                    None]  # (N, L, L) 标记需要掩码的残基对。如果两个残基中只是一个是生成区域残基，则需要标记，之后删除对应信息，避免信息泄露；否则不用标记
        dihed = pairwise_dihedrals(pos_atoms)  # (N, L, L, 2) phi psi的相对坐标
        feat_dihed = self.dihedral_embed(dihed)  # # 纯数学计算。角度值+sin高频编码+cos高频编码
        if pair_structure_mask is not None:
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]

        relregion = region_type[:, :, None] * (REGION_NUM + 1) + region_type[:, None, :]  # (N, L, L)
        feat_relregion = self.relregion_embed(relregion)  # (N, L, L, feat_dim//2)

        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed, feat_relregion], dim=-1)
        feat_all = self.out_mlp(feat_all)  # (N, L, L, F)

        feat_all = feat_all * mask_pair[:, :, :, None]  # 填充AA的信息之前都是正常算，不额外干预，质感与生成区域数据不要泄露。在这里统一将填充AA相关信息置为0

        return feat_all

