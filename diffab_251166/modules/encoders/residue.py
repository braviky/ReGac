import torch
import torch.nn as nn

from diffab_251166.modules.common.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles
from diffab_251166.modules.common.layers import AngularEncoding
from diffab_251166.utils.protein.constants import BBHeavyAtom, AA, REGION_NUM, CHAIN_NUM


class ResidueEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms  # 15
        self.max_aa_types = max_aa_types  # 22
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)  # (15,128)
        self.dihed_embed = AngularEncoding()
        self.chain_type_embed = nn.Embedding(CHAIN_NUM + 1, feat_dim // 2, padding_idx=0)    # 1: Heavy, 2: Light, 3: Ag (0: padding)
        self.region_type_embed = nn.Embedding(REGION_NUM + 1, feat_dim // 2, padding_idx=0)    # 1-16: CDR, FR, epitope, non_epitope (0: padding)
        infeat_dim = feat_dim + (self.max_aa_types*max_num_atoms*3) + self.dihed_embed.get_out_dim(3) + feat_dim  # 原始输入特征：aa特征+原子特征+角度特征+类型特征
        self.mlp = nn.Sequential(  # 4个MLP层构成的mlp块
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),  # infeat_dim->256
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),  # 256->128
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),  # 128->128
            nn.Linear(feat_dim, feat_dim)  # 128->128
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, fragment_type, region_type, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa:         (N, L).
            res_nb:     (N, L).
            chain_nb:   (N, L).
            pos_atoms:  (N, L, A, 3).  # A原子数=15
            mask_atoms: (N, L, A). Heavy atom存在情况
            fragment_type:  (N, L).
            structure_mask: (N, L), mask out unknown structures to generate.
            sequence_mask:  (N, L), mask out unknown amino acids to generate.
        """
        N, L = aa.size()  # N 复合物数量，L：统一后AA序列长度（如果N=1，则L为改复合物实际AA序列长度）
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L) 标记的是真实的AA，为0的是填充的AA

        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        if sequence_mask is not None:  # 将生成区域AA 替换为UNK，避免生成数据隐式泄露到其它数据中
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_feat = self.aatype_embed(aa) # (N, L, feat(128))  self.aatype_embed->nn.Embedding)

        R = construct_3d_basis( # 根据每个AA的全局坐标，得到每个AA的局部坐标系，原点在每个AA的CA处
            pos_atoms[:, :, BBHeavyAtom.CA],   # (N, L, 3) 每个AA的Cα坐标
            pos_atoms[:, :, BBHeavyAtom.C], 
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)    # (N, L, A, 3)  将每个AA原子的全局坐标转化为各自的局部坐标：R是每个AA以CA为远点构建的坐标系；t是每个AA的CA坐标；pos_atoms是每个AA的所有原子全局坐标
        crd_mask = mask_atoms[:, :, :, None].expand_as(crd)
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))  # 将AA没有实际原子的局部坐标 置为0

        aa_expand  = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd[:, :, None, :, :].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, self.max_aa_types*self.max_num_atoms*3)  # 每种AA类型都有15*3个元素标识坐标。只有在AA类型对应的区域 才有具体的局部坐标（不存在的原子坐标仍然为0），其余位置全为0
        if structure_mask is not None:
            crd_feat = crd_feat * structure_mask[:, :, None]  # 生成区域坐标全部都是0

        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # (N, L, 3, dihed/3)  (N, L, 3, 角度值+sin高频位置编码+cos高频位置编码)
        dihed_feat = dihed_feat.reshape(N, L, -1)  # 3个角度特征拉平 得到最终的角度特征
        if structure_mask is not None:
            dihed_mask = torch.logical_and(  # 生成区域左右个一个AA都没mask。因为生成区域第一个、最后一个AA算某个角度的时候会用到上一个AA的坐标。因为整两个AA也包含一些生成区域的信息，所以要剔除
                structure_mask,
                torch.logical_and(
                    torch.roll(structure_mask, shifts=+1, dims=1), 
                    torch.roll(structure_mask, shifts=-1, dims=1)
                ),
            )   # Avoid slight data leakage via dihedral angles of anchor residues
            dihed_feat = dihed_feat * dihed_mask[:, :, None]

        chain_type_feat = self.chain_type_embed(fragment_type) # (N, L, feat/2)
        region_type_feat = self.region_type_embed(region_type) # (N, L, feat/2)
        out_feat = self.mlp(torch.cat([aa_feat, crd_feat, dihed_feat, chain_type_feat, region_type_feat], dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]  # 填充对象的特征都是0
        return out_feat
