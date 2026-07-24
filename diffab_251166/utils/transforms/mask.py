import torch
import random
from typing import List, Optional

from ..protein import constants
from ._base import register_transform
from copy import deepcopy as dcp

def random_shrink_extend(flag, min_length=5, shrink_limit=1, extend_limit=2):
    first, last = continuous_flag_to_range(flag)  # 掩码区域中最小和最大的索引位置
    length = flag.sum().item()  # 计算掩码区域的长度
    if (length - 2*shrink_limit) < min_length:
        shrink_limit = 0
    first_ext = max(0, first-random.randint(-shrink_limit, extend_limit))
    last_ext = min(last+random.randint(-shrink_limit, extend_limit), flag.size(0)-1)
    flag_ext = flag.clone()
    flag_ext[first_ext : last_ext+1] = True
    return flag_ext


def continuous_flag_to_range(flag):
    first = (torch.arange(0, flag.size(0))[flag]).min().item()  # flag.size(0)是对应chains上的AA个数
    last = (torch.arange(0, flag.size(0))[flag]).max().item()
    return first, last  # 找到对应区域的起始和终止索引位置


@register_transform('mask_single_cdr')
class MaskSingleCDR(object):  # 只对1个chain的1个cdr进行掩码

    def __init__(self, selection=None, augmentation=False, antibody_structure_generate_mode='cdr_only'):  # 源码augmentation=True
        super().__init__()
        cdr_str_to_enum = {
            'H1': constants.CDR.H1,
            'H2': constants.CDR.H2,
            'H3': constants.CDR.H3,
            'L1': constants.CDR.L1,
            'L2': constants.CDR.L2,
            'L3': constants.CDR.L3,
            'H_CDR1': constants.CDR.H1,
            'H_CDR2': constants.CDR.H2,
            'H_CDR3': constants.CDR.H3,
            'L_CDR1': constants.CDR.L1,
            'L_CDR2': constants.CDR.L2,
            'L_CDR3': constants.CDR.L3,
            'CDR3': 'CDR3',     # H3 first, then fallback to L3
        }
        assert selection is None or selection in cdr_str_to_enum
        self.selection = cdr_str_to_enum.get(selection, None)
        self.augmentation = augmentation
        assert antibody_structure_generate_mode in ['cdr_only', 'cdr_and_fr']
        self.antibody_structure_generate_mode = antibody_structure_generate_mode

    def perform_masking_(self, data, selection=None):
        cdr_flag = data['cdr_flag']

        if selection is None:  # 如果没有指定生成的cdr，随机选择1个cdr。而非全部生成
            cdr_all = cdr_flag[cdr_flag > 0].unique().tolist()
            cdr_to_mask = random.choice(cdr_all)
        else:  # 如果指定了生成的cdr，直接使用
            cdr_to_mask = selection

        cdr_to_mask_flag = cdr_flag * (cdr_flag == cdr_to_mask)  # 需要被掩码的区域 是True，其余为False
        if self.augmentation:  # 生成区域随机扩展/缩小
            cdr_to_mask_flag = random_shrink_extend(cdr_to_mask_flag.to(torch.bool))

        cdr_first, cdr_last = continuous_flag_to_range(cdr_to_mask_flag.to(torch.bool))
        left_idx = max(0, cdr_first - 1)  # 生成区域左侧的AA索引【外侧，不是生成区域。除非生成区域本身已经到最AA序列始端了】
        right_idx = min(data['aa'].size(0) - 1, cdr_last + 1)  # 生成区域右侧一共AA的索引【外侧，不是生成区域。除非生成区域本身已经到最AA序列末端了】
        anchor_flag = torch.zeros(data['aa'].shape, dtype=torch.bool)
        anchor_flag[left_idx] = True  # 标识生成区域左端
        anchor_flag[right_idx] = True  # 标识生成区域右端

        data['generate_flag'] = cdr_to_mask_flag.to(torch.int)
        data['anchor_flag'] = anchor_flag

        if self.antibody_structure_generate_mode == 'cdr_only':
            data['structure_generate_flag'] = cdr_to_mask_flag.to(torch.int)
        elif self.antibody_structure_generate_mode == 'cdr_and_fr':
            data['structure_generate_flag'] = (cdr_flag > 0).to(torch.int)

    def __call__(self, structure):  # 只对1个chain的1个cdr进行掩码
        if self.selection is None: # 如果没有指定生成的cdr，随机选择H/L一个
            ab_data = []
            if structure['heavy'] is not None:
                ab_data.append(structure['heavy'])
            if structure['light'] is not None:
                ab_data.append(structure['light'])
            data_to_mask = random.choice(ab_data)  # 随机选择生成chain
            sel = None  # 具体生成的cdr 为None
        elif self.selection in (constants.CDR.H1, constants.CDR.H2, constants.CDR.H3, ):  # 生成对象为cdrh
            data_to_mask = structure['heavy']  # 生成对象为H
            sel = int(self.selection)  # 生成的cdr
        elif self.selection in (constants.CDR.L1, constants.CDR.L2, constants.CDR.L3, ): # 生成对象为cdrl
            data_to_mask = structure['light']  # 生成对象为L
            sel = int(self.selection) # 生成的cdr
        elif self.selection == 'CDR3':  # 如果只指定生成cdr3，但没有指定生成的chain。有限悬着cdrh3；如果没有h链，则生成cdrl3
            if structure['heavy'] is not None:
                data_to_mask = structure['heavy']
                sel = constants.CDR.H3
            else:
                data_to_mask = structure['light']
                sel = constants.CDR.L3

        self.perform_masking_(data_to_mask, selection=sel)

        chains = ['heavy', 'light']
        for c in chains:
            if structure[c] is not None:
                if 'generate_flag' in structure[c]: # 表明该chain完全没掩码
                    gen_mask = structure[c]['generate_flag'].to(torch.bool)
                    structure[c]['fix_cdr_flag'] = dcp(structure[c]['cdr_flag']).to(torch.int)
                    structure[c]['fix_cdr_flag'][gen_mask] = 0
                    assert (structure[c]['fix_cdr_flag'].to(torch.bool) & gen_mask).sum() == 0, \
                        f"structure[c]['fix_cdr_flag'] {structure[c]['fix_cdr_flag']}, gen_flag {structure[c]['generate_flag']}"
                    assert torch.equal(structure[c]['fix_cdr_flag'] | structure[c]['generate_flag'],
                                       structure[c]['cdr_flag'].to(torch.int)), \
                        f"structure[c]['fix_cdr_flag'] {structure[c]['fix_cdr_flag']}, gen_flag {structure[c]['generate_flag']}"

                else:
                    structure[c]['generate_flag'] = torch.zeros(structure[c]['cdr_flag'].shape, dtype=torch.int)
                    structure[c]['fix_cdr_flag'] = dcp(structure[c]['cdr_flag']).to(torch.int)

                if 'structure_generate_flag' not in structure[c]:
                    if self.antibody_structure_generate_mode == 'cdr_only':
                        structure[c]['structure_generate_flag'] = structure[c]['generate_flag'].clone()
                    elif self.antibody_structure_generate_mode == 'cdr_and_fr':
                        structure[c]['structure_generate_flag'] = (structure[c]['cdr_flag'] > 0).to(torch.int)
        return structure


@register_transform('mask_multiple_cdrs')
class MaskMultipleCDRs(object): # 对H、L 2个chain的个多cdr同时掩码

    def __init__(self, selection: Optional[List[str]]=None, augmentation=False, antibody_structure_generate_mode='cdr_only'):  # # 源码augmentation=True
        super().__init__()
        cdr_str_to_enum = {
            'H1': constants.CDR.H1,
            'H2': constants.CDR.H2,
            'H3': constants.CDR.H3,
            'L1': constants.CDR.L1,
            'L2': constants.CDR.L2,
            'L3': constants.CDR.L3,
            'H_CDR1': constants.CDR.H1,
            'H_CDR2': constants.CDR.H2,
            'H_CDR3': constants.CDR.H3,
            'L_CDR1': constants.CDR.L1,
            'L_CDR2': constants.CDR.L2,
            'L_CDR3': constants.CDR.L3,
        }
        if selection is not None:  # 生成区域list
            self.selection = [cdr_str_to_enum[s] for s in selection]
        else:
            self.selection = None
        self.augmentation = augmentation
        assert antibody_structure_generate_mode in ['cdr_only', 'cdr_and_fr']
        self.antibody_structure_generate_mode = antibody_structure_generate_mode

    def mask_one_cdr_(self, data, cdr_to_mask):
        cdr_flag = data['cdr_flag']

        cdr_to_mask_flag = cdr_flag * (cdr_flag == cdr_to_mask)  # 需要被掩码的区域 是True，其余为False
        if self.augmentation:  # 生成区域随机扩展/缩小
            cdr_to_mask_flag = random_shrink_extend(cdr_to_mask_flag.to(torch.bool))

        cdr_first, cdr_last = continuous_flag_to_range(cdr_to_mask_flag.to(torch.bool))
        left_idx = max(0, cdr_first-1)  # 生成区域左侧的AA索引【外侧，不是生成区域。除非生成区域本身已经到最AA序列始端了】
        right_idx = min(data['aa'].size(0)-1, cdr_last+1) # 生成区域右侧一共AA的索引【外侧，不是生成区域。除非生成区域本身已经到最AA序列末端了】
        anchor_flag = torch.zeros(data['aa'].shape, dtype=torch.bool)
        anchor_flag[left_idx] = True  # 标识生成区域左端
        anchor_flag[right_idx] = True # 标识生成区域右端

        if 'generate_flag' not in data:
            data['generate_flag'] = cdr_to_mask_flag.to(torch.bool)
            data['anchor_flag'] = anchor_flag
        else:   # 如果已经有generate——flag说明是多个cdr mask,按位或。得到多个cdr和和对应生成区域两端标识
            data['generate_flag'] |= cdr_to_mask_flag.to(torch.bool)
            data['anchor_flag'] |= anchor_flag

    def mask_for_one_chain_(self, data):  # 在1条链上mask 1个cdr
        cdr_flag = data['cdr_flag']  # dict_keys(['chain_id', 'resseq', 'icode', 'res_nb', 'aa', 'pos_heavyatom', 'mask_heavyatom', 'cdr_flag', 'H1_seq', 'H2_seq', 'H3_seq'])
        cdr_all = cdr_flag[cdr_flag > 0].unique().tolist()  # cdr区域编号。heavy[1,2,3] light[4,5,6]
    
        num_cdrs_to_mask = random.randint(1, len(cdr_all))   # 如果有多个cdr区域，随机选择要掩码的cdr区域数量

        if self.selection is not None:   # 如果指定了生成的cdr，则和数据本身具备的cdr取交集
            cdrs_to_mask = list(set(cdr_all).intersection(self.selection))
        else:  # 如果数据本身有多个cdr，但是没有指定生成的cdr_all。则随机选择要掩码的cdr区域数量、随机选择cdr
            random.shuffle(cdr_all)  # 随机选择要掩码的cdr区域
            cdrs_to_mask = cdr_all[:num_cdrs_to_mask]

        for cdr_to_mask in cdrs_to_mask:  # 将每个生成cdr区域都进行generate_flag标注、anchor_flag标注【生成区域就是增强后的cdr，而anchor是生成区域的两端】
            self.mask_one_cdr_(data, cdr_to_mask)  # 增加generate_flag和anchor_flag标识生成区域和生成区域两端点

    def __call__(self, structure):

        if 'heavy' in structure and structure.get('heavy') is not None:
            self.mask_for_one_chain_(structure['heavy'])
            cdr_gen_mask = structure['heavy']['generate_flag'].to(torch.bool)
            if self.antibody_structure_generate_mode == 'cdr_and_fr':
                structure['heavy']['structure_generate_flag'] = cdr_gen_mask | structure['heavy']['fr_flag'].to(torch.bool)
            else:
                structure['heavy']['structure_generate_flag'] = cdr_gen_mask.to(torch.int)
            structure['heavy']['fix_cdr_flag'] = dcp(structure['heavy']['cdr_flag']).to(torch.int)
            structure['heavy']['fix_cdr_flag'][cdr_gen_mask] = 0
            structure['heavy']['fr_flag'][cdr_gen_mask] = 0

            if 'light' in structure and structure.get('light') is not None:
                self.mask_for_one_chain_(structure['light'])
                cdr_gen_mask = structure['light']['generate_flag'].to(torch.bool)
                if self.antibody_structure_generate_mode == 'cdr_and_fr':
                    structure['light']['structure_generate_flag'] = cdr_gen_mask | structure['light']['fr_flag'].to(torch.bool)
                else:
                    structure['light']['structure_generate_flag'] = cdr_gen_mask.to(torch.int)
                structure['light']['fix_cdr_flag'] = dcp(structure['light']['cdr_flag']).to(torch.int)
                structure['light']['fix_cdr_flag'][cdr_gen_mask] = 0
                structure['light']['fr_flag'][cdr_gen_mask] = 0
        elif 'light' in structure and structure.get('light') is not None:
            self.mask_for_one_chain_(structure['light'])
            cdr_gen_mask = structure['light']['generate_flag'].to(torch.bool)
            if self.antibody_structure_generate_mode == 'cdr_and_fr':
                structure['light']['structure_generate_flag'] = cdr_gen_mask | structure['light']['fr_flag'].to(torch.bool)
            else:
                structure['light']['structure_generate_flag'] = cdr_gen_mask.to(torch.int)
            structure['light']['fix_cdr_flag'] = dcp(structure['light']['cdr_flag']).to(torch.int)
            structure['light']['fix_cdr_flag'][cdr_gen_mask] = 0
            structure['light']['fr_flag'][cdr_gen_mask] = 0
        else:
            if 'cdr_flag' in structure:
                if 'generate_flag' not in structure:
                    structure['generate_flag'] = torch.zeros(structure['cdr_flag'].shape, dtype=torch.int)
                if 'anchor_flag' not in structure:
                    structure['anchor_flag'] = torch.zeros(structure['cdr_flag'].shape, dtype=torch.int)

                cdr_all = self._get_available_cdrs(structure['cdr_flag'])
                if self.selection is not None:
                    cdrs_to_mask = list(set(cdr_all).intersection(self.selection))
                else:
                    cdrs_to_mask = cdr_all

                for cdr_to_mask in cdrs_to_mask:
                    self.mask_one_cdr_(structure, cdr_to_mask)

                cdr_gen_mask = structure['generate_flag'].to(torch.bool)
                if self.antibody_structure_generate_mode == 'cdr_and_fr':
                    if 'fr_flag' in structure:
                        structure['structure_generate_flag'] = cdr_gen_mask | structure['fr_flag'].to(torch.bool)
                    else:
                        structure['structure_generate_flag'] = cdr_gen_mask.to(torch.int)
                else:
                    structure['structure_generate_flag'] = cdr_gen_mask.to(torch.int)

                if 'fix_cdr_flag' not in structure:
                    structure['fix_cdr_flag'] = dcp(structure['cdr_flag']).to(torch.int)
                    structure['fix_cdr_flag'][cdr_gen_mask] = 0
                if 'fr_flag' in structure:
                    structure['fr_flag'][cdr_gen_mask] = 0
            else:
                pass

            if 'generate_flag' not in structure:
                structure['generate_flag'] = torch.zeros(structure['cdr_flag'].shape, dtype=torch.int)
            if 'anchor_flag' not in structure:
                structure['anchor_flag'] = torch.zeros(structure['cdr_flag'].shape, dtype=torch.int)

            cdr_all = self._get_available_cdrs(structure['cdr_flag'])
            if self.selection is not None:
                cdrs_to_mask = list(set(cdr_all).intersection(self.selection))
            else:
                cdrs_to_mask = cdr_all

            for cdr_to_mask in cdrs_to_mask:
                self.mask_one_cdr_(structure, cdr_to_mask)

            cdr_gen_mask = structure['generate_flag'].to(torch.bool)
            if self.antibody_structure_generate_mode == 'cdr_and_fr':
                if 'fr_flag' in structure:
                    structure['structure_generate_flag'] = cdr_gen_mask | structure['fr_flag'].to(torch.bool)
                else:
                    structure['structure_generate_flag'] = cdr_gen_mask.to(torch.int)
            else:
                structure['structure_generate_flag'] = cdr_gen_mask.to(torch.int)

            if 'fix_cdr_flag' not in structure:
                structure['fix_cdr_flag'] = dcp(structure['cdr_flag']).to(torch.int)
            structure['fix_cdr_flag'][cdr_gen_mask] = 0

        return structure

class MaskAntibody(object):

    def mask_ab_chain_(self, data):
        data['generate_flag'] = torch.ones(data['aa'].shape, dtype=torch.bool)

    def __call__(self, structure):
        pos_ab_alpha = []
        if structure['heavy'] is not None:
            self.mask_ab_chain_(structure['heavy'])
            pos_ab_alpha.append(
                structure['heavy']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            )
        if structure['light'] is not None:
            self.mask_ab_chain_(structure['light'])
            pos_ab_alpha.append(
                structure['light']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            )
        pos_ab_alpha = torch.cat(pos_ab_alpha, dim=0)   # (L_Ab, 3)

        if structure['antigen'] is not None:
            pos_ag_alpha = structure['antigen']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            ag_ab_dist = torch.cdist(pos_ag_alpha, pos_ab_alpha)    # (L_Ag, L_Ab)
            nn_ab_dist = ag_ab_dist.min(dim=1)[0]   # (L_Ag)
            contact_flag = (nn_ab_dist <= 6.0)      # (L_Ag)
            if contact_flag.sum().item() == 0:
                contact_flag[nn_ab_dist.argmin()] = True

            anchor_idx = torch.multinomial(contact_flag.float(), num_samples=1).item()
            anchor_flag = torch.zeros(structure['antigen']['aa'].shape, dtype=torch.bool)
            anchor_flag[anchor_idx] = True
            structure['antigen']['anchor_flag'] = anchor_flag
            structure['antigen']['contact_flag'] = contact_flag
        
        return structure


@register_transform('remove_antigen')
class RemoveAntigen:

    def __call__(self, structure):
        structure['antigen'] = None
        structure['antigen_seqmap'] = None
        return structure
