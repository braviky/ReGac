import torch

from ..protein import constants
from ._base import register_transform


@register_transform('merge_chains')
class MergeChains(object): # 为每个AA分配对应chain的整数id。并合并heavy light antigen的数据，为一个整体。合并前是分开的、相互独立的，合并是按照key合并为一个整体

    def __init__(self):
        super().__init__()

    def assign_chain_number_(self, data_list):  # 根据对应符合物占据chain的总数，为每个AA分配整数id，依次是heavy\light\antigen【antigen如果涉及两条chain,分别分配一个id】
        chains = set()
        for data in data_list:  # data = structure['heavy']\structure['light']\structure['antigen']
            chains.update(data['chain_id'])
        chains = {c: i for i, c in enumerate(chains)}  # eg chain_id {A: 0, B: 1, C: 2}

        for data in data_list:
            data['chain_nb'] = torch.LongTensor([
                chains[c] for c in data['chain_id']
            ])

    def _data_attr(self, data, name):
        if name in ('generate_flag', 'anchor_flag', 'structure_generate_flag') and name not in data:
            return torch.zeros(data['aa'].shape, dtype=torch.bool)
        elif name in ('cdr_flag', 'fr_flag', 'fix_cdr_flag') and name not in data:  # # antigen没有cdr区域 对应位置置为0
            return torch.zeros_like(
                data['aa'],
            )
        else:
            return data[name]

    def __call__(self, structure):
        data_list = []
        if structure['heavy'] is not None:
            structure['heavy']['fragment_type'] = torch.full_like(  # 创建一个与输入张量形状相同、但所有元素都被填充为指定值的新张量
                structure['heavy']['aa'],
                fill_value = constants.Fragment.Heavy,  # 1
            )
            h_region_type = structure['heavy']['cdr_flag'] + structure['heavy']['fr_flag']
            assert h_region_type.min() > 0 and h_region_type.max() <= 14  , f"h_region_type.min() (f{h_region_type.min()}) == 0 or h_region_type.max()(f{h_region_type.max()}) > 11. Heavy region {h_region_type}"
            structure['heavy']['region_type'] = h_region_type.long()
            data_list.append(structure['heavy'])

        if structure['light'] is not None:
            structure['light']['fragment_type'] = torch.full_like( # 创建一个与输入张量形状相同、但所有元素都被填充为指定值的新张量
                structure['light']['aa'],
                fill_value = constants.Fragment.Light,  # 2
            )
            l_region_type = structure['light']['cdr_flag'] + structure['light']['fr_flag']
            structure['light']['region_type'] = l_region_type.long()
            assert l_region_type.min() > 0 and l_region_type.max() <= 14  , f"l_region_type.min() (f{l_region_type.min()}) == 0 or l_region_type.max()(f{l_region_type.max()}) > 11. Light region {l_region_type}"
            data_list.append(structure['light'])

        if structure['antigen'] is not None:
            structure['antigen']['fragment_type'] = torch.full_like( # 创建一个与输入张量形状相同、但所有元素都被填充为指定值的新张量
                structure['antigen']['aa'],
                fill_value = constants.Fragment.Antigen,  # 3
            )
            structure['antigen']['region_type'] = torch.full_like(
                structure['antigen']['aa'],
                fill_value = constants.AG.NON_EPI,  # 17
            ).long()
            data_list.append(structure['antigen'])

        self.assign_chain_number_(data_list)  # data_list包含H、L、antigen数据。为复合物的每个chain分配一个整数id 按heavy light antigen的顺序

        list_props = {
            'chain_id': [],
            'icode': [],
        }
        tensor_props = {  # 合并后作为整体的keys。heavy/light的具体的H/L1-3 seq舍弃了。
            'chain_nb': [],
            'resseq': [],
            'res_nb': [],
            'aa': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'bfactor': [],
            'generate_flag': [],
            'structure_generate_flag': [],
            'cdr_flag': [],
            'fr_flag': [],
            'fix_cdr_flag': [],
            'anchor_flag': [],
            'region_type': [],
            'fragment_type': [],
        }

        for data in data_list:
            for k in list_props.keys():
                list_props[k].append(self._data_attr(data, k))
            for k in tensor_props.keys():
                tensor_props[k].append(self._data_attr(data, k))  # 依次返回H L antigen对象的value张量，构成列表。如果改key不存在，则返回与该chain长于一样的zeros

        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}  # 按照H L antigen的顺序合并
        data_out = {
            **list_props,
            **tensor_props,
        }
        if "id" in structure:
            data_out["id"] = structure["id"]
        return data_out

