import math
import torch
from torch.utils.data._utils.collate import default_collate


DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_id': ' ', 
    'icode': ' ',
}

DEFAULT_NO_PADDING = {
    'origin',
}
class PaddingCollate(object):  # 将每个batch进行填充，'origin'不填充，’aa‘用21填充、'chian_id'用' '填充，'icode'用’ ‘填充。其余的数据用0填充（会把0转化为与填充对象同类型的数据，eg bool）

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key  # 整个batch中参照的最大长度来自哪个key的数据
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):  # x是数据，n是需要填充的长度,value是填充的值
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)  # 会转化为和x相同类型的数据
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))  # n- len(x) 补多少个； value 用什么值补充
            return x + pad  # 补充内容追加到后面
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n): # pad是为了整个batch长度对齐。l是当前数据的长度，n是整个batch的长度，都是True和False的拼接，标记原始位置，和填充位置
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key, v):
        if key not in self.pad_values:  # key是否有固定的pad值  self.pad_values={'aa': 21, 'chain_id': ' ', 'icode': ' '}
            if isinstance(v, torch.Tensor):
                zero = torch.tensor(0, dtype=v.dtype, device=v.device)
            else:
                zero = 0
            return zero
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])  # data_list中最长的AA数量【多个复合物中 局部数据的最长AA数量】
        keys = self._get_common_keys(data_list)  # 获取所有数据的公共key
        if self.eight:  # 局部数据中的最大AA数量是否必须是8的倍数
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:  # 从右侧填充。除了不用补齐（self.no_padding），即所有复合物中数量一致（origin 中心点坐标）的数据。要将每个batch中数据其余各个key下的数据补齐。因为数据实际上是以AA为单位保存的，对齐到整个批次里最长的AA数量。其中个别key需要用专门的数据补齐（{'aa': 21, 'chain_id': ' ', 'icode': ' '}），其余都用0补齐
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k, v)) if k not in self.no_padding else v  # self.no_padding={'origin'} 局部数据/anchor AAs的中心点坐标
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)  # 原始数据的标识，1指的是原始数据位置，0指的是补充以对齐整个batch的数据位置
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def apply_patch_to_tensor(x_full, x_patch, patch_idx):  # 将局部区域的AA信息替换原始x_full的AA信息，构成新的、完整的复合物
    """
    Args:
        x_full:  (N, ...)
        x_patch: (M, ...)
        patch_idx:  (M, )
    Returns:
        (N, ...)
    """
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full
