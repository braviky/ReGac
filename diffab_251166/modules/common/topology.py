import torch
import torch.nn.functional as F


def get_consecutive_flag(chain_nb, res_nb, mask):
    """
    判断每个AA是否和后一个AA在序列上相邻（residue-i 与 residue-(i+1) 的关系）。
    需要满足：1) 残基编号差为1；2) 在同一条链上。

    Args:
        chain_nb: 链编号，shape (B, L)
        res_nb: 残基序号，shape (B, L)
        mask: 有效残基掩码，shape (B, L)

    Returns:
        consec: BoolTensor (B, L-1)，consec[b, i] 表示 residue-i 是否与 residue-(i+1) 连续
                i 的范围是 [0, L-2]，对应 residue 0~L-2 与下一个残基的关系
    """
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()   # (B, L-1)
    same_chain = (chain_nb[:, 1:] == chain_nb[:, :-1])
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])
    return consec


def get_terminus_flag(chain_nb, res_nb, mask): # 补个N端 C端的flag
    consec = get_consecutive_flag(chain_nb, res_nb, mask)  # 连续AAflag，False的地方说明AA在序列上不连续了。填充位置的AA对应值为False
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)  # 左端加1个true，consec左移，相当于整个完整序列中考虑最左侧一个真实的AA
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)  # 右端加1个True，consec左移，相当于整个完整序列中考虑最右侧一个真实的AA
    return N_term_flag, C_term_flag
