import torch
import logging
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from easydict import EasyDict

from .constants import (
    AA, max_num_heavyatoms,
    restype_to_heavyatom_names,
    BBHeavyAtom
)

class ParsingException(Exception):
    pass


B_FACTOR_MIN = 5.0     # 最低合理值（高质量晶体）
B_FACTOR_MAX = 200.0   # 最高合理值（柔性区域）
B_FACTOR_WARN_MIN = 10.0  # 警告阈值：过低
B_FACTOR_WARN_MAX = 150.0  # 警告阈值：过高


def validate_b_factor(b_factor_tensor, mask_heavyatom=None, pdb_id="", chain_id=""):
    """
    验证B-factor值是否在合理范围内。

    Args:
        b_factor_tensor: (L, n_atoms) - B-factor值
        mask_heavyatom: (L, n_atoms) - 原子存在mask
        pdb_id: PDB文件ID（用于日志）
        chain_id: 链ID（用于日志）

    Returns:
        is_valid: bool - 是否通过验证
        message: str - 验证信息（如果通过则为空）
    """
    if b_factor_tensor is None or b_factor_tensor.numel() == 0:
        return True, ""

    if mask_heavyatom is not None:
        valid_b_factors = b_factor_tensor[mask_heavyatom]
    else:
        valid_b_factors = b_factor_tensor[b_factor_tensor > 0]

    if valid_b_factors.numel() == 0:
        return True, "No valid B-factor values"

    b_mean = valid_b_factors.mean().item()
    b_min = valid_b_factors.min().item()
    b_max = valid_b_factors.max().item()

    if b_min < B_FACTOR_MIN or b_max > B_FACTOR_MAX:
        msg = (f"[B-factor-Warning] {pdb_id}:{chain_id} B-factor out of range: "
               f"mean={b_mean:.1f}, min={b_min:.1f}, max={b_max:.1f}. "
               f"Expected range: [{B_FACTOR_MIN}, {B_FACTOR_MAX}]")
        logging.warning(msg)
        return False, msg

    if b_mean < B_FACTOR_WARN_MIN or b_mean > B_FACTOR_WARN_MAX:
        msg = (f"[B-factor-Warn] {pdb_id}:{chain_id} B-factor unusual: "
               f"mean={b_mean:.1f} (expected ~{B_FACTOR_WARN_MIN}-{B_FACTOR_WARN_MAX}). "
               f"min={b_min:.1f}, max={b_max:.1f}")
        logging.warning(msg)
        return True, msg

    return True, ""


def _get_residue_heavyatom_info(res: Residue):  # 返回15*3的torch矩阵，每个位置表示AA对应的原子坐标，存在的原子是绝对真实坐标，不存在的是0；(15,)指示原子是否真实存在，是为1，不是为0【1对用有坐标，0对应坐标为000】
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float) ## max_num_heavyatoms=15
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool) ## [15]
    b_factor = torch.zeros([max_num_heavyatoms, ], dtype=torch.float)  # 新增：B-factor [15]
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):  # 找到res对应的所有重原子
        if atom_name == '': continue
        if atom_name in res:
            atom = res[atom_name]
            pos_heavyatom[idx] = torch.tensor(atom.get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True  # 标记当前原子是否是真实存在的
            b_factor[idx] = atom.get_bfactor()  # 新增：提取B-factor
    return pos_heavyatom, mask_heavyatom, b_factor  # res在PDB中真实的重原子坐标，以及标识该原子坐标是否存在，以及B-factor


def parse_biopython_structure(entity, unknown_threshold=1.0, max_resseq=None):  # 单独解析H/L/antigen。因为是imgt chothia编号好的结构，因此可根据PDB中的标号判断是否在cdr区间：解析时会跳过没有骨干原子的AA。entity指的是某类链条（H/L/antigen）的结构数据（来自pdb文件的，包括chain上的所有AA、每个AA上的原子以及坐标）
    chains = Selection.unfold_entities(entity, 'C')  # 以chain为单位展开每个复合物(entity)
    chains.sort(key=lambda c: c.get_id())  # 按照链名进行排序
    data = EasyDict({  # 复合物上的AA数据信息
        'chain_id': [],
        'resseq': [], 'icode': [], 'res_nb': [],
        'aa': [],
        'pos_heavyatom': [], 'mask_heavyatom': [],
        'bfactor': [],  # 新增：B-factor字段
    })
    tensor_types = {
        'resseq': torch.LongTensor,  # PDB文件中的编号
        'res_nb': torch.LongTensor,  # 从1开始的相对编号
        'aa': torch.LongTensor,  # chain中对应的整数序列
        'pos_heavyatom': torch.stack,  # AA坐标矩阵构成的序列 15*3的矩阵
        'mask_heavyatom': torch.stack,  # AA原子是否存在的标志矩阵 15*1的矩阵
        'bfactor': torch.stack,  # 新增：B-factor矩阵 15*1
    }

    count_aa, count_unk = 0, 0
    for i, chain in enumerate(chains):
        seq_this = 0   # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # 按照PDB文件中的resseq-icode 序列号和插入码升序排序
        for _, res in enumerate(residues):  # 一条chain中以AA为单位的数据，只保留到PDB序列号<max_resseq的res，包括AA的全局标识、全局序列号(reseq)、局部序列号（res_nb。保持和原始一样的序列距离，只不过从0开始编号）、全局15个原子坐标、是否真实存在的原子标记
            resseq_this = int(res.get_id()[1])  # 获取当前残基的在PDB文件中对应的序列号
            if max_resseq is not None and resseq_this > max_resseq:
                continue

            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue  # 只保留有骨干结构的AA
            restype = AA(resname)  # 转化为整数数字
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            data.chain_id.append(chain.get_id())

            data.aa.append(restype) # Will be automatically cast to torch.long

            pos_heavyatom, mask_heavyatom, b_factor = _get_residue_heavyatom_info(res)  # 获取每个AA的原子坐标（15*3）；mask指示对应位置是否有原子坐标，1表示有对应坐标，0表示没有原子对应为000；b_factor是B-factor
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)
            data.bfactor.append(b_factor)  # 新增：保存B-factor

            resseq_this = int(res.get_id()[1])  # AA在复合物中的序列编号
            icode_this = res.get_id()[2]  # AA在复合物中的插入码
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        raise ParsingException('No parsed residues.')

    if (count_unk / count_aa) >= unknown_threshold:  # 如果未知残基（被标记为AA.UNK）的残基数量 和 确定残基的数量比超过设定的阈值，认为这个复合物有问题，因为太多未知残基了
        raise ParsingException(
            f'Too many unknown residues, threshold {unknown_threshold:.2f}.'
        )

    seq_map = {}
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        seq_map[(chain_id, resseq, icode)] = i  # {（chain_id, resseq, icode）', idx}是AA的在PDB文件中的唯一标识， resseq对应PDB文件中的编号。seq_map是每个AA的在原始（resseq）or 局部序列（res_nb）中的索引

    for key, convert_fn in tensor_types.items():  # 仅仅转化数据类型
        data[key] = convert_fn(data[key])

    if 'bfactor' in data and data['bfactor'].numel() > 0:
        first_chain_id = data['chain_id'][0] if len(data['chain_id']) > 0 else "unknown"
        validate_b_factor(
            b_factor_tensor=data['bfactor'],
            mask_heavyatom=data['mask_heavyatom'],
            pdb_id="",  # 可以从外部传入
            chain_id=str(first_chain_id)
        )

    return data, seq_map  # 得到以AA为单位的数据。最终的res序列不一定时PDB文件中的，因为删除了坐标缺失的res；AA在整个结构中的真实标识->索引的map
