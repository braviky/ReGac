

import os
import random
import logging
import datetime
import pandas as pd
import joblib
import pickle
import lmdb
import subprocess
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from itertools import chain, combinations
import json
from ..utils.protein import parsers, constants
from ._base import register_dataset

VAL_RADIO = 0.1


def _is_main_process():
    """Check if current process is the main process (rank 0) in distributed training."""
    rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    try:
        return int(rank) == 0
    except (ValueError, TypeError):
        return True


def log_info(msg):
    """Print message only from main process to avoid duplicate output in distributed training."""
    if _is_main_process():
        print(msg)


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein',
    'protein | protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0
TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]
TEST_DIFFAB_PATH = f'./test_diffab_pdb_names.txt'
TEST_RABD_PATH = f'./test_radb_pdb_names.txt'

RABD_CHAIN_OVERRIDES = {
    ('3uzq', 'A', 'a', 'B'): ('a', 'A', ('B',)),
}

RABD_LEGACY_ID_ALIASES = {
    '2ghw_B_B_A': '2ghw_B_b_A',
    '3h3b_C_C_B': '3h3b_c_C_B',
    '3uzq_A_A_B': '3uzq_a_A_B',
    '3uzq_A_a_B': '3uzq_a_A_B',
}

RABD_SAFE_LEGACY_STRUCTURE_IDS = {
    '2ghw_B_b_A': ('2ghw_B_B_A',),
    '3h3b_c_C_B': ('3h3b_C_C_B',),
    '3uzq_a_A_B': ('3uzq_A_A_B', '3uzq_A_a_B'),
}


def _legacy_candidates_for_rabd_id(entry_id):
    candidates = []
    canonical_id = _canonicalize_rabd_id(entry_id)
    if canonical_id != entry_id:
        candidates.append(canonical_id)
    candidates.extend(RABD_SAFE_LEGACY_STRUCTURE_IDS.get(canonical_id, ()))
    candidates.extend(RABD_SAFE_LEGACY_STRUCTURE_IDS.get(entry_id, ()))
    for legacy_id, mapped_id in RABD_LEGACY_ID_ALIASES.items():
        if mapped_id == canonical_id and legacy_id not in candidates:
            candidates.append(legacy_id)
    return list(dict.fromkeys(candidates))



def _make_sabdab_entry_id(pdbcode, heavy_chain, light_chain, ag_chains):
    return f"{pdbcode}_{heavy_chain}_{light_chain}_{''.join(ag_chains)}"


def _legacy_case_collapsed_chains(heavy_chain, light_chain):
    legacy_h, legacy_l = heavy_chain, light_chain
    if legacy_h.upper() == legacy_l:
        legacy_h = legacy_h.upper()
    if legacy_l.upper() == legacy_h:
        legacy_l = legacy_l.upper()
    return legacy_h, legacy_l


def _canonicalize_rabd_chains(pdbcode, heavy_chain, light_chain, ag_chains):
    key = (str(pdbcode).lower(), heavy_chain, light_chain, ''.join(ag_chains))
    if key in RABD_CHAIN_OVERRIDES:
        new_h, new_l, new_ag = RABD_CHAIN_OVERRIDES[key]
        return new_h, new_l, list(new_ag)
    return heavy_chain, light_chain, ag_chains


def _canonicalize_rabd_id(entry_id):
    return RABD_LEGACY_ID_ALIASES.get(entry_id, entry_id)


def get_all_antigen_subsets(ag_chains_str):
    s = list(ag_chains_str)
    subsets = chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    return ["".join(item) for item in subsets]

def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])  # 把每个AA的数字表示 转化为单char的表示


def get_cdr_indices(cdr_flag, cdr_type):
    """Returns (start, end) indices of a CDR region."""
    cdr_mask = (cdr_flag == cdr_type)
    if not cdr_mask.any():  # No such CDR found
        return (None, None)
    indices = torch.where(cdr_mask)[0]
    return (indices[0].item(), indices[-1].item() + 1)  # (start, end)


def PDB_scheme_to_parser(flag_size, pdb_seq_idx, chain_type, CDRDef):
    cdr_flag = torch.zeros(flag_size, dtype=torch.long)
    fr_flag = torch.zeros(flag_size, dtype=torch.long)

    skipped_positions = []  # 记录无法识别的位置
    for position, idx in pdb_seq_idx.items():
        resseq = position[1]
        cdr_type = CDRDef.to_cdr(chain_type, resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
            fr_flag[idx] = 0
        else:
            fr_type = CDRDef.to_fr(chain_type, resseq)
            if fr_type is None:
                skipped_positions.append((position, resseq))
                continue
            fr_flag[idx] = fr_type
            cdr_flag[idx] = 0
    
    if skipped_positions:
        skipped_res = [str(r) for (_, r) in skipped_positions[:5]]  # 只显示前5个
        logging.warning(
            f"Skipped {len(skipped_positions)} positions not in FR definitions "
            f"(chain {chain_type}): {skipped_res}{'...' if len(skipped_positions) > 5 else ''}"
        )
    
    valid_mask = (cdr_flag > 0) | (fr_flag > 0)  # 有效的残基位置（CDR或FR）
    if valid_mask.any():
        cdr_flag_not_valid = torch.where(cdr_flag >= 1, 0, 1).to(torch.bool)[valid_mask]
        fr_flag_valid = (fr_flag > 0).to(torch.bool)[valid_mask]
        assert torch.equal(cdr_flag_not_valid, fr_flag_valid), \
            f"Mismatch between cdr_flag and fr_flag for chain {chain_type}"
    return cdr_flag, fr_flag


def numbering_scheme_to_parser(data, chain_type, numbering_scheme, CDRDef):  ## chain_type 'H' or 'L';
    aa_seq = _aa_tensor_to_sequence(data)  # aa_seq是AA序列 字符串
    try:
        from anarci import anarci
        results = anarci([(chain_type, aa_seq)], scheme=numbering_scheme)
        numbering = results[0][0][0][0]  # [(num, aa), ...]
        use_anarci = numbering is not None

    except Exception as e:
        logging.warning(f"ANARCI failed on H chain: {e}")
        use_anarci = False

    aa_nmb_list = [aa for (_, aa) in numbering]  # 提取 AA 序列，包括 gap
    aa_nmb_seq = ''.join(aa_nmb_list)

    aa_fv = aa_nmb_seq.replace('-', '')
    assert aa_fv in aa_seq, \
        f"Numbering sequence without gaps '{aa_fv}' not in original sequence '{aa_seq}'"
    fv_start = aa_seq.find(aa_fv)

    if fv_start == -1:
        raise ValueError(f"ANARCI Fv region '{aa_fv}' not found in sequence '{aa_seq}'")

    fv_end = fv_start + len(aa_fv)  # 直接作为切片的end即可，即比之际的索引多1个

    seq_len = len(aa_seq)
    cdr_flag = torch.zeros(seq_len, dtype=torch.long)
    fr_flag = torch.zeros(seq_len, dtype=torch.long)

    seq_idx = fv_start  # 原序列索引
    for (num, _), aa_aa in numbering:
        if aa_aa == '-':
            continue

        while seq_idx < fv_end and aa_seq[seq_idx] != aa_aa:
            seq_idx += 1
        if seq_idx >= fv_end:
            raise ValueError(f"AA mismatch between numbering and sequence at {aa_aa}")

        cdr_type = CDRDef.to_cdr(chain_type, num)
        fr_type = CDRDef.to_fr(chain_type, num)
        if cdr_type is not None:
            cdr_flag[seq_idx] = cdr_type
        elif fr_type is not None:
            fr_flag[seq_idx] = fr_type  # 改为整数1
        else:
            raise ValueError(f"Unsupported numbering scheme in {seq_idx} {data}: {numbering_scheme}")

        seq_idx += 1
    return cdr_flag, fr_flag


def _label_heavy_chain_cdr_fr(data, seq_map, max_cdr3_length=30,
                           numbering_scheme='chothia'):  # 在解析的内容data中，加入cdr区域对应的序列、整个序列上cdr的flag 0表示非cdr，1、2、3表示H1、H2、H3
    if data is None or seq_map is None:
        return data, seq_map

    if numbering_scheme.lower() == "chothia":
        CDRDef = constants.ChothiaCDRRange
    elif numbering_scheme.lower() == "imgt":
        CDRDef = constants.IMGTCDRRange
    else:
        raise ValueError(f"Unsupported numbering scheme: {numbering_scheme}")


    cdr_flag, fr_flag = PDB_scheme_to_parser(flag_size=data['aa'].shape, pdb_seq_idx=seq_map, chain_type='H',
                                             CDRDef=CDRDef)

    data['cdr_flag'] = cdr_flag  # 每个AA的flag属于[0,1,2,3]表示非CDR、H1、H2、H3
    data['fr_flag'] = fr_flag
    data['H1_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.H1])
    data['H2_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.H2])
    data['H3_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.H3])

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()




    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map


def _label_light_chain_cdr_fr(data, seq_map, max_cdr3_length=30, numbering_scheme='chothia'):
    if data is None or seq_map is None:
        return data, seq_map

    if numbering_scheme.lower() == "chothia":
        CDRDef = constants.ChothiaCDRRange
    elif numbering_scheme.lower() == "imgt":
        CDRDef = constants.IMGTCDRRange
    else:
        raise ValueError(f"Unsupported numbering scheme: {numbering_scheme}")


    cdr_flag, fr_flag = PDB_scheme_to_parser(flag_size=data['aa'].shape, pdb_seq_idx=seq_map, chain_type='L',
                                             CDRDef=CDRDef)
    data['cdr_flag'] = cdr_flag  # 每个AA的flag属于[0,1,2,3]表示非CDR、L1、L2、L3
    data['fr_flag'] = fr_flag

    data['L1_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.L1])
    data['L2_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.L2])
    data['L3_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.L3])

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()

    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map


def preprocess_sabdab_structure(task,
                                counters):  # task = {id: '', entry: {}, pdb_path: ''}；对该task/pdb对应的PDB文件是否合规、该pdb信息是否完整，进一步筛选，不满足的数据，返回为None、满足的返回结构信息
    entry = task['entry']
    number_scheme = task['number_scheme']
    pdb_path = task['pdb_path']
    parser = PDBParser(QUIET=True)
    parsed = {
        'id': entry['id'],  # 自定义的id。pdb_Hchains_Lchains_Agchains
        'heavy': None,  # None or list字典 以AA为单位被解析的重链数据。每个key的元素长度=过滤无重原子后的heavy序列长度
        'heavy_seqmap': None,  # None or list字典 长度=过滤无重原子后的heavy序列长度
        'light': None,  # None or list字典 以AA为单位被解析的轻链数据  每个key的元素长度=过滤无重原子后的light序列长度
        'light_seqmap': None,  # None or list字典 长度=过滤无重原子后的light序列长度
        'antigen': None,  # # None or list字典 以AA为单位被解析的数据 每个key的元素长度=过滤无重原子后的antigen序列长度
        'antigen_seqmap': None,
    }

    try:


        model = parser.get_structure(id, pdb_path)[
            0]  # model指的是对结构信息进行建模后的数据。四对pdb文件结构化后的结果，包含每条chain，链上的AA，AA上的原子信息。即把pdb文件转化为了结构数据
        if entry['H_chain'] is not None:
            (  # 如果chr3过长或者为被检测到，则为parsed的对应项['heavy' 'heavy_seqmap'为]None
                parsed['heavy'],
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr_fr(*parsers.parse_biopython_structure(
                model[entry['H_chain']],  # entry['H_chain'] H_chain的名称。从model中提取这条链条上的结构信息（pdb中的AA 每个AA上的原子 以及坐标等等）
                max_resseq=constants.MAX_RESLEN[number_scheme]['H']  # 113 Chothia, end of Heavy chain Fv
            ))

        if entry['L_chain'] is not None:
            (  # 如果chr3过长或者为被检测到，则为parsed的对应项['light' 'light_seqmap'为]None
                parsed['light'],  # 和heavy chain一样
                parsed['light_seqmap']
            ) = _label_light_chain_cdr_fr(*parsers.parse_biopython_structure(  # 和heavy chain一样
                model[entry['L_chain']],  # L_chain的名称
                max_resseq=constants.MAX_RESLEN[number_scheme]['L']  # 106 Chothia, end of Light chain Fv
            ))

        if parsed['heavy'] is None and parsed['light'] is None:  # 过滤掉重链和轻链的cdr3长度都有问题的的对象
            counters['issue_heavy & light_miss'] += 1
            raise ValueError('Neither valid H-chain or L-chain is found.')

        if len(entry['ag_chains']) > 0:  # 如果没有antigen的话，则antigen和antigen_seqmap均为None
            chains = [model[c] for c in entry['ag_chains']]
            (
                parsed['antigen'],  # list
                parsed['antigen_seqmap']  # list
            ) = parsers.parse_biopython_structure(
                chains)  # 如果是多条链。chains为列表。分别解析其AA数据，使用seqmap将其链接为一个整体。但是对于每个chain，其局部AA序列都是从res_nb=0开始的


    except (
            PDBExceptions.PDBConstructionException,
            parsers.ParsingException,
            KeyError,
            ValueError,
    ) as e:
        if isinstance(e, parsers.ParsingException):
            counters['issue_sequnk'] += 1
        elif isinstance(e, PDBExceptions.PDBConstructionException):  # 过滤PDB文件不合规/结构异常的复合物对象，因此结构返回为None
            log_info(f'mmcif_parse: {pdb_path} {{{str(e)}}}')
            counters['issue_file'] += 1
        logging.warning('[{}] {}: {}'.format(
            task['id'],
            e.__class__.__name__,
            str(e)
        ))
        return None  # 过滤掉复合物中 重链和轻链同时缺失的对象，返回None

    return parsed  # 可能存在部分项为None (['heavy']['heavy_seqmap']['light']['light_seqmap'] ['antigen']['antigen_seqmap'])的情况



def save_test_df(df, test_indices, save_dir, data_name, data_from_path):
    test_data_df = df.iloc[test_indices].reset_index(drop=True)

    test_data_csv = os.path.join(save_dir, data_name + '.csv')  # 替换为你的输出CSV路径
    os.makedirs(os.path.dirname(test_data_csv), exist_ok=True)

    test_data_df.to_csv(test_data_csv, index=False, sep=',')

    if not data_from_path:  # 数据不是来源于某个文件
        test_data_txt = os.path.join(save_dir, data_name + '.txt')  # 输出TXT路径
        os.makedirs(os.path.dirname(test_data_txt), exist_ok=True)
        with open(test_data_txt, 'w', encoding='utf-8') as f:
            for t_id in self.ref_test_ids:
                f.write(t_id + '\n')


class SAbDabDataset(Dataset):
    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB

    _cache = {}

    def __init__(
            self,
            summary_path='./data/sabdab_summary_all.tsv',
            total_data_dir='./data/all_structures/chothia',
            processed_dir='./data/processed',
            split='train',
            data_number_scheme='chothia',
            split_seed=2022,
            transform=None,
            reset_strc=True,
            test_data='',  ## str 'data/test_diffab_251166.txt' or diffab
            val_split_mode='sample',  # 'sample' or 'cluster'
            val_ratio=0.1,  # 样本比例或cluster比例
            val_epitope_source='native',  # 'native' or 'cross_complex'
    ):
        super().__init__()
        self.split = split
        self.test_data = test_data
        self.val_split_mode = val_split_mode  # 'sample' or 'cluster'
        self.val_ratio = val_ratio
        self.val_epitope_source = val_epitope_source
        log_info(f'split {split};;test_data {test_data};;val_split_mode {val_split_mode};;val_ratio {val_ratio};;val_epitope_source {val_epitope_source}')

        self.ref_test_ids = []
        self.fact_test_ids = []
        self.requested_test_ids = self._read_requested_test_ids(test_data)

        self.summary_path = summary_path
        self.total_data_dir = total_data_dir
        self.reset_strc = reset_strc
        assert total_data_dir.split('/')[
                   -1] == data_number_scheme, f'total_data_dir {total_data_dir}, number_scheme {data_number_scheme}'
        self.data_number_scheme = data_number_scheme
        if not os.path.exists(total_data_dir):
            raise FileNotFoundError(
                f"SAbDab structures not found in {total_data_dir}. "
                "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        cache_key = f"{processed_dir}_{test_data}_{reset_strc}"

        base_cache_loaded = False
        split_cache_loaded = False

        if not reset_strc and os.path.exists(self._disk_cache_base_path):
            base_cache_loaded = self._load_base_disk_cache()
            if base_cache_loaded and self._base_cache_matches_requested_test_ids():
                log_info(f"[SAbDabDataset] Loaded base disk cache, skipping reprocessing")
            elif base_cache_loaded:
                log_info(
                    f"[SAbDabDataset] Ignoring stale base disk cache for requested test ids: "
                    f"requested={len(self.requested_test_ids)}, cached_ref={len(self.ref_test_ids)}"
                )
                base_cache_loaded = False

        if not base_cache_loaded:
            if cache_key not in SAbDabDataset._cache or reset_strc:
                log_info(f"[SAbDabDataset] Cache miss for key: {cache_key}, loading data...")
                self.sabdab_entries = None
                self._load_sabdab_entries()

                self.db_conn = None
                self.db_ids = None
                self._load_structures(reset_strc)

                self.clusters = None
                self.id_to_cluster = None
                if self.split != 'test':
                    self._load_clusters(reset_strc)
                    self._export_full_dataset_with_clusters()
                    self._export_test_json()

                SAbDabDataset._cache[cache_key] = {
                    'sabdab_entries': self.sabdab_entries,
                    'db_ids': self.db_ids,
                    'clusters': self.clusters,
                    'id_to_cluster': self.id_to_cluster,
                    'test_full_nm': self.test_full_nm,
                    'fact_test_ids': self.fact_test_ids,
                    'ref_test_ids': self.ref_test_ids,
                }

                if not reset_strc:
                    self._save_base_disk_cache()
                    log_info(f"[SAbDabDataset] Data cached with key: {cache_key}")
            else:
                log_info(f"[SAbDabDataset] Cache hit for key: {cache_key}, reusing shared data...")
                cached = SAbDabDataset._cache[cache_key]
                self.sabdab_entries = cached['sabdab_entries']
                self.db_ids = cached['db_ids']
                self.clusters = cached['clusters']
                self.id_to_cluster = cached['id_to_cluster']
                self.test_full_nm = cached.get('test_full_nm', [])
                self.fact_test_ids = cached.get('fact_test_ids', [])
                self.ref_test_ids = cached.get('ref_test_ids', [])
                self.db_conn = None  # db_conn 是每个实例独立的

        if not reset_strc:
            split_cache_loaded = self._load_split_disk_cache()
            if split_cache_loaded:
                log_info(f"[SAbDabDataset] Using cached {self.split} split")

        if not split_cache_loaded:
            self.ids_in_split = None  # [id1, id2, ...] train/val/test列表
            self._load_split(split, split_seed)  # split是指分割的数据集
            self._export_to_json_lines()

            if not reset_strc:
                self._save_split_disk_cache()

        self.transform = transform

    def _read_requested_test_ids(self, test_data):
        if not test_data or '.' not in str(test_data) or not os.path.exists(test_data):
            return []
        ids = []
        with open(test_data, 'r', encoding='utf-8') as f:
            for line in f:
                data_id = line.strip()
                if data_id:
                    ids.append(_canonicalize_rabd_id(data_id))
        return list(dict.fromkeys(ids))

    def _base_cache_matches_requested_test_ids(self):
        if self.split != 'test' or not self.requested_test_ids:
            return True
        cached_ref = [_canonicalize_rabd_id(x) for x in getattr(self, 'ref_test_ids', [])]
        return set(self.requested_test_ids).issubset(set(cached_ref))

    def _extract_to_standard_format(self, data, entry):
        if data is None: return None
        res = {
            "pdb": entry['pdbcode'], "heavy_chain": entry.get('H_chain'), "light_chain": entry.get('L_chain'),
            "antigen_chains": entry['ag_chains'],
            "pdb_data_path": os.path.abspath(os.path.join(self.total_data_dir, f"{entry['pdbcode']}.pdb")),
            "numbering": self.data_number_scheme, "pre_numbered": True,
            "heavy_chain_seq": None, "light_chain_seq": None, "antigen_seqs": [],
            "cdrh1_pos": None, "cdrh1_seq": None, "cdrh2_pos": None, "cdrh2_seq": None, "cdrh3_pos": None,
            "cdrh3_seq": None,
            "cdrl1_pos": None, "cdrl1_seq": None, "cdrl2_pos": None, "cdrl2_seq": None, "cdrl3_pos": None,
            "cdrl3_seq": None,
        }
        for ch_key, prefix in [('heavy', 'cdrh'), ('light', 'cdrl')]:
            if data.get(ch_key):
                d = data[ch_key]
                res[f"{ch_key}_chain_seq"] = _aa_tensor_to_sequence(d['aa'])
                for i in range(1, 4):
                    tag = f"{prefix.upper()[3]}{i}"  # 提取 H1, H2, L1 等
                    res[f"{prefix}{i}_pos"] = list(get_cdr_indices(d['cdr_flag'], getattr(constants.CDR, tag)))
                    res[f"{prefix}{i}_seq"] = d.get(f"{tag}_seq")
        if data.get('antigen'):
            ag_dict = {}
            ag_seq = _aa_tensor_to_sequence(data['antigen']['aa'])
            for (cid, _, _), idx in data['antigen_seqmap'].items():
                ag_dict.setdefault(cid, []).append(ag_seq[idx])
            res["antigen_seqs"] = ["".join(v) for v in ag_dict.values()]
        return res

    def _export_full_dataset_with_clusters(self):
        output_file = os.path.join(self.processed_dir, "full_dataset_with_clusters.csv")

        if os.path.exists(output_file) and not self.reset_strc:
            return

        id_to_entry = {e['id']: e for e in self.sabdab_entries}

        all_data = []
        for data_id in tqdm(self.db_ids, desc="Exporting full dataset CSV"):
            data = self.get_structure(data_id)
            entry = id_to_entry.get(data_id)

            if data and entry:
                formatted = self._extract_to_standard_format(data, entry)
                if formatted:
                    formatted['cluster_id'] = self.id_to_cluster.get(data_id)
                    all_data.append(formatted)

        if all_data:
            df = pd.DataFrame(all_data)
            cols = ['pdb', 'cluster_id'] + [c for c in df.columns if c not in ['pdb', 'cluster_id']]
            df[cols].to_csv(output_file, index=False)
            logging.info(f"Full dataset with clusters exported to {output_file}")

    def _export_to_json_lines(self):
        """生成当前split的JSON文件（原有逻辑）"""
        output_file = os.path.join(self.processed_dir, f"{self.split}.json")
        id_to_entry = {e['id']: e for e in self.sabdab_entries}
        with open(output_file, 'w') as f:
            for data_id in self.ids_in_split:
                data, entry = self.get_structure(data_id), id_to_entry.get(data_id)
                if data and entry:
                    formatted = self._extract_to_standard_format(data, entry)
                    if formatted: f.write(json.dumps(formatted) + '\n')

    def _export_test_json(self):
        """额外生成test.json（使用fact_test_ids）"""
        output_file = os.path.join(self.processed_dir, "test.json")
        if os.path.exists(output_file) and not self.reset_strc:
            log_info(f"[SAbDabDataset] test.json already exists, skipping")
            return
        log_info(f"[SAbDabDataset] Generating test.json with {len(self.fact_test_ids)} entries...")
        id_to_entry = {e['id']: e for e in self.sabdab_entries}
        with open(output_file, 'w') as f:
            for data_id in self.fact_test_ids:
                data, entry = self.get_structure(data_id), id_to_entry.get(data_id)
                if data and entry:
                    formatted = self._extract_to_standard_format(data, entry)
                    if formatted: f.write(json.dumps(formatted) + '\n')
        log_info(f"[SAbDabDataset] test.json generated")


    def _save_base_disk_cache(self):
        """Save base data (shared across splits) to disk cache."""
        cache_data = {
            'sabdab_entries': self.sabdab_entries,
            'db_ids': self.db_ids,
            'clusters': self.clusters,
            'id_to_cluster': self.id_to_cluster,
            'ref_test_ids': self.ref_test_ids,
            'fact_test_ids': self.fact_test_ids,
            'test_full_nm': getattr(self, 'test_full_nm', []),
        }
        with open(self._disk_cache_base_path, 'wb') as f:
            pickle.dump(cache_data, f)
        log_info(f"[SAbDabDataset] Saved base disk cache to {self._disk_cache_base_path}")

    def _save_split_disk_cache(self):
        """Save split-specific data to disk cache."""
        cache_data = {
            'ids_in_split': self.ids_in_split,
            'requested_test_ids': getattr(self, 'requested_test_ids', []),
            'test_data': self.test_data,
        }
        split_path = self._disk_cache_split_path(self.split)
        with open(split_path, 'wb') as f:
            pickle.dump(cache_data, f)
        log_info(f"[SAbDabDataset] Saved {self.split} split cache to {split_path}")

    def _load_base_disk_cache(self):
        """Load base data (shared across splits) from disk cache. Returns True if successful."""
        if not os.path.exists(self._disk_cache_base_path):
            return False

        try:
            with open(self._disk_cache_base_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.sabdab_entries = cache_data['sabdab_entries']
            self.db_ids = cache_data['db_ids']
            self.clusters = cache_data.get('clusters')
            self.id_to_cluster = cache_data.get('id_to_cluster')
            self.ref_test_ids = cache_data.get('ref_test_ids', [])
            self.fact_test_ids = cache_data.get('fact_test_ids', [])
            self.test_full_nm = cache_data.get('test_full_nm', [])
            self.db_conn = None

            log_info(f"[SAbDabDataset] Loaded base disk cache: {len(self.sabdab_entries)} entries, {len(self.db_ids)} structures")
            return True
        except Exception as e:
            log_info(f"[SAbDabDataset] Failed to load base disk cache: {e}")
            return False

    def _load_split_disk_cache(self):
        """Load split-specific data from disk cache. Returns True if successful."""
        split_path = self._disk_cache_split_path(self.split)
        if not os.path.exists(split_path):
            return False

        try:
            with open(split_path, 'rb') as f:
                cache_data = pickle.load(f)

            raw_ids_in_split = cache_data['ids_in_split']
            self.ids_in_split = [_canonicalize_rabd_id(data_id) for data_id in raw_ids_in_split]
            cached_requested = [_canonicalize_rabd_id(x) for x in cache_data.get('requested_test_ids', [])]
            if self.split == 'test' and self.requested_test_ids:
                cached_set = set(self.ids_in_split)
                requested_set = set(self.requested_test_ids)
                if cached_requested and set(cached_requested) != requested_set:
                    log_info(
                        f"[SAbDabDataset] Ignoring stale {self.split} split cache: "
                        f"cached requested ids differ from current test_data"
                    )
                    return False
                if not requested_set.issubset(cached_set):
                    missing = sorted(requested_set - cached_set)
                    log_info(
                        f"[SAbDabDataset] Ignoring stale {self.split} split cache: "
                        f"missing requested ids {missing}"
                    )
                    return False
            if self.ids_in_split != raw_ids_in_split:
                replacements = [
                    (old_id, new_id)
                    for old_id, new_id in zip(raw_ids_in_split, self.ids_in_split)
                    if old_id != new_id
                ]
                log_info(f"[SAbDabDataset] Canonicalized legacy split IDs: {replacements}")
            log_info(f"[SAbDabDataset] Loaded {self.split} split cache: {len(self.ids_in_split)} samples")
            return True
        except Exception as e:
            log_info(f"[SAbDabDataset] Failed to load {self.split} split cache: {e}")
            return False

    def _load_sabdab_entries(self):  # 提取all_structure.tsv中的数据信息，保留关键数据，保存在entries_all列表中。每个entry是一个字典
        df = pd.read_csv(self.summary_path, sep='\t')  # (18684, 30)
        entries_all = []
        ref_test_entries = []
        ref_test_indices = []  # 用于保存原始test 的索引

        fact_test_entries = []
        fact_test_indices = []  # 用于保存过滤后的test 的索引

        check_test_info = []
        if '.' in self.test_data: # 是文件
            with open(self.test_data, 'r', encoding='utf-8') as f:
                for line in f:
                    id = _canonicalize_rabd_id(line.strip())
                    if id:
                        self.ref_test_ids.append(id)

        self.test_full_nm = []
        for i, row in tqdm(
                df.iterrows(),
                dynamic_ncols=True,
                desc='Loading entries',
                total=len(df),
        ):
            raw_h, raw_l = nan_to_empty_string(row['Hchain']), nan_to_empty_string(row['Lchain'])
            ag_chains = split_sabdab_delimited_str(nan_to_empty_string(row['antigen_chain']))
            h, l, ag_chains = _canonicalize_rabd_chains(row['pdb'], raw_h, raw_l, ag_chains)

            entry_id = _make_sabdab_entry_id(row['pdb'], h, l, ag_chains)
            raw_entry_id = _make_sabdab_entry_id(row['pdb'], raw_h, raw_l, ag_chains)
            legacy_h, legacy_l = _legacy_case_collapsed_chains(raw_h, raw_l)

            resolution = parse_sabdab_resolution(row['resolution'])
            entry = {  # 将all_structure.tsv中每行数据的重要信息进行保存。信息是说明/描述，不是具体数据
                'id': entry_id,  # 自定义id  原始id_Hchains_Lchains_Antigen
                'pdbcode': row['pdb'],  # 原始id
                'number_scheme': self.data_number_scheme,
                'H_chain': nan_to_none(h),  # Hchain的名称C D...
                'L_chain': nan_to_none(l),  # Lchain的名称E F...
                'ag_chains': ag_chains,
                'ag_type': nan_to_none(row['antigen_type']),
                'ag_name': nan_to_none(row['antigen_name']),
                'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                'resolution': resolution,
                'method': row['method'],
                'scfv': row['scfv'],
            }

            extend_ids = [entry_id] + [f"{row['pdb']}_{h}_{l}_{a}" for a in get_all_antigen_subsets(''.join(ag_chains))]
            for alias_h, alias_l in ((raw_h, raw_l), (legacy_h, legacy_l)):
                alias_entry_id = _make_sabdab_entry_id(row['pdb'], alias_h, alias_l, ag_chains)
                if alias_entry_id != entry_id:
                    extend_ids += [alias_entry_id] + [
                        f"{row['pdb']}_{alias_h}_{alias_l}_{a}"
                        for a in get_all_antigen_subsets(''.join(ag_chains))
                    ]
            extend_ids = list(dict.fromkeys(extend_ids))


            is_test = False
            matched_test_id = None
            if self.ref_test_ids:# 说明是给定的测试集数据
                for eid in extend_ids:  #
                    if eid in self.ref_test_ids:
                        is_test = True
                        matched_test_id = eid
                        self.test_full_nm.append(eid)
                        break  # 找到一个匹配就停止

            else:
                if entry['ag_name'] in TEST_ANTIGENS:
                    is_test = True

            if is_test and matched_test_id is not None:
                entry['id'] = _canonicalize_rabd_id(matched_test_id)

            if is_test:
                ref_test_entries.append(entry)
                ref_test_indices.append(i)

            if entry['ag_type'] in ALLOWED_AG_TYPES:
                if is_test:
                    fact_test_entries.append(entry)
                    fact_test_indices.append(i)
                    self.fact_test_ids.append(entry['id'])
                if entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD:  # 过滤掉不符合条件的数据。根据抗原类型、分辨率
                    entries_all.append(entry)


        log_info(f'===total entry{len(df)}, filter by agtype && resolution {len(entries_all)}')
        assert len(df) >= len(entries_all)

        test_from_path = True if '.' in self.test_data else False
        save_test_df(df, ref_test_indices, self.processed_dir, 'ref_test_data', data_from_path=test_from_path)
        save_test_df(df, fact_test_indices, self.processed_dir, 'fact_test_data', data_from_path=test_from_path)

        if self.split == 'test':
            entries_all = ref_test_entries

        self.sabdab_entries = entries_all  # 列表。只保留了符合要求的复合物信息entry(dict)。如果目标是测试集，则直接使用测试集的数据；否则是初步过滤的数据entry
        assert len(self.ref_test_ids) >= len(
            self.fact_test_ids), f"self.ref_test_ids {len(self.ref_test_ids)}, self.fact_test_ids {len(self.fact_test_ids)}"

        if len(self.ref_test_ids) > len(self.fact_test_ids):
            diff = [x for x in self.ref_test_ids if x not in self.fact_test_ids]
            log_info(diff)  # [1, 3]
            for e_id in diff:
                for e in ref_test_entries:
                    if e_id == e['id']:
                        check_test_info.append({e['id']: {"ag_type": e['ag_type'], "resolution": e['resolution']}})
                        break
            log_info(f"=============diff=============\n{diff}\n=============self.ref_test_ids=========\n{self.ref_test_ids}\n\
            ===========self.fact_test_ids=============\n{self.fact_test_ids}\n===================check_test_info=========\n{check_test_info}")

    def _load_structures(self,
                         reset_strc):  # 解析结构，保留H L上有重原子坐标的AA 真实序列编号、局部序列编号、坐标、标记cdr_flag fr_flag H1-3seqs L1-3seqs
        if not os.path.exists(self._structure_cache_path) or reset_strc:
            if os.path.exists(self._structure_cache_path):
                os.unlink(self._structure_cache_path)
            self._preprocess_structures()

        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f) ## 满足agtype && resolution && 能够解析符合要求的cdr3的对象的pdbid
        self.sabdab_entries = list(  # 根据实际通过解析后过滤的数据，对保留的pdb条目信息进行在一次过滤，使得条目信息和保留的解析数据一致
            filter(
                lambda e: e['id'] in self.db_ids,
                self.sabdab_entries
            )
        )
        log_info(f"=======Total {len(self.db_ids)}=========After filting pdb w/o cdr3: {len(self.sabdab_entries)}")

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')

    @property
    def _disk_cache_base_path(self):
        """Path for base disk cache (shared across splits)."""
        return os.path.join(self.processed_dir, 'dataset_cache.pkl')

    def _disk_cache_split_path(self, split):
        """Path for split-specific disk cache."""
        return os.path.join(self.processed_dir, f'dataset_cache_{split}.pkl')

    def _preprocess_structures(self):
        tasks = []
        cnt_total = len(self.sabdab_entries)  # 由resolution和ag_type过滤后的结果
        cnt_notexist = 0
        notexist_pdb = []
        for entry in self.sabdab_entries:
            pdb_path = os.path.join(self.total_data_dir, '{}.pdb'.format(entry['pdbcode']))
            if not os.path.exists(pdb_path):
                cnt_notexist += 1
                notexist_pdb.append(entry['pdbcode'])
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            tasks.append({  # 搬出实际需要处理的任务对象。因为有些没有pdb文件，所以跳过
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
                'number_scheme': entry['number_scheme']
            })
        log_info(f'==========Parsed total {cnt_total};; not_exit pdb {cnt_notexist}; indeed parsed {len(tasks)}')
        with open('notexist_pdb.txt', 'w') as f:
            for item in notexist_pdb:
                f.write(str(item) + '\n')
        from multiprocessing import Manager
        manager = Manager()
        counters = manager.dict({
            'issue_file': 0,
            'issue_sequnk': 0,
            'issue_heavy & light_miss': 0
        })

        data_list = joblib.Parallel(
            n_jobs=max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_sabdab_structure)(task, counters)  # 对经过过滤最终留下的每个task依次解析
            for task in tqdm(tasks, dynamic_ncols=True,
                             desc='Preprocess (parse strcture [cdr_flag, fr_flag, H1-3 seqs, L1-3 seqs])')
        )
        num_filter = 0
        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []  # 从重链、轻链中正确提取出cdr3的id
        with db_conn.begin(write=True, buffers=True) as txn:  # 写入：id名称+对应解析后的数据
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:  # 说明data_list有None。这些为None的数据是重链轻链同时缺失的复合物对象
                    num_filter += 1  # 重链轻链同时缺失
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'),
                        pickle.dumps(data))  # txn.put(key, value) 存储数据到数据库中，id是key，data是value。LMDB要求key和value必须是二进制字节类型

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)  # 正确提取cdr3的对象的id

        log_info(f"===========Total to parse: {len(data_list)}, PDB_parsed error {num_filter} -> final {len(ids)}.\
         filter counter issues: issue_file {counters['issue_file']}, sequnk {counters['issue_sequnk']}, issue_heavy & light_miss' {counters['issue_heavy & light_miss']}")

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset_strc):  # mmseqs根据相似度计算完cluster后，每个cluster有个name，记录id和cluster的对应关系
        if not os.path.exists(self._cluster_path) or reset_strc:
            self._create_clusters()  # 开启命名seq子程序，得到每个复合物id和cluster的对应关系

        clusters, id_to_cluster = {}, {}  # clusters保留各个claster中对应的复合物id，长度为cluster的不个数；id_to_cluster记录每个id对应的cluster，长度为复合物个数
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):  # mmseq根据重链或者轻链的cdr3序列相似度【如果复合物有被解析的轻链和重链。只用重链；如果没有重连才使用轻链】
        cdr_records = []
        cnt_total = 0
        cnt_heavy = 0
        cnt_light = 0
        log_info(f'===================self.db_ids {len(self.db_ids)}...')
        for id in self.db_ids:  # 所有符合agtype和resolution要求，且能够解析出cdr3的数据（只要总数据集一样，该数据都一样。与测试集无关）
            cnt_total += 1
            structure = self.get_structure(id)
            if structure['heavy']:
                cnt_heavy += 1
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['heavy']['H3_seq']),
                    id=structure['id'],
                    name='',
                    description='',
                ))
            elif structure['light']:
                cnt_light += 1
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['light']['L3_seq']),
                    id=structure['id'],
                    name='',
                    description='',
                ))
        fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
        SeqIO.write(cdr_records, fasta_path,
                    'fasta')  # 聚簇基于的chain保存，直接写成fasta格式(light heavy混合。有heavy的用heavy，有light的用light)
        log_info(f'====================== Totally data {cnt_total}; heavy cdr {cnt_heavy}; light cdr3 {cnt_light} ')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),  # 聚簇对象
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',  # 序列相似度
            '-c', '0.8',  # 两个序列相似序列对于原序列的覆盖度
            '--cov-mode', '1',  # 双向覆盖
        ])
        subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)

    def _load_split(self, split, split_seed):  # 根据mmseq2聚簇后的结果进行数据集分割。目前的数据是完整数据.split不同self.sabdab_entries不同
        assert split in ('train', 'val', 'test')

        if split == 'test':
            self.ids_in_split = self.fact_test_ids
        else:
            test_relevant_clusters = set([
                self.id_to_cluster[id]
                for id in self.fact_test_ids
                if id in self.id_to_cluster
            ])
            log_info(
                f'factual test complex {len(self.fact_test_ids)};;test_clusters {len(test_relevant_clusters)}')
            excluded_by_similarity = [eid for eid in self.db_ids if self.id_to_cluster.get(eid) in test_relevant_clusters]
            log_info(f"Total objects matching test clusters: {len(excluded_by_similarity)}")
            log_info(f"=== Test Related Clusters Statistics ===")
            cluster_stats = []
            for c_id in test_relevant_clusters:
                size = len(self.clusters[c_id])
                cluster_stats.append({'cluster_id': c_id, 'size': size})
                log_info(f"Cluster: {c_id} | Size: {size}")

            total_excluded = sum(len(self.clusters[c_id]) for c_id in test_relevant_clusters)
            log_info(f"Total entries excluded from training due to test overlap: {total_excluded}")
            log_info(self.fact_test_ids)
            ids_train_val = [  # test和train data完全不用和test在一个cluster中的数据
                entry['id']  # 复合物id
                for entry in self.sabdab_entries
                if self.id_to_cluster[entry['id']] not in test_relevant_clusters
            ]

            total_train_val = len(ids_train_val)
            log_info(f"Total train+val samples after excluding test clusters: {total_train_val}")

            available_clusters = set()
            for entry_id in ids_train_val:
                if entry_id in self.id_to_cluster:
                    available_clusters.add(self.id_to_cluster[entry_id])
            available_clusters = sorted(list(available_clusters))
            log_info(f"Available clusters for train/val split: {len(available_clusters)}")

            if self.val_split_mode == 'cluster':
                val_cluster_num = max(1, int(len(available_clusters) * self.val_ratio))

                if val_cluster_num > 1:
                    log_info(f"Using cluster-based split: val_ratio={self.val_ratio}, val_clusters={val_cluster_num}")
                    random.Random(split_seed).shuffle(available_clusters)
                    val_clusters = set(available_clusters[:val_cluster_num])
                    train_clusters = set(available_clusters[val_cluster_num:])

                    ids_val = [eid for eid in ids_train_val if self.id_to_cluster.get(eid) in val_clusters]
                    ids_train = [eid for eid in ids_train_val if self.id_to_cluster.get(eid) in train_clusters]

                    log_info(f"Cluster-based split: {len(ids_val)} val samples from {len(val_clusters)} clusters, "
                          f"{len(ids_train)} train samples from {len(train_clusters)} clusters")

                    if split == 'val':
                        self.ids_in_split = ids_val
                    else:
                        self.ids_in_split = ids_train
                else:
                    log_info(f"WARNING: Only {val_cluster_num} cluster(s) available for validation (need > 1). "
                          f"Falling back to sample-based split.")
                    val_num = max(int(total_train_val * self.val_ratio), 32)
                    random.Random(split_seed).shuffle(ids_train_val)

                    if split == 'val':
                        self.ids_in_split = ids_train_val[:val_num]
                    else:
                        self.ids_in_split = ids_train_val[val_num:]
            else:
                log_info(f"Using sample-based split: val_ratio={self.val_ratio}")
                val_num = max(int(total_train_val * self.val_ratio), 32)
                random.Random(split_seed).shuffle(ids_train_val)

                if split == 'val':
                    self.ids_in_split = ids_train_val[:val_num]
                else:
                    self.ids_in_split = ids_train_val[val_num:]

            log_info(f"Final {split} split: {len(self.ids_in_split)} samples")

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, id):
        self._connect_db()
        with self.db_conn.begin() as txn:
            raw = txn.get(id.encode())
            legacy_id = None
            if raw is None:
                for candidate_id in _legacy_candidates_for_rabd_id(id):
                    raw = txn.get(candidate_id.encode())
                    if raw is not None:
                        legacy_id = candidate_id
                        break
            if raw is None:
                raise KeyError(f"Structure ID {id} not found in LMDB cache")
            data = pickle.loads(raw)
            if legacy_id is not None:
                data = data.copy()
                data['id'] = id
            return data

    def __len__(self):
        return len(self.ids_in_split)

    def _get_cross_complex_epitope(self, data_id, data):
        """
        从同一簇的其他复合物中获取替代表位
        模拟推理时：表位来自其他复合物状态
        """
        if not hasattr(self, 'id_to_cluster') or self.id_to_cluster is None:
            return data  # 没有聚类信息，返回原数据

        cluster_id = self.id_to_cluster.get(data_id)
        if cluster_id is None:
            return data  # 当前数据不在聚类中，返回原数据

        cluster_members = self.clusters.get(cluster_id, [])
        other_members = [m for m in cluster_members if m != data_id]

        if len(other_members) == 0:
            return data  # 簇中只有自己，返回原数据

        import random
        source_id = random.choice(other_members)
        source_data = self.get_structure(source_id)

        if source_data is None or 'antigen' not in source_data:
            return data  # 源数据无效，返回原数据

        new_data = data.copy()

        if 'antigen' not in data or data['antigen'] is None:
            return data

        new_data['antigen'] = source_data['antigen']
        new_data['antigen_seqmap'] = source_data.get('antigen_seqmap', {})


        return new_data

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)

        if self.transform is not None:
            data = self.transform(data)

        if self.split == 'val' and self.val_epitope_source == 'cross_complex':
            data = self._get_cross_complex_epitope(id, data)

        return data  # 最后返回的都是patch后的数据


@register_dataset('sabdab')
def get_sabdab_dataset(cfg, transform):
    return SAbDabDataset(
        summary_path=cfg.summary_path,
        total_data_dir=cfg.total_data_dir,
        processed_dir=cfg.processed_dir,
        split=cfg.split,
        split_seed=cfg.get('split_seed', 2022),
        transform=transform,
        test_data=cfg.test_data,
        reset_strc=cfg.reset_strc,
        data_number_scheme=cfg.data_number_scheme,
        val_split_mode=cfg.get('val_split_mode', 'sample'),
        val_ratio=cfg.get('val_ratio', 0.1)
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed')
    parser.add_argument('--reset_strc', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset_strc:
        sure = input('Sure to reset_strc? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SAbDabDataset(
        processed_dir=args.processed_dir,
        split=args.split,
        reset_strc=args.reset_strc
    )
    log_info(dataset[0])
    log_info(f"len(dataset) {len(dataset)}, len(dataset.clusters) {len(dataset.clusters)}")


