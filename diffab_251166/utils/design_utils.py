import fcntl
import os

from diffab_251166.utils.transforms import *
from diffab_251166.utils.inference import *
from diffab_251166.utils.protein.constants import Fragment, resindex_to_ressymb
from diffab_251166.utils.protein.writers import save_pdb


_DEFAULT_MASK_REGION_NAME = 'HCDR1_HCDR2_HCDR3_LCDR1_LCDR2_LCDR3'


def _scalar_to_int(value):
    if hasattr(value, 'item'):
        return int(value.item())
    return int(value)


def _normalize_cdr_label(cdr_name):
    if cdr_name is None:
        return None
    return str(cdr_name).replace('_', '')


def _resolve_mask_region_name(variant):
    if variant.get('cdrs'):
        return '_'.join(_normalize_cdr_label(cdr) for cdr in variant['cdrs'])
    if variant.get('cdr'):
        return _normalize_cdr_label(variant['cdr'])
    return _DEFAULT_MASK_REGION_NAME


def _write_fasta(path, seq_map):
    with open(path, 'w') as handle:
        for label in ('H', 'L', 'A'):
            handle.write(f'>{label}\n{seq_map.get(label, "")}\n')


def _append_item_id_once(item_ids_path, structure_id):
    os.makedirs(os.path.dirname(item_ids_path), exist_ok=True)
    with open(item_ids_path, 'a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = {line.strip() for line in handle if line.strip()}
        if structure_id not in existing:
            handle.write(f'{structure_id}\n')
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def export_eval_records(run_dir, structure_id, data_native, variant, case_dir=None):
    if case_dir is not None:
        os.makedirs(case_dir, exist_ok=True)
        save_pdb(data_native, os.path.join(case_dir, 'reference.pdb'))

    os.makedirs(run_dir, exist_ok=True)
    item_ids_path = os.path.join(run_dir, 'item_ids.txt')
    pdb_records_dir = os.path.join(run_dir, 'pdb_records')
    fasta_records_dir = os.path.join(run_dir, 'fasta_records')
    mask_region = _resolve_mask_region_name(variant)
    fasta_mask_dir = os.path.join(run_dir, 'fasta_records_mask', mask_region)

    os.makedirs(pdb_records_dir, exist_ok=True)
    os.makedirs(fasta_records_dir, exist_ok=True)
    os.makedirs(fasta_mask_dir, exist_ok=True)

    aa_native = data_native['aa']
    aa_variant = variant['data']['aa']
    generate_flag = variant['data']['generate_flag']
    fragment_type = data_native['fragment_type']

    if len(aa_native) != len(aa_variant) or len(aa_native) != len(generate_flag):
        raise ValueError(
            f'Length mismatch for {structure_id}: '
            f'native={len(aa_native)}, variant={len(aa_variant)}, generate_flag={len(generate_flag)}'
        )

    seq_map = {'H': [], 'L': [], 'A': []}
    mask_map = {'H': [], 'L': [], 'A': []}
    fragment_to_label = {
        int(Fragment.Heavy): 'H',
        int(Fragment.Light): 'L',
        int(Fragment.Antigen): 'A',
    }

    for idx in range(len(aa_native)):
        frag_label = fragment_to_label.get(_scalar_to_int(fragment_type[idx]))
        if frag_label is None:
            continue
        aa_symbol = resindex_to_ressymb[_scalar_to_int(aa_native[idx])]
        seq_map[frag_label].append(aa_symbol)
        is_generated = bool(_scalar_to_int(generate_flag[idx]))
        mask_map[frag_label].append('X' if is_generated else aa_symbol)

    seq_map = {key: ''.join(value) for key, value in seq_map.items()}
    mask_map = {key: ''.join(value) for key, value in mask_map.items()}

    save_pdb(data_native, os.path.join(pdb_records_dir, f'{structure_id}.pdb'))
    _write_fasta(os.path.join(fasta_records_dir, f'{structure_id}.fasta'), seq_map)
    _write_fasta(os.path.join(fasta_mask_dir, f'{structure_id}.fasta'), mask_map)
    _append_item_id_once(item_ids_path, structure_id)

def create_data_variants(config, structure_factory, heavy_id, light_id):  # 创建生成对象，核心是添加generate_flag和anchor_flag。 # 单cdr生成可能有多个【配置文件中每个cdr都是一个独立的生成对象，因为是单cdr生成】，但是多cdr生成只有一个【不同的配置文件只是决定了该数据同时生成的cdr区域不同】
    structure = structure_factory()  # 提取parsed信息
    structure_id = structure['id']

    data_variants = []  # single_cdr是当cdr分别生成，因此会对应不同的变量，他们的generate_flag和anchor_flag不同
    if config.mode == 'single_cdr': # 单个cdr生成，具体分别生成多少个cdr取决于config.sampling.cdrs, which指定了同时生成的cdr区域
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))  # 计算两个集合的交集：find_cdrs 根据['cdr_flag']计算当前数据有几个cdr；config.sampling.cdrs配置文件中有多少个
        for cdr_name in cdrs:  # 每个cdr的生成都是一个变量对象
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append({
                'data': data_var,
                'name': f'{structure_id}-{cdr_name}',
                'tag': f'{cdr_name}',
                'cdr': cdr_name,
                'heavy_id': heavy_id,
                'light_id': light_id,
                'residue_first': residue_first,  # reference
                'residue_last': residue_last,
                'residue_range_list': [[residue_first, residue_last]],
            })
    elif config.mode == 'multiple_cdrs':  # 6个cdr同时生成。【具体多少个取决于config.sampling.cdrs, which指定了同时生成的cdr区域】
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))  # 计算两个集合的交集：find_cdrs 根据['cdr_flag']计算当前数据有几个cdr；config.sampling.cdrs配置文件中有多少个
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())  # 'chain_id', 'icode', 'chain_nb', 'resseq', 'res_nb', 'aa', 'pos_heavyatom', 'mask_heavyatom', 'generate_flag', 'cdr_flag', 'anchor_flag', 'fragment_type']
        residue_range_list = get_residue_first_last_multi(data_var)
        data_variants.append({  # 对指定的多个cdr进行mask，即标注每个生成区域的generate_flag和anchor_flag
            'data': data_var,
            'name': f'{structure_id}-MultipleCDRs',
            'tag': 'MultipleCDRs',
            'cdrs': cdrs,
            'heavy_id': heavy_id,
            'light_id': light_id,
            'residue_first': None,
            'residue_last': None,
            'residue_range_list': residue_range_list
        })
    elif config.mode == 'full': # 生成全抗体
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
            'residue_range_list': None
        })
    elif config.mode == 'abopt':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)  # 生成区域里第一个AA（chain_id, resseq, icode）；生成区域里最后一个AA（chain_id, resseq, icode）
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                    'residue_range_list': [[residue_first, residue_last]]
                })
    elif config.mode in ('abopt_multicdrs', 'abopt_multiple_cdrs', 'abopt_full_cdrs'):
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        if not cdrs:
            raise ValueError(f'No CDRs selected for full-CDR optimization: {structure_id}')
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        residue_range_list = get_residue_first_last_multi(data_var)
        for opt_step in config.sampling.optimize_steps:
            data_variants.append({
                'data': data_var,
                'name': f'{structure_id}-MultipleCDRs-O{opt_step}',
                'tag': f'MultipleCDRs-O{opt_step}',
                'cdrs': cdrs,
                'heavy_id': heavy_id,
                'light_id': light_id,
                'opt_step': opt_step,
                'residue_first': None,
                'residue_last': None,
                'residue_range_list': residue_range_list
            })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')
    return data_variants  # 单cdr生成可能有多个【配置文件中每个cdr都是一个独立的生成对象，因为是单cdr生成】，但是多cdr生成只有一个【不同的配置文件只是决定了该数据同时生成的cdr区域不同】
