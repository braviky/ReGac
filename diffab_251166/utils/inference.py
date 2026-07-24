import torch
from .noise_mode import apply_noise_mode_masks, is_epitope_noise_mode, resolve_noise_mode_masks
from .protein import constants


def find_cdrs(structure):
    cdrs = []
    if structure['heavy'] is not None:
        flag = structure['heavy']['cdr_flag']
        if int(constants.CDR.H1) in flag:
            cdrs.append('H_CDR1')
        if int(constants.CDR.H2) in flag:
            cdrs.append('H_CDR2')
        if int(constants.CDR.H3) in flag:
            cdrs.append('H_CDR3')

    if structure['light'] is not None:
        flag = structure['light']['cdr_flag']
        if int(constants.CDR.L1) in flag:
            cdrs.append('L_CDR1')
        if int(constants.CDR.L2) in flag:
            cdrs.append('L_CDR2')
        if int(constants.CDR.L3) in flag:
            cdrs.append('L_CDR3')
    
    return cdrs  # 确定生成哪个cdr区域


def get_residue_first_last(data):
    loop_flag = data['generate_flag']
    loop_idx = torch.arange(loop_flag.size(0))[loop_flag]
    idx_first, idx_last = loop_idx.min().item(), loop_idx.max().item()
    residue_first = (data['chain_id'][idx_first], data['resseq'][idx_first].item(), data['icode'][idx_first])
    residue_last = (data['chain_id'][idx_last], data['resseq'][idx_last].item(), data['icode'][idx_last])
    return residue_first, residue_last

def get_residue_first_last_multi(data):
    generate_flag = data['generate_flag'] # [N]
    cdr_flag = data['cdr_flag']   # [N]，范围 0~6，对应 [None, H1, H2, H3, L1, L2, L3]

    idx_all = torch.arange(len(generate_flag))
    idx_gen = idx_all[generate_flag.to(torch.bool)]

    if idx_gen.numel() == 0:
        return [None] * 6  # 没有生成区域

    split_indices = torch.where(torch.diff(idx_gen) > 1)[0] + 1
    splits = torch.tensor_split(idx_gen, split_indices)
    print(f'splits {splits}')

    cdr_ranges = [None] * 6

    for seg in splits:
        start_idx = seg[0].item()
        end_idx = seg[-1].item()

        seg_cdr_flags = cdr_flag[seg]
        seg_cdr_nonzero = seg_cdr_flags[seg_cdr_flags != 0]
        print(f'seg_cdr_flags {seg_cdr_flags};;seg_cdr_nonzero {seg_cdr_nonzero}')
        if len(seg_cdr_nonzero) == 0:
            continue  # 不属于任何 CDR，跳过

        cdr_id = int((torch.mode(seg_cdr_nonzero).values.item())) # 1~6，对应H1~L3
        idx_in_list = cdr_id - 1  # 索引从0开始
        print(f'cdr_id {cdr_id}, idx_in_list {idx_in_list}')
        if cdr_ranges[idx_in_list] is not None:
            continue

        residue_first = (
            data['chain_id'][start_idx],
            data['resseq'][start_idx].item(),
            data['icode'][start_idx]
        )
        residue_last = (
            data['chain_id'][end_idx],
            data['resseq'][end_idx].item(),
            data['icode'][end_idx]
        )

        cdr_ranges[idx_in_list] = [residue_first, residue_last]

    return cdr_ranges




def resolve_inference_noise_mode(config=None, cli_noise_mode=None, default='cdr_only'):
    if cli_noise_mode:
        return cli_noise_mode

    sampling = None
    if config is not None:
        if isinstance(config, dict):
            sampling = config.get('sampling', None)
        else:
            sampling = getattr(config, 'sampling', None)

    if sampling is not None:
        if isinstance(sampling, dict):
            noise_mode = sampling.get('noise_mode', None)
        else:
            noise_mode = getattr(sampling, 'noise_mode', None)
        if noise_mode:
            return noise_mode

    return default


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _extract_training_noise_sampling(config):
    model_cfg = _cfg_get(config, 'model')
    modal_training = _cfg_get(model_cfg, 'modal_training')
    noise_sampling = _cfg_get(modal_training, 'noise_sampling', {}) if modal_training is not None else {}
    return dict(noise_sampling) if noise_sampling is not None else {}


def _extract_training_noise_mode(config, default='cdr_only'):
    noise_sampling = _extract_training_noise_sampling(config)
    train_noise_mode = noise_sampling.get('noise_mode', default)
    return train_noise_mode or default


def extract_inference_noise_sampling_base(config):
    base = _extract_training_noise_sampling(config)
    sampling = _cfg_get(config, 'sampling')
    for key in (
        'offset',
        'power',
        'epitope_time_strategy',
        'epitope_time_min_ratio',
        'epitope_time_max_ratio',
        'epitope_time_power',
        'epitope_time_min_gap',
        'induced_fit_max_t',
        'induced_fit_unfreeze_t_max',
    ):
        value = _cfg_get(sampling, key, None)
        if value is not None:
            base[key] = value
    return base


def resolve_effective_inference_noise_mode(config, requested_noise_mode, default='cdr_only'):
    requested_mode = requested_noise_mode or default
    train_noise_mode = _extract_training_noise_mode(config, default=default)
    if is_epitope_noise_mode(requested_mode) and not is_epitope_noise_mode(train_noise_mode):
        return 'cdr_only'
    return requested_mode


def build_inference_mode_plan(config, requested_modes):
    plan = []
    effective_to_entry = {}
    for requested_mode in requested_modes:
        effective_mode = resolve_effective_inference_noise_mode(config, requested_mode)
        entry = effective_to_entry.get(effective_mode)
        if entry is None:
            entry = {
                'requested_modes': [requested_mode],
                'effective_mode': effective_mode,
            }
            effective_to_entry[effective_mode] = entry
            plan.append(entry)
        else:
            entry['requested_modes'].append(requested_mode)
    return plan


def build_inference_modal_training(
    noise_mode,
    source_noise_sampling=None,
    offset=40,
    power=1.4,
    epitope_time_strategy='induced_fit_late',
    epitope_time_min_ratio=0.35,
    epitope_time_max_ratio=0.80,
    epitope_time_power=1.5,
    epitope_time_min_gap=1,
    induced_fit_max_t=20,
    induced_fit_unfreeze_t_max=20,
):
    noise_sampling = dict(source_noise_sampling) if source_noise_sampling is not None else {}
    effective_noise_mode = noise_mode or noise_sampling.get('noise_mode', 'cdr_only')
    noise_sampling['noise_mode'] = effective_noise_mode
    noise_sampling.setdefault('offset', offset)
    noise_sampling.setdefault('power', power)

    if is_epitope_noise_mode(effective_noise_mode):
        noise_sampling.setdefault('epitope_time_strategy', epitope_time_strategy)
        noise_sampling.setdefault('epitope_time_min_ratio', epitope_time_min_ratio)
        noise_sampling.setdefault('epitope_time_max_ratio', epitope_time_max_ratio)
        noise_sampling.setdefault('epitope_time_power', epitope_time_power)
        noise_sampling.setdefault('epitope_time_min_gap', epitope_time_min_gap)
        noise_sampling.setdefault('induced_fit_max_t', induced_fit_max_t)
        noise_sampling.setdefault('induced_fit_unfreeze_t_max', induced_fit_unfreeze_t_max)

    return {
        'noise_sampling': noise_sampling
    }


def get_effective_structure_generate_flag(data, noise_mode):
    resolved = resolve_noise_mode_masks(
        data,
        noise_mode=noise_mode,
        assert_epitope_mask=is_epitope_noise_mode(noise_mode),
        context='inference',
    )
    return resolved['structure_generate_flag'].to(torch.bool)


def apply_inference_noise_mode(data, noise_mode):
    apply_noise_mode_masks(
        data,
        noise_mode=noise_mode,
        assert_epitope_mask=is_epitope_noise_mode(noise_mode),
        context='inference',
    )
    data['inference_noise_mode'] = noise_mode or 'cdr_only'
    return data


class RemoveNative(object):  # 剔除序列。就是把目标区域（generate_flag）的AA替换为AA.UNK; 提出结果，就额是把对应AA的15给原子坐标替换为高斯噪声，并*10扩展成均值0 方差为10的噪声

    def __init__(self, remove_structure, remove_sequence):
        super().__init__()
        self.remove_structure = remove_structure
        self.remove_sequence = remove_sequence

    def __call__(self, data):
        generate_flag = data['generate_flag'].clone().to(torch.bool)

        if self.remove_sequence: # 剔除序列。就是把目标区域（generate_flag）的AA替换为AA.UNK;
            data['aa'] = torch.where(
                generate_flag,
                torch.full_like(data['aa'], fill_value=int(constants.AA.UNK)),    # Is loop
                data['aa']
            )

        if self.remove_structure:
            pass

        return data
