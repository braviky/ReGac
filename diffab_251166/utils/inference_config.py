import copy

import yaml


DEFAULT_CDRS = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def to_plain_config(value):
    if isinstance(value, dict):
        return {k: to_plain_config(v) for k, v in value.items()}
    if hasattr(value, 'items'):
        return {k: to_plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_config(v) for v in value]
    return copy.deepcopy(value)


def get_patch_around_anchor_kwargs(config):
    dataset_cfg = cfg_get(config, 'dataset')
    for split_name in ('test', 'val', 'train'):
        split_cfg = cfg_get(dataset_cfg, split_name)
        transforms = cfg_get(split_cfg, 'transform', []) or []
        for transform_cfg in transforms:
            if cfg_get(transform_cfg, 'type') != 'patch_around_anchor':
                continue
            kwargs = to_plain_config(transform_cfg)
            kwargs.pop('type', None)
            return kwargs
    return {}


def resolve_modal_noise_sampling_config(config, noise_mode=None):
    model_cfg = cfg_get(config, 'model')
    dataset_cfg = cfg_get(config, 'dataset')
    modal_candidates = [
        cfg_get(model_cfg, 'modal_training'),
        cfg_get(cfg_get(dataset_cfg, 'val'), 'modal_training'),
        cfg_get(cfg_get(dataset_cfg, 'test'), 'modal_training'),
        cfg_get(cfg_get(dataset_cfg, 'train'), 'modal_training'),
    ]
    for modal_cfg in modal_candidates:
        noise_sampling = to_plain_config(cfg_get(modal_cfg, 'noise_sampling', None))
        if noise_sampling:
            noise_sampling['noise_mode'] = noise_mode or noise_sampling.get('noise_mode', 'cdr_only')
            return noise_sampling
    return {
        'noise_mode': noise_mode or 'cdr_only',
        'offset': 40,
        'power': 1.4,
    }


def build_inference_model_config(config, checkpoint_path, noise_mode, noise_sampling_config=None):
    model_cfg = to_plain_config(cfg_get(config, 'model', {}))
    if not model_cfg:
        raise ValueError('config.model is required to build inference config')

    model_cfg['checkpoint'] = checkpoint_path

    modal_training_cfg = to_plain_config(cfg_get(model_cfg, 'modal_training', {})) or {}
    resolved_noise_sampling = to_plain_config(noise_sampling_config) if noise_sampling_config else None
    if not resolved_noise_sampling:
        resolved_noise_sampling = resolve_modal_noise_sampling_config(config, noise_mode)
    resolved_noise_sampling['noise_mode'] = noise_mode or resolved_noise_sampling.get('noise_mode', 'cdr_only')

    modal_training_cfg['enabled'] = bool(modal_training_cfg.get('enabled', True))
    modal_training_cfg['noise_sampling'] = resolved_noise_sampling
    model_cfg['modal_training'] = modal_training_cfg
    return model_cfg


def build_inference_config_dict(
    config,
    checkpoint_path,
    noise_mode,
    sample_names,
    num_gen=2,
    patch_kwargs=None,
    noise_sampling_config=None,
):
    patch_kwargs = patch_kwargs or get_patch_around_anchor_kwargs(config)
    antigen_patch_mode = patch_kwargs.get('antigen_patch_mode', 'soft_rigid')

    dataset_cfg = (
        to_plain_config(cfg_get(cfg_get(config, 'dataset'), 'test'))
        or to_plain_config(cfg_get(cfg_get(config, 'dataset'), 'val'))
        or to_plain_config(cfg_get(cfg_get(config, 'dataset'), 'train'))
        or {}
    )
    dataset_cfg['split'] = 'test'
    dataset_cfg['transform'] = [
        {'type': 'patch_around_anchor', 'antigen_patch_mode': antigen_patch_mode}
    ]

    sampling_cfg = to_plain_config(cfg_get(config, 'sampling', {})) or {}
    seed = sampling_cfg.get('seed', cfg_get(cfg_get(config, 'train'), 'seed', 2022))
    cdrs = sampling_cfg.get('cdrs', DEFAULT_CDRS)
    return {
        'mode': cfg_get(config, 'mode', 'multiple_cdrs'),
        'model': build_inference_model_config(
            config,
            checkpoint_path=checkpoint_path,
            noise_mode=noise_mode,
            noise_sampling_config=noise_sampling_config,
        ),
        'sampling': {
            'seed': seed,
            'sample_structure': sampling_cfg.get('sample_structure', True),
            'sample_sequence': sampling_cfg.get('sample_sequence', True),
            'cdrs': cdrs,
            'num_samples': num_gen,
            'noise_mode': noise_mode,
            'sample_names': sample_names,
        },
        'dataset': {
            'test': dataset_cfg,
        },
    }


def write_inference_config(config_dict, output_dir, noise_mode):
    config_path = f'{output_dir}/inference_config_{noise_mode}.yml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
    return config_path
