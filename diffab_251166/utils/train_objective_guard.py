from diffab_251166.utils.noise_mode import is_epitope_noise_mode


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def validate_training_objective_alignment(config):
    model_cfg = _cfg_get(config, 'model')
    modal_training = _cfg_get(model_cfg, 'modal_training')
    noise_sampling = _cfg_get(modal_training, 'noise_sampling', {}) if modal_training is not None else {}
    train_noise_mode = _cfg_get(noise_sampling, 'noise_mode', 'cdr_only')
    if not is_epitope_noise_mode(train_noise_mode):
        return

    train_cfg = _cfg_get(config, 'train')
    loss_weights = _cfg_get(train_cfg, 'loss_weights', {}) or {}
    epitope_keys = ('epitope_rot', 'epitope_pos', 'epitope_bone', 'epitope_omega')
    positive = []
    for key in epitope_keys:
        value = _cfg_get(loss_weights, key, 0.0)
        try:
            value = float(value)
        except Exception:
            value = 0.0
        if value > 0.0:
            positive.append((key, value))

    if positive:
        return

    raise ValueError(
        'Training objective mismatch: noise_mode requests epitope denoising '
        f'({train_noise_mode}) but all epitope loss weights are zero. '
        'Use a dedicated cdr_epitope config or set at least one of '
        'epitope_rot / epitope_pos / epitope_bone / epitope_omega > 0.'
    )
