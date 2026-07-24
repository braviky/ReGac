import torch


_NOISE_MODE_DEBUG_COUNTER = {}


def has_noise_mode_token(noise_mode, token):
    if isinstance(noise_mode, str):
        return token in noise_mode
    if isinstance(noise_mode, (list, tuple)):
        return any(isinstance(mode, str) and (token in mode) for mode in noise_mode)
    return False


def is_cdr_noise_mode(noise_mode):
    return has_noise_mode_token(noise_mode, 'cdr')


def is_epitope_noise_mode(noise_mode):
    return has_noise_mode_token(noise_mode, 'epitope')


def _clone_tensor(value):
    if torch.is_tensor(value):
        return value.clone()
    return value


def _get_raw_value(data, raw_key, key):
    if raw_key in data:
        return data[raw_key]
    return data.get(key, None)


def _ensure_raw_masks(data):
    for key in ('antigen_mask', 'antigen_soft_mask', 'antigen_rigid_mask', 'structure_generate_flag'):
        raw_key = f'{key}_raw'
        if key in data and raw_key not in data:
            data[raw_key] = _clone_tensor(data[key])


def _maybe_debug_log(message):
    try:
        from diffab_251166.modules.common.bfactor_utils import DEBUG_WEIGHTS, debug_log
        if DEBUG_WEIGHTS:
            debug_log(message)
            return
    except Exception:
        pass
    print(message)


def _residue_count(mask):
    if mask is None:
        return 0
    return int(mask.to(torch.bool).sum().item())


def _assert_noise_mode_invariants(
    noise_mode,
    generate_flag,
    antigen_mask,
    raw_soft_mask,
    raw_rigid_mask,
    effective_soft_mask,
    effective_rigid_mask,
    effective_structure_flag,
    context='',
):
    assert generate_flag.shape == antigen_mask.shape == raw_soft_mask.shape == raw_rigid_mask.shape, (
        f"{context}: mask shape mismatch for noise_mode={noise_mode}"
    )
    assert effective_soft_mask.shape == effective_rigid_mask.shape == effective_structure_flag.shape == generate_flag.shape, (
        f"{context}: effective mask shape mismatch for noise_mode={noise_mode}"
    )
    assert not torch.logical_and(generate_flag, antigen_mask).any(), (
        f"{context}: generate_flag overlaps antigen residues for noise_mode={noise_mode}"
    )
    assert not torch.logical_and(raw_soft_mask, raw_rigid_mask).any(), (
        f"{context}: raw antigen_soft_mask overlaps antigen_rigid_mask for noise_mode={noise_mode}"
    )
    assert torch.equal(torch.logical_or(raw_soft_mask, raw_rigid_mask), antigen_mask), (
        f"{context}: raw antigen_soft_mask/antigen_rigid_mask must partition antigen_mask for noise_mode={noise_mode}"
    )

    if is_epitope_noise_mode(noise_mode):
        assert effective_soft_mask.any(), (
            f"{context}: epitope noise mode requires non-empty effective antigen_soft_mask for noise_mode={noise_mode}"
        )
        assert not torch.logical_and(effective_soft_mask, effective_rigid_mask).any(), (
            f"{context}: effective antigen_soft_mask overlaps antigen_rigid_mask for noise_mode={noise_mode}"
        )
        assert torch.equal(torch.logical_or(effective_soft_mask, effective_rigid_mask), antigen_mask), (
            f"{context}: effective antigen_soft_mask/antigen_rigid_mask must partition antigen_mask for noise_mode={noise_mode}"
        )
        assert torch.equal(torch.logical_and(effective_structure_flag, effective_soft_mask), effective_soft_mask), (
            f"{context}: effective structure_generate_flag must include the full epitope mask for noise_mode={noise_mode}"
        )
    else:
        assert not effective_soft_mask.any(), (
            f"{context}: cdr_only-like mode must zero effective antigen_soft_mask for noise_mode={noise_mode}"
        )
        assert torch.equal(effective_rigid_mask, antigen_mask), (
            f"{context}: cdr_only-like mode must treat all antigen residues as rigid for noise_mode={noise_mode}"
        )
        assert not torch.logical_and(effective_structure_flag, antigen_mask).any(), (
            f"{context}: cdr_only-like mode must exclude antigen residues from structure_generate_flag for noise_mode={noise_mode}"
        )


def _maybe_log_noise_mode(context, noise_mode, use_epitope, generate_flag, antigen_mask, effective_soft_mask, effective_rigid_mask, effective_structure_flag):
    key = (str(context), str(noise_mode), bool(use_epitope))
    count = _NOISE_MODE_DEBUG_COUNTER.get(key, 0) + 1
    _NOISE_MODE_DEBUG_COUNTER[key] = count
    if count > 2 and count % 1000 != 0:
        return
    message = (
        f"[NoiseMode/{context or 'unknown'}] apply#{count} "
        f"mode={noise_mode or 'cdr_only'} use_epitope={use_epitope} "
        f"cdr={_residue_count(generate_flag)} antigen={_residue_count(antigen_mask)} "
        f"soft={_residue_count(effective_soft_mask)} rigid={_residue_count(effective_rigid_mask)} "
        f"struct={_residue_count(effective_structure_flag)} "
        f"struct_antigen={_residue_count(torch.logical_and(effective_structure_flag, antigen_mask))}"
    )
    _maybe_debug_log(message)


def resolve_noise_mode_masks(data, noise_mode, assert_epitope_mask=False, context=''):
    generate_flag = data['generate_flag'].to(torch.bool)
    antigen_mask_src = _get_raw_value(data, 'antigen_mask_raw', 'antigen_mask')
    soft_mask_src = _get_raw_value(data, 'antigen_soft_mask_raw', 'antigen_soft_mask')
    rigid_mask_src = _get_raw_value(data, 'antigen_rigid_mask_raw', 'antigen_rigid_mask')
    structure_flag_src = _get_raw_value(data, 'structure_generate_flag_raw', 'structure_generate_flag')

    antigen_mask = antigen_mask_src.to(torch.bool) if antigen_mask_src is not None else torch.zeros_like(generate_flag)
    raw_soft_mask = soft_mask_src.to(torch.bool) if soft_mask_src is not None else torch.zeros_like(antigen_mask)
    raw_rigid_mask = rigid_mask_src.to(torch.bool) if rigid_mask_src is not None else torch.logical_and(antigen_mask, ~raw_soft_mask)
    raw_structure_flag = structure_flag_src.to(torch.bool) if structure_flag_src is not None else generate_flag.clone()

    antibody_structure_flag = torch.logical_and(raw_structure_flag, ~raw_soft_mask)
    use_epitope = is_epitope_noise_mode(noise_mode)

    if use_epitope:
        effective_soft_mask = raw_soft_mask
        effective_rigid_mask = raw_rigid_mask
        if assert_epitope_mask and not effective_soft_mask.any():
            mode_name = noise_mode if noise_mode is not None else 'cdr_epitope'
            context_prefix = f'{context}: ' if context else ''
            raise AssertionError(
                f"{context_prefix}noise_mode={mode_name} requires antigen_soft_mask to contain epitope residues."
            )
        effective_structure_flag = torch.logical_or(antibody_structure_flag, effective_soft_mask)
    else:
        effective_soft_mask = torch.zeros_like(raw_soft_mask)
        effective_rigid_mask = antigen_mask
        effective_structure_flag = antibody_structure_flag

    _assert_noise_mode_invariants(
        noise_mode=noise_mode or 'cdr_only',
        generate_flag=generate_flag,
        antigen_mask=antigen_mask,
        raw_soft_mask=raw_soft_mask,
        raw_rigid_mask=raw_rigid_mask,
        effective_soft_mask=effective_soft_mask,
        effective_rigid_mask=effective_rigid_mask,
        effective_structure_flag=effective_structure_flag,
        context=context,
    )

    soft_dtype = soft_mask_src.dtype if soft_mask_src is not None else torch.bool
    rigid_dtype = rigid_mask_src.dtype if rigid_mask_src is not None else (antigen_mask_src.dtype if antigen_mask_src is not None else torch.bool)
    structure_dtype = structure_flag_src.dtype if structure_flag_src is not None else data['generate_flag'].dtype

    return {
        'noise_mode': noise_mode or 'cdr_only',
        'use_epitope': use_epitope,
        'antigen_soft_mask': effective_soft_mask.to(soft_dtype),
        'antigen_rigid_mask': effective_rigid_mask.to(rigid_dtype),
        'structure_generate_flag': effective_structure_flag.to(structure_dtype),
    }


def apply_noise_mode_masks(data, noise_mode, assert_epitope_mask=False, context=''):
    _ensure_raw_masks(data)
    resolved = resolve_noise_mode_masks(
        data,
        noise_mode=noise_mode,
        assert_epitope_mask=assert_epitope_mask,
        context=context,
    )
    data['antigen_soft_mask'] = resolved['antigen_soft_mask']
    data['antigen_rigid_mask'] = resolved['antigen_rigid_mask']
    data['structure_generate_flag'] = resolved['structure_generate_flag']
    data['effective_noise_mode'] = resolved['noise_mode']
    antigen_mask = _get_raw_value(data, 'antigen_mask_raw', 'antigen_mask')
    if antigen_mask is None:
        antigen_mask = torch.zeros_like(data['generate_flag'].to(torch.bool))
    _maybe_log_noise_mode(
        context=context,
        noise_mode=resolved['noise_mode'],
        use_epitope=resolved['use_epitope'],
        generate_flag=data['generate_flag'].to(torch.bool),
        antigen_mask=antigen_mask.to(torch.bool),
        effective_soft_mask=data['antigen_soft_mask'].to(torch.bool),
        effective_rigid_mask=data['antigen_rigid_mask'].to(torch.bool),
        effective_structure_flag=data['structure_generate_flag'].to(torch.bool),
    )
    return data
