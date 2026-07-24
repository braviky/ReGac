def unwrap_checkpoint_state_dict(checkpoint):
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ('module', 'model', 'state_dict'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        cleaned[clean_key] = value
    return cleaned


def _split_allowed(keys, allowed_prefixes, allowed_fragments=()):
    allowed = []
    blocked = []
    prefixes = tuple(allowed_prefixes or ())
    fragments = tuple(allowed_fragments or ())
    for key in keys:
        if (prefixes and any(key.startswith(prefix) for prefix in prefixes)) or (fragments and any(fragment in key for fragment in fragments)):
            allowed.append(key)
        else:
            blocked.append(key)
    return allowed, blocked


def load_state_dict_or_raise(
    model,
    state_dict,
    *,
    allowed_missing_prefixes=(),
    allowed_unexpected_prefixes=(),
    context='checkpoint',
):
    incompatible = model.load_state_dict(state_dict, strict=False)
    default_allowed_missing_fragments = (
        '.graph_time_edge_weight',
        '.graph_time_edge_offset',
    )
    missing_allowed, missing_blocked = _split_allowed(
        incompatible.missing_keys,
        allowed_missing_prefixes,
        default_allowed_missing_fragments,
    )
    unexpected_allowed, unexpected_blocked = _split_allowed(incompatible.unexpected_keys, allowed_unexpected_prefixes)

    if missing_blocked or unexpected_blocked:
        details = [
            f'{context} state_dict mismatch:',
            f'  blocked missing keys ({len(missing_blocked)}): {missing_blocked}',
            f'  blocked unexpected keys ({len(unexpected_blocked)}): {unexpected_blocked}',
        ]
        if missing_allowed:
            details.append(f'  allowed missing keys ({len(missing_allowed)}): {missing_allowed}')
        if unexpected_allowed:
            details.append(f'  allowed unexpected keys ({len(unexpected_allowed)}): {unexpected_allowed}')
        raise RuntimeError('\n'.join(details))

    return {
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'allowed_missing_keys': missing_allowed,
        'allowed_unexpected_keys': unexpected_allowed,
    }
