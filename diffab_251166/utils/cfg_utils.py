import torch


def mask_conditioning_features(res_feat, pair_feat, keep_mask):
    keep_mask = keep_mask.to(dtype=torch.bool)

    res_feat_masked = None
    if res_feat is not None:
        res_feat_masked = res_feat * keep_mask.unsqueeze(-1).to(res_feat.dtype)

    pair_feat_masked = None
    if pair_feat is not None:
        pair_keep_mask = (keep_mask.unsqueeze(1) & keep_mask.unsqueeze(2)).unsqueeze(-1)
        pair_feat_masked = pair_feat * pair_keep_mask.to(pair_feat.dtype)

    return res_feat_masked, pair_feat_masked


def build_cfg_unconditional_inputs(res_feat, pair_feat, region_type, epitope_region_ids):
    mask_epitope = region_type == epitope_region_ids[0]
    for region_id in epitope_region_ids[1:]:
        mask_epitope = mask_epitope | (region_type == region_id)

    keep_mask = ~mask_epitope
    res_feat_uncond, pair_feat_uncond = mask_conditioning_features(res_feat, pair_feat, keep_mask)
    return res_feat_uncond, pair_feat_uncond, mask_epitope, keep_mask
