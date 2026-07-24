import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab_251166.modules.common.geometry import construct_3d_basis
from diffab_251166.modules.common.so3 import rotation_to_so3vec
from diffab_251166.modules.encoders.residue import ResidueEmbedding
from diffab_251166.modules.encoders.pair import PairEmbedding
from diffab_251166.modules.diffusion.dpm_full import FullDPM
from diffab_251166.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom, FLAGS
from diffab_251166.utils.noise_mode import apply_noise_mode_masks, is_epitope_noise_mode
from ._base import register_model

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


def _to_plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) if isinstance(v, dict) or hasattr(v, 'items') else v for k, v in value.items()}
    if hasattr(value, 'items'):
        return {k: _to_plain_dict(v) if isinstance(v, dict) or hasattr(v, 'items') else v for k, v in value.items()}
    return dict(value)


def _inject_epitope_var_sched_options(diffusion_config, modal_config):
    if not modal_config:
        return diffusion_config
    noise_sampling = _to_plain_dict(modal_config.get('noise_sampling', {}))
    if not noise_sampling:
        return diffusion_config

    var_sched_opt = {}
    if 'epitope_offset' in noise_sampling:
        var_sched_opt['epitope_offset'] = noise_sampling['epitope_offset']
    elif 'offset' in noise_sampling:
        var_sched_opt['epitope_offset'] = noise_sampling['offset']

    if 'epitope_power' in noise_sampling:
        var_sched_opt['epitope_power'] = noise_sampling['epitope_power']
    elif 'epitope_gamma' in noise_sampling:
        var_sched_opt['epitope_power'] = noise_sampling['epitope_gamma']
    elif 'power' in noise_sampling:
        var_sched_opt['epitope_power'] = noise_sampling['power']

    if not var_sched_opt:
        return diffusion_config

    diffusion_config = dict(diffusion_config)
    for transition_key in ('trans_pos_opt', 'trans_rot_opt'):
        transition_cfg = _to_plain_dict(diffusion_config.get(transition_key, {}))
        current_var_sched = _to_plain_dict(transition_cfg.get('var_sched_opt', {}))
        for key, value in var_sched_opt.items():
            current_var_sched.setdefault(key, value)
        transition_cfg['var_sched_opt'] = current_var_sched
        diffusion_config[transition_key] = transition_cfg
    return diffusion_config


@register_model('diffab')
class DiffusionAntibodyDesign(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]  # 15
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        cond_cfg = cfg.get('conditioner', {})

        diffusion_config = dict(cfg.diffusion)

        train_weights = {}
        if hasattr(cfg, '_parent') and hasattr(cfg._parent, 'train'):
            train_cfg = cfg._parent.train
            if hasattr(train_cfg, 'loss_weights'):
                train_weights = dict(train_cfg.loss_weights)

        diffusion_weights = {}
        if 'loss_weights' in cfg.diffusion:
            diffusion_weights = dict(cfg.diffusion.loss_weights)

        final_weights = {}
        for key, value in train_weights.items():
            final_weights[key] = value
        for key, value in diffusion_weights.items():
            if key not in final_weights:
                final_weights[key] = value
            elif final_weights.get(key) != value:
                import warnings
                warnings.warn(
                    f"Loss weight conflict: '{key}' in train.loss_weights={final_weights.get(key)} "
                    f"is preserved over diffusion.loss_weights={value}. "
                    f"To use diffusion value, remove it from train.loss_weights."
                )

        if final_weights:
            diffusion_config['loss_weights'] = final_weights

        diffusion_config = _inject_epitope_var_sched_options(diffusion_config, cfg.get('modal_training', None))


        if final_weights:
            try:
                from diffab_251166.modules.common.bfactor_utils import debug_log, DEBUG_WEIGHTS
                if DEBUG_WEIGHTS:
                    debug_log(f"[Diffab] Final loss_weights passed to FullDPM: {final_weights}")
            except:
                pass

        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            cfg.region_feat_dim,
            cond_cfg,
            **diffusion_config,
        )

        self.modal_config = cfg.get('modal_training', None)

    def encode(self, batch, remove_structure=True, remove_sequence=True): # remove_sequence 控制生成区域的序列和结构不泄露
        if remove_structure and "pos_heavyatom" in batch and "pos_heavyatom_native" not in batch:
            batch["pos_heavyatom_native"] = batch["pos_heavyatom"].clone()
        sequence_context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
            ~(batch['generate_flag'] > 0),
        )
        structure_context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
            ~(batch['structure_generate_flag'] > 0),
        )
        structure_mask = structure_context_mask if remove_structure else None
        sequence_mask = sequence_context_mask if remove_sequence else None

        res_feat_ctx = self.residue_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            fragment_type=batch['fragment_type'],
            region_type=batch['region_type'],
            structure_mask=structure_mask, # 训练过程中 需要掩住，推理过程不用
            sequence_mask=sequence_mask # 训练过程中 需要掩住，推理过程不用
        )

        pair_feat_ctx = self.pair_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            region_type=batch['region_type'],
            structure_mask=structure_mask, # 训练过程中 需要掩住，推理过程不用
            sequence_mask=sequence_mask # 训练过程中 需要掩住，推理过程不用
        )
        R = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
                               batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
                               batch['pos_heavyatom'][:, :, BBHeavyAtom.N])
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]


        return res_feat_ctx, pair_feat_ctx, R, p

    def forward(self, batch):
        remove_structure = self.cfg.get('train_structure', True)  # 训练和验证都掩码
        remove_sequence = self.cfg.get('train_sequence', True)

        if self.training:
            if self.modal_config:
                batch['modal_training'] = {
                    'noise_sampling': dict(self.modal_config['noise_sampling'])
                }
            else:
                batch['modal_training'] = None
        else:
            val_modal_config = batch.get('val_modal_training', None)
            train_noise_mode = self.modal_config['noise_sampling'].get('noise_mode', 'cdr_only') if self.modal_config else 'cdr_only'

            if val_modal_config:
                noise_sampling = dict(val_modal_config.get('noise_sampling', {}))
                val_noise_mode = noise_sampling.get('noise_mode', 'cdr_only')
                val_has_epitope = val_noise_mode == 'cdr_epitope' or (isinstance(val_noise_mode, list) and 'cdr_epitope' in val_noise_mode)

                if val_has_epitope and train_noise_mode == 'cdr_only':
                    degraded_sampling = dict(noise_sampling)
                    degraded_sampling['noise_mode'] = ['cdr_only'] if isinstance(val_noise_mode, list) else 'cdr_only'
                    batch['modal_training'] = {
                        'noise_sampling': degraded_sampling
                    }
                else:
                    batch['modal_training'] = {
                        'noise_sampling': noise_sampling
                    }
            else:
                if self.modal_config:
                    batch['modal_training'] = {
                        'noise_sampling': dict(self.modal_config['noise_sampling'])
                    }
                else:
                    batch['modal_training'] = None

        noise_sampling = batch.get('modal_training', {}).get('noise_sampling', {}) if batch.get('modal_training') else {}
        noise_mode = noise_sampling.get('noise_mode', 'cdr_only')
        apply_noise_mode_masks(
            batch,
            noise_mode=noise_mode,
            assert_epitope_mask=is_epitope_noise_mode(noise_mode),
            context='train' if self.training else 'val',
        )

        res_feat_ctx, pair_feat_ctx, R_0, p_0 = self.encode(
            batch,
            remove_structure=remove_structure,
            remove_sequence=remove_sequence
        )

        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        mask_dict = {
            'mask_cdr': batch['generate_flag'] > 0,
            'mask_soft_antigen': batch['antigen_soft_mask'],
            'mask_full_antigen': batch['antigen_mask'],
        }
        assert torch.equal(mask_dict['mask_soft_antigen'].to(torch.bool), batch['antigen_soft_mask'].to(torch.bool)), \
            "mask_dict['mask_soft_antigen'] must match batch['antigen_soft_mask']"
        if is_epitope_noise_mode(noise_mode):
            assert mask_dict['mask_soft_antigen'].any(), "epitope mode requires non-empty mask_soft_antigen in forward()"
        else:
            assert not mask_dict['mask_soft_antigen'].any(), "cdr_only mode must pass an empty mask_soft_antigen in forward()"
        bfactor = batch['bfactor']
        loss_dict, weight_info, loss_dict_raw = self.diffusion(
            batch, v_0, p_0, s_0, res_feat_ctx, pair_feat_ctx, bfactor, mask_dict,
            t_dict=None,  # 让 diffusion 内部决定采样策略
        )

        return loss_dict, weight_info, loss_dict_raw

    @torch.no_grad()
    def sample(self, batch, sample_opt=None):
        sample_opt = sample_opt or {'sample_structure': True, 'sample_sequence': True, 'pbar': False}
        noise_sampling = batch.get('modal_training', {}).get('noise_sampling', {}) if batch.get('modal_training') else {}
        noise_mode = noise_sampling.get('noise_mode', 'cdr_only')
        apply_noise_mode_masks(
            batch,
            noise_mode=noise_mode,
            assert_epitope_mask=is_epitope_noise_mode(noise_mode),
            context='sample',
        )

        mask_struct_generate = batch['structure_generate_flag'] > 0
        mask_seq_generate = batch['generate_flag'] > 0
        mask_res = batch['mask']


        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=sample_opt.get('sample_structure', True),
            remove_sequence=sample_opt.get('sample_sequence', True)
        )

        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        mask_dict = {
            'mask_cdr': batch['generate_flag'] > 0,
            'mask_soft_antigen': batch['antigen_soft_mask'],
            'mask_full_antigen': batch['antigen_mask'],
            'mask_res':batch['mask']
        }
        assert torch.equal(mask_dict['mask_soft_antigen'].to(torch.bool), batch['antigen_soft_mask'].to(torch.bool)), \
            "mask_dict['mask_soft_antigen'] must match batch['antigen_soft_mask']"
        if is_epitope_noise_mode(noise_mode):
            assert mask_dict['mask_soft_antigen'].any(), "epitope mode requires non-empty mask_soft_antigen in sample()"
        else:
            assert not mask_dict['mask_soft_antigen'].any(), "cdr_only mode must pass an empty mask_soft_antigen in sample()"
        bfactor = batch['bfactor']

        traj = self.diffusion.sample(
            batch, v_0, p_0, s_0, res_feat, pair_feat, mask_seq_generate, mask_struct_generate, mask_dict, bfactor,
            pbar=sample_opt.get('pbar', False),
            compute_loss=sample_opt.get('compute_loss', False),
        )
        return traj

    @torch.no_grad()
    def optimize(self, batch, opt_step, optimize_opt=None):
        optimize_opt = optimize_opt or {'sample_structure': True, 'sample_sequence': True, 'pbar': False}
        noise_sampling = batch.get('modal_training', {}).get('noise_sampling', {}) if batch.get('modal_training') else {}
        noise_mode = noise_sampling.get('noise_mode', 'cdr_only')
        apply_noise_mode_masks(
            batch,
            noise_mode=noise_mode,
            assert_epitope_mask=is_epitope_noise_mode(noise_mode),
            context='optimize',
        )
        mask_generate = batch['generate_flag'] > 0
        mask_res = batch['mask']

        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=optimize_opt.get('sample_structure', True),
            remove_sequence=optimize_opt.get('sample_sequence', True)
        )

        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(
            batch, v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_generate, mask_res,
            sample_structure=optimize_opt.get('sample_structure', True),
            sample_sequence=optimize_opt.get('sample_sequence', True),
            pbar=optimize_opt.get('pbar', False),
        )

        return traj
