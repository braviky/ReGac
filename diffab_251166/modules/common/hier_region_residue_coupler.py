import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab_251166.utils.protein.constants import AG, BBHeavyAtom, CDR, REGION_NUM


def _region_onehot(region_type, num_regions):
    region_type = region_type.to(torch.long).clamp(min=0, max=num_regions)
    return F.one_hot(region_type, num_classes=num_regions + 1)[:, :, 1:].float()


class HierarchicalRegionResidueCoupler(nn.Module):
    """Residue-region bidirectional coupling for graph-asynchronous CDR denoising.

    Residual updates add intra-CDR consistency, condition-anchored residue
    interactions, and top-down region feedback on top of TDGN.
    """

    def __init__(self, node_dim, region_dim, num_regions=REGION_NUM, opt=None):
        super().__init__()
        opt = opt or {}
        self.num_regions = int(opt.get("num_regions", num_regions))
        self.enabled = bool(opt.get("enabled", False))
        self.residual_scale = float(opt.get("residual_scale", 0.35))
        self.condition_distance_scale = float(opt.get("condition_distance_scale", 10.0))
        self.condition_contact_cutoff = float(opt.get("condition_contact_cutoff", 8.0))
        self.use_distance_bias = bool(opt.get("use_distance_bias", True))
        self.contact_latent_enabled = bool(opt.get("contact_latent_enabled", True))
        self.contact_supervision_enabled = bool(opt.get("contact_supervision_enabled", True))
        self.contact_supervision_cutoff = float(
            opt.get("contact_supervision_cutoff", self.condition_contact_cutoff)
        )
        self.contact_feedback_scale = float(opt.get("contact_feedback_scale", 0.25))

        self.res_norm = nn.LayerNorm(node_dim)
        self.region_norm = nn.LayerNorm(region_dim)
        self.res_to_region = nn.Linear(node_dim, region_dim)
        self.region_to_res = nn.Linear(region_dim, node_dim)

        self.intra_q = nn.Linear(node_dim, node_dim)
        self.intra_k = nn.Linear(node_dim, node_dim)
        self.intra_v = nn.Linear(node_dim, node_dim)
        self.intra_out = nn.Linear(node_dim, node_dim)

        self.cond_q = nn.Linear(node_dim, node_dim)
        self.cond_k = nn.Linear(node_dim, node_dim)
        self.cond_v = nn.Linear(node_dim, node_dim)
        self.cond_out = nn.Linear(node_dim, node_dim)
        self.cond_to_region = nn.Linear(node_dim, region_dim)

        self.region_q = nn.Linear(region_dim, region_dim)
        self.region_k = nn.Linear(region_dim, region_dim)
        self.region_v = nn.Linear(region_dim, region_dim)
        self.region_msg = nn.Linear(region_dim, region_dim)

        self.contact_q = nn.Linear(region_dim, region_dim)
        self.contact_k = nn.Linear(region_dim, region_dim)
        self.contact_v = nn.Linear(region_dim, region_dim)
        self.contact_latent_proj = nn.Sequential(
            nn.Linear(region_dim * 2, region_dim),
            nn.SiLU(),
            nn.Linear(region_dim, region_dim),
        )
        self.region_contact_head = nn.Linear(region_dim, 1)
        self.residue_contact_head = nn.Sequential(
            nn.Linear(node_dim * 3, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
        )
        self.contact_to_res = nn.Linear(region_dim, node_dim)
        self.contact_feedback_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )

        self.top_q = nn.Linear(node_dim, region_dim)
        self.top_k = nn.Linear(region_dim, region_dim)
        self.top_v = nn.Linear(region_dim, node_dim)
        self.top_out = nn.Linear(node_dim, node_dim)

        self.film = nn.Linear(region_dim, node_dim * 2)
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 4, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )

        self.register_buffer("region_prior", self._build_region_prior(self.num_regions))
        self.last_state = None

    @staticmethod
    def _build_region_prior(num_regions):
        prior = torch.zeros(num_regions, num_regions)
        cdr_ids = [int(x) for x in CDR]
        epi_ids = [int(AG.EPI_CORE), int(AG.EPI_RIM)]
        non_epi = int(AG.NON_EPI)
        for i in cdr_ids:
            for j in cdr_ids:
                if i != j and i <= num_regions and j <= num_regions:
                    prior[i - 1, j - 1] = 1.0
            for j in epi_ids:
                if i <= num_regions and j <= num_regions:
                    prior[i - 1, j - 1] = 1.0
                    prior[j - 1, i - 1] = 1.0
        for i in epi_ids:
            for j in epi_ids + [non_epi]:
                if i != j and i <= num_regions and j <= num_regions:
                    prior[i - 1, j - 1] = 1.0
                    prior[j - 1, i - 1] = 1.0
        return prior

    def _pool_regions(self, x, onehot, valid_mask):
        mask = onehot * valid_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        pooled = torch.bmm(mask.transpose(1, 2), x) / denom
        valid = mask.sum(dim=1) > 0
        return pooled, valid

    def _masked_attention(self, q, k, v, mask, bias=None):
        dim = q.size(-1)
        logits = torch.bmm(q, k.transpose(1, 2)) / (dim ** 0.5)
        if bias is not None:
            logits = logits + bias
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.bmm(attn, v)
        return out, attn

    def _condition_bias(self, p_t, query_mask, condition_mask):
        if p_t is None or not self.use_distance_bias:
            return None
        dist = torch.cdist(p_t.float(), p_t.float())
        scale = max(self.condition_distance_scale, 1e-6)
        bias = -dist / scale
        return bias.masked_fill(~(query_mask.unsqueeze(2) & condition_mask.unsqueeze(1)), 0.0)

    def _cdr_region_bool(self, device):
        ids = torch.arange(1, self.num_regions + 1, device=device)
        out = torch.zeros_like(ids, dtype=torch.bool)
        for rid in [int(CDR.H1), int(CDR.H2), int(CDR.H3), int(CDR.L1), int(CDR.L2), int(CDR.L3)]:
            out = out | (ids == rid)
        return out

    def _epi_region_bool(self, device):
        ids = torch.arange(1, self.num_regions + 1, device=device)
        return (ids == int(AG.EPI_CORE)) | (ids == int(AG.EPI_RIM))

    def _safe_bce_with_logits(self, logits, target, mask):
        if mask is not None:
            if not bool(mask.any().item()):
                return logits.sum() * 0.0
            logits = logits[mask]
            target = target[mask]
        return F.binary_cross_entropy_with_logits(logits, target.float())

    def _build_contact_targets(self, region_type, onehot, valid_mask, region_valid, region_aux_inputs):
        if not self.contact_latent_enabled or not self.contact_supervision_enabled or region_aux_inputs is None:
            return None
        pos_heavyatom = region_aux_inputs.get("pos_heavyatom")
        if pos_heavyatom is None or pos_heavyatom.dim() != 4:
            return None

        device = region_type.device
        ca_pos = pos_heavyatom.to(device=device, dtype=torch.float32)[:, :, BBHeavyAtom.CA, :]
        atom_mask = region_aux_inputs.get("mask_heavyatom")
        if atom_mask is not None and atom_mask.dim() == 3:
            ca_valid = atom_mask.to(device=device, dtype=torch.bool)[:, :, BBHeavyAtom.CA]
        else:
            ca_valid = valid_mask

        cdr_res = self._cdr_region_bool(device)[region_type.long().clamp(min=1, max=self.num_regions) - 1]
        cdr_res = cdr_res & valid_mask & ca_valid
        epi_res = self._epi_region_bool(device)[region_type.long().clamp(min=1, max=self.num_regions) - 1]
        antigen_soft = region_aux_inputs.get("antigen_soft_mask_raw")
        if antigen_soft is not None and tuple(antigen_soft.shape) == tuple(epi_res.shape):
            epi_res = epi_res | antigen_soft.to(device=device, dtype=torch.bool)
        epi_res = epi_res & valid_mask & ca_valid

        if not bool(cdr_res.any().item()) or not bool(epi_res.any().item()):
            return None

        dist = torch.cdist(ca_pos, ca_pos)
        valid_pair = ca_valid.unsqueeze(2) & ca_valid.unsqueeze(1)
        contact_pair = (dist <= self.contact_supervision_cutoff) & valid_pair

        residue_target = (contact_pair & cdr_res.unsqueeze(2) & epi_res.unsqueeze(1)).any(dim=-1).float()
        region_contact_target, _ = self._pool_regions(residue_target.unsqueeze(-1), onehot, valid_mask)
        region_contact_target = region_contact_target.squeeze(-1).clamp(0.0, 1.0)

        member = onehot * (valid_mask & ca_valid).float().unsqueeze(-1)
        region_pair_count = torch.bmm(member.transpose(1, 2), torch.bmm(contact_pair.float(), member))
        region_pair_target = (region_pair_count > 0).float()

        cdr_region = self._cdr_region_bool(device)
        epi_region = self._epi_region_bool(device)
        region_ids = torch.arange(self.num_regions, device=device)
        non_self = region_ids[:, None] != region_ids[None, :]
        pair_label_mask = (
            (cdr_region[:, None] & epi_region[None, :]) |
            (epi_region[:, None] & cdr_region[None, :]) |
            (cdr_region[:, None] & cdr_region[None, :] & non_self)
        )
        pair_label_mask = pair_label_mask.unsqueeze(0)
        pair_label_mask = pair_label_mask & region_valid.unsqueeze(1) & region_valid.unsqueeze(2)

        region_label_mask = region_valid & cdr_region.unsqueeze(0)
        return {
            "residue_contact_target": residue_target,
            "residue_label_mask": cdr_res,
            "region_contact_target": region_contact_target,
            "region_label_mask": region_label_mask,
            "region_pair_target": region_pair_target,
            "region_pair_label_mask": pair_label_mask,
        }

    def forward(self, res_feat, region_type, structure_generate_flag, valid_mask, p_t=None, region_aux_inputs=None):
        if not self.enabled:
            self.last_state = None
            return res_feat, {}

        valid_mask = valid_mask.to(torch.bool)
        generate_mask = (structure_generate_flag > 0) & valid_mask
        condition_mask = valid_mask & (~generate_mask)
        cdr_region_mask = (
            (region_type == int(CDR.H1)) |
            (region_type == int(CDR.H2)) |
            (region_type == int(CDR.H3)) |
            (region_type == int(CDR.L1)) |
            (region_type == int(CDR.L2)) |
            (region_type == int(CDR.L3))
        ) & generate_mask

        onehot = _region_onehot(region_type, self.num_regions)
        x = self.res_norm(res_feat)

        same_region = region_type.unsqueeze(2) == region_type.unsqueeze(1)
        intra_mask = same_region & cdr_region_mask.unsqueeze(1) & cdr_region_mask.unsqueeze(2)
        intra_out, intra_attn = self._masked_attention(
            self.intra_q(x), self.intra_k(x), self.intra_v(x), intra_mask
        )
        intra_out = self.intra_out(intra_out)

        cond_pair_mask = cdr_region_mask.unsqueeze(2) & condition_mask.unsqueeze(1)
        cond_bias = self._condition_bias(p_t, cdr_region_mask, condition_mask)
        cond_out, cond_attn = self._masked_attention(
            self.cond_q(x), self.cond_k(x), self.cond_v(x), cond_pair_mask, bias=cond_bias
        )
        cond_out = self.cond_out(cond_out)

        region_tokens, region_valid = self._pool_regions(self.res_to_region(x), onehot, valid_mask)
        contact_region, _ = self._pool_regions(self.cond_to_region(cond_out), onehot, valid_mask)
        region_tokens = region_tokens + contact_region

        r = self.region_norm(region_tokens)
        q = self.region_q(r)
        k = self.region_k(r)
        v = self.region_v(r)
        region_mask = region_valid.unsqueeze(1) & region_valid.unsqueeze(2)
        prior = self.region_prior.to(region_tokens.device).unsqueeze(0)
        region_logits = torch.bmm(q, k.transpose(1, 2)) / (r.size(-1) ** 0.5)
        region_logits = region_logits + prior
        region_logits = region_logits.masked_fill(~region_mask, float("-inf"))
        region_attn = F.softmax(region_logits, dim=-1)
        region_attn = torch.nan_to_num(region_attn, nan=0.0)
        region_msg = self.region_msg(torch.bmm(region_attn, v))
        region_out = region_tokens + region_msg

        if self.contact_latent_enabled:
            c_q = self.contact_q(self.region_norm(region_out))
            c_k = self.contact_k(self.region_norm(region_out))
            c_v = self.contact_v(region_out)
            region_pair_contact_logits = torch.bmm(c_q, c_k.transpose(1, 2)) / (c_q.size(-1) ** 0.5)
            region_pair_contact_logits = region_pair_contact_logits + prior
            contact_logits_masked = region_pair_contact_logits.masked_fill(~region_mask, float("-inf"))
            contact_attn = F.softmax(contact_logits_masked, dim=-1)
            contact_attn = torch.nan_to_num(contact_attn, nan=0.0)
            contact_msg = torch.bmm(contact_attn, c_v)
            contact_latent = self.contact_latent_proj(torch.cat([region_out, contact_msg], dim=-1))
            contact_latent = contact_latent * region_valid.unsqueeze(-1).float()
            region_contact_logits = self.region_contact_head(contact_latent).squeeze(-1)
            region_contact_prob = torch.sigmoid(region_contact_logits) * region_valid.float()
        else:
            B, R, _ = region_out.shape
            region_pair_contact_logits = region_out.new_zeros(B, R, R)
            contact_latent = region_out.new_zeros(region_out.shape)
            region_contact_logits = region_out.new_zeros(B, R)
            region_contact_prob = region_contact_logits

        own_region = torch.bmm(onehot, region_out)
        own_contact_latent = torch.bmm(onehot, contact_latent)
        top_mask = valid_mask.unsqueeze(2) & region_valid.unsqueeze(1)
        top_out, top_attn = self._masked_attention(
            self.top_q(x), self.top_k(region_out), self.top_v(region_out), top_mask
        )
        top_out = self.top_out(top_out)
        scale_shift = self.film(own_region)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        top_film = x * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        residue_contact_logits = self.residue_contact_head(torch.cat([x, cond_out, top_out], dim=-1)).squeeze(-1)
        contact_residual = self.contact_to_res(own_contact_latent)
        contact_gate = self.contact_feedback_gate(torch.cat([x, contact_residual], dim=-1))
        contact_feedback = self.contact_feedback_scale * contact_gate * contact_residual

        gate_input = torch.cat([x, intra_out, cond_out, top_out + top_film], dim=-1)
        gate = self.gate(gate_input)
        delta = gate * (intra_out + cond_out + top_out + top_film + contact_feedback)
        delta = delta * cdr_region_mask.unsqueeze(-1).float()
        out = res_feat + self.residual_scale * delta
        out = out * valid_mask.unsqueeze(-1).float()

        epitope_context_mask = (
            (region_type == int(AG.EPI_CORE)) |
            (region_type == int(AG.EPI_RIM))
        ) & condition_mask
        contact_mass = (cond_attn * epitope_context_mask.unsqueeze(1).float()).sum(dim=-1)
        contact_mass = contact_mass * cdr_region_mask.float()
        region_contact_mass, _ = self._pool_regions(contact_mass.unsqueeze(-1), onehot, valid_mask)
        region_contact_mass = region_contact_mass.squeeze(-1)
        if p_t is not None:
            dist = torch.cdist(p_t.float(), p_t.float())
            contact_hits = (dist <= self.condition_contact_cutoff) & cond_pair_mask
            residue_contact_hit = contact_hits.any(dim=-1).float()
            region_contact_hit, _ = self._pool_regions(residue_contact_hit.unsqueeze(-1), onehot, valid_mask)
            region_contact_hit = region_contact_hit.squeeze(-1)
        else:
            region_contact_hit = region_contact_mass.new_zeros(region_contact_mass.shape)

        self.last_state = {
            "region_tokens": region_out.detach(),
            "region_valid": region_valid.detach(),
            "region_attn": region_attn.detach(),
            "region_contact_mass": region_contact_mass.detach(),
            "region_contact_hit": region_contact_hit.detach(),
            "region_contact_prob": region_contact_prob.detach(),
            "region_pair_contact_prob": torch.sigmoid(region_pair_contact_logits).detach(),
            "contact_latent": contact_latent.detach(),
            "residue_condition_attn": cond_attn.detach(),
            "residue_intra_attn": intra_attn.detach(),
            "topdown_attn": top_attn.detach(),
            "residue_contact_prob": torch.sigmoid(residue_contact_logits).detach(),
            "gate": gate.detach(),
            "contact_gate": contact_gate.detach(),
        }

        aux = {
            "monitor_hgacd_contact_mass": region_contact_mass.mean(),
            "monitor_hgacd_contact_hit": region_contact_hit.mean(),
            "monitor_hgacd_contact_prob": region_contact_prob.mean(),
            "monitor_hgacd_pair_contact_prob": torch.sigmoid(region_pair_contact_logits).mean(),
            "monitor_hgacd_contact_gate": contact_gate[cdr_region_mask].mean() if cdr_region_mask.any() else contact_gate.mean() * 0.0,
            "monitor_hgacd_gate": gate[cdr_region_mask].mean() if cdr_region_mask.any() else gate.mean() * 0.0,
        }
        targets = self._build_contact_targets(region_type, onehot, valid_mask, region_valid, region_aux_inputs)
        if targets is not None:
            region_loss = self._safe_bce_with_logits(
                region_contact_logits,
                targets["region_contact_target"],
                targets["region_label_mask"],
            )
            residue_loss = self._safe_bce_with_logits(
                residue_contact_logits,
                targets["residue_contact_target"],
                targets["residue_label_mask"],
            )
            pair_loss = self._safe_bce_with_logits(
                region_pair_contact_logits,
                targets["region_pair_target"],
                targets["region_pair_label_mask"],
            )
            residue_prob = torch.sigmoid(residue_contact_logits).unsqueeze(-1)
            pooled_residue_prob, _ = self._pool_regions(residue_prob, onehot, valid_mask)
            pooled_residue_prob = pooled_residue_prob.squeeze(-1)
            consistency_mask = targets["region_label_mask"]
            if bool(consistency_mask.any().item()):
                consistency_loss = F.mse_loss(
                    region_contact_prob[consistency_mask],
                    pooled_residue_prob[consistency_mask],
                )
            else:
                consistency_loss = region_contact_logits.sum() * 0.0
            aux.update({
                "region_hgacd_contact_mass": region_loss,
                "region_hgacd_residue_contact": residue_loss,
                "region_hgacd_pair_contact": pair_loss,
                "region_hgacd_contact_consistency": consistency_loss,
                "monitor_hgacd_contact_target": targets["region_contact_target"][targets["region_label_mask"]].mean()
                if bool(targets["region_label_mask"].any().item()) else region_loss.detach() * 0.0,
            })
        return out, aux
