import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneEvidenceField(nn.Module):
    def __init__(self,
                 feature_channels,
                 token_channels,
                 hidden_channels=64,
                 gate_init=-2.1972246,
                 detach_sample=True,
                 loss_weight=0.02,
                 pos_weight=10.0,
                 target_radius=1):
        super(LaneEvidenceField, self).__init__()
        self.detach_sample = bool(detach_sample)
        self.loss_weight = float(loss_weight)
        self.pos_weight = float(pos_weight)
        self.target_radius = int(target_radius)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        hidden_channels = int(hidden_channels)
        self.evidence_head = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1))
        self.sample_proj = nn.Sequential(
            nn.Conv2d(1, token_channels, 1),
            nn.GELU(),
            nn.Conv2d(token_channels, token_channels, 1))
        self.out_norm = nn.LayerNorm(token_channels)
        self.out_proj = nn.Linear(token_channels, token_channels)
        nn.init.constant_(self.evidence_head[-1].bias, -4.0)
        nn.init.zeros_(self.sample_proj[-1].weight)
        nn.init.zeros_(self.sample_proj[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, tokens, dense_features, grid_us, grid_vs, valid_mask=None):
        batch_size, channels, num_lanes, num_points = tokens.shape
        evidence_logits = self.evidence_head(dense_features)
        grid = torch.stack([grid_us, grid_vs], dim=-1)
        grid = grid.reshape(batch_size, num_lanes, num_points, 2)
        sampled = F.grid_sample(evidence_logits, grid, padding_mode='zeros')
        sampled_for_fuse = sampled.detach() if self.detach_sample else sampled
        evidence_embed = self.sample_proj(sampled_for_fuse)
        point_tokens = tokens + evidence_embed
        point_tokens = point_tokens.permute(0, 2, 3, 1).contiguous()
        residual = self.out_proj(self.out_norm(point_tokens))
        residual = residual.permute(0, 3, 1, 2).contiguous()
        if valid_mask is not None:
            residual = residual * valid_mask[:, None].to(residual.dtype)
        tokens = tokens + self.gate_logit.sigmoid().to(tokens.dtype) * residual
        return tokens, evidence_logits, sampled

    @staticmethod
    def project_points(matrix, xs, ys, zs):
        ones = torch.ones_like(zs)
        coordinates = torch.stack([xs, ys, zs, ones], dim=1)
        trans = torch.bmm(matrix, coordinates)
        return trans[:, 0, :] / trans[:, 2, :], trans[:, 1, :] / trans[:, 2, :]

    def build_target(self, evidence_logits, gt_3dlanes, project_matrixes, y_max):
        batch_size, _, height, width = evidence_logits.shape
        target = evidence_logits.new_zeros((batch_size, 1, height, width))
        if gt_3dlanes is None:
            return target
        if isinstance(gt_3dlanes, (list, tuple)):
            if len(gt_3dlanes) == 0:
                return target
            lane_batches = gt_3dlanes
            gt_anchor_len = int((lane_batches[0].shape[-1] - 5) // 3)
            lane_device = lane_batches[0].device
            lane_dtype = lane_batches[0].dtype
        else:
            lane_batches = [gt_3dlanes[idx] for idx in range(gt_3dlanes.shape[0])]
            gt_anchor_len = int((gt_3dlanes.shape[-1] - 5) // 3)
            lane_device = gt_3dlanes.device
            lane_dtype = gt_3dlanes.dtype
        if gt_anchor_len <= 0:
            return target
        y_values = torch.arange(
            1, gt_anchor_len + 1, device=lane_device, dtype=lane_dtype)
        valid_y = y_values <= float(y_max)
        z_offset = 5 + gt_anchor_len
        vis_offset = 5 + gt_anchor_len * 2

        for batch_idx, lanes in enumerate(lane_batches[:batch_size]):
            lanes = lanes[lanes[:, 1] > 0]
            if lanes.numel() == 0:
                continue
            xs = lanes[:, 5:5 + gt_anchor_len]
            zs = lanes[:, z_offset:z_offset + gt_anchor_len]
            vis = (lanes[:, vis_offset:vis_offset + gt_anchor_len] > 0.5)
            vis = vis & valid_y.view(1, -1)
            if not bool(vis.any()):
                continue
            ys = y_values.view(1, -1).expand_as(xs)
            us, vs = self.project_points(
                project_matrixes[batch_idx:batch_idx + 1],
                xs.reshape(1, -1), ys.reshape(1, -1), zs.reshape(1, -1))
            us = us.reshape_as(xs)
            vs = vs.reshape_as(xs)
            valid = vis & torch.isfinite(us) & torch.isfinite(vs)
            valid = valid & (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
            if not bool(valid.any()):
                continue
            u_idx = us[valid].round().long().clamp(0, width - 1)
            v_idx = vs[valid].round().long().clamp(0, height - 1)
            for du in range(-self.target_radius, self.target_radius + 1):
                for dv in range(-self.target_radius, self.target_radius + 1):
                    uu = (u_idx + du).clamp(0, width - 1)
                    vv = (v_idx + dv).clamp(0, height - 1)
                    target[batch_idx, 0, vv, uu] = 1.
        return target

    def loss(self, logits_all, gt_3dlanes, gt_project_matrix, project_builder,
             y_max):
        total_loss = None
        total_pos = 0.
        total_prob = 0.
        count = 0
        for iter_logits in logits_all:
            for evidence_logits in iter_logits:
                if evidence_logits is None:
                    continue
                feat_size = evidence_logits.shape[-2:]
                project_matrixes = torch.stack(
                    project_builder(gt_project_matrix, feat_size), dim=0)
                target = self.build_target(
                    evidence_logits, gt_3dlanes, project_matrixes, y_max)
                weights = 1. + target * (self.pos_weight - 1.)
                loss = F.binary_cross_entropy_with_logits(
                    evidence_logits.float(), target.float(), reduction='none')
                loss = (loss * weights.float()).mean()
                total_loss = loss if total_loss is None else total_loss + loss
                total_pos += target.sum().detach()
                total_prob += evidence_logits.sigmoid().mean().detach()
                count += 1
        if count == 0:
            return {}, {}
        batch_size = len(gt_3dlanes) if isinstance(gt_3dlanes, (list, tuple)) \
            else int(gt_3dlanes.shape[0])
        losses = {'lane_evidence_loss': total_loss * self.loss_weight / count}
        metrics = {
            'batch_lane_evidence_pos': total_pos / max(float(batch_size), 1.0),
            'batch_lane_evidence_prob': total_prob / count,
        }
        return losses, metrics
