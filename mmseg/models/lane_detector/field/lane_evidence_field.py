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
                 target_radius=1,
                 geometry_norm=(20.0, 100.0, 10.0),
                 bev_enabled=False,
                 bev_loss_weight=0.01,
                 bev_pos_weight=8.0,
                 bev_target_radius=1,
                 bev_height=128,
                 bev_width=48,
                 bev_x_min=-10.0,
                 bev_x_max=10.0,
                 bev_y_min=3.0,
                 bev_y_max=103.0,
                 tube_enabled=False,
                 tube_x_offsets=(-0.5, -0.25, 0.25, 0.5),
                 tube_z_offsets=(-0.3, 0.3)):
        super(LaneEvidenceField, self).__init__()
        self.detach_sample = bool(detach_sample)
        self.loss_weight = float(loss_weight)
        self.pos_weight = float(pos_weight)
        self.target_radius = int(target_radius)
        self.geometry_norm = tuple(float(v) for v in geometry_norm)
        self.bev_enabled = bool(bev_enabled)
        self.bev_loss_weight = float(bev_loss_weight)
        self.bev_pos_weight = float(bev_pos_weight)
        self.bev_target_radius = int(bev_target_radius)
        self.bev_height = int(bev_height)
        self.bev_width = int(bev_width)
        self.bev_x_min = float(bev_x_min)
        self.bev_x_max = float(bev_x_max)
        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)
        self.tube_enabled = bool(tube_enabled)
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
        if self.bev_enabled:
            self.bev_head = nn.Sequential(
                nn.Conv2d(feature_channels, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, 1))
            self.bev_sample_proj = nn.Sequential(
                nn.Conv2d(1, token_channels, 1),
                nn.GELU(),
                nn.Conv2d(token_channels, token_channels, 1))
        else:
            self.bev_head = None
            self.bev_sample_proj = None
        if self.tube_enabled:
            offsets = self.build_tube_offsets(tube_x_offsets, tube_z_offsets)
            self.register_buffer(
                'tube_offsets', torch.tensor(offsets, dtype=torch.float32),
                persistent=False)
            self.tube_sample_proj = nn.Sequential(
                nn.Conv2d(len(offsets), token_channels, 1),
                nn.GELU(),
                nn.Conv2d(token_channels, token_channels, 1))
        else:
            self.tube_sample_proj = None
        self.out_norm = nn.LayerNorm(token_channels)
        self.out_proj = nn.Linear(token_channels, token_channels)
        nn.init.constant_(self.evidence_head[-1].bias, -4.0)
        nn.init.zeros_(self.sample_proj[-1].weight)
        nn.init.zeros_(self.sample_proj[-1].bias)
        if self.bev_enabled:
            nn.init.constant_(self.bev_head[-1].bias, -4.0)
            nn.init.zeros_(self.bev_sample_proj[-1].weight)
            nn.init.zeros_(self.bev_sample_proj[-1].bias)
        if self.tube_enabled:
            nn.init.zeros_(self.tube_sample_proj[-1].weight)
            nn.init.zeros_(self.tube_sample_proj[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def build_tube_offsets(tube_x_offsets, tube_z_offsets):
        offsets = [(0.0, 0.0)]
        offsets.extend((float(offset), 0.0) for offset in tube_x_offsets)
        offsets.extend((0.0, float(offset)) for offset in tube_z_offsets)
        return offsets

    def geometry_to_bev_grid(self, geometry):
        x_norm, y_norm, _ = self.geometry_norm
        raw_x = geometry[..., 0] * x_norm
        raw_y = geometry[..., 1] * y_norm
        grid_x = 2. * (raw_x - self.bev_x_min) / (
            self.bev_x_max - self.bev_x_min) - 1.
        grid_y = 2. * (raw_y - self.bev_y_min) / (
            self.bev_y_max - self.bev_y_min) - 1.
        valid = (grid_x > -1.) & (grid_x < 1.) & (grid_y > -1.) & (grid_y < 1.)
        return torch.stack([grid_x, grid_y], dim=-1), valid

    def build_metric_bev_features(self, dense_features, project_matrixes):
        if project_matrixes is None:
            return None
        batch_size = dense_features.shape[0]
        device = dense_features.device
        dtype = dense_features.dtype
        xs = torch.linspace(
            self.bev_x_min, self.bev_x_max, self.bev_width,
            device=device, dtype=dtype)
        ys = torch.linspace(
            self.bev_y_min, self.bev_y_max, self.bev_height,
            device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        flat_x = xx.reshape(1, -1).expand(batch_size, -1)
        flat_y = yy.reshape(1, -1).expand(batch_size, -1)
        flat_z = torch.zeros_like(flat_x)
        ones = torch.ones_like(flat_z)
        coordinates = torch.stack([flat_x, flat_y, flat_z, ones], dim=1)
        trans = torch.bmm(
            project_matrixes.to(device=device, dtype=dtype), coordinates)
        depth = trans[:, 2, :]
        us = trans[:, 0, :] / depth.clamp_min(1e-6)
        vs = trans[:, 1, :] / depth.clamp_min(1e-6)
        norm_u = (us / dense_features.shape[-1] - 0.5) * 2.
        norm_v = (vs / dense_features.shape[-2] - 0.5) * 2.
        valid = (depth > 1e-6) & torch.isfinite(norm_u) & torch.isfinite(norm_v)
        valid = valid & (norm_u > -1.) & (norm_u < 1.)
        valid = valid & (norm_v > -1.) & (norm_v < 1.)
        norm_u = torch.where(valid, norm_u, norm_u.new_full(norm_u.shape, 2.))
        norm_v = torch.where(valid, norm_v, norm_v.new_full(norm_v.shape, 2.))
        grid = torch.stack([norm_u, norm_v], dim=-1)
        grid = grid.reshape(batch_size, self.bev_height, self.bev_width, 2)
        bev_features = F.grid_sample(
            dense_features, grid, padding_mode='zeros', align_corners=False)
        return bev_features * valid.reshape(
            batch_size, 1, self.bev_height, self.bev_width).to(dtype)

    def build_tube_samples(self, evidence_logits, geometry, project_matrixes):
        if project_matrixes is None:
            return None, None
        x_norm, y_norm, z_norm = self.geometry_norm
        batch_size, _, height, width = evidence_logits.shape
        offsets = self.tube_offsets.to(device=geometry.device, dtype=geometry.dtype)
        raw_x = geometry[..., 0] * x_norm
        raw_y = geometry[..., 1] * y_norm
        raw_z = geometry[..., 2] * z_norm
        sample_x = raw_x[..., None] + offsets[:, 0]
        sample_y = raw_y[..., None].expand_as(sample_x)
        sample_z = raw_z[..., None] + offsets[:, 1]
        flat_x = sample_x.reshape(batch_size, -1)
        flat_y = sample_y.reshape(batch_size, -1)
        flat_z = sample_z.reshape(batch_size, -1)
        ones = torch.ones_like(flat_z)
        coordinates = torch.stack([flat_x, flat_y, flat_z, ones], dim=1)
        trans = torch.bmm(
            project_matrixes.to(device=geometry.device, dtype=geometry.dtype),
            coordinates)
        depth = trans[:, 2, :]
        us = trans[:, 0, :] / depth.clamp_min(1e-6)
        vs = trans[:, 1, :] / depth.clamp_min(1e-6)
        norm_u = (us / width - 0.5) * 2.
        norm_v = (vs / height - 0.5) * 2.
        valid = (depth > 1e-6) & torch.isfinite(norm_u) & torch.isfinite(norm_v)
        valid = valid & (norm_u > -1.) & (norm_u < 1.)
        valid = valid & (norm_v > -1.) & (norm_v < 1.)
        norm_u = torch.where(valid, norm_u, norm_u.new_full(norm_u.shape, 2.))
        norm_v = torch.where(valid, norm_v, norm_v.new_full(norm_v.shape, 2.))
        grid = torch.stack([norm_u, norm_v], dim=-1)
        num_lanes, num_points = geometry.shape[1], geometry.shape[2]
        grid = grid.reshape(batch_size, num_lanes, num_points * len(offsets), 2)
        sampled = F.grid_sample(
            evidence_logits, grid, padding_mode='zeros', align_corners=False)
        sampled = sampled.reshape(batch_size, 1, num_lanes, num_points, len(offsets))
        sampled = sampled.permute(0, 4, 2, 3, 1).squeeze(-1).contiguous()
        valid = valid.reshape(batch_size, num_lanes, num_points, len(offsets))
        valid = valid.permute(0, 3, 1, 2).contiguous()
        return sampled * valid.to(sampled.dtype), valid

    def forward(self, tokens, dense_features, grid_us, grid_vs,
                valid_mask=None, geometry=None, project_matrixes=None):
        batch_size, channels, num_lanes, num_points = tokens.shape
        evidence_logits = self.evidence_head(dense_features)
        grid = torch.stack([grid_us, grid_vs], dim=-1)
        grid = grid.reshape(batch_size, num_lanes, num_points, 2)
        sampled = F.grid_sample(
            evidence_logits, grid, padding_mode='zeros', align_corners=False)
        sampled_for_fuse = sampled.detach() if self.detach_sample else sampled
        evidence_embed = self.sample_proj(sampled_for_fuse)
        bev_logits = None
        if self.bev_enabled and geometry is not None:
            bev_features = self.build_metric_bev_features(
                dense_features, project_matrixes)
            if bev_features is None:
                bev_features = F.interpolate(
                    dense_features,
                    size=(self.bev_height, self.bev_width),
                    mode='bilinear',
                    align_corners=False)
            bev_logits = self.bev_head(bev_features)
            bev_grid, bev_valid = self.geometry_to_bev_grid(
                geometry.detach() if self.detach_sample else geometry)
            bev_sampled = F.grid_sample(
                bev_logits, bev_grid, padding_mode='zeros', align_corners=True)
            bev_for_fuse = bev_sampled.detach() if self.detach_sample else bev_sampled
            bev_embed = self.bev_sample_proj(bev_for_fuse)
            bev_embed = bev_embed * bev_valid[:, None].to(bev_embed.dtype)
            evidence_embed = evidence_embed + bev_embed
        tube_sampled = None
        tube_valid = None
        if self.tube_enabled and geometry is not None:
            tube_sampled, tube_valid = self.build_tube_samples(
                evidence_logits, geometry, project_matrixes)
            if tube_sampled is not None:
                if valid_mask is not None:
                    tube_valid = tube_valid & valid_mask[:, None]
                    tube_sampled = tube_sampled * tube_valid.to(tube_sampled.dtype)
                tube_for_fuse = tube_sampled.detach() if self.detach_sample else tube_sampled
                tube_embed = self.tube_sample_proj(tube_for_fuse)
                if valid_mask is not None:
                    tube_embed = tube_embed * valid_mask[:, None].to(tube_embed.dtype)
                evidence_embed = evidence_embed + tube_embed
        point_tokens = tokens + evidence_embed
        point_tokens = point_tokens.permute(0, 2, 3, 1).contiguous()
        residual = self.out_proj(self.out_norm(point_tokens))
        residual = residual.permute(0, 3, 1, 2).contiguous()
        if valid_mask is not None:
            residual = residual * valid_mask[:, None].to(residual.dtype)
        tokens = tokens + self.gate_logit.sigmoid().to(tokens.dtype) * residual
        logits = {
            'image': evidence_logits,
            'bev': bev_logits,
            'tube': tube_sampled,
            'tube_valid': tube_valid,
        }
        return tokens, logits, sampled

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

    def build_bev_target(self, bev_logits, gt_3dlanes, y_max):
        batch_size, _, height, width = bev_logits.shape
        target = bev_logits.new_zeros((batch_size, 1, height, width))
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
        valid_y = ((y_values <= float(y_max)) &
                   (y_values >= self.bev_y_min) &
                   (y_values <= self.bev_y_max))
        vis_offset = 5 + gt_anchor_len * 2

        for batch_idx, lanes in enumerate(lane_batches[:batch_size]):
            lanes = lanes[lanes[:, 1] > 0]
            if lanes.numel() == 0:
                continue
            xs = lanes[:, 5:5 + gt_anchor_len]
            vis = (lanes[:, vis_offset:vis_offset + gt_anchor_len] > 0.5)
            valid = vis & valid_y.view(1, -1)
            valid = valid & (xs >= self.bev_x_min) & (xs <= self.bev_x_max)
            if not bool(valid.any()):
                continue
            ys = y_values.view(1, -1).expand_as(xs)
            u_float = (xs[valid] - self.bev_x_min) / (
                self.bev_x_max - self.bev_x_min) * (width - 1)
            v_float = (ys[valid] - self.bev_y_min) / (
                self.bev_y_max - self.bev_y_min) * (height - 1)
            u_idx = u_float.round().long().clamp(0, width - 1)
            v_idx = v_float.round().long().clamp(0, height - 1)
            for du in range(-self.bev_target_radius, self.bev_target_radius + 1):
                for dv in range(-self.bev_target_radius, self.bev_target_radius + 1):
                    uu = (u_idx + du).clamp(0, width - 1)
                    vv = (v_idx + dv).clamp(0, height - 1)
                    target[batch_idx, 0, vv, uu] = 1.
        return target

    @staticmethod
    def unpack_logits(logits):
        if isinstance(logits, dict):
            return (logits.get('image'), logits.get('bev'), logits.get('tube'),
                    logits.get('tube_valid'))
        return logits, None, None, None

    def loss(self, logits_all, gt_3dlanes, gt_project_matrix, project_builder,
             y_max):
        total_loss = None
        total_bev_loss = None
        total_pos = 0.
        total_prob = 0.
        total_bev_pos = 0.
        total_bev_prob = 0.
        total_tube_valid = 0.
        total_tube_center_prob = 0.
        total_tube_max_prob = 0.
        count = 0
        bev_count = 0
        tube_count = 0
        metric_batch_size = None
        for iter_logits in logits_all:
            for logits in iter_logits:
                evidence_logits, bev_logits, tube_logits, tube_valid = (
                    self.unpack_logits(logits))
                if (evidence_logits is None or gt_project_matrix is None or
                        project_builder is None):
                    image_loss = None
                else:
                    feat_size = evidence_logits.shape[-2:]
                    project_matrixes = torch.stack(
                        project_builder(gt_project_matrix, feat_size), dim=0)
                    target = self.build_target(
                        evidence_logits, gt_3dlanes, project_matrixes, y_max)
                    weights = 1. + target * (self.pos_weight - 1.)
                    image_loss = F.binary_cross_entropy_with_logits(
                        evidence_logits.float(), target.float(), reduction='none')
                    image_loss = (image_loss * weights.float()).mean()
                    total_pos += target.sum().detach()
                    total_prob += evidence_logits.sigmoid().mean().detach()
                    metric_batch_size = evidence_logits.shape[0]
                    count += 1
                if image_loss is not None:
                    total_loss = (image_loss if total_loss is None
                                  else total_loss + image_loss)
                if bev_logits is not None:
                    bev_target = self.build_bev_target(bev_logits, gt_3dlanes, y_max)
                    bev_weights = 1. + bev_target * (self.bev_pos_weight - 1.)
                    bev_loss = F.binary_cross_entropy_with_logits(
                        bev_logits.float(), bev_target.float(), reduction='none')
                    bev_loss = (bev_loss * bev_weights.float()).mean()
                    total_bev_loss = (bev_loss if total_bev_loss is None
                                      else total_bev_loss + bev_loss)
                    total_bev_pos += bev_target.sum().detach()
                    total_bev_prob += bev_logits.sigmoid().mean().detach()
                    metric_batch_size = bev_logits.shape[0]
                    bev_count += 1
                if tube_logits is not None and tube_valid is not None:
                    valid = tube_valid.to(tube_logits.dtype)
                    point_valid = valid.max(dim=1, keepdim=True).values
                    tube_prob = tube_logits.sigmoid()
                    total_tube_valid += (valid.sum() / max(float(tube_logits.shape[0]), 1.0)).detach()
                    total_tube_center_prob += (
                        (tube_prob[:, :1] * valid[:, :1]).sum() /
                        valid[:, :1].sum().clamp_min(1.)).detach()
                    total_tube_max_prob += (
                        (tube_prob.max(dim=1, keepdim=True).values * point_valid).sum() /
                        point_valid.sum().clamp_min(1.)).detach()
                    metric_batch_size = tube_logits.shape[0]
                    tube_count += 1
        if count == 0 and bev_count == 0:
            return {}, {}
        if gt_3dlanes is None:
            batch_size = int(metric_batch_size or 1)
        elif isinstance(gt_3dlanes, (list, tuple)):
            batch_size = len(gt_3dlanes)
        else:
            batch_size = int(gt_3dlanes.shape[0])
        losses = {}
        metrics = {}
        if count > 0:
            losses['lane_evidence_loss'] = total_loss * self.loss_weight / count
            metrics['batch_lane_evidence_pos'] = total_pos / max(float(batch_size), 1.0)
            metrics['batch_lane_evidence_prob'] = total_prob / count
        if bev_count > 0:
            losses['bev_evidence_loss'] = (
                total_bev_loss * self.bev_loss_weight / bev_count)
            metrics['batch_bev_evidence_pos'] = (
                total_bev_pos / max(float(batch_size), 1.0))
            metrics['batch_bev_evidence_prob'] = total_bev_prob / bev_count
        if tube_count > 0:
            metrics['batch_tube_valid'] = total_tube_valid / tube_count
            metrics['batch_tube_center_prob'] = total_tube_center_prob / tube_count
            metrics['batch_tube_max_prob'] = total_tube_max_prob / tube_count
        return losses, metrics
