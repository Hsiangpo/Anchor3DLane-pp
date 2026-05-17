import torch
import torch.nn as nn
import torch.nn.functional as F


class RoadLatticeHead(nn.Module):
    def __init__(self,
                 anchor_len=20,
                 hidden_channels=64,
                 max_delta_x=0.25,
                 max_delta_z=0.1,
                 quality_logit_scale=1.0,
                 geometry_norm=(30.0, 10.0),
                 **kwargs):
        super(RoadLatticeHead, self).__init__()
        self.anchor_len = int(anchor_len)
        self.max_delta_x = float(max_delta_x)
        self.max_delta_z = float(max_delta_z)
        self.quality_logit_scale = float(quality_logit_scale)
        self.geometry_norm = tuple(float(v) for v in geometry_norm)
        hidden_channels = int(hidden_channels)
        if self.anchor_len <= 0:
            raise ValueError('anchor_len must be positive')
        if hidden_channels != 64:
            raise ValueError('MTC-RLH checkpoint expects hidden_channels=64')
        self.loss_weight = float(kwargs.pop('loss_weight', 0.0))
        self.quality_loss_weight = float(kwargs.pop('quality_loss_weight', 0.0))
        self.order_loss_weight = float(kwargs.pop('order_loss_weight', 0.0))
        self.pos_distance = float(kwargs.pop('pos_distance', 0.45))
        self.decoy_distance = float(kwargs.pop('decoy_distance', 1.2))
        self.min_lateral_gap = float(kwargs.pop('min_lateral_gap', 0.05))
        self.anchor_steps = tuple(kwargs.pop('anchor_steps', ()))
        if kwargs:
            raise ValueError(f'Unsupported road_lattice_head options: {sorted(kwargs.keys())}')

        self.delta_head = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.anchor_len * 2))
        self.quality_head = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    def encode_geometry(self, proposals):
        x_norm, z_norm = self.geometry_norm
        x_values = proposals[..., 5:5 + self.anchor_len] / x_norm
        z_values = proposals[..., 5 + self.anchor_len:5 + self.anchor_len * 2] / z_norm
        vis_values = proposals[..., 5 + self.anchor_len * 2:5 + self.anchor_len * 3]
        vis_values = vis_values.clamp(0.0, 1.0)
        stats = torch.stack([
            x_values.mean(dim=-1),
            x_values.std(dim=-1, unbiased=False),
            z_values.mean(dim=-1),
            vis_values.mean(dim=-1),
        ], dim=-1)
        features = torch.cat([x_values, z_values, vis_values, stats], dim=-1)
        if features.shape[-1] != 64:
            raise ValueError('MTC-RLH geometry encoder must produce 64 channels')
        return features

    def forward(self, proposals):
        features = self.encode_geometry(proposals)
        raw_delta = self.delta_head(features).reshape(
            proposals.shape[0], proposals.shape[1], self.anchor_len, 2)
        delta_x = raw_delta[..., 0].tanh() * self.max_delta_x
        delta_z = raw_delta[..., 1].tanh() * self.max_delta_z
        quality_logits = self.quality_head(features).squeeze(-1)

        warped = proposals.clone()
        warped[..., 5:5 + self.anchor_len] += delta_x
        warped[..., 5 + self.anchor_len:5 + self.anchor_len * 2] += delta_z
        metrics = {
            'batch_mtc_delta_x_abs': delta_x.detach().abs().mean(),
            'batch_mtc_delta_z_abs': delta_z.detach().abs().mean(),
            'batch_mtc_quality_logit': quality_logits.detach().mean(),
        }
        return warped, quality_logits, metrics
