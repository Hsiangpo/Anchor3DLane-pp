import torch
import torch.nn as nn


class LanePointFieldTransformer(nn.Module):
    def __init__(self,
                 channels,
                 num_heads=4,
                 ffn_ratio=2.0,
                 dropout=0.0,
                 gate_init=-2.1972246,
                 use_geometry=True,
                 detach_geometry=True):
        super(LanePointFieldTransformer, self).__init__()
        if channels % num_heads != 0:
            raise ValueError('channels must be divisible by num_heads')
        self.channels = int(channels)
        self.use_geometry = bool(use_geometry)
        self.detach_geometry = bool(detach_geometry)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

        self.lane_norm = nn.LayerNorm(channels)
        self.lateral_norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.out_norm = nn.LayerNorm(channels)
        self.lane_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True)
        self.lateral_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True)
        hidden = max(int(channels * ffn_ratio), channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels))
        self.out_proj = nn.Linear(channels, channels)
        if self.use_geometry:
            self.geometry_proj = nn.Sequential(
                nn.Linear(3, channels),
                nn.GELU(),
                nn.Linear(channels, channels))
        else:
            self.geometry_proj = None
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def safe_key_padding_mask(valid_mask):
        if valid_mask is None:
            return None
        invalid = ~valid_mask
        flat_invalid = invalid.reshape(-1, invalid.shape[-1]).clone()
        all_invalid = flat_invalid.all(dim=1)
        if bool(all_invalid.any()):
            flat_invalid[all_invalid] = False
        return flat_invalid

    def add_geometry(self, point_tokens, geometry):
        if self.geometry_proj is None or geometry is None:
            return point_tokens
        if self.detach_geometry:
            geometry = geometry.detach()
        return point_tokens + self.geometry_proj(geometry.to(point_tokens.dtype))

    def forward(self, features, valid_mask=None, geometry=None):
        batch_size, channels, num_lanes, num_points = features.shape
        point_tokens = features.permute(0, 2, 3, 1).contiguous()
        point_tokens = self.add_geometry(point_tokens, geometry)

        lane_tokens = self.lane_norm(point_tokens).reshape(
            batch_size * num_lanes, num_points, channels)
        lane_mask = None
        if valid_mask is not None:
            lane_mask = self.safe_key_padding_mask(valid_mask.reshape(
                batch_size * num_lanes, num_points))
        lane_out = self.lane_attn(
            lane_tokens, lane_tokens, lane_tokens,
            key_padding_mask=lane_mask, need_weights=False)[0]
        lane_out = lane_out.reshape(batch_size, num_lanes, num_points, channels)
        point_tokens = point_tokens + lane_out

        lateral_tokens = self.lateral_norm(point_tokens).permute(
            0, 2, 1, 3).contiguous().reshape(batch_size * num_points,
                                             num_lanes, channels)
        lateral_mask = None
        if valid_mask is not None:
            lateral_mask = self.safe_key_padding_mask(valid_mask.permute(
                0, 2, 1).contiguous().reshape(batch_size * num_points, num_lanes))
        lateral_out = self.lateral_attn(
            lateral_tokens, lateral_tokens, lateral_tokens,
            key_padding_mask=lateral_mask, need_weights=False)[0]
        lateral_out = lateral_out.reshape(batch_size, num_points, num_lanes,
                                          channels).permute(0, 2, 1, 3)
        point_tokens = point_tokens + lateral_out
        point_tokens = point_tokens + self.ffn(self.ffn_norm(point_tokens))

        residual = self.out_proj(self.out_norm(point_tokens))
        residual = residual.permute(0, 3, 1, 2).contiguous()
        if valid_mask is not None:
            residual = residual * valid_mask[:, None].to(residual.dtype)
        return features + self.gate_logit.sigmoid().to(features.dtype) * residual
