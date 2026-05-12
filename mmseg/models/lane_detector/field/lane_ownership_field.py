import math

import torch
import torch.nn as nn


class LaneOwnershipField(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels=None,
                 relation_hidden=64,
                 gate_init=-2.1972246,
                 use_geometry=True,
                 detach_geometry=True):
        super(LaneOwnershipField, self).__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels or channels)
        self.use_geometry = bool(use_geometry)
        self.detach_geometry = bool(detach_geometry)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

        self.point_norm = nn.LayerNorm(channels)
        self.lane_norm = nn.LayerNorm(channels)
        self.query_proj = nn.Linear(channels, self.hidden_channels)
        self.key_proj = nn.Linear(channels, self.hidden_channels)
        self.value_proj = nn.Linear(channels, channels)
        self.out_norm = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, channels)
        if self.use_geometry:
            self.geometry_proj = nn.Sequential(
                nn.Linear(3, channels),
                nn.GELU(),
                nn.Linear(channels, channels))
            self.relation_proj = nn.Sequential(
                nn.Linear(3, relation_hidden),
                nn.GELU(),
                nn.Linear(relation_hidden, 1))
            nn.init.zeros_(self.relation_proj[-1].weight)
            nn.init.zeros_(self.relation_proj[-1].bias)
        else:
            self.geometry_proj = None
            self.relation_proj = None
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def add_geometry(self, point_tokens, geometry):
        if self.geometry_proj is None or geometry is None:
            return point_tokens
        if self.detach_geometry:
            geometry = geometry.detach()
        return point_tokens + self.geometry_proj(geometry.to(point_tokens.dtype))

    @staticmethod
    def masked_lane_pool(point_tokens, valid_mask):
        if valid_mask is None:
            return point_tokens.mean(dim=2)
        weights = valid_mask.to(point_tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (point_tokens * weights).sum(dim=2) / denom

    def relation_bias(self, geometry, valid_mask):
        if self.relation_proj is None or geometry is None:
            return None
        if self.detach_geometry:
            geometry = geometry.detach()
        lane_geometry = self.masked_lane_pool(geometry.to(dtype=torch.float32), valid_mask)
        relation = (lane_geometry[:, :, None, :] - lane_geometry[:, None, :, :]).abs()
        return self.relation_proj(relation.to(dtype=geometry.dtype)).squeeze(-1)

    def forward(self, features, valid_mask=None, geometry=None):
        batch_size, channels, num_lanes, num_points = features.shape
        point_tokens = features.permute(0, 2, 3, 1).contiguous()
        point_tokens = self.add_geometry(point_tokens, geometry)

        lane_tokens = self.masked_lane_pool(self.point_norm(point_tokens), valid_mask)
        lane_tokens = self.lane_norm(lane_tokens)
        queries = self.query_proj(lane_tokens)
        keys = self.key_proj(lane_tokens)
        owner_logits = torch.matmul(queries, keys.transpose(1, 2))
        owner_logits = owner_logits / math.sqrt(float(self.hidden_channels))
        owner_logits = 0.5 * (owner_logits + owner_logits.transpose(1, 2))
        relation_bias = self.relation_bias(geometry, valid_mask)
        if relation_bias is not None:
            owner_logits = owner_logits + relation_bias

        attention_logits = owner_logits
        if valid_mask is not None:
            lane_valid = valid_mask.any(dim=2)
            invalid = ~lane_valid
            attention_logits = attention_logits.masked_fill(
                invalid[:, None, :], torch.finfo(attention_logits.dtype).min)
            all_invalid = invalid.all(dim=1)
            if bool(all_invalid.any()):
                attention_logits[all_invalid] = owner_logits[all_invalid]
        owner_weights = attention_logits.softmax(dim=-1)
        lane_context = torch.matmul(owner_weights, self.value_proj(lane_tokens))
        context = lane_context[:, :, None, :].expand(batch_size, num_lanes,
                                                     num_points, channels)
        residual = self.out_proj(self.out_norm(point_tokens + context))
        residual = residual.permute(0, 3, 1, 2).contiguous()
        if valid_mask is not None:
            residual = residual * valid_mask[:, None].to(residual.dtype)
        output = features + self.gate_logit.sigmoid().to(features.dtype) * residual
        return output, owner_logits
