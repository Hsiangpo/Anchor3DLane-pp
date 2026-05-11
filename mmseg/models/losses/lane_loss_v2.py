# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2025 TuSimple
# @Time    : 2025/05/07
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_assigner
from .kornia_focal import FocalLoss
from .utils import get_class_weight, weight_reduce_loss
from .matcher import HungarianMatcher
from .focal_loss import FocalLossSigmoid

@LOSSES.register_module()
class LaneLossV2(nn.Module):
    def __init__(self,
                 focal_alpha=0.25,
                 focal_gamma=2.,
                 anchor_len=10,
                 gt_anchor_len=200,
                 anchor_steps=[],
                 weighted_ce=False,
                 use_sigmoid=False,
                 loss_weights=None,
                 anchor_assign=False,
                 delta = 0.2,
                 ds = 10,
                 curve_geom_axes=('x',),
                 curve_slope_weight=1.0,
                 curve_curv_weight=0.4,
                 accept_vis_thresh=0.7,
                 accept_dist_thresh=1.5,
                 accept_cover_thresh=0.75,
                 accept_purity_thresh=0.75,
                 accept_neg_cover_thresh=0.5,
                 accept_neg_purity_thresh=0.5,
                 accept_hard_score_thresh=0.35,
                 accept_pos_weight=1.0,
                 accept_neg_weight=1.0,
                 dup_rank_margin=0.1,
                 dup_rank_max_pairs=4,
                 frt_enable=False,
                 frt_min_width=0.6,
                 frt_default_width=2.0,
                 frt_max_width=2.5,
                 frt_slack=0.15,
                 frt_dirty_thresh=0.10,
                 frt_rank_weight=0.25,
                 assign_cfg=None):
        super(LaneLossV2, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.anchor_len = anchor_len
        self.anchor_steps = np.array(anchor_steps) - 1
        self.gt_anchor_len = gt_anchor_len
        self.use_sigmoid = use_sigmoid

        self.weighted_ce = weighted_ce
        self.loss_weights = loss_weights
        self.anchor_assign = anchor_assign
        self.lane_prior = 'reg_losses_prior' in loss_weights.keys()
        self.consist = ('consist_losses' in loss_weights.keys())
        self.curve_geom = ('curve_geom_losses' in loss_weights.keys())
        self.accept_calib = ('accept_calib_losses' in loss_weights.keys())
        self.dup_rank = ('dup_rank_losses' in loss_weights.keys())
        self.curve_geom_axes = tuple(curve_geom_axes)
        assert len(self.anchor_steps) == self.anchor_len
        assert set(self.curve_geom_axes).issubset({'x', 'z'})
        self.curve_slope_weight = float(curve_slope_weight)
        self.curve_curv_weight = float(curve_curv_weight)
        self.accept_vis_thresh = float(accept_vis_thresh)
        self.accept_dist_thresh = float(accept_dist_thresh)
        self.accept_cover_thresh = float(accept_cover_thresh)
        self.accept_purity_thresh = float(accept_purity_thresh)
        self.accept_neg_cover_thresh = float(accept_neg_cover_thresh)
        self.accept_neg_purity_thresh = float(accept_neg_purity_thresh)
        self.accept_hard_score_thresh = float(accept_hard_score_thresh)
        self.accept_pos_weight = float(accept_pos_weight)
        self.accept_neg_weight = float(accept_neg_weight)
        self.dup_rank_margin = float(dup_rank_margin)
        self.dup_rank_max_pairs = int(dup_rank_max_pairs)
        self.frt_enable = bool(frt_enable)
        self.frt_min_width = float(frt_min_width)
        self.frt_default_width = float(frt_default_width)
        self.frt_max_width = float(frt_max_width)
        self.frt_slack = float(frt_slack)
        self.frt_dirty_thresh = float(frt_dirty_thresh)
        self.frt_rank_weight = float(frt_rank_weight)
        self.register_buffer('anchor_y_steps_tensor',
                             torch.as_tensor(anchor_steps, dtype=torch.float32),
                             persistent=False)
        self.fp16_enabled = False
        self.delta = delta
        self.ds = ds
        self.assigner = HungarianMatcher(anchor_len=self.anchor_len, **assign_cfg)

    def zero_loss(self, tensor):
        return tensor.sum() * 0

    def masked_smooth_l1(self, pred, target, mask):
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        mask = mask.to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def axis_curve_geom_loss(self, pred, target, vis):
        zero = self.zero_loss(pred)
        if pred.shape[1] < 2:
            return zero

        y_steps = self.anchor_y_steps_tensor.to(device=pred.device, dtype=pred.dtype)
        dy = (y_steps[1:] - y_steps[:-1]).clamp_min(1e-6)
        pred_slope = (pred[:, 1:] - pred[:, :-1]) / dy.view(1, -1)
        target_slope = (target[:, 1:] - target[:, :-1]) / dy.view(1, -1)
        slope_mask = (vis[:, 1:] > 0.5) & (vis[:, :-1] > 0.5)

        total = zero
        if self.curve_slope_weight > 0:
            total = total + self.curve_slope_weight * self.masked_smooth_l1(
                pred_slope, target_slope, slope_mask)

        if self.curve_curv_weight > 0 and pred.shape[1] >= 3:
            mid_dy = ((dy[1:] + dy[:-1]) * 0.5).clamp_min(1e-6)
            pred_curv = (pred_slope[:, 1:] - pred_slope[:, :-1]) / mid_dy.view(1, -1)
            target_curv = (target_slope[:, 1:] - target_slope[:, :-1]) / mid_dy.view(1, -1)
            curv_mask = slope_mask[:, 1:] & slope_mask[:, :-1]
            total = total + self.curve_curv_weight * self.masked_smooth_l1(
                pred_curv, target_curv, curv_mask)

        return total

    def curve_geom_loss(self, x_pred, z_pred, x_target, z_target, vis_target):
        losses = []
        if 'x' in self.curve_geom_axes:
            losses.append(self.axis_curve_geom_loss(x_pred, x_target, vis_target))
        if 'z' in self.curve_geom_axes:
            losses.append(self.axis_curve_geom_loss(z_pred, z_target, vis_target))
        if not losses:
            return self.zero_loss(x_pred)
        return sum(losses) / len(losses)

    def proposal_scores(self, proposals):
        cls_logits = proposals[:, 5 + self.anchor_len * 3:]
        if self.use_sigmoid:
            return cls_logits.sigmoid().max(dim=1)[0]
        return 1 - cls_logits.softmax(dim=1)[:, 0]

    def refined_visibility_mask(self, vis_pred):
        visible = vis_pred >= self.accept_vis_thresh
        flag_l = visible.cumsum(dim=1)
        flag_r = visible.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        return (flag_l > 0) & (flag_r > 0)

    def build_ribbon_widths(self, x_target, vis_target):
        target_count, step_count = x_target.shape
        widths = x_target.new_full(x_target.shape, self.frt_default_width)
        has_neighbor = torch.zeros(x_target.shape, dtype=torch.bool, device=x_target.device)
        if target_count < 2:
            return widths.clamp(self.frt_min_width, self.frt_max_width), has_neighbor

        for step_idx in range(step_count):
            step_vis = vis_target[:, step_idx]
            if int(step_vis.sum().item()) < 2:
                continue
            step_x = x_target[:, step_idx]
            gap = (step_x[:, None] - step_x[None, :]).abs()
            invalid = (~step_vis[:, None]) | (~step_vis[None, :])
            gap = gap.masked_fill(invalid, 1e6)
            gap.fill_diagonal_(1e6)
            nearest = gap.min(dim=1)[0]
            reliable = step_vis & (nearest < 1e5)
            widths[reliable, step_idx] = nearest[reliable] * 0.5
            has_neighbor[reliable, step_idx] = True
        return widths.clamp(self.frt_min_width, self.frt_max_width), has_neighbor

    def ribbon_pair_violation(self, x_pred, pred_mask, x_target, vis_target):
        if x_target.shape[0] == 0:
            return x_pred.new_zeros((x_pred.shape[0], 0))
        widths, has_neighbor = self.build_ribbon_widths(x_target, vis_target)
        pair_mask = pred_mask[:, None, :] & vis_target[None, :, :] & has_neighbor[None, :, :]
        excess = (x_pred[:, None, :] - x_target[None, :, :]).abs()
        excess = F.relu(excess - widths[None, :, :] - self.frt_slack)
        denom = pair_mask.to(excess.dtype).sum(dim=2).clamp_min(1.0)
        return (excess * pair_mask.to(excess.dtype)).sum(dim=2) / denom

    def acceptance_labels(self, proposals, target):
        num_props = proposals.shape[0]
        device = proposals.device
        labels = proposals.new_zeros(num_props)
        valid = torch.zeros(num_props, dtype=torch.bool, device=device)
        accepted = torch.zeros(num_props, dtype=torch.bool, device=device)
        best_target = torch.full((num_props,), -1, dtype=torch.long, device=device)
        best_cost = proposals.new_full((num_props,), 1e6)
        base_scores = self.proposal_scores(proposals)

        if target.shape[0] == 0:
            hard_neg = base_scores > self.accept_hard_score_thresh
            valid[hard_neg] = True
            return labels, valid, accepted, best_target, best_cost, labels.new_tensor(0.)

        x_pred = proposals[:, 5:5+self.anchor_len]
        z_pred = proposals[:, 5+self.anchor_len:5+self.anchor_len*2]
        vis_pred = proposals[:, 5+self.anchor_len*2:5+self.anchor_len*3]
        pred_mask = self.refined_visibility_mask(vis_pred)
        x_target = target[:, 5:5+self.anchor_len]
        z_target = target[:, 5+self.anchor_len:5+self.anchor_len*2]
        vis_target = target[:, 5+self.anchor_len*2:5+self.anchor_len*3] > 0.5

        dist = ((x_pred[:, None, :] - x_target[None, :, :]) ** 2 +
                (z_pred[:, None, :] - z_target[None, :, :]) ** 2).clamp_min(1e-8).sqrt()
        pair_mask = pred_mask[:, None, :] & vis_target[None, :, :]
        hit = (dist <= self.accept_dist_thresh) & pair_mask
        hit_count = hit.to(proposals.dtype).sum(dim=2)
        gt_visible = vis_target.to(proposals.dtype).sum(dim=1).clamp_min(1.0)
        pred_visible = pred_mask.to(proposals.dtype).sum(dim=1).clamp_min(1.0)
        cover = hit_count / gt_visible.view(1, -1)
        purity = hit_count / pred_visible.view(-1, 1)
        pair_accept = (cover >= self.accept_cover_thresh) & (purity >= self.accept_purity_thresh)
        if self.frt_enable:
            ribbon_violation = self.ribbon_pair_violation(
                x_pred, pred_mask, x_target, vis_target)
            pair_clean = ribbon_violation <= self.frt_dirty_thresh
            pair_accept_clean = pair_accept & pair_clean
            frt_dirty = (pair_accept & ~pair_clean).any(dim=1)
        else:
            ribbon_violation = dist.new_zeros(pair_accept.shape)
            pair_accept_clean = pair_accept
            frt_dirty = torch.zeros(num_props, dtype=torch.bool, device=device)

        hit_cost = (dist * hit.to(dist.dtype)).sum(dim=2) / hit_count.clamp_min(1.0)
        if self.frt_enable:
            hit_cost = hit_cost + self.frt_rank_weight * ribbon_violation
        pair_cost = torch.where(pair_accept_clean, hit_cost, dist.new_full(hit_cost.shape, 1e6))
        min_cost, min_idx = pair_cost.min(dim=1)
        accepted = pair_accept_clean.any(dim=1)
        labels[accepted] = 1.
        valid[accepted] = True
        best_target[accepted] = min_idx[accepted]
        best_cost[accepted] = min_cost[accepted]

        max_cover = cover.max(dim=1)[0]
        max_purity = purity.max(dim=1)[0]
        hard_neg = ((base_scores > self.accept_hard_score_thresh) &
                    (max_cover < self.accept_neg_cover_thresh) &
                    (max_purity < self.accept_neg_purity_thresh))
        valid[hard_neg] = True
        dirty_count = (frt_dirty & ~accepted).to(labels.dtype).sum()
        return labels, valid, accepted, best_target, best_cost, dirty_count

    def duplicate_rank_loss(self, accept_logit, accepted, best_target, best_cost, target_count):
        rank_losses = []
        pair_count = accept_logit.new_tensor(0.)
        for tgt_idx in range(target_count):
            lane_mask = accepted & (best_target == tgt_idx)
            if int(lane_mask.sum().item()) < 2:
                continue
            inds = lane_mask.nonzero(as_tuple=False).flatten()
            order = best_cost[inds].argsort()
            best = inds[order[0]]
            others = inds[order[1:1+self.dup_rank_max_pairs]]
            if others.numel() == 0:
                continue
            rank_losses.append(F.relu(self.dup_rank_margin + accept_logit[others] - accept_logit[best]).mean())
            pair_count = pair_count + accept_logit.new_tensor(float(others.numel()))
        if not rank_losses:
            return self.zero_loss(accept_logit), pair_count
        return sum(rank_losses) / len(rank_losses), pair_count

    def accept_calibration_loss(self, proposals, target, accept_logit):
        if accept_logit is None:
            zero = self.zero_loss(proposals)
            return zero, zero, zero, zero, zero, zero, zero

        accept_logit = accept_logit.reshape(-1)
        with torch.no_grad():
            labels, valid, accepted, best_target, best_cost, frt_dirty = self.acceptance_labels(
                proposals.detach(), target.detach())
        if bool(valid.any()):
            weights = torch.where(labels[valid] > 0.5,
                                  accept_logit.new_tensor(self.accept_pos_weight),
                                  accept_logit.new_tensor(self.accept_neg_weight))
            bce = F.binary_cross_entropy_with_logits(
                accept_logit[valid], labels[valid], reduction='none')
            accept_loss = (bce * weights).sum() / weights.sum().clamp_min(1.0)
        else:
            accept_loss = self.zero_loss(accept_logit)

        if self.dup_rank:
            rank_loss, rank_pairs = self.duplicate_rank_loss(
                accept_logit, accepted, best_target, best_cost, target.shape[0])
        else:
            rank_loss = self.zero_loss(accept_logit)
            rank_pairs = accept_logit.new_tensor(0.)

        pos_count = accepted.to(accept_logit.dtype).sum()
        neg_count = (valid & ~accepted).to(accept_logit.dtype).sum()
        ignore_count = (~valid).to(accept_logit.dtype).sum()
        return accept_loss, rank_loss, pos_count, neg_count, ignore_count, rank_pairs, frt_dirty

    def init_accept_stats(self, proposals):
        zero = proposals.new_tensor(0.)
        return dict(loss=zero, rank_loss=zero, pos=zero, neg=zero,
                    ignore=zero, rank_pairs=zero, frt_dirty=zero)

    def update_accept_stats(self, stats, proposals, target, accept_logit):
        if stats is None:
            return
        accept_loss, rank_loss, pos_count, neg_count, ignore_count, rank_pairs, frt_dirty = \
            self.accept_calibration_loss(proposals, target, accept_logit)
        stats['loss'] = stats['loss'] + accept_loss
        stats['rank_loss'] = stats['rank_loss'] + rank_loss
        stats['pos'] = stats['pos'] + pos_count
        stats['neg'] = stats['neg'] + neg_count
        stats['ignore'] = stats['ignore'] + ignore_count
        stats['rank_pairs'] = stats['rank_pairs'] + rank_pairs
        stats['frt_dirty'] = stats['frt_dirty'] + frt_dirty
        
    def forward(self, proposals_list, targets):
        if self.use_sigmoid:
            focal_loss = FocalLossSigmoid(alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='none')
        else:
            focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        cls_losses = 0
        reg_losses_x = 0
        reg_losses_z = 0
        reg_losses_vis = 0
        if self.lane_prior:
            reg_losses_prior = 0
        if self.consist:
            consist_losses = 0 
        if self.curve_geom:
            curve_geom_losses = 0
        accept_stats = self.init_accept_stats(proposals_list[0][0]) if self.accept_calib else None
        valid_imgs = len(targets)
        total_positives = 0
        total_negatives = 0
        for idx in range(len(proposals_list)):
            proposals = proposals_list[idx][0]
            num_clses = proposals.shape[1] - 5 - self.anchor_len * 3
            anchors = proposals_list[idx][1]
            accept_logit = proposals_list[idx][2] if len(proposals_list[idx]) > 2 else None
            target = targets[idx]
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] > 0]   # [N, 605]
            if len(target) == 0:
                empty_target = proposals.new_zeros((0, 5 + self.anchor_len * 3))
                self.update_accept_stats(accept_stats, proposals, empty_target, accept_logit)
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, 5+self.anchor_len*3:]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.lane_prior:
                    reg_losses_prior += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.consist:
                    consist_losses += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.curve_geom:
                    curve_geom_losses += self.zero_loss(cls_pred)
                continue
            # Gradients are also not necessary for the positive & negative matching
            x_indices = torch.tensor(self.anchor_steps).to(torch.long).to(target.device) + 5
            z_indices = x_indices + self.gt_anchor_len
            vis_indices = x_indices + self.gt_anchor_len * 2
            x_target = target.index_select(1, x_indices)
            z_target = target.index_select(1, z_indices)
            vis_target = target.index_select(1, vis_indices)   # [N, 10]
            target = torch.cat((target[:, :5], x_target, z_target, vis_target), dim=1)   # [N, 35]
            self.update_accept_stats(accept_stats, proposals, target, accept_logit)
            with torch.no_grad():
                if self.anchor_assign:
                    anchor_assign = torch.cat([anchors, proposals[:, 65:]], 1)
                    indices_src, indices_tgt = self.assigner(anchor_assign, target, use_sigmoid=self.use_sigmoid)
                else:
                    indices_src, indices_tgt = self.assigner(proposals, target, use_sigmoid=self.use_sigmoid)

            positives = proposals[indices_src]
            num_positives = len(positives)
            total_positives += num_positives
            negatives_mask = torch.ones(proposals.shape[0], dtype=torch.bool, device=proposals.device)
            negatives_mask[indices_src] = False
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)
            total_negatives += num_negatives

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0  # avoid dividing zeros
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.lane_prior:
                    reg_losses_prior += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.consist:
                    consist_losses += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.curve_geom:
                    curve_geom_losses += self.zero_loss(cls_pred)
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = target[indices_tgt][:, 1]
            cls_pred = all_proposals[:, 5+self.anchor_len*3:]  # [N, C]

            # Regression targets
            x_pred = positives[:, 5:5+self.anchor_len]   # [N, l]
            z_pred = positives[:, 5+self.anchor_len:5+self.anchor_len*2]   # [N, l]
            vis_pred = positives[:, 5+self.anchor_len*2:5+self.anchor_len*3]  # [N, l]
            prior_pred = positives[:, 2:5]

            with torch.no_grad():
                target = target[indices_tgt]
                x_target = target[:, 5:5+self.anchor_len]
                z_target = target[:, 5+self.anchor_len:5+self.anchor_len*2]
                vis_target = target[:, 5+self.anchor_len*2:5+self.anchor_len*3]
                prior_target = target[:, 2:5]
                valid_points = vis_target.sum()

            # Loss calc
            reg_loss_x = smooth_l1_loss(x_pred, x_target)
            reg_loss_x = reg_loss_x * vis_target  #  * scores # [N, l]
            reg_losses_x += reg_loss_x.sum() / valid_points
            reg_loss_z = smooth_l1_loss(z_pred, z_target)
            reg_loss_z = reg_loss_z * vis_target # * scores
            reg_losses_z += reg_loss_z.sum() / valid_points
            reg_loss_vis = smooth_l1_loss(vis_pred, vis_target)
            reg_losses_vis += reg_loss_vis.mean()
            cls_loss = focal_loss(cls_pred, cls_target)
            if self.lane_prior:
                prior_loss = smooth_l1_loss(prior_pred, prior_target).mean()
                reg_losses_prior += prior_loss
            if self.consist:
                xr = x_pred.unsqueeze(1)  # [N, 1, l]
                xl = x_pred.unsqueeze(0)  # [1, N, l]
                cos = x_pred.new_zeros(x_pred.shape[0], self.anchor_len - 1)
                cos = 5 / ((x_pred[..., 1:] - x_pred[..., :-1]) ** 2 + 25) ** 0.5
                cos = torch.cat([cos[..., 0:1].clone(), cos], -1)
                distance = (xr - xl) * cos.detach()  # [N, N, l]
                distance_mean = distance.mean(-1, keepdims=True)  # [N, N, 1]
                distance_delta = (distance - distance_mean).abs()[..., self.ds:]  # [N, N, l']
                distance_mask = distance_delta < self.delta
                consist_loss = (distance_delta * distance_mask).sum(-1) / (distance_mask.sum(-1) + 1e-6)
                consist_loss = consist_loss.triu(diagonal=1)
                consist_losses += consist_loss.sum() / (num_positives * (num_positives - 1) / 2 + 1e-6)
            if self.curve_geom:
                curve_geom_losses += self.curve_geom_loss(
                    x_pred, z_pred, x_target, z_target, vis_target)
            
            if self.use_sigmoid:
                cls_losses += cls_loss.sum() / num_positives / num_clses
            else:
                cls_losses += cls_loss.sum() / num_positives

        # Batch mean
        cls_losses = cls_losses / valid_imgs
        reg_losses_x = reg_losses_x / valid_imgs
        reg_losses_z = reg_losses_z / valid_imgs
        reg_losses_vis = reg_losses_vis / valid_imgs

        losses = {'cls_loss': cls_losses, 'reg_losses_x': reg_losses_x, 'reg_losses_z': reg_losses_z, 'reg_losses_vis': reg_losses_vis}

        if self.lane_prior:
            reg_losses_prior = reg_losses_prior / valid_imgs
            losses['reg_losses_prior'] = reg_losses_prior

        if self.consist:
            consist_losses = consist_losses / valid_imgs
            losses['consist_losses'] = consist_losses
        if self.curve_geom:
            curve_geom_losses = curve_geom_losses / valid_imgs
            losses['curve_geom_losses'] = curve_geom_losses
        if self.accept_calib:
            accept_calib_losses = accept_stats['loss'] / valid_imgs
            losses['accept_calib_losses'] = accept_calib_losses
        if self.dup_rank:
            dup_rank_losses = accept_stats['rank_loss'] / valid_imgs
            losses['dup_rank_losses'] = dup_rank_losses

        for k in losses.keys():
            losses[k] = losses[k] * self.loss_weights[k]

        bs = len(proposals_list)
        result = {'losses':losses, 'batch_positives': total_positives / bs, 'batch_negatives': total_negatives / bs}
        if self.accept_calib:
            result['batch_accept_pos'] = accept_stats['pos'] / bs
            result['batch_accept_neg'] = accept_stats['neg'] / bs
            result['batch_accept_ignore'] = accept_stats['ignore'] / bs
        if self.dup_rank:
            result['batch_dup_rank_pairs'] = accept_stats['rank_pairs'] / bs
        if self.accept_calib and self.frt_enable:
            result['batch_frt_dirty'] = accept_stats['frt_dirty'] / bs
        return result
