from ...builder import LANENET2S
from ..anchor_3dlane_pp import Anchor3DLanePP
from .road_lattice_head import RoadLatticeHead


@LANENET2S.register_module()
class Anchor3DLanePPMTCRLH(Anchor3DLanePP):
    def __init__(self, road_lattice_head=None, **kwargs):
        super(Anchor3DLanePPMTCRLH, self).__init__(**kwargs)
        self.build_road_lattice_head(road_lattice_head)
        if getattr(self, 'accept_calib_freeze_base', False):
            self.freeze_base_for_accept_calib()

    def build_road_lattice_head(self, road_lattice_head):
        self.road_lattice_head = None
        self.road_lattice_head_enabled = False
        self.road_lattice_head_apply = 'final'
        if road_lattice_head is None:
            return
        cfg = road_lattice_head.copy()
        if not cfg.pop('enabled', True):
            return
        self.road_lattice_head_enabled = True
        self.road_lattice_head_apply = cfg.pop('apply', 'final')
        cfg.setdefault('anchor_len', self.anchor_len)
        cfg.setdefault('geometry_norm', (self.x_norm, self.z_norm))
        self.road_lattice_head = RoadLatticeHead(**cfg)

    def frozen_base_trainable_prefixes(self):
        prefixes = list(super(Anchor3DLanePPMTCRLH, self).frozen_base_trainable_prefixes())
        if getattr(self, 'road_lattice_head_enabled', False):
            prefixes.append('road_lattice_head.')
        return tuple(prefixes)

    def train(self, mode=True):
        super(Anchor3DLanePPMTCRLH, self).train(mode)
        if mode and getattr(self, 'accept_calib_freeze_base', False):
            if getattr(self, 'road_lattice_head_enabled', False):
                self.road_lattice_head.train(True)
        return self

    def use_road_lattice_head(self, feat_idx, iter_idx):
        if not getattr(self, 'road_lattice_head_enabled', False):
            return False
        if self.road_lattice_head_apply == 'final':
            return feat_idx == self.feat_num - 1 and iter_idx == self.iter_reg - 1
        if self.road_lattice_head_apply == 'all':
            return True
        raise ValueError(f'Unsupported road_lattice_head_apply: {self.road_lattice_head_apply}')

    def apply_road_lattice(self, reg_proposals, accept_logits):
        warped, quality_logits, _ = self.road_lattice_head(reg_proposals)
        quality_logits = quality_logits * self.road_lattice_head.quality_logit_scale
        if accept_logits is None:
            accept_logits = quality_logits
        else:
            accept_logits = accept_logits + quality_logits.to(accept_logits.dtype)
        return warped, accept_logits

    def get_proposals(self, project_matrixes, anchor_feat, feat_idx, proposals_prev,
                      feat_size, iter_idx, reg_prior=False):
        reg_proposals, cur_anchors, accept_logits, ownership_logits, evidence_logits = (
            super(Anchor3DLanePPMTCRLH, self).get_proposals(
                project_matrixes, anchor_feat, feat_idx, proposals_prev,
                feat_size, iter_idx, reg_prior))
        if self.use_road_lattice_head(feat_idx, iter_idx):
            reg_proposals, accept_logits = self.apply_road_lattice(
                reg_proposals, accept_logits)
        return reg_proposals, cur_anchors, accept_logits, ownership_logits, evidence_logits
