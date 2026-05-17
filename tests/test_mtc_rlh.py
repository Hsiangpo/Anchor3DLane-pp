import os
import unittest

import torch

from mmcv import Config

from mmseg.models import build_lanedetector
from mmseg.models.lane_detector.road_lattice import RoadLatticeHead


class RoadLatticeHeadTest(unittest.TestCase):

    def make_proposals(self, batch_size=2, num_lanes=30, anchor_len=20):
        width = 5 + anchor_len * 3 + 21
        proposals = torch.zeros(batch_size, num_lanes, width)
        proposals[..., 5:5 + anchor_len] = torch.linspace(
            -2.0, 2.0, anchor_len).reshape(1, 1, anchor_len)
        proposals[..., 5 + anchor_len:5 + anchor_len * 2] = 0.2
        proposals[..., 5 + anchor_len * 2:5 + anchor_len * 3] = 1.0
        return proposals

    def test_forward_shape_and_neutral_initialization(self):
        head = RoadLatticeHead(anchor_len=20)
        proposals = self.make_proposals()

        warped, quality_logits, metrics = head(proposals)

        self.assertEqual(tuple(warped.shape), tuple(proposals.shape))
        self.assertEqual(tuple(quality_logits.shape), (2, 30))
        self.assertTrue(torch.allclose(warped, proposals))
        self.assertTrue(torch.allclose(quality_logits, torch.zeros_like(quality_logits)))
        self.assertIn('batch_mtc_delta_x_abs', metrics)

    def test_delta_clamp(self):
        head = RoadLatticeHead(anchor_len=20, max_delta_x=0.25, max_delta_z=0.1)
        proposals = self.make_proposals()
        with torch.no_grad():
            head.delta_head[-1].bias[:20].fill_(100.0)
            head.delta_head[-1].bias[20:].fill_(-100.0)

        warped, _, _ = head(proposals)
        dx = warped[..., 5:25] - proposals[..., 5:25]
        dz = warped[..., 25:45] - proposals[..., 25:45]

        self.assertLessEqual(float(dx.detach().abs().max()), 0.250001)
        self.assertLessEqual(float(dz.detach().abs().max()), 0.100001)

    def test_checkpoint_head_keys_can_load_strictly(self):
        checkpoint_path = os.environ.get('MTC_RLH_CKPT')
        if not checkpoint_path:
            self.skipTest('MTC_RLH_CKPT is not set')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = {
            key.replace('road_lattice_head.', ''): value
            for key, value in checkpoint['state_dict'].items()
            if key.startswith('road_lattice_head.')
        }

        head = RoadLatticeHead(anchor_len=20)
        load_result = head.load_state_dict(state_dict, strict=True)

        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        self.assertEqual(len(state_dict), 10)


class Anchor3DLanePPMTCRLHTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = Config.fromfile(
            'configs_v2/openlane/experiments/anchor3dlane++_r50x2_mtc_rlh.py')
        cls.model = build_lanedetector(cfg.model)

    def test_final_only_gate(self):
        model = self.model

        self.assertFalse(model.use_road_lattice_head(0, 0))
        self.assertTrue(model.use_road_lattice_head(
            model.feat_num - 1, model.iter_reg - 1))

    def test_quality_logits_inject_accept_logits(self):
        model = self.model
        proposals = torch.zeros(2, model.anchor_num,
                                5 + model.anchor_len * 3 + model.num_category)
        accept_logits = torch.zeros(2, model.anchor_num)
        with torch.no_grad():
            model.road_lattice_head.quality_head[-1].weight.zero_()
            model.road_lattice_head.quality_head[-1].bias.fill_(2.0)

        _, injected = model.apply_road_lattice(proposals, accept_logits)

        self.assertTrue(torch.allclose(injected, torch.full_like(injected, 2.0)))

    def test_frozen_trainable_prefixes_include_road_lattice(self):
        prefixes = self.model.frozen_base_trainable_prefixes()

        self.assertIn('road_lattice_head.', prefixes)

    def test_full_mtc_checkpoint_consumes_road_lattice_keys(self):
        checkpoint_path = os.environ.get('MTC_RLH_CKPT')
        if not checkpoint_path:
            self.skipTest('MTC_RLH_CKPT is not set')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        load_result = self.model.load_state_dict(
            checkpoint['state_dict'], strict=False)
        unexpected = sorted(load_result.unexpected_keys)

        self.assertEqual(load_result.missing_keys, [])
        self.assertNotIn('road_lattice_head.delta_head.0.weight', unexpected)
        self.assertEqual(unexpected, ['lane_evidence_field.tube_offsets'])


if __name__ == '__main__':
    unittest.main()
