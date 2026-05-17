_base_ = './anchor3dlane++_r50x2_ptef.py'

model = dict(
    type='Anchor3DLanePPMTCRLH',
    road_lattice_head=dict(
        enabled=True,
        apply='final',
        hidden_channels=64,
        max_delta_x=0.25,
        max_delta_z=0.1,
        loss_weight=0.04,
        quality_loss_weight=0.04,
        quality_logit_scale=1.0,
        order_loss_weight=0.01,
        pos_distance=0.45,
        decoy_distance=1.2,
        min_lateral_gap=0.05,
        anchor_steps=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                      55, 60, 65, 70, 75, 80, 85, 90, 95, 100]))

data = dict(samples_per_gpu=3)
optimizer = dict(type='Adam', lr=1e-5)
lr_config = dict(policy='step', step=[8000], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)

load_from = 'output/openlane/experiments/ptef_r50x2/iter_10000.pth'
work_dir = 'output/openlane/experiments/mtc_rlh_r50x2'
