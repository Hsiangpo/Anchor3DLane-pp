_base_ = './anchor3dlane++_r50x2_lef_dff.py'

model = dict(
    lane_evidence_field=dict(
        enabled=True,
        apply='final',
        trainable=True,
        hidden_channels=64,
        gate_init=-2.1972246,
        detach_sample=True,
        loss_weight=0.02,
        pos_weight=12.0,
        target_radius=1,
        bev_enabled=True,
        bev_loss_weight=0.01,
        bev_pos_weight=8.0,
        bev_target_radius=1,
        bev_height=128,
        bev_width=48,
        bev_x_min=-10.0,
        bev_x_max=10.0,
        bev_y_min=3.0,
        bev_y_max=103.0))

data = dict(samples_per_gpu=3)
optimizer = dict(type='Adam', lr=1e-5)
lr_config = dict(policy='step', step=[8000], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)

load_from = 'output/openlane/experiments/lef_dff_r50x2/iter_10000.pth'
work_dir = 'output/openlane/experiments/pgb_od_r50x2'
