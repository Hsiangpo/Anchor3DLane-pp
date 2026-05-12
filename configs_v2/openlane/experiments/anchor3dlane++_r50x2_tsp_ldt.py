_base_ = './anchor3dlane++_r50x2_ptef.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    to_rgb=True)
input_size = (720, 960)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(input_size[1], input_size[0]), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='MaskGenerate', input_size=input_size),
    dict(type='AngleCalculate'),
    dict(type='LaneFormat'),
    dict(type='Collect', keys=[
        'img', 'img_metas', 'gt_3dlanes', 'gt_project_matrix', 'mask',
        'tsp_teacher', 'tsp_valid']),
]

model = dict(
    lane_evidence_field=dict(
        tsp_enabled=True,
        tsp_channels=4,
        tsp_loss_weight=0.03,
        tsp_occ_weight=1.0,
        tsp_sdf_weight=0.6,
        tsp_z_weight=0.2,
        tsp_quality_weight=0.5,
        tsp_pos_weight=8.0))

data = dict(
    samples_per_gpu=40,
    train=dict(
        type='OpenlaneTSPDataset',
        tsp_cache_dir='lane3d_1000_v1.3/tsp_ldt_cache',
        tsp_channels=4,
        tsp_height=128,
        tsp_width=48,
        pipeline=train_pipeline))

optimizer = dict(type='Adam', lr=1e-5)
lr_config = dict(policy='step', step=[8000], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)

load_from = 'output/openlane/experiments/ptef_r50x2/iter_10000.pth'
work_dir = 'output/openlane/experiments/tsp_ldt_r50x2'
