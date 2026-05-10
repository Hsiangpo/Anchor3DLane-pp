_base_ = '../anchor3dlane++_r50x2.py'

data = dict(samples_per_gpu=3)

model = dict(
    strip_sample=True,
    strip_offsets=[-1.0, 0.0, 1.0],
    strip_center_bias=2.0,
    strip_apply='final',
)

optimizer = dict(type='Adam', lr=2e-5)
lr_config = dict(policy='step', step=[8000], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=2500, max_keep_ckpts=2)

load_from = 'checkpoints/Openlane/openlane_anchor3dlane++_r50x2.pth'
work_dir = 'output/openlane/experiments/strip_sample_r50x2'
