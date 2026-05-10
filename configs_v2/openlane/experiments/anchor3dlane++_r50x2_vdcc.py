_base_ = './anchor3dlane++_r50x2_strip_sample.py'

anchor_y_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                  55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
anchor_len = len(anchor_y_steps)

lane_loss_prior = dict(
    type='LaneLossV2',
    loss_weights=dict(cls_loss=1,
                      reg_losses_x=1,
                      reg_losses_z=1,
                      reg_losses_vis=1,
                      reg_losses_prior=20,
                      consist_losses=0.1),
    assign_cfg=dict(
        cost_class=3.0,
        cost_dist=1.0,
    ),
    anchor_len=anchor_len,
    anchor_steps=anchor_y_steps,
    anchor_assign=False,
    delta=0.1,
    ds=10)

lane_loss = dict(
    type='LaneLossV2',
    loss_weights=dict(cls_loss=1,
                      reg_losses_x=1,
                      reg_losses_z=1,
                      reg_losses_vis=1),
    assign_cfg=dict(
        cost_class=3.0,
        cost_dist=1.0,
    ),
    anchor_len=anchor_len,
    anchor_steps=anchor_y_steps,
    anchor_assign=False,
    delta=0.1,
    ds=10)

lane_loss_vdcc = dict(
    type='LaneLossV2',
    loss_weights=dict(cls_loss=1,
                      reg_losses_x=1,
                      reg_losses_z=1,
                      reg_losses_vis=1,
                      curve_geom_losses=0.05),
    assign_cfg=dict(
        cost_class=3.0,
        cost_dist=1.0,
    ),
    anchor_len=anchor_len,
    anchor_steps=anchor_y_steps,
    anchor_assign=False,
    delta=0.1,
    ds=10,
    curve_geom_axes=('x',),
    curve_slope_weight=1.0,
    curve_curv_weight=0.4)

model = dict(
    loss_lane=[[lane_loss_prior, lane_loss_prior],
               [lane_loss, lane_loss],
               [lane_loss, lane_loss_vdcc]])

optimizer = dict(type='Adam', lr=1e-5)
lr_config = dict(policy='step', step=[8000], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=2500, max_keep_ckpts=2)

load_from = 'output/openlane/experiments/strip_sample_r50x2/iter_10000.pth'
work_dir = 'output/openlane/experiments/vdcc_r50x2'
