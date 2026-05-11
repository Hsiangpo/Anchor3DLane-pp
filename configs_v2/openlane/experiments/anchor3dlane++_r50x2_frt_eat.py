_base_ = './anchor3dlane++_r50x2_eac_dr.py'

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

lane_loss_frt_eat = dict(
    type='LaneLossV2',
    loss_weights=dict(cls_loss=1,
                      reg_losses_x=1,
                      reg_losses_z=1,
                      reg_losses_vis=1,
                      curve_geom_losses=0.05,
                      accept_calib_losses=0.08,
                      dup_rank_losses=0.02),
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
    curve_curv_weight=0.4,
    accept_vis_thresh=0.7,
    accept_dist_thresh=1.5,
    accept_cover_thresh=0.75,
    accept_purity_thresh=0.75,
    accept_neg_cover_thresh=0.45,
    accept_neg_purity_thresh=0.45,
    accept_hard_score_thresh=0.35,
    accept_pos_weight=1.0,
    accept_neg_weight=1.0,
    dup_rank_margin=0.1,
    dup_rank_max_pairs=4,
    frt_enable=True,
    frt_min_width=0.6,
    frt_default_width=2.0,
    frt_max_width=2.5,
    frt_slack=0.15,
    frt_dirty_thresh=0.10,
    frt_rank_weight=0.25)

model = dict(
    accept_calib=dict(
        enabled=True,
        apply='final',
        detach=True,
        freeze_base=True,
        bias_init=4.0,
        hidden_channels=64,
        score_floor=0.75,
        nms_power=1.0),
    loss_lane=[[lane_loss_prior, lane_loss_prior],
               [lane_loss, lane_loss],
               [lane_loss, lane_loss_frt_eat]])

data = dict(samples_per_gpu=3)
optimizer = dict(type='Adam', lr=5e-5)
lr_config = dict(policy='step', step=[6000], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)

load_from = 'output/openlane/experiments/eac_dr_r50x2/iter_10000.pth'
work_dir = 'output/openlane/experiments/frt_eat_r50x2'
