[系统状态]：Anchor3DLane-pp is deployed and the official checkpoint baseline is reproduced on OpenLane validation, current stable F1 = 62.521; ApolloSim is not mounted yet.

# Anchor3DLane-pp Agent Notes

## Goal
- Minimum target: OpenLane-v1.2 Anchor3DLane++ F1 must exceed 65 before the goal can be marked complete.
- Minimum research standard: the final successful solution must contain at least 3-5 real innovation points, not just one-off tuning.
- Challenge target: reach F1 70 with at least 5-8 real innovation points.
- Baseline reproduction target: official Anchor3DLane++ camera-only ResNet-50 720x960 F1 = 62.4.

## Current Structure
- Remote path: `/root/autodl-tmp/Anchor3DLane-pp` on `anchor3dlane-seetacloud`.
- Active code branch: `anchor3dlane-pp`, tracking `origin/anchor3dlane++`.
- `configs/openlane/`: OpenLane training and evaluation configs.
- `configs_v2/openlane/anchor3dlane++_r50x2.py`: official camera-only baseline config for OpenLane-v1.2 F1 = 62.4.
- `mmseg/models/lane_detector/`: Anchor3DLane and Anchor3DLane++ model implementation.
- `mmseg/datasets/lane_datasets/`: dataset loaders and OpenLane evaluation integration.
- `tools/train.py` and `tools/test.py`: baseline training and evaluation entry points.
- `docs/实验记录.md`: only record successful experiments with F1 improvement above baseline.
- `docs/失败路线.md`: record failed or regressed ideas so later iterations avoid them.

## Current Progress
- SSH alias `anchor3dlane-seetacloud` works without password.
- The project is installed on the remote server with `python setup.py develop --no-deps`.
- Runtime uses the server base Python with `torch 2.8.0+cu128`, `torchvision 0.23.0+cu128`, and `mmcv 1.7.1`.
- Compatibility patches are present for NumPy 2.x aliases, missing `mmcv-full` focal loss extension, missing MSDA extension fallback, optional ONCE/spconv imports, and DCNv2 via `torchvision.ops.deform_conv2d`.
- Official weights are present remotely:
  - `pretrained/resnet50_v1c-2cccc1ad.pth`
  - `checkpoints/Openlane/openlane_anchor3dlane++_r50x2.pth`
- The official checkpoint loads into the current `Anchor3DLanePP` model.
- Official checkpoint validation evaluation completed:
  - PID file: `/root/autodl-tmp/Anchor3DLane-pp/logs/baseline_eval.pid`
  - Latest log: `/root/autodl-tmp/Anchor3DLane-pp/logs/baseline_eval_20260510_053432.log`
  - Command: `PORT=29601 bash tools/dist_test.sh configs_v2/openlane/anchor3dlane++_r50x2.py checkpoints/Openlane/openlane_anchor3dlane++_r50x2.pth 2 --show-dir output/openlane/baseline_eval`
  - Result file: `/root/autodl-tmp/Anchor3DLane-pp/output/openlane/baseline_eval/evaluation_result.json`
  - Metrics: F1 `0.6240448139487519`, recall `0.613268850798542`, precision `0.6352062468171787`, category accuracy `0.9337897428296154`.
- Current stable score:
  - Experiment: `strip_sample` / projection-aware local lane-strip sampling.
  - Result file: `/root/autodl-tmp/Anchor3DLane-pp/output/openlane/experiments/strip_sample_r50x2_eval/evaluation_result.json`.
  - Metrics: F1 `0.6252112086013109`, recall `0.6098164672986526`, precision `0.6414033570289482`, category accuracy `0.9344681247959707`.
  - Improvement over official checkpoint baseline: `+0.001166394652559` F1, or `+0.116639` points on the 0-100 reporting scale.
- Completed score-improvement experiments:
  - `vis_bce` changed visibility supervision from SmoothL1 to BCE with positive weighting.
  - Formal 2-GPU training completed from `logs/train_vis_bce_retry_20260510_082812.log` and produced `output/openlane/experiments/vis_bce_r50x2/iter_10000.pth`.
  - Formal validation evaluation completed from `logs/eval_vis_bce_20260510_091100.log`.
  - Result file: `/root/autodl-tmp/Anchor3DLane-pp/output/openlane/experiments/vis_bce_r50x2_eval/evaluation_result.json`.
  - Metrics: F1 `0.6235742314274739`, recall `0.6226701761143566`, precision `0.624480915754884`, category accuracy `0.9352928519977174`.
  - Decision: regressed below the stable baseline F1 `0.6240448139487519`; failed experiment code was rolled back and the failure was documented in `docs/失败路线.md`.
  - Official prediction `test_conf` sweep from 0.30 to 0.60 also did not improve F1 and is documented in `docs/失败路线.md`.
  - `quality_calib` added quality-aware confidence calibration on the last feature level and final refinement only.
  - Formal 2-GPU training completed from `logs/train_quality_calib_resume_20260510_100110.log` and produced `output/openlane/experiments/quality_calib_r50x2/iter_10000.pth`.
  - Formal validation evaluation completed from `logs/eval_quality_calib_20260510_103444.log`.
  - Result file: `/root/autodl-tmp/Anchor3DLane-pp/output/openlane/experiments/quality_calib_r50x2_eval/evaluation_result.json`.
  - Metrics: F1 `0.6227141985407065`, recall `0.6076266290043012`, precision `0.6385701047829496`, category accuracy `0.9337999389775974`.
  - Decision: regressed below the stable baseline F1 `0.6240448139487519`; failed experiment code was rolled back and the failure was documented in `docs/失败路线.md`.
  - `eval_match` added evaluator-inspired clipped 3D distance cost to Hungarian assignment only.
  - Formal 2-GPU training completed from `logs/train_eval_match_20260510_110304.log` and produced `output/openlane/experiments/eval_match_r50x2/iter_10000.pth`.
  - Formal validation evaluation completed from `logs/eval_eval_match_20260510_114446.log`.
  - Result file: `/root/autodl-tmp/Anchor3DLane-pp/output/openlane/experiments/eval_match_r50x2_eval/evaluation_result.json`.
  - Metrics: F1 `0.6206571803570284`, recall `0.6117780488326806`, precision `0.629797844865972`, category accuracy `0.9319983736531815`.
  - Decision: regressed below the stable baseline F1 `0.6240448139487519`; failed experiment code was rolled back and the failure was documented in `docs/失败路线.md`.
  - Avoid repeating standalone evaluator-distance Hungarian matching or only tuning `cost_eval` / `eval_dist_th`.
- Latest failed experiment verification notes:
  - Code review subagent found no blocking OOM, DDP, tensor-shape, or config issue.
  - Remote config load passed with `samples_per_gpu=2`, `max_iters=10000`, `cost_eval=0.5`, `cost_vis=0.0`, loading `checkpoints/Openlane/openlane_anchor3dlane++_r50x2.pth`.
  - Remote matcher sanity passed: default matcher output equals the previous cost formula; enabled eval cost returns finite `[30, 6]` matrices.
  - Remote 2-GPU DDP smoke passed for 1 and 20 iterations. The 20-iter run had finite losses and gradients, logged memory about 24.7GB, and no OOM/Traceback/NaN.
- Current score-improvement experiment:
  - Selected idea: projection-aware local lane-strip sampling.
  - Rationale: the original `Anchor3DLanePP` samples one image feature point per projected 3D anchor point. A small projection error can miss thin lane evidence, especially far away. This experiment samples three horizontal offsets around the projected point and learns a light softmax aggregation, while keeping loss, matcher, visibility supervision, and test thresholds unchanged.
  - Code: `mmseg/models/lane_detector/anchor_3dlane_pp.py` adds optional `strip_sample`, `strip_offsets`, `strip_center_bias`, and `strip_apply`.
  - Config: `configs_v2/openlane/experiments/anchor3dlane++_r50x2_strip_sample.py`.
  - Conservative setting: `strip_apply='final'`, so only the final refinement output uses strip sampling; default baseline configs keep `strip_sample=False`.
  - Local checks passed: `python -m py_compile mmseg/models/lane_detector/anchor_3dlane_pp.py`, `git diff --check`, and `rg` confirmed only one active `get_proposals` definition.
  - Code review subagent found no blocking OOM, DDP unused-parameter, tensor-shape, checkpoint-load, or gradient issue after the duplicate dead `get_proposals` definition was removed.
  - Remote config load and model build passed; `strip_weight_logits` starts with softmax weights about `[0.107, 0.787, 0.107]`.
  - Remote 2-GPU DDP smoke passed for 1 and 20 iterations at `samples_per_gpu=2`.
  - Batch-size probe passed at `samples_per_gpu=3`; 20-iter DDP smoke had no hard errors, finite loss/grad, and memory about 27.4GB per GPU, so the formal run should use `samples_per_gpu=3`.
  - Formal 2-GPU training completed with PID file `logs/train_strip_sample.pid`, log `logs/train_strip_sample_20260510_121835.log`, `samples_per_gpu=3`, and work dir `output/openlane/experiments/strip_sample_r50x2`.
  - Final checkpoint saved at `output/openlane/experiments/strip_sample_r50x2/iter_10000.pth`; `iter_5000.pth` and `iter_7500.pth` were kept by `max_keep_ckpts=2`; memory was about 27.4GB and no hard errors were found.
  - Formal validation evaluation completed with PID file `logs/eval_strip_sample.pid`, log `logs/eval_strip_sample_20260510_131335.log`, and result path `output/openlane/experiments/strip_sample_r50x2_eval/evaluation_result.json`.
  - Decision: improved over stable baseline F1 `0.6240448139487519`; keep the implementation, record it in `docs/实验记录.md`, and commit/push this stable version.
- Dataset status:
  - OpenLane source is reachable from the remote server through `hf-mirror.com`.
  - OpenLane validation list has 39981 entries and cache has 39981 `.pkl` files.
  - Validation images are under `data/OpenLane/images/validation`.
  - OpenLane raw training image archives are about 88.7G; raw validation image archives are about 22.3G.
  - `/root/autodl-tmp` is the only writable data mount and is 50G total. `/autodl-pub` is read-only and has no reusable OpenLane/ApolloSim dataset directory.
  - Current OpenLane training image workaround uses `/dev/shm/openlane_train_images_a` for `images_training_0..10`, `/root/openlane_train_images_b` for `images_training_11..15`, and symlinks segments into `data/OpenLane/images/training`.
  - OpenLane training image extraction is complete with 798 symlinked segment directories.
  - OpenLane training cache/list generation completed with 136496 samples.
  - Current preparation script: `/root/autodl-tmp/jobs/prepare_openlane_train_split.sh`; it must check `images_training_N/segment-*` per archive before skipping extraction, otherwise later archives can be skipped incorrectly.
  - No dataset preparation job is currently running.
  - ApolloSim still needs a larger writable disk or an external mounted dataset path.

## Agent Workflow
- Main agent owns environment setup, training orchestration, scoring decisions, Git commits, and documentation.
- Use 2-3 `gpt-5.5 xhigh` subagents for research before each score-improvement experiment.
- Use one `gpt-5.5 xhigh` subagent for code review before starting training after code changes.
- When prompting subagents, explicitly tell them they are subagents and must not spawn or call further subagents.

## Rules
- Do not repeat environment deployment after the status line says deployment and baseline reproduction are complete.
- Baseline compatibility patches and documentation were committed after reproducing F1 = 62.4.
- Before each new experiment, read `docs/失败路线.md`.
- Commit and push only after F1 improves above the current stable score and is above baseline.
- If training fails or F1 regresses, revert to the last stable high-score commit and document the failure in `docs/失败路线.md`.
- Do not create new files with version suffixes such as `_v1` or `_v2`; record experiment versions in docs instead.
- Roll back failed implementation code promptly; keep only the stable code and documented lessons.
- Keep GPU utilization practical: avoid wastefully small settings after smoke tests, but do not exceed the known memory-safe envelope.
