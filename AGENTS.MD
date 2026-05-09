[系统状态]：Anchor3DLane-pp is deployed and the official checkpoint baseline is reproduced on OpenLane validation, current stable F1 = 62.4; full OpenLane training data and ApolloSim are still blocked by writable storage.

# Anchor3DLane-pp Agent Notes

## Goal
- Minimum target: OpenLane-v1.2 Anchor3DLane++ F1 must exceed 65 before the goal can be marked complete.
- Challenge target: reach F1 70.
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
- Dataset status:
  - OpenLane source is reachable from the remote server through `hf-mirror.com`.
  - OpenLane validation list has 39981 entries and cache has 39981 `.pkl` files.
  - Validation images are under `data/OpenLane/images/validation`.
  - OpenLane raw training image archives are about 88.7G; raw validation image archives are about 22.3G.
  - `/root/autodl-tmp` is the only writable data mount and is 50G total. `/autodl-pub` is read-only and has no reusable OpenLane/ApolloSim dataset directory.
  - Full OpenLane training data cannot be mounted on the current writable disk without larger storage or an external dataset mount.
  - The server has about 503GiB RAM and `/dev/shm` is 62G by default. A larger temporary RAM-backed image mount may be possible after the current evaluation, but it is not verified yet and would not persist across restarts.
  - ApolloSim and full OpenLane train data still need a larger writable disk or an external mounted dataset path.

## Agent Workflow
- Main agent owns environment setup, training orchestration, scoring decisions, Git commits, and documentation.
- Use 2-3 `gpt-5.5 xhigh` subagents for research before each score-improvement experiment.
- Use one `gpt-5.5 xhigh` subagent for code review before starting training after code changes.

## Rules
- Do not repeat environment deployment after the status line says deployment and baseline reproduction are complete.
- Baseline compatibility patches and documentation were committed after reproducing F1 = 62.4.
- Before each new experiment, read `docs/失败路线.md`.
- Commit and push only after F1 improves above the current stable score and is above baseline.
- If training fails or F1 regresses, revert to the last stable high-score commit and document the failure in `docs/失败路线.md`.
