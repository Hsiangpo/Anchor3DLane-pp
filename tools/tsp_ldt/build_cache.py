import argparse
import os
import pickle
from collections import defaultdict

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def parse_args():
    parser = argparse.ArgumentParser(description='Build TSP-LDT teacher cache.')
    parser.add_argument('--data-root', default='data/OpenLane')
    parser.add_argument('--data-list', default='lane3d_1000_v1.3/data_lists/training.txt')
    parser.add_argument('--cache-dir', default='lane3d_1000_v1.3/cache_dense')
    parser.add_argument('--out-dir', default='lane3d_1000_v1.3/tsp_ldt_cache')
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--x-min', type=float, default=-10.0)
    parser.add_argument('--x-max', type=float, default=10.0)
    parser.add_argument('--y-min', type=float, default=3.0)
    parser.add_argument('--y-max', type=float, default=103.0)
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--max-y-shift', type=float, default=8.0)
    parser.add_argument('--y-shift-step', type=float, default=0.5)
    parser.add_argument('--match-thresh', type=float, default=2.0)
    parser.add_argument('--sigma', type=float, default=0.7)
    parser.add_argument('--sdf-clip', type=float, default=3.0)
    parser.add_argument('--min-support', type=int, default=24)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_ids(data_root, data_list):
    path = os.path.join(data_root, data_list)
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def group_by_segment(sample_ids):
    segments = defaultdict(list)
    for sample_id in sample_ids:
        parts = sample_id.split('/')
        segments['/'.join(parts[:-1])].append(sample_id)
    for key in segments:
        segments[key].sort(key=lambda item: int(os.path.basename(item)))
    return segments


def load_cache(data_root, cache_dir, sample_id):
    filename = os.path.join(data_root, cache_dir, sample_id + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


def extract_points(lanes, x_min, x_max, y_min, y_max):
    if lanes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    gt_len = (lanes.shape[1] - 5) // 6
    y_values = np.arange(1, gt_len + 1, dtype=np.float32)
    points = []
    for lane in lanes:
        if lane[1] <= 0:
            continue
        xs = lane[5:5 + gt_len]
        zs = lane[5 + 2 * gt_len:5 + 3 * gt_len]
        vis = lane[5 + 4 * gt_len:5 + 5 * gt_len] > 0.5
        valid = vis & np.isfinite(xs) & np.isfinite(zs)
        valid &= (xs >= x_min - 3.0) & (xs <= x_max + 3.0)
        valid &= (y_values >= y_min) & (y_values <= y_max)
        valid &= (np.abs(xs) < 50.0) & (np.abs(zs) < 20.0)
        if valid.sum() < 2:
            continue
        lane_points = np.stack([xs[valid], y_values[valid], zs[valid],
                                np.ones(valid.sum(), dtype=np.float32)], axis=1)
        points.append(lane_points)
    if not points:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32)


def nearest_query(tree, query_xy, ref_xy):
    if cKDTree is not None:
        return tree.query(query_xy, k=1)
    diff = query_xy[:, None, :] - ref_xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    idx = dist.argmin(axis=1)
    return dist[np.arange(len(query_xy)), idx], idx


def align_points(neighbor, current, args):
    if len(neighbor) < args.min_support or len(current) < args.min_support:
        return None, None
    ref_xy = current[:, :2]
    tree = cKDTree(ref_xy) if cKDTree is not None else None
    best = None
    y_shifts = np.arange(-args.max_y_shift,
                         args.max_y_shift + args.y_shift_step * 0.5,
                         args.y_shift_step)
    for dy in y_shifts:
        shifted = neighbor.copy()
        shifted[:, 1] += dy
        in_range = (shifted[:, 1] >= args.y_min) & (shifted[:, 1] <= args.y_max)
        if in_range.sum() < args.min_support:
            continue
        query = shifted[in_range]
        dist, idx = nearest_query(tree, query[:, :2], ref_xy)
        close = dist < args.match_thresh
        if close.sum() < args.min_support:
            continue
        matched = current[idx[close]]
        query_close = query[close]
        dx = np.median(matched[:, 0] - query_close[:, 0])
        dz = np.median(matched[:, 2] - query_close[:, 2])
        query_refined = query.copy()
        query_refined[:, 0] += dx
        refined_dist, refined_idx = nearest_query(tree, query_refined[:, :2], ref_xy)
        refined_close = refined_dist < args.match_thresh
        support = int(refined_close.sum())
        if support < args.min_support:
            continue
        score = float(np.mean(np.minimum(refined_dist[refined_close], args.match_thresh)))
        candidate = (
            score, -support, dx, dy, dz, refined_dist, refined_idx,
            in_range, refined_close)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None, None
    _, neg_support, dx, dy, dz, refined_dist, refined_idx, in_range, refined_close = best
    aligned = neighbor[in_range].copy()
    aligned[:, 0] += dx
    aligned[:, 1] += dy
    aligned[:, 2] += dz
    aligned = aligned[refined_close].copy()
    confidence = np.clip(1.0 - refined_dist[refined_close] / args.match_thresh, 0.0, 1.0)
    aligned[:, 3] *= 0.7 * confidence.astype(np.float32)
    info = {'support': int(-neg_support), 'dx': float(dx), 'dy': float(dy), 'dz': float(dz)}
    return aligned, info


def make_teacher(points, args):
    h, w = args.height, args.width
    teacher = np.zeros((4, h, w), dtype=np.float32)
    valid = np.zeros((1, h, w), dtype=np.float32)
    if len(points) == 0:
        return teacher, valid
    xs = np.linspace(args.x_min, args.x_max, w, dtype=np.float32)
    ys = np.linspace(args.y_min, args.y_max, h, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    tree = cKDTree(points[:, :2]) if cKDTree is not None else None
    dist, idx = nearest_query(tree, grid, points[:, :2])
    dist_map = dist.reshape(h, w)
    nearest = points[idx].reshape(h, w, 4)
    occ = np.exp(-0.5 * (dist_map / args.sigma) ** 2)
    sdf = np.clip(dist_map / args.sdf_clip, 0.0, 1.0)
    z_norm = np.clip(nearest[..., 2] / 5.0, -1.0, 1.0)
    quality = np.clip(nearest[..., 3], 0.0, 1.0) * occ
    teacher[0] = occ
    teacher[1] = sdf
    teacher[2] = z_norm
    teacher[3] = quality
    valid[0] = 1.0
    return teacher, valid


def build_one(data_root, cache_dir, sample_id, segment_ids, index, args):
    current_obj = load_cache(data_root, cache_dir, sample_id)
    current = extract_points(current_obj['gt_3dlanes'], args.x_min, args.x_max,
                             args.y_min, args.y_max)
    all_points = [current]
    align_infos = []
    left = max(0, index - args.window)
    right = min(len(segment_ids), index + args.window + 1)
    for neighbor_id in segment_ids[left:right]:
        if neighbor_id == sample_id:
            continue
        try:
            neighbor_obj = load_cache(data_root, cache_dir, neighbor_id)
        except FileNotFoundError:
            continue
        neighbor = extract_points(neighbor_obj['gt_3dlanes'], args.x_min, args.x_max,
                                  args.y_min, args.y_max)
        aligned, info = align_points(neighbor, current, args)
        if aligned is not None:
            all_points.append(aligned)
            align_infos.append(info)
    merged = np.concatenate([p for p in all_points if len(p) > 0], axis=0) \
        if any(len(p) > 0 for p in all_points) else current
    teacher, valid = make_teacher(merged, args)
    return teacher, valid, align_infos, len(current), len(merged)


def main():
    args = parse_args()
    sample_ids = load_ids(args.data_root, args.data_list)
    segments = group_by_segment(sample_ids)
    out_root = os.path.join(args.data_root, args.out_dir)
    todo = sample_ids[args.start:]
    if args.limit > 0:
        todo = todo[:args.limit]
    segment_index = {}
    for ids in segments.values():
        for idx, sample_id in enumerate(ids):
            segment_index[sample_id] = (ids, idx)
    os.makedirs(out_root, exist_ok=True)
    built = 0
    align_support = []
    for pos, sample_id in enumerate(todo, start=args.start):
        out_file = os.path.join(out_root, sample_id + '.npz')
        if os.path.exists(out_file) and not args.overwrite:
            continue
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        segment_ids, index = segment_index[sample_id]
        teacher, valid, infos, cur_count, merged_count = build_one(
            args.data_root, args.cache_dir, sample_id, segment_ids, index, args)
        support = np.array([info['support'] for info in infos], dtype=np.float32)
        support_mean = float(support.mean()) if support.size else 0.0
        align_support.append(support_mean)
        np.savez_compressed(
            out_file,
            tsp_teacher=teacher.astype(np.float16),
            tsp_valid=valid.astype(np.uint8),
            frame_count=np.array([len(infos) + 1], dtype=np.uint8),
            current_points=np.array([cur_count], dtype=np.int32),
            merged_points=np.array([merged_count], dtype=np.int32),
            align_support=np.array([support_mean], dtype=np.float32))
        built += 1
        if built % 200 == 0 or built == 1:
            avg = float(np.mean(align_support[-200:])) if align_support else 0.0
            print('built {} samples, latest global index {}, avg_support {:.2f}'.format(
                built, pos, avg), flush=True)
    print('done built={}, mean_align_support={:.2f}'.format(
        built, float(np.mean(align_support)) if align_support else 0.0))


if __name__ == '__main__':
    main()
