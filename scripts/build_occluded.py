"""Generate occluded (partial) inputs from the colored GT — the completion task's
inputs. Ports occluded.ipynb to a CLI. Viewpoint-crop like PoinTr: remove the
crop_ratio points nearest a random viewpoint; the rest is the partial, the removed
region is the (labelled) missing GT.

Input : gt_s3/<synset>/<model>.ply  (colored, fixed-size GT — build_gt.py output)
Output: occluded_occ/<difficulty>/<synset>/<model>.ply         (partial input)
        occluded_occ/<difficulty>/<synset>/<model>_missing.ply (removed region)
        occluded_occ/manifest.csv
difficulty = simple(0.25) / moderate(0.50) / hard(0.75).

Usage
-----
    python scripts/build_occluded.py \
        --in-dir  $PCC_DATA_ROOT/gt_s3 \
        --out-dir $PCC_DATA_ROOT/occluded_occ \
        [--synset 02691156] [--limit 5] [--views 1]
"""
import os
import sys
import glob
import zlib
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT  # noqa: E402

RATIOS = {"simple": 0.25, "moderate": 0.50, "hard": 0.75}


def load_ply_xyzrgb(ply_path):
    import open3d as o3d
    pc = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pc.points); rgb = np.asarray(pc.colors)
    if rgb.shape[0] != xyz.shape[0]:
        rgb = np.zeros_like(xyz)
    return np.concatenate([xyz, rgb], 1).astype(np.float32)


def save_ply_xyzrgb(arr, out_ply):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(arr[:, 3:6], 0, 1).astype(np.float64))
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pc)


def separate_point_cloud(points, crop_ratio, viewpoint=None, seed=None):
    """Remove the crop_ratio points nearest a viewpoint (on the unit sphere).
    Returns (partial (M,6), missing (N-M,6)). Color rides along."""
    N = len(points); num_crop = int(round(N * crop_ratio))
    if num_crop <= 0:
        return points.copy(), points[:0].copy()
    xyz = points[:, :3]
    c = xyz.mean(0)
    xyzn = (xyz - c) / (np.linalg.norm(xyz - c, axis=1).max() + 1e-12)
    if viewpoint is None:
        v = np.random.default_rng(seed).standard_normal(3)
    else:
        v = np.asarray(viewpoint, float)
    center = v / np.linalg.norm(v)
    idx = np.argsort(np.linalg.norm(xyzn - center[None], axis=1))
    return points[idx[num_crop:]], points[idx[:num_crop]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(DATA_ROOT, "gt_s3"),
                    help="colored GT root: <synset>/<model>.ply")
    ap.add_argument("--out-dir", default=os.path.join(DATA_ROOT, "occluded_occ"))
    ap.add_argument("--synset", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N models")
    ap.add_argument("--views", type=int, default=1, help="partials per (model, difficulty)")
    ap.add_argument("--difficulties", nargs="+", default=list(RATIOS),
                    help="subset of: simple moderate hard")
    ap.add_argument("--no-missing", action="store_true", help="do NOT save the removed region")
    args = ap.parse_args()

    synsets = [args.synset] if args.synset else sorted(
        d for d in os.listdir(args.in_dir) if os.path.isdir(os.path.join(args.in_dir, d)))

    rows, done = [], 0
    for syn in synsets:
        gt_plys = sorted(glob.glob(os.path.join(args.in_dir, syn, "*.ply")))
        for gp in gt_plys:
            mid = os.path.splitext(os.path.basename(gp))[0]
            pts = load_ply_xyzrgb(gp)
            for dname in args.difficulties:
                ratio = RATIOS[dname]
                for v in range(args.views):
                    seed = zlib.crc32(f"{mid}_{dname}_{v}".encode()) & 0x7FFFFFFF
                    partial, missing = separate_point_cloud(pts, ratio, seed=seed)
                    tag = dname if args.views == 1 else f"{dname}_v{v}"
                    out = os.path.join(args.out_dir, tag, syn, mid + ".ply")
                    save_ply_xyzrgb(partial, out)
                    if not args.no_missing:
                        save_ply_xyzrgb(missing, os.path.join(args.out_dir, tag, syn, mid + "_missing.ply"))
                    rows.append((syn, mid, tag, ratio, v, seed, len(partial), len(missing), gp))
            done += 1
            if done % 20 == 0:
                print(f"  {done} models occluded...")
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    man = os.path.join(args.out_dir, "manifest.csv")
    with open(man, "w") as f:
        f.write("synset,model,difficulty,crop_ratio,view,seed,n_partial,n_missing,gt_ply\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\ndone. {len(rows)} partials from {done} GT clouds -> {args.out_dir}\nmanifest -> {man}")


if __name__ == "__main__":
    main()
