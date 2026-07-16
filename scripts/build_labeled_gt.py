"""Build `labeled_s3`: fuse ShapeNet-Part per-point part labels onto the colored
`gt_s3` ground truth, so each GT point carries (xyz, rgb, part).

Why
---
The part-aware colored-completion idea needs a semantic part label per point. The
colored GT (`gt_s3`, xyzrgb) comes from ShapeNetCore meshes; ShapeNet-Part provides
per-point labels for the SAME models but in its own frame. This script recovers the
frame transform (reusing `align_partseg.py`), transfers labels by nearest neighbour,
and writes one self-contained `.npz` per model.

Output
------
    <out-dir>/<synset>/<model>.npz   with  xyz (N,3) f32, rgb (N,3) f32, part (N,) i16
    <out-dir>/labeled_manifest.csv   (synset, model, n, n_parts, residual, reflection, verdict)

ShapeNet-Part
-------------
Needs `shapenetcore_partanno_segmentation_benchmark_v0_normal/<synset>/<model>.txt`
(row = x y z nx ny nz label). Only models present in BOTH gt-dir and part-dir are built.

Usage
-----
    python scripts/build_labeled_gt.py \
        --gt-dir   $PCC_DATA_ROOT/gt_s3 \
        --part-dir $PCC_DATA_ROOT/shapenet_part \
        --out-dir  $PCC_DATA_ROOT/labeled_s3 \
        [--synset 02691156] [--limit 2] [--save-viz $PCC_DATA_ROOT/labeled_s3/_viz]

Start small: `--synset 02691156 --limit 2 --save-viz ...`, eyeball the part-coloured
HTML to confirm alignment, THEN drop --limit to build everything.
"""
import os
import sys
import glob
import argparse
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_partseg import load_partseg_txt, normalize, best_alignment  # noqa: E402


# ---------------------------------------------------------------- IO
def load_colored_ply(path):
    """Colored PLY -> (xyz (N,3), rgb (N,3) in [0,1])."""
    import open3d as o3d
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points, np.float64)
    rgb = np.asarray(pc.colors, np.float64)
    if len(rgb) != len(xyz):
        rgb = np.zeros_like(xyz)
    return xyz, rgb


# ---------------------------------------------------------------- core (tested)
def transfer_labels(gt_xyz, part_xyz, part_lab):
    """Transfer ShapeNet-Part labels onto gt points.
    Returns (gt_part (N,) i16, residual_chamfer, is_reflection)."""
    d, R, refl = best_alignment(part_xyz, gt_xyz)[0]
    pn, _, _ = normalize(part_xyz)
    gn, _, _ = normalize(gt_xyz)
    part_in_gt = pn @ R.T                          # part -> gt-normalized frame
    _, idx = cKDTree(part_in_gt).query(gn, k=1)     # each gt point <- nearest part point
    return part_lab[idx].astype(np.int16), float(d), bool(refl)


# ---------------------------------------------------------------- viz
def save_part_viz(path, xyz, part):
    """Plotly HTML colouring points by part id (to eyeball alignment)."""
    import plotly.graph_objects as go
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
               "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324", "#800000"]
    col = [palette[int(p) % len(palette)] for p in part]
    fig = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                                 marker=dict(size=1.8, color=col)))
    fig.update_layout(height=560, scene_aspectmode="data", margin=dict(l=0, r=0, t=0, b=0))
    fig.write_html(path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True, help="gt_s3 root: <synset>/<model>.ply (colored)")
    ap.add_argument("--part-dir", required=True, help="ShapeNet-Part root: <synset>/<model>.txt")
    ap.add_argument("--out-dir", required=True, help="labeled_s3 root to write <synset>/<model>.npz")
    ap.add_argument("--synset", default=None, help="restrict to one synset (e.g. 02691156)")
    ap.add_argument("--limit", type=int, default=None, help="only build the first N models")
    ap.add_argument("--save-viz", default=None, help="dir: also write a part-coloured HTML per model")
    ap.add_argument("--weak", type=float, default=0.15, help="residual above this = flag alignment WEAK")
    args = ap.parse_args()

    synsets = [args.synset] if args.synset else sorted(
        d for d in os.listdir(args.gt_dir) if os.path.isdir(os.path.join(args.gt_dir, d)))

    rows, built, weak = [], 0, 0
    for syn in synsets:
        gt_plys = sorted(glob.glob(os.path.join(args.gt_dir, syn, "*.ply")))
        for gp in gt_plys:
            mid = os.path.splitext(os.path.basename(gp))[0]
            part_txt = os.path.join(args.part_dir, syn, mid + ".txt")
            if not os.path.exists(part_txt):
                continue                                  # no label for this model -> skip
            gt_xyz, gt_rgb = load_colored_ply(gp)
            part_xyz, part_lab = load_partseg_txt(part_txt)
            gt_part, resid, refl = transfer_labels(gt_xyz, part_xyz, part_lab)
            is_weak = resid > args.weak
            weak += is_weak

            out_syn = os.path.join(args.out_dir, syn)
            os.makedirs(out_syn, exist_ok=True)
            np.savez_compressed(os.path.join(out_syn, mid + ".npz"),
                                xyz=gt_xyz.astype(np.float32), rgb=gt_rgb.astype(np.float32),
                                part=gt_part)
            if args.save_viz:
                os.makedirs(args.save_viz, exist_ok=True)
                save_part_viz(os.path.join(args.save_viz, f"{syn}_{mid}.html"), gt_xyz, gt_part)
            rows.append((syn, mid, len(gt_xyz), len(np.unique(gt_part)), round(resid, 4),
                         refl, "WEAK" if is_weak else "ok"))
            built += 1
            print(f"[{'WEAK' if is_weak else ' ok '}] {syn}/{mid}  n={len(gt_xyz)} "
                  f"parts={len(np.unique(gt_part))} residual={resid:.4f}")
            if args.limit and built >= args.limit:
                break
        if args.limit and built >= args.limit:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    man = os.path.join(args.out_dir, "labeled_manifest.csv")
    with open(man, "w") as f:
        f.write("synset,model,n,n_parts,residual,reflection,verdict\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\nbuilt {built} labeled clouds ({weak} flagged WEAK - eyeball those). manifest -> {man}")
    if weak:
        print("WEAK = alignment not a clean axis swap; check the viz, may need ICP polish.")


if __name__ == "__main__":
    main()
