"""Verify (and recover) the frame alignment between ShapeNet-Part and a ShapeNetCore
mesh, for a single model.

Why this exists
---------------
ShapeNet-Part gives per-point *labels* but NO color; color comes from the textured
ShapeNetCore mesh. To fuse them (paint color + carry part labels on ONE cloud), the
two must live in the same coordinate frame. ShapeNet-Part's points are in their own
normalized frame, which usually differs from `model_normalized.obj` by an axis
convention (a signed axis permutation: rotation and/or reflection). This tool finds
that discrete transform and reports the residual, so we know the fusion is sound
BEFORE building s2_v2 on top of it.

Method
------
Both point sets are centered and RMS-scaled (frame-free), then we search all 48
signed axis permutations (24 rotations + 24 reflections) for the one minimising a
symmetric Chamfer distance to the mesh surface. A small, well-separated residual =
alignment locked to that transform; a large residual for every candidate = the
mismatch is not a pure axis swap (needs continuous ICP) -> escalate.

CLI
---
    # align to the mesh surface (canonical):
    python scripts/align_partseg.py --part <model_id>.txt --mesh model_normalized.obj
    # or align to an existing sampled cloud (e.g. your s2 GT ply), no trimesh needed:
    python scripts/align_partseg.py --part <model_id>.txt --ref-cloud gt.ply
    # dump overlay plys to eyeball in Plotly:
    python scripts/align_partseg.py --part ... --mesh ... --save-overlay out_dir/

Deps: numpy, scipy (+ trimesh for --mesh, open3d for --ref-cloud / --save-overlay).
"""
import os
import sys
import itertools
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- IO
def load_partseg_txt(path):
    """ShapeNet-Part `_normal` row = `x y z nx ny nz label`. Returns (xyz, label)."""
    a = np.loadtxt(path).astype(np.float64)
    if a.ndim == 1:
        a = a[None, :]
    xyz = a[:, :3]
    label = a[:, -1].astype(int) if a.shape[1] >= 7 else np.zeros(len(a), int)
    return xyz, label


def mesh_surface_sample(obj_path, n):
    """Uniform surface sample of a (possibly multi-geometry) OBJ -> (n,3)."""
    import trimesh
    scene = trimesh.load(obj_path, process=False)
    geoms = [g for g in (scene.geometry.values() if hasattr(scene, "geometry") else [scene])
             if len(getattr(g, "faces", []))]
    mesh = trimesh.util.concatenate(geoms)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts, dtype=np.float64)


def load_cloud(ply_path, n=None):
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(ply_path).points, dtype=np.float64)
    if n and len(pts) > n:
        pts = pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]
    return pts


# ---------------------------------------------------------------- geometry
def normalize(pts):
    """Center at centroid, scale by RMS radius -> frame-free. Returns (pts_n, c, s)."""
    c = pts.mean(0)
    q = pts - c
    s = np.sqrt((q ** 2).sum(1).mean()) + 1e-12
    return q / s, c, s


def signed_perms():
    """All 48 signed axis-permutation matrices (24 rotations + 24 reflections)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for sx, sy, sz in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            M[0, perm[0]], M[1, perm[1]], M[2, perm[2]] = sx, sy, sz
            mats.append(M)
    return mats


def chamfer(a, b):
    """Symmetric mean nearest-neighbour distance between two point sets."""
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float(da.mean() + db.mean())


def best_alignment(part_xyz, ref_xyz):
    """Search signed axis permutations; return sorted [(chamfer, R, is_reflection)]."""
    p, _, _ = normalize(part_xyz)
    r, _, _ = normalize(ref_xyz)
    scored = []
    for R in signed_perms():
        d = chamfer(p @ R.T, r)                 # apply R to every part point
        scored.append((d, R, float(np.linalg.det(R)) < 0))
    scored.sort(key=lambda t: t[0])
    return scored


# ---------------------------------------------------------------- CLI
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Recover ShapeNet-Part <-> ShapeNetCore frame alignment.")
    ap.add_argument("--part", required=True, help="ShapeNet-Part _normal .txt for one model")
    ap.add_argument("--mesh", help="model_normalized.obj (color/geometry reference)")
    ap.add_argument("--ref-cloud", help="OR a sampled ply already in mesh frame (e.g. s2 GT)")
    ap.add_argument("--n", type=int, default=8192, help="reference sample size")
    ap.add_argument("--save-overlay", help="dir: write part_aligned.ply + ref_sample.ply")
    args = ap.parse_args()

    if not (args.mesh or args.ref_cloud):
        ap.error("give --mesh model_normalized.obj OR --ref-cloud gt.ply")

    part_xyz, part_lab = load_partseg_txt(args.part)
    ref_xyz = mesh_surface_sample(args.mesh, args.n) if args.mesh else load_cloud(args.ref_cloud, args.n)

    print(f"part points : {len(part_xyz):>7}   labels present: {len(np.unique(part_lab))} "
          f"({sorted(np.unique(part_lab).tolist())})")
    print(f"ref  points : {len(ref_xyz):>7}   ({'mesh sample' if args.mesh else 'cloud'})\n")

    scored = best_alignment(part_xyz, ref_xyz)
    best_d, best_R, best_refl = scored[0]
    second_d = scored[1][0]
    print("top-3 candidate transforms (lower chamfer = better):")
    for d, R, refl in scored[:3]:
        print(f"  chamfer={d:.4f}  {'reflection' if refl else 'rotation  '}  "
              f"R.flat=[{','.join(f'{v:+.0f}' for v in R.flat)}]")

    ratio = second_d / (best_d + 1e-12)
    print(f"\nbest chamfer   : {best_d:.4f}   ({'REFLECTION' if best_refl else 'rotation'})")
    print(f"separation     : {ratio:.2f}x vs 2nd best")
    ok = best_d < 0.15 and ratio > 1.3
    print("verdict        :", "ALIGNMENT LOCKED - safe to fuse" if ok else
          "WEAK - not a pure axis swap; add ICP polish or inspect the overlay")
    print("chosen R (part -> mesh frame, on RMS-normalized coords):")
    print(best_R.astype(int))

    if args.save_overlay:
        import open3d as o3d
        os.makedirs(args.save_overlay, exist_ok=True)
        p, _, _ = normalize(part_xyz)
        r, _, _ = normalize(ref_xyz)
        for name, pts, col in [("part_aligned", p @ best_R.T, [1, 0, 0]),
                               ("ref_sample", r, [0.6, 0.6, 0.6])]:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(np.tile(col, (len(pts), 1)))
            o3d.io.write_point_cloud(os.path.join(args.save_overlay, name + ".ply"), pc)
        print("overlay ->", args.save_overlay, "(red=aligned part, grey=reference)")


if __name__ == "__main__":
    sys.exit(main())
