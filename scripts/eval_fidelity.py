"""Mesh-fidelity evaluation for sampled colored point clouds.

Answers: "how faithfully does a sampled cloud reproduce its source mesh?"
Five metrics, all LOWER = BETTER:

  speckle     local color variance among kNN        color noise (our metric for the
                                                     ShapeNet double-face artifact)
  uniformity  CV of nearest-neighbor distance        even spread     (~ PU-Net NUC)
  p2f         mean point-to-surface distance          points on surface (~ Metro / PU-GAN)
  coverage    mean surface->cloud distance / bbox     surface covered   (~ PU-GAN bidir.)
  color_dE    mean CIE76 ΔE vs mesh texture            color correctness (CIELAB + texel lookup)

Reference color is read from the CLEAN exterior mesh via barycentric + texel lookup,
so a point that picked up an interior face's color scores a high ΔE.

Deps: open3d, trimesh, rtree, pillow, scipy, numpy  (+ pymeshlab only for --mesh mode).
CLI:
    python scripts/eval_fidelity.py --cloud cloud.ply --ref clean.obj
    python scripts/eval_fidelity.py --cloud cloud.ply --mesh model_normalized.obj   # builds ref
"""
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- color (CIELAB ΔE)
def srgb_to_lab(rgb):
    rgb = np.clip(np.asarray(rgb, float), 0, 1)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)

def delta_e(a, b):
    return np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=1)

# ---------------------------------------------------------------- reference color from mesh
def load_reference(clean_obj):
    """Load the clean exterior mesh as (list-of-geometries, concatenated-mesh)."""
    import trimesh
    scene = trimesh.load(clean_obj, process=False)
    geoms = [g for g in (scene.geometry.values() if hasattr(scene, "geometry") else [scene])
             if len(getattr(g, "faces", []))]
    return geoms, trimesh.util.concatenate(geoms)

def _sample_img(img, uv):
    arr = np.asarray(img.convert("RGB")); H, W = arr.shape[:2]
    u = (uv[:, 0] % 1.0) * (W - 1)
    v = (1.0 - (uv[:, 1] % 1.0)) * (H - 1)               # OBJ uv origin bottom-left -> flip v
    xi = np.clip(u.astype(int), 0, W - 1); yi = np.clip(v.astype(int), 0, H - 1)
    return arr[yi, xi].astype(float) / 255.0

def _geom_color(g, tri, cp):
    import trimesh
    vis = g.visual
    mat = getattr(vis, "material", None)
    img = getattr(mat, "image", None)
    uv = getattr(vis, "uv", None)
    bary = trimesh.triangles.points_to_barycentric(g.triangles[tri], cp)
    if img is not None and uv is not None:                       # textured face -> texel lookup
        uvp = np.einsum("ij,ijk->ik", bary, np.asarray(uv)[g.faces[tri]])
        return _sample_img(img, uvp)
    mc = getattr(mat, "main_color", None)                        # flat Kd material
    if mc is not None:
        return np.tile(np.asarray(mc[:3], float) / 255.0, (len(tri), 1))
    return np.full((len(tri), 3), 0.5)

def _ref_color_and_dist(geoms, ps):
    import trimesh
    N = len(ps)
    best_d = np.full(N, np.inf); best_c = np.zeros((N, 3))
    for g in geoms:
        cp, d, tri = trimesh.proximity.closest_point(g, ps)
        better = d < best_d
        if not better.any():
            continue
        c = _geom_color(g, tri, cp)
        best_c[better] = c[better]; best_d[better] = d[better]
    return best_c, best_d

# ---------------------------------------------------------------- the 5 metrics
def evaluate(cloud_ply, clean_obj, n_probe=15000, k=12, seed=0):
    """Return the 5-metric fidelity dict for one cloud vs its clean reference mesh."""
    import open3d as o3d, trimesh
    pc = o3d.io.read_point_cloud(cloud_ply)
    pts, col = np.asarray(pc.points), np.asarray(pc.colors)
    geoms, mesh_all = load_reference(clean_obj)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), min(n_probe, len(pts)), replace=False)
    ps, cs = pts[idx], col[idx]
    tree = cKDTree(pts)

    _, nn = tree.query(ps, k=k)
    speckle = float(col[nn].std(axis=1).mean())
    d2, _ = tree.query(ps, k=2)
    nnd = d2[:, 1]; uniformity = float(nnd.std() / (nnd.mean() + 1e-12))

    with np.errstate(divide="ignore", invalid="ignore"):
        ref, dist = _ref_color_and_dist(geoms, ps)
        d_e = delta_e(cs, ref)

    diag = float(np.linalg.norm(mesh_all.bounds[1] - mesh_all.bounds[0]))
    surf, _ = trimesh.sample.sample_surface(mesh_all, n_probe)
    cov, _ = tree.query(surf, k=1)

    return dict(points=int(len(pts)),
                speckle=speckle,
                uniformity=uniformity,
                p2f=float(np.nanmean(dist)),
                coverage=float(cov.mean() / diag),
                color_dE=float(np.nanmean(d_e)))

# ---------------------------------------------------------------- optional: build the reference
# pymeshlab's ambient_occlusion can SEGFAULT on some meshes (kills the whole process).
# Run it in a subprocess so a crash is isolated and just fails this one model.
_AO_HELPER_SRC = (
    "import sys, os, numpy as np, pymeshlab\n"
    "in_dir, in_file, fm_out, fq_out = sys.argv[1:5]\n"
    "os.chdir(in_dir)\n"
    "ms = pymeshlab.MeshSet(); ms.load_new_mesh(in_file)\n"
    "ms.compute_scalar_ambient_occlusion()\n"
    "m = ms.mesh(0)\n"
    "np.save(fm_out, m.face_matrix()); np.save(fq_out, m.face_scalar_array())\n"
)

def _ambient_occlusion(obj_path):
    """Per-face ambient occlusion (face_matrix, face_scalar) via an isolated subprocess."""
    import subprocess, tempfile
    in_dir, in_file = os.path.split(obj_path)
    with tempfile.TemporaryDirectory() as td:
        helper = os.path.join(td, "_ao.py"); open(helper, "w").write(_AO_HELPER_SRC)
        fm_out, fq_out = os.path.join(td, "fm.npy"), os.path.join(td, "fq.npy")
        subprocess.run([sys.executable, helper, in_dir, in_file, fm_out, fq_out],
                       capture_output=True, timeout=600)
        if not (os.path.exists(fm_out) and os.path.exists(fq_out)):
            raise RuntimeError("ambient occlusion failed (pymeshlab crashed on this mesh)")
        return np.load(fm_out), np.load(fq_out)

def build_clean_reference(obj_path, out_dir):
    """EPFL/MMSPG interior-face removal -> a clean exterior OBJ usable as color reference.
    Same logic as sample_s2's EPFL path. Needs pymeshlab. Returns the clean .obj path."""
    import shutil
    in_dir = os.path.dirname(obj_path)
    fm, fq = _ambient_occlusion(obj_path)   # crash-isolated pymeshlab

    keys = np.sort(fm, axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ks = keys[order]; change = np.any(ks[1:] != ks[:-1], axis=1)
    gs = np.concatenate(([0], np.nonzero(change)[0] + 1)); ge = np.concatenate((gs[1:], [len(order)]))
    drop = np.zeros(fm.shape[0], bool)
    for s, e in zip(gs, ge):
        mem = order[s:e]
        if len(mem) > 1:
            mem = np.sort(mem); keep = mem[np.argmax(fq[mem])]
            drop[mem[mem != keep]] = True

    lines = _ensure_corr(open(obj_path).read().splitlines(), fm)
    if lines is None:
        raise RuntimeError("pymeshlab/obj face mismatch")
    ne = [(i, x) for i, x in enumerate(lines) if len(x) > 0]
    fidx = np.asarray([i for i, x in ne if x[0] == "f"])
    clean = list(np.delete(np.asarray(lines), fidx[drop]))

    cdir = os.path.join(out_dir, "models"); os.makedirs(cdir, exist_ok=True)
    mtl_i, mtl_line = next((i, x) for i, x in enumerate(clean) if "mtllib" in x)
    mtl_file = next(t for t in mtl_line.split() if t.endswith(".mtl"))
    shutil.copyfile(os.path.join(in_dir, mtl_file), os.path.join(cdir, "clean.mtl"))
    clean[mtl_i] = "mtllib clean.mtl"
    cobj = os.path.join(cdir, "clean.obj"); open(cobj, "w").write("\n".join(clean))
    for line in open(os.path.join(cdir, "clean.mtl")).read().splitlines():
        if "map_Kd" in line:
            tex = line.split()[1]; rel, name = os.path.split(tex)
            dd = os.path.join(cdir, rel); os.makedirs(dd, exist_ok=True)
            shutil.copyfile(os.path.join(in_dir, tex), os.path.join(dd, name))
    return cobj

def _ensure_corr(obj_lines, fm):
    ne = [(i, x) for i, x in enumerate(obj_lines) if len(x) > 0]
    fobj = [[x.split()[1].split("/")[0], x.split()[2].split("/")[0], x.split()[3].split("/")[0]]
            for _, x in ne if x[0] == "f"]
    fobj = np.asarray(fobj, dtype=int) - 1
    if np.array_equal(fm, fobj):
        return obj_lines
    fidx = [i for i, x in ne if x[0] == "f"]; dropped = []
    while fobj.shape[0] > fm.shape[0]:
        corr = (fm == fobj[:fm.shape[0], :]).all(axis=1)
        if not corr.all():
            d = np.nonzero(~corr)[0][0]; fobj = np.delete(fobj, d, axis=0); dropped.append(fidx.pop(d))
        else:
            dropped += fidx[fm.shape[0]:]; fobj = fobj[:fm.shape[0], :]
    return None if not np.array_equal(fm, fobj) else list(np.delete(np.asarray(obj_lines), dropped))

# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse, tempfile
    ap = argparse.ArgumentParser(description="Mesh-fidelity metrics for a sampled colored cloud.")
    ap.add_argument("--cloud", required=True, help="sampled colored PLY")
    ap.add_argument("--ref", help="clean exterior OBJ (color reference)")
    ap.add_argument("--mesh", help="original OBJ; build the clean reference from it (needs pymeshlab)")
    args = ap.parse_args()

    ref = args.ref
    if ref is None:
        if not args.mesh:
            ap.error("give --ref clean.obj OR --mesh original.obj")
        ref = build_clean_reference(args.mesh, tempfile.mkdtemp(prefix="pcc_ref_"))

    m = evaluate(args.cloud, ref)
    print(f"{'metric':12}{'value':>12}   (lower = better)")
    for key in ("points", "speckle", "uniformity", "p2f", "coverage", "color_dE"):
        print(f"{key:12}{m[key]:12.4g}")
