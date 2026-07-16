"""Build the colored GT end-to-end in ONE step: dense colored sample -> de-speckle
(dual-face removal) -> FPS downsample to a fixed size. Consolidates sample_s1/s2/s3.

Why staged internally (not "just sample 8192")
-----------------------------------------------
De-speckling (dual-face removal) needs a DENSE cloud to tell the true surface
from back-face speckle, and it must run BEFORE downsampling (it deletes points, so
the count is variable). So the order dense-sample -> de-speckle -> FPS is mandatory;
this script just chains it and, by default, only writes the final fixed-size GT
(pass --keep-dense to also keep the ~2^20 clean cloud = the old data_s2).

Pipeline per model
-------------------
  1) pymeshlab ambient occlusion -> per-face visibility
  2) among faces sharing the same 3 vertices, keep the most-visible one (drop dual faces)
  3) rewrite a clean .obj (+ mtl/textures) and sample it dense with CloudCompare
  4) color-aware FPS -> fixed-size GT (geometry base + optional color-edge budget)

Requirements (NOT installed here — this is the pipeline, not env setup):
  CloudCompare (desktop binary), pymeshlab, open3d, scipy, numpy. On Colab/Ubuntu:
  `apt-get install -y cloudcompare xvfb && pip install pymeshlab open3d`.

Usage
-----
    python scripts/build_gt.py \
        --mesh-root $PCC_DATA_ROOT/shapenet_extracted \  # <synset>/<model>/models/model_normalized.obj
        --out-dir   $PCC_DATA_ROOT \
        [--synset 02691156] [--limit 2] [--n-gt 8192] [--color-frac 0.0] [--keep-dense]
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT  # noqa: E402


# ---------------------------------------------------------------- CloudCompare
def find_cc_bin():
    for c in (shutil.which("CloudCompare"), shutil.which("cloudcompare"),
              "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare"):
        if c and os.path.exists(c):
            return c
    sys.exit("CloudCompare not found. Colab/Ubuntu: apt-get install -y cloudcompare xvfb; "
             "macOS: brew install --cask cloudcompare.")


_XVFB = "xvfb-run -a " if shutil.which("xvfb-run") else ""


def _cc_sample(cc_bin, obj_path, n_points):
    """Run CloudCompare mesh sampling; return the produced *_SAMPLED_POINTS.ply path."""
    cmd = (f'{_XVFB}{cc_bin} -SILENT -AUTO_SAVE OFF -C_EXPORT_FMT PLY '
           f'-O "{obj_path}" -SAMPLE_MESH POINTS {n_points} -SAVE_CLOUDS')
    subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    produced = [p for p in glob.glob(os.path.join(os.path.dirname(obj_path), "*.ply"))
                if "SAMPLED_POINTS" in os.path.basename(p)]
    if not produced:
        raise RuntimeError(f"CloudCompare produced no PLY for {obj_path}")
    return produced[0]


# ---------------------------------------------------------------- de-speckle (dual-face removal)
def _write_ao_helper(work_dir):
    """pymeshlab ambient_occlusion is state-heavy -> run it in a subprocess for stability."""
    path = os.path.join(work_dir, "_ao_helper.py")
    with open(path, "w") as fh:
        fh.write(
            "import sys, numpy as np, pymeshlab\n"
            "in_dir, in_file, fm_file, fq_file = sys.argv[1:5]\n"
            "import os; os.chdir(in_dir)\n"
            "ms = pymeshlab.MeshSet(); ms.load_new_mesh(in_file)\n"
            "ms.compute_scalar_ambient_occlusion()\n"
            "m = ms.mesh(0)\n"
            "np.save(fm_file, m.face_matrix())\n"
            "np.save(fq_file, m.face_scalar_array())\n"
        )
    return path


def _duplicate_faces_to_delete(face_matrix, face_quality, threshold=0.0):
    """Dual-face logic, vectorized: faces sharing the same 3 vertices -> keep the highest-
    occlusion one, drop the rest; also drop faces with occlusion < threshold."""
    keys = np.sort(face_matrix, axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ks = keys[order]
    change = np.any(ks[1:] != ks[:-1], axis=1)
    grp_start = np.concatenate(([0], np.nonzero(change)[0] + 1))
    grp_end = np.concatenate((grp_start[1:], [len(order)]))
    to_delete = np.zeros(face_matrix.shape[0], dtype=bool)
    for s, e in zip(grp_start, grp_end):
        members = order[s:e]
        if len(members) == 1:
            continue
        members = np.sort(members)
        keep = members[np.argmax(face_quality[members])]
        to_delete[members[members != keep]] = True
    return to_delete | (face_quality < threshold)


def _ensure_face_correspondence(obj_lines, face_matrix):
    """Align the raw .obj face list with pymeshlab's face_matrix (pymeshlab may skip
    faces on load). Returns aligned obj_lines, or None if irreconcilable."""
    nonempty = [(i, x) for i, x in enumerate(obj_lines) if len(x) > 0]
    fobj = [[x.split()[1].split('/')[0], x.split()[2].split('/')[0],
             x.split()[3].split('/')[0]] for _, x in nonempty if x[0] == 'f']
    fobj = np.asarray(fobj, dtype=int) - 1
    if np.array_equal(face_matrix, fobj):
        return obj_lines
    fidx = [i for i, x in nonempty if x[0] == 'f']
    dropped = []
    while fobj.shape[0] > face_matrix.shape[0]:
        corr = (face_matrix == fobj[:face_matrix.shape[0], :]).all(axis=1)
        if not corr.all():
            d = np.nonzero(np.logical_not(corr))[0][0]
            fobj = np.delete(fobj, d, axis=0)
            dropped.append(fidx.pop(d))
        else:
            dropped += fidx[face_matrix.shape[0]:]
            fobj = fobj[:face_matrix.shape[0], :]
    if not np.array_equal(face_matrix, fobj):
        return None
    return list(np.delete(np.asarray(obj_lines), dropped))


def sample_despeckled_dense(obj_path, out_ply, cc_bin, work_dir, n_points, ao_helper,
                            occlusion_threshold=0.0):
    """Clean-mesh colored sampling -> dense de-speckled PLY at out_ply."""
    in_dir, in_file = os.path.split(obj_path)
    fm_file, fq_file = "face_matrix.npy", "face_quality.npy"
    subprocess.run(["python", ao_helper, in_dir, in_file, fm_file, fq_file],
                   capture_output=True, timeout=600)
    fm_path, fq_path = os.path.join(in_dir, fm_file), os.path.join(in_dir, fq_file)
    if not (os.path.exists(fm_path) and os.path.exists(fq_path)):
        raise RuntimeError("ambient_occlusion produced no output (pymeshlab failed).")
    face_matrix = np.load(fm_path); face_quality = np.load(fq_path)
    os.remove(fm_path); os.remove(fq_path)

    drop = _duplicate_faces_to_delete(face_matrix, face_quality, occlusion_threshold)
    with open(obj_path) as f:
        obj_lines = f.read().splitlines()
    obj_lines = _ensure_face_correspondence(obj_lines, face_matrix)
    if obj_lines is None:
        raise RuntimeError("face mismatch between pymeshlab and .obj; skipping model.")
    nonempty = [(i, x) for i, x in enumerate(obj_lines) if len(x) > 0]
    fidx = np.asarray([i for i, x in nonempty if x[0] == 'f'])
    obj_lines = list(np.delete(np.asarray(obj_lines), fidx[drop]))

    clean_dir = os.path.join(work_dir, "_clean", "models")
    shutil.rmtree(os.path.join(work_dir, "_clean"), ignore_errors=True)
    os.makedirs(clean_dir, exist_ok=True)
    mtl_i, mtl_line = next((i, x) for i, x in enumerate(obj_lines) if "mtllib" in x)
    mtl_file = next(t for t in mtl_line.split() if t.endswith(".mtl"))
    clean_mtl = "clean.mtl"
    shutil.copyfile(os.path.join(in_dir, mtl_file), os.path.join(clean_dir, clean_mtl))
    obj_lines[mtl_i] = f"mtllib {clean_mtl}"
    clean_obj = os.path.join(clean_dir, "clean.obj")
    with open(clean_obj, "w") as f:
        f.write("\n".join(obj_lines))
    with open(os.path.join(clean_dir, clean_mtl)) as f:
        for line in f.read().splitlines():
            if "map_Kd" in line:
                tex = line.split()[1]
                rel, name = os.path.split(tex)
                dst = os.path.join(clean_dir, rel); os.makedirs(dst, exist_ok=True)
                shutil.copyfile(os.path.join(in_dir, tex), os.path.join(dst, name))

    produced = _cc_sample(cc_bin, clean_obj, n_points)
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    shutil.move(produced, out_ply)
    shutil.rmtree(os.path.join(work_dir, "_clean"), ignore_errors=True)
    return out_ply


# ---------------------------------------------------------------- downsample
def farthest_point_sampling(points, n_samples, seed=None):
    N = points.shape[0]
    if n_samples >= N:
        return np.arange(N)
    xyz = points[:, :3]; rng = np.random.default_rng(seed)
    sel = np.zeros(n_samples, dtype=np.int64); dist = np.full(N, np.inf)
    sel[0] = rng.integers(N)
    for i in range(1, n_samples):
        dist = np.minimum(dist, np.sum((xyz - xyz[sel[i - 1]]) ** 2, axis=1))
        sel[i] = np.argmax(dist)
    return sel


def color_gradient(points, k=16):
    from scipy.spatial import cKDTree
    xyz, rgb = points[:, :3], points[:, 3:6]
    _, idx = cKDTree(xyz).query(xyz, k=min(k + 1, len(xyz)))
    return np.linalg.norm(rgb - rgb[idx[:, 1:]].mean(axis=1), axis=1)


def color_aware_sample(points, n_total, color_frac=0.0, k=16, seed=42):
    """Downsample to n_total (n_total,6). color_frac>0 draws the extra toward color edges."""
    rng = np.random.default_rng(seed); N = points.shape[0]
    if n_total >= N:
        return points.copy()
    n_color = int(n_total * color_frac); n_geo = n_total - n_color
    geo_idx = farthest_point_sampling(points, n_geo, seed=seed)
    if n_color == 0:
        return points[geo_idx]
    mask = np.ones(N, dtype=bool); mask[geo_idx] = False
    rem = np.nonzero(mask)[0]; grad = color_gradient(points, k=k)[rem]
    p = None if grad.sum() <= 0 else grad / grad.sum()
    col_idx = rng.choice(rem, size=min(n_color, len(rem)), replace=False, p=p)
    return points[np.concatenate([geo_idx, col_idx])]


# ---------------------------------------------------------------- IO
def load_ply_xyzrgb(ply_path):
    import open3d as o3d
    pc = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pc.points); rgb = np.asarray(pc.colors)
    if rgb.shape[0] != xyz.shape[0]:
        rgb = np.zeros_like(xyz)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def save_ply_xyzrgb(arr, out_ply):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(arr[:, 3:6], 0, 1).astype(np.float64))
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pc)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-root", required=True,
                    help="extracted ShapeNetCore: <synset>/<model>/models/model_normalized.obj")
    ap.add_argument("--out-dir", default=DATA_ROOT, help="writes gt_s3/ (+ data_s2/ if --keep-dense)")
    ap.add_argument("--synset", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-dense", type=int, default=1_048_576, help="dense sample size (2^20)")
    ap.add_argument("--n-gt", type=int, default=8192, help="fixed GT size after FPS")
    ap.add_argument("--color-frac", type=float, default=0.0, help="0=pure FPS; >0 adds color-edge budget")
    ap.add_argument("--occlusion-threshold", type=float, default=0.0)
    ap.add_argument("--keep-dense", action="store_true", help="also keep the ~2^20 de-speckled cloud (data_s2)")
    args = ap.parse_args()

    cc_bin = find_cc_bin()
    work = os.path.join(args.out_dir, "_build_gt_work"); os.makedirs(work, exist_ok=True)
    ao_helper = _write_ao_helper(work)
    gt_root = os.path.join(args.out_dir, "gt_s3")
    dense_root = os.path.join(args.out_dir, "data_s2")

    # accept category folders named airplane/car/chair OR by synset id; output stays synset-keyed
    SYNSET_BY_NAME = {"airplane": "02691156", "car": "02958343", "chair": "03001627"}
    to_synset = lambda name: SYNSET_BY_NAME.get(name.lower(), name)

    folders = sorted(d for d in os.listdir(args.mesh_root)
                     if os.path.isdir(os.path.join(args.mesh_root, d)))
    if args.synset:  # filter by name or synset id
        folders = [f for f in folders if args.synset in (f, to_synset(f))]

    def find_obj(model_dir):
        """model dir -> its .obj (prefer model_normalized.obj), searched recursively."""
        objs = glob.glob(os.path.join(model_dir, "**", "*.obj"), recursive=True)
        if not objs:
            return None
        norm = [o for o in objs if os.path.basename(o) == "model_normalized.obj"]
        return (norm or sorted(objs))[0]

    rows, done, failed = [], 0, 0
    for folder in folders:
        syn = to_synset(folder)
        model_dirs = sorted(d for d in glob.glob(os.path.join(args.mesh_root, folder, "*"))
                            if os.path.isdir(d))
        for md in model_dirs:
            mid = os.path.basename(md)
            obj = find_obj(md)
            if obj is None:
                print(f"[skip] {folder}/{mid}: .obj bulunamadı"); continue
            dense_ply = (os.path.join(dense_root, syn, mid + ".ply") if args.keep_dense
                         else os.path.join(work, "_dense.ply"))
            try:
                sample_despeckled_dense(obj, dense_ply, cc_bin, work, args.n_dense, ao_helper,
                                        args.occlusion_threshold)
                dense = load_ply_xyzrgb(dense_ply)
                gt = color_aware_sample(dense, args.n_gt, color_frac=args.color_frac)
                gt_ply = os.path.join(gt_root, syn, mid + ".ply")
                save_ply_xyzrgb(gt, gt_ply)
                ok_color = float(gt[:, 3:6].std()) > 0.01
                rows.append((syn, mid, len(gt), round(float(gt[:, 3:6].std()), 4), ok_color))
                done += 1
                print(f"[ ok ] {syn}/{mid}  dense={len(dense)} -> gt={len(gt)}  color_ok={ok_color}")
            except Exception as e:
                failed += 1
                print(f"[FAIL] {syn}/{mid}: {e}")
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break

    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(gt_root, exist_ok=True)
    with open(os.path.join(gt_root, "gt_manifest.csv"), "w") as f:
        f.write("synset,model,n,color_std,color_ok\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\nbuilt {done} GT clouds ({failed} failed). n_gt={args.n_gt} color_frac={args.color_frac}")
    print(f"GT -> {gt_root}" + (f"  | dense -> {dense_root}" if args.keep_dense else "  (dense discarded)"))


if __name__ == "__main__":
    main()
