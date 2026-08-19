"""Does this dataset carry enough part-colour signal to answer our question?

Run this BEFORE spending GPU on a new dataset. It measures, per category, how much
of the colour a point's PART LABEL explains -- by comparing two oracles that both
get the ground-truth answer handed to them:

    oracle-part    every point painted its own part's mean colour
    global-mean    every point painted the model's single mean colour
    headroom       global-mean - oracle-part   (what the part label buys, in dE)
    ratio          oracle-part / global-mean   (near 1.0 = parts say nothing)

Both are upper bounds: no part-CONSTANT colouring method can score below oracle-part.
A category with near-zero headroom cannot distinguish a part-aware method from a
part-blind one, so a null result there says something about the data, not the method.

WHAT THIS DOES NOT BOUND. A model that uses the part label as a *conditioning signal*
rather than as a colour lookup can beat oracle-part -- ours does, on airplane/simple
(14.90 vs a 15.37 ceiling). So low headroom is a warning that the comparison is
compressed, not proof that part-conditioning is useless. Read it alongside the
measured method spread, never instead of it.

Usage:
    python scripts/headroom.py                             # ours, from $PCC_DATA_ROOT/labeled_s3
    python scripts/headroom.py --root /path/to/densepoint --format ply
    python scripts/headroom.py --csv docs/headroom.csv
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT  # noqa: E402

# ShapeNet synsets we benchmark; anything else is reported under its own folder name.
SYNSETS = {"02691156": "airplane", "02958343": "car", "03001627": "chair"}

# A model whose mean per-point channel spread is below this is effectively greyscale:
# there is almost no colour for ANY method to get right or wrong.
GREY_CHROMA = 0.05


def srgb_to_lab(rgb):
    rgb = np.clip(rgb, 0, 1)
    lin = np.where(rgb > .04045, ((rgb + .055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
    x = (lin @ m.T) / np.array([.95047, 1., 1.08883])
    d = 6 / 29
    f = np.where(x > d ** 3, np.cbrt(x), x / (3 * d ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def delta_e(a, b):
    return np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=1)


# ---------------------------------------------------------------- readers
def read_npz(path):
    """Our labeled_s3 format: xyz (N,3) f32, rgb (N,3) f32 in [0,1], part (N,) i16."""
    z = np.load(path)
    part = z["part"].astype(np.int64) if "part" in z else None
    return np.clip(z["rgb"].astype(np.float64), 0, 1), part


def read_ply(path):
    """Minimal binary PLY reader -- avoids depending on plyfile for a diagnostic.

    Handles the layout DensePoint ships (xyz f32 / rgb uchar / normals f32 / label i32)
    and any other binary_little_endian vertex block built from scalar properties."""
    T = {"float": ("f4", 4), "float32": ("f4", 4), "double": ("f8", 8), "float64": ("f8", 8),
         "uchar": ("u1", 1), "uint8": ("u1", 1), "char": ("i1", 1), "int8": ("i1", 1),
         "ushort": ("u2", 2), "uint16": ("u2", 2), "short": ("i2", 2), "int16": ("i2", 2),
         "uint": ("u4", 4), "uint32": ("u4", 4), "int": ("i4", 4), "int32": ("i4", 4)}
    with open(path, "rb") as fh:
        fmt, n, props, in_vertex = None, 0, [], False
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if not line:
                raise ValueError(f"{path}: header ended without end_header")
            tok = line.split()
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError(f"{path}: list property in vertex block")
                props.append((tok[2], tok[1]))
            elif tok[0] == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: only binary_little_endian supported, got {fmt}")
        dt = np.dtype([(nm, T[ty][0]) for nm, ty in props])
        v = np.frombuffer(fh.read(dt.itemsize * n), dtype=dt, count=n)

    names = set(v.dtype.names)
    if {"red", "green", "blue"} <= names:
        # uchar channels are 0..255; float channels are already 0..1
        chan = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64)
        rgb = chan / 255.0 if chan.max() > 1.5 else chan
    else:
        return None, None                       # no colour: nothing to measure
    lab = next((c for c in ("label", "part", "part_label", "seg", "scalar_label")
                if c in names), None)
    return np.clip(rgb, 0, 1), (v[lab].astype(np.int64) if lab else None)


READERS = {"npz": ("*.npz", read_npz), "ply": ("*.ply", read_ply)}


# ---------------------------------------------------------------- measurement
def measure(rgb, part):
    """-> (oracle-part dE, global-mean dE, chroma spread). oracle is None without parts."""
    glob_de = float(delta_e(np.repeat(rgb.mean(0)[None], len(rgb), 0), rgb).mean())
    chroma = float(np.mean(rgb.max(1) - rgb.min(1)))
    if part is None:
        return None, glob_de, chroma
    pred = np.empty_like(rgb)
    for p in np.unique(part):
        m = part == p
        pred[m] = rgb[m].mean(0)
    return float(delta_e(pred, rgb).mean()), glob_de, chroma


def scan(folder, pattern, reader, limit):
    files = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    if limit:
        files = files[:limit]
    orc, glb, grey, parts = [], [], 0, set()
    for f in files:
        try:
            rgb, part = reader(f)
        except Exception as e:                  # one unreadable file must not kill a sweep
            print(f"    [skip] {os.path.basename(f)}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if rgb is None:
            continue
        if part is not None:
            parts.update(np.unique(part).tolist())
        o, g, c = measure(rgb, part)
        if o is not None:
            orc.append(o)
        glb.append(g)
        grey += c < GREY_CHROMA
    return orc, glb, grey, parts, len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(DATA_ROOT, "labeled_s3"),
                    help="folder holding per-category subfolders")
    ap.add_argument("--format", choices=sorted(READERS), default="npz")
    ap.add_argument("--limit", type=int, default=None, help="models per category (default all)")
    ap.add_argument("--csv", help="also write the table here")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"not a folder: {args.root}")
    pattern, reader = READERS[args.format]

    cats = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    if not cats:
        cats = [args.root]
    print(f"root: {args.root}   format: {args.format}\n")
    hdr = (f"{'category':14s} {'n':>4s} {'oracle':>8s} {'global':>8s} {'headroom':>9s} "
           f"{'ratio':>6s} {'grey':>6s} {'parts':>6s}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for c in cats:
        base = os.path.basename(c)
        name = SYNSETS.get(base, SYNSETS.get(base.split("_")[0], base))[:14]
        orc, glb, grey, parts, nf = scan(c, pattern, reader, args.limit)
        if not glb:
            print(f"{name:14s} {nf:4d}  -- no readable coloured models --")
            continue
        o = float(np.mean(orc)) if orc else float("nan")
        g = float(np.mean(glb))
        print(f"{name:14s} {len(glb):4d} {o:8.2f} {g:8.2f} {g-o:9.2f} {o/g:6.2f} "
              f"{100*grey/len(glb):5.1f}% {len(parts):6d}")
        rows.append(dict(category=name, n=len(glb), oracle_part=o, global_mean=g,
                         headroom=g - o, ratio=o / g,
                         grey_frac=grey / len(glb), n_parts=len(parts)))

    print("\nheadroom = what the part label buys, in dE.  ratio near 1.00 = parts say nothing.")
    print("Both are ORACLE bounds on part-CONSTANT colouring. A model that uses the part")
    print("label as a conditioning signal can beat them -- so read this next to the measured")
    print("method spread, never instead of it.")

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
