"""Toy Stage-0 for part-aware colored completion: does part-consistent coloring
beat nearest-neighbour colour propagation when the "parts are colour-consistent"
assumption holds (exactly, or only approximately)?

Idea being tested
-----------------
The completion task is: xyzrgb partial -> xyzrgb complete. Geometry can be filled
by an uncolored net (PoinTr); the open question is how to COLOUR the newly filled
points. The current baseline copies colour from the nearest *visible* point
(NN-propagation). It fails exactly where a filled point's nearest visible neighbour
belongs to a *different* part (e.g. a mostly-occluded backrest inherits seat colour).
The proposed fix: label every filled point with its part, and colour it from that
part's own (visible) colour model. This script isolates that claim.

Three colouring strategies compared
-----------------------------------
* baseline  : NN colour from the nearest visible point.
* method    : mean colour of the point's part over the *visible* points (oracle seg).
* ceiling   : mean colour of the point's part over the *full* cloud -- the best any
              per-part-constant rule could do; equals the irreducible within-part
              colour variance (0 iff colour is perfectly flat).
gap(baseline, method)  = cross-part colour-bleed removed by the idea.
gap(method,  ceiling)  = cost of estimating a part's colour from visible points only
                         (blows up when a part is barely/never visible).

Staging (this file = Stage 0)
-----------------------------
Oracle geometry (the "completed" points are the true xyz of the occluded region) and
oracle segmentation, so the ONLY thing under test is the colouring rule. `colour_noise`
makes the assumption *approximately* true; `--sweep` averages over objects, occlusion
ratios and viewpoints. Later stages swap in PoinTr geometry and a learned PointNet
part-seg -- each re-introduces one error source, one at a time.

Run:
    python scripts/toy_part_color.py                 # single object, ΔE table + HTML
    python scripts/toy_part_color.py --sweep         # crop 25/50/75% curve -> CSV + HTML
"""
import os
import sys
import argparse
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import subdir  # noqa: E402


# ------------------------------------------------------------------ toy geometry
def box_surface(center, size, n, rng):
    """Sample n points on the 6 faces of an axis-aligned box (area-weighted)."""
    center = np.asarray(center, float); size = np.asarray(size, float)
    hx, hy, hz = size / 2.0
    areas = np.array([hy * hz, hy * hz, hx * hz, hx * hz, hx * hy, hx * hy]) * 4
    counts = np.random.default_rng(rng.integers(1 << 30)).multinomial(n, areas / areas.sum())
    pts = []
    for face, k in enumerate(counts):
        if k == 0:
            continue
        u = rng.uniform(-1, 1, k); v = rng.uniform(-1, 1, k)
        if face == 0:   p = np.stack([np.full(k, 1.0), u, v], 1)
        elif face == 1: p = np.stack([np.full(k, -1.0), u, v], 1)
        elif face == 2: p = np.stack([u, np.full(k, 1.0), v], 1)
        elif face == 3: p = np.stack([u, np.full(k, -1.0), v], 1)
        elif face == 4: p = np.stack([u, v, np.full(k, 1.0)], 1)
        else:           p = np.stack([u, v, np.full(k, -1.0)], 1)
        pts.append(p * np.array([hx, hy, hz]) + center)
    return np.concatenate(pts, 0)


# part template: name -> (center, size). Colour is assigned per-object (palette).
PARTS = [
    ("seat",  (0.0, 0.90, 0.00), (2.0, 0.15, 2.0)),
    ("back",  (0.0, 1.65, -0.93), (2.0, 1.5, 0.15)),
    ("leg_a", (-0.9, 0.45, -0.9), (0.16, 0.9, 0.16)),
    ("leg_b", (0.9, 0.45, -0.9), (0.16, 0.9, 0.16)),
    ("leg_c", (-0.9, 0.45, 0.9), (0.16, 0.9, 0.16)),
    ("leg_d", (0.9, 0.45, 0.9), (0.16, 0.9, 0.16)),
]
NAMES = [p[0] for p in PARTS]


def build_toy(n=8192, seed=0, jitter=0.0, colour_noise=0.0, fixed_palette=None):
    """Return xyz (N,3), rgb (N,3), part (N,) int.

    jitter        : per-object geometry jitter (fraction) for a mini-set of objects.
    colour_noise  : per-point Gaussian sigma on rgb -> assumption only *approximately* true.
    fixed_palette : (P,3) flat colours; if None a random palette is drawn per object.
    """
    rng = np.random.default_rng(seed)
    palette = fixed_palette if fixed_palette is not None else rng.uniform(0.12, 0.9, (len(PARTS), 3))
    sizes = np.array([s for _, _, s in PARTS], float)
    area = 2 * (sizes[:, 0] * sizes[:, 1] + sizes[:, 1] * sizes[:, 2] + sizes[:, 0] * sizes[:, 2])
    counts = rng.multinomial(n, area / area.sum())
    xyz, rgb, part = [], [], []
    for pid, ((name, c, s), k) in enumerate(zip(PARTS, counts)):
        if k == 0:
            continue
        c = np.asarray(c, float) * (1 + rng.uniform(-jitter, jitter, 3))
        s = np.asarray(s, float) * (1 + rng.uniform(-jitter, jitter, 3))
        p = box_surface(c, s, k, rng)
        col = palette[pid] + (rng.normal(0, colour_noise, (k, 3)) if colour_noise else 0.0)
        xyz.append(p); rgb.append(np.clip(col, 0, 1)); part.append(np.full(k, pid))
    return np.concatenate(xyz), np.concatenate(rgb), np.concatenate(part)


# ------------------------------------------------------------------ occlusion
def occlude(xyz, crop_ratio=0.5, viewpoint=None, seed=0):
    """PoinTr seprate_point_cloud: drop the crop_ratio points nearest a viewpoint
    direction (on the RMS-normalised sphere). Returns (visible_idx, missing_idx)."""
    n = len(xyz); num_crop = int(round(n * crop_ratio))
    c = xyz.mean(0); r = np.linalg.norm(xyz - c, axis=1).max() + 1e-12
    xyzn = (xyz - c) / r
    v = np.random.default_rng(seed).standard_normal(3) if viewpoint is None else np.asarray(viewpoint, float)
    center = v / np.linalg.norm(v)
    idx = np.argsort(np.linalg.norm(xyzn - center[None], axis=1))
    return idx[num_crop:], idx[:num_crop]


# ------------------------------------------------------------------ colouring rules
def colour_nn(vis_xyz, vis_rgb, query_xyz):
    _, i = cKDTree(vis_xyz).query(query_xyz, k=1)
    return vis_rgb[i], i


def part_means(rgb, part, n_parts, fallback):
    means = np.tile(fallback, (n_parts, 1))
    absent = []
    for p in range(n_parts):
        m = part == p
        if m.any():
            means[p] = rgb[m].mean(0)
        else:
            absent.append(p)
    return means, absent


# ------------------------------------------------------------------ metrics
def srgb_to_lab(rgb):
    rgb = np.clip(rgb, 0, 1)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883]); d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def delta_e(a, b):
    return np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=1)


def colour_scene(xyz, rgb, part, vis, miss):
    """Colour the missing region 3 ways; return per-strategy mean ΔE + diagnostics."""
    q_xyz, q_rgb, q_part = xyz[miss], rgb[miss], part[miss]
    glob = rgb[vis].mean(0)
    nn_rgb, nn_src = colour_nn(xyz[vis], rgb[vis], q_xyz)
    vis_means, absent = part_means(rgb[vis], part[vis], len(NAMES), glob)   # method (visible)
    full_means, _ = part_means(rgb, part, len(NAMES), glob)                 # ceiling (full)
    return {
        "baseline": float(delta_e(nn_rgb, q_rgb).mean()),
        "method": float(delta_e(vis_means[q_part], q_rgb).mean()),
        "ceiling": float(delta_e(full_means[q_part], q_rgb).mean()),
        "wrong_part_frac": float((part[vis][nn_src] != q_part).mean()),
        "fully_occluded": bool(len(set(absent) & set(np.unique(q_part).tolist()))),
        "colours": (nn_rgb, vis_means[q_part]),
    }


# ------------------------------------------------------------------ single-object report
def run_single(args):
    xyz, rgb, part = build_toy(args.n, seed=0, colour_noise=args.colour_noise)
    vis, miss = occlude(xyz, args.crop, args.viewpoint)
    print(f"toy: {len(xyz)} pts, {len(NAMES)} parts   crop={args.crop}  colour_noise={args.colour_noise}")
    print("missing-point count per part:",
          {NAMES[p]: int((part[miss] == p).sum()) for p in range(len(NAMES))})
    r = colour_scene(xyz, rgb, part, vis, miss)
    print("\n--- mean ΔE(Lab) on the completed (missing) region ---")
    print(f"  baseline (NN-propagation) : {r['baseline']:7.3f}")
    print(f"  method   (part-mean, vis) : {r['method']:7.3f}")
    print(f"  ceiling  (part-mean, full): {r['ceiling']:7.3f}")
    print(f"  baseline drew colour cross-part for {r['wrong_part_frac']*100:.1f}% of missing points"
          + ("   ! a part was fully occluded (fallback fired)" if r["fully_occluded"] else ""))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        nn_rgb, pm_rgb = r["colours"]
        path = os.path.join(args.out, "toy_part_color.html")
        write_panels(path, [
            ("partial (input)", xyz[vis], rgb[vis]),
            ("baseline: NN colour", np.vstack([xyz[vis], xyz[miss]]), np.vstack([rgb[vis], nn_rgb])),
            ("method: part-mean", np.vstack([xyz[vis], xyz[miss]]), np.vstack([rgb[vis], pm_rgb])),
            ("GT", xyz, rgb),
        ])
        print("\nviz ->", path)


# ------------------------------------------------------------------ sweep
def run_sweep(args):
    crops = [0.25, 0.5, 0.75]
    rows = []                      # (crop, strategy, mean, std)
    fig_data = {"baseline": [], "method": [], "ceiling": []}
    print(f"sweep: {args.objects} objects x {args.views} viewpoints, colour_noise={args.colour_noise}")
    for crop in crops:
        acc = {"baseline": [], "method": [], "ceiling": []}
        wrong, occl = [], []
        for o in range(args.objects):
            xyz, rgb, part = build_toy(args.n, seed=100 + o, jitter=args.jitter,
                                       colour_noise=args.colour_noise)
            for v in range(args.views):
                vis, miss = occlude(xyz, crop, seed=1000 * o + v)
                r = colour_scene(xyz, rgb, part, vis, miss)
                for k in acc:
                    acc[k].append(r[k])
                wrong.append(r["wrong_part_frac"]); occl.append(r["fully_occluded"])
        line = f"crop {int(crop*100):>2}% |"
        for k in ("baseline", "method", "ceiling"):
            a = np.array(acc[k]); rows.append((crop, k, a.mean(), a.std()))
            fig_data[k].append((crop, a.mean(), a.std()))
            line += f"  {k}={a.mean():6.2f}±{a.std():4.2f}"
        line += f"  | cross-part={np.mean(wrong)*100:4.1f}%  fully-occl-scenes={np.mean(occl)*100:4.1f}%"
        print(line)

    out = args.out or subdir("eval")
    os.makedirs(out, exist_ok=True)
    csv = os.path.join(out, "toy_color_sweep.csv")
    with open(csv, "w") as f:
        f.write("crop,strategy,delta_e_mean,delta_e_std\n")
        for crop, k, m, s in rows:
            f.write(f"{crop},{k},{m:.4f},{s:.4f}\n")
    print("\ncsv ->", csv)
    write_curve(os.path.join(out, "toy_color_sweep.html"), fig_data)
    print("viz ->", os.path.join(out, "toy_color_sweep.html"))


# ------------------------------------------------------------------ viz helpers
def write_panels(path, panels):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(panels), specs=[[{"type": "scene"}] * len(panels)],
                        subplot_titles=[t for t, _, _ in panels])
    for col, (_, xyz, rgb) in enumerate(panels, 1):
        c = ["rgb(%d,%d,%d)" % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb]
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                                   marker=dict(size=1.8, color=c)), 1, col)
    fig.update_layout(height=520, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    for i in range(1, len(panels) + 1):
        fig.layout["scene" if i == 1 else f"scene{i}"].aspectmode = "data"
    fig.write_html(path)


def write_curve(path, fig_data):
    import plotly.graph_objects as go
    fig = go.Figure()
    label = {"baseline": "baseline (NN-propagation)", "method": "method (part-mean, oracle seg)",
             "ceiling": "ceiling (per-part-constant best)"}
    for k, pts in fig_data.items():
        x = [c * 100 for c, _, _ in pts]; y = [m for _, m, _ in pts]; e = [s for _, _, s in pts]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label[k],
                                 error_y=dict(type="data", array=e, visible=True)))
    fig.update_layout(xaxis_title="occlusion %", yaxis_title="mean ΔE(Lab) on completed region",
                      height=460, template="plotly_white")
    fig.write_html(path)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="run the crop 25/50/75% mini-set sweep")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--crop", type=float, default=0.5, help="single-object mode occlusion ratio")
    ap.add_argument("--viewpoint", type=float, nargs=3, default=[0.15, 0.55, -1.0])
    ap.add_argument("--colour-noise", dest="colour_noise", type=float, default=0.03,
                    help="per-point rgb sigma; 0 => assumption exactly true")
    ap.add_argument("--objects", type=int, default=12, help="sweep: distinct objects")
    ap.add_argument("--views", type=int, default=4, help="sweep: viewpoints per object")
    ap.add_argument("--jitter", type=float, default=0.1, help="sweep: per-object geometry jitter")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    (run_sweep if args.sweep else run_single)(args)


if __name__ == "__main__":
    main()
