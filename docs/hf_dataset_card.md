---
license: other
license_name: shapenet-derived
license_link: https://shapenet.org/terms
task_categories:
  - other
tags:
  - point-cloud
  - point-cloud-completion
  - 3d
  - shapenet
  - colored-point-cloud
size_categories:
  - 1K<n<10K
---

# Colored point-cloud completion — PoinTr-protocol occlusions

Colored ground truth and occluded partial inputs for **colored** point-cloud completion,
built from textured ShapeNetCore meshes. Three categories, 599 models, 1797 partials.

What makes this different from the usual completion sets: the ground truth carries
**per-point color sampled from the mesh texture**, and it is de-speckled before sampling,
so the color is actually correct rather than plausible-looking.

## Contents

```
gt_s3/<synset>/<model_id>.ply                          colored GT, 8192 pts, xyzrgb
gt_s3/gt_manifest.csv
occluded_occ/<difficulty>/<synset>/<model_id>.ply          partial input
occluded_occ/<difficulty>/<synset>/<model_id>_missing.ply  removed region
occluded_occ/manifest.csv
```

`difficulty` ∈ `simple` (25 % removed) · `moderate` (50 %) · `hard` (75 %).

For every model **`partial ∪ missing = gt`** exactly — the two files are a partition of the
same 8192 points, so the removed region is itself labelled ground truth, not an
approximation.

| synset | category | models | partials |
|---|---|---|---|
| 02691156 | airplane | 200 | 600 |
| 02958343 | car | 199 | 597 |
| 03001627 | chair | 200 | 600 |
| **total** | | **599** | **1797** |

One car model (`15fcfe91d44c0e15e5c9256f048d92d2`) is absent — its texture is missing from
the source ShapeNetCore archive.

## How the ground truth was built

A naive colored mesh sample is *wrong*, not merely noisy: ShapeNet meshes carry
double-sided faces, so a sampler picks back-face colors and produces speckle. The fix is
part of the pipeline:

1. **pymeshlab ambient occlusion** → per-face visibility.
2. **Dual-face removal** — among faces sharing the same three vertices, keep the most
   visible one (Lazzarotto & Ebrahimi, EPFL/MMSPG,
   [arXiv:2201.06935](https://arxiv.org/abs/2201.06935)). On a single test model this alone
   moves color ΔE from **13.1 → 0.08**.
3. **Dense color sample** of the cleaned mesh (CloudCompare, 2¹⁷ points).
4. **Color-aware farthest-point sampling** → fixed 8192 points.

De-speckling has to run on the mesh *before* sampling, so the order is not optional.

## Occlusion protocol

**PoinTr / ShapeNet-55 style**, ported from `seprate_point_cloud`: draw a random unit
viewpoint `v`, sort points by distance to `v`, remove the nearest `crop_ratio · N`.

This is *not* the PCN protocol. PCN back-projects depth renders from 8 viewpoints and
models a real range scan; this one removes a **known fraction** of the surface and is meant
for controlled difficulty sweeps. They are different task definitions — a benchmark wants
both, and results are not interchangeable.

`occluded_occ/manifest.csv` records `synset, model, difficulty, crop_ratio, view, seed,
n_partial, n_missing, gt_ply`. The `seed` makes every partial reproducible from the GT alone.

## Loading

```python
import open3d as o3d, numpy as np

def load(p):
    pc = o3d.io.read_point_cloud(p)
    return np.asarray(pc.points), np.asarray(pc.colors)   # (N,3) xyz, (N,3) rgb in [0,1]

xyz_in,  rgb_in  = load("occluded_occ/moderate/02691156/<model>.ply")          # input
xyz_tgt, rgb_tgt = load("occluded_occ/moderate/02691156/<model>_missing.ply")  # target
xyz_gt,  rgb_gt  = load("gt_s3/02691156/<model>.ply")                          # input ∪ target
```

## Suggested metric

Color error on the **missing region only** — mean ΔE in CIELAB between prediction and
ground truth. Evaluating over the whole cloud hides the task: the visible part is given.

Note that ΔE is a *distortion* metric and provably rewards the conditional mean, so it
favours averaging baselines over generative ones (distortion–perception trade-off, Blau &
Michaeli, 2018). Report a distribution metric alongside it.

## Provenance and license

Derived from **ShapeNetCore v2** meshes and **ShapeNet-Part** model lists. Only derived
point clouds are published — the source meshes are **not** redistributed. ShapeNet's
original terms govern anything derived from it; read them before redistributing.

Produced by `scripts/build_gt.py` → `scripts/build_occluded.py`.

## Related

A companion dataset by the same project covers the **PCN protocol** (8 depth-render views
per model) over the same three categories:
[`efeyenice/pc-completion-data`](https://huggingface.co/datasets/efeyenice/pc-completion-data).
