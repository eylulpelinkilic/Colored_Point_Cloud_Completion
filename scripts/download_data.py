"""Fetch project data into PCC_DATA_ROOT (resolved by config.py).

Two very different kinds of data:

1. RAW ShapeNet meshes (airplane / car / chair) — third-party and *gated*. NOT
   redistributed by this repo. To fetch them you must (a) accept the ShapeNet
   license on HuggingFace and (b) provide YOUR OWN token via the HF_TOKEN env var.
   The token is never stored in the repo.

2. The DERIVED benchmark (colored GT + occluded partials) — this project's own
   contribution, published as a PUBLIC HuggingFace dataset. Downloads with NO token.

Usage:
    export PCC_DATA_ROOT=/path/to/data          # optional; defaults to ./data
    export HF_TOKEN=hf_xxx                       # only needed for the raw ShapeNet step
    export HF_DATASET_REPO=<user>/<dataset>      # your published derived dataset
    python scripts/download_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT  # noqa: E402

from huggingface_hub import snapshot_download, hf_hub_download  # noqa: E402

SHAPENET_REPO = "ShapeNet/ShapeNetCore"                       # gated, third-party
DERIVED_REPO  = os.environ.get("HF_DATASET_REPO", "<your-hf-username>/colored-shapenet-completion")
SYNSETS       = ["02691156", "02958343", "03001627"]         # airplane, car, chair

# ShapeNet-Part (geometry + per-point part labels; NO color) — used to bring part
# structure into the colored benchmark. Same model_ids as ShapeNetCore, but a
# DIFFERENT coordinate frame -> align to the textured mesh first (scripts/align_partseg.py).
PARTSEG_REPO = os.environ.get("PARTSEG_HF_REPO", "ShapeSplats/sharing")   # HF mirror
PARTSEG_ZIP  = "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"  # 709MB, gated
PARTSEG_DIR  = "shapenetcore_partanno_segmentation_benchmark_v0_normal"       # unzipped top dir


def fetch_shapenet():
    """Raw ShapeNet zips -> DATA_ROOT/shapenetcore/  (needs HF_TOKEN + license accepted)."""
    token = os.environ.get("HF_TOKEN")
    dst = os.path.join(DATA_ROOT, "shapenetcore")
    if all(os.path.exists(os.path.join(dst, f"{s}.zip")) for s in SYNSETS):
        print("[ok]   ShapeNet zips already present ->", dst)
        return
    if not token:
        print("[skip] ShapeNet is gated. Accept its license on HuggingFace and set "
              "HF_TOKEN to fetch the airplane/car/chair zips.")
        return
    snapshot_download(repo_id=SHAPENET_REPO, repo_type="dataset", local_dir=dst,
                      allow_patterns=[f"{s}.zip" for s in SYNSETS], token=token)
    print("[ok]   ShapeNet zips ->", dst)


def fetch_partseg():
    """ShapeNet-Part labels -> DATA_ROOT/shapenet_part/  (needs HF_TOKEN, gated).

    Per-shape .txt = `x y z nx ny nz label`, global part id 0-49, no color. The
    color still comes from the textured ShapeNetCore mesh; this only supplies part
    structure. NOTE: its points live in ShapeNet-Part's own normalized frame, which
    generally does NOT match model_normalized.obj -> run scripts/align_partseg.py
    on one model per category before mass sampling.
    """
    import zipfile
    dst = os.path.join(DATA_ROOT, "shapenet_part")
    if os.path.isdir(os.path.join(dst, PARTSEG_DIR)):
        print("[ok]   ShapeNet-Part already present ->", dst)
        return
    token = os.environ.get("HF_TOKEN")
    try:
        zpath = hf_hub_download(repo_id=PARTSEG_REPO, repo_type="dataset",
                                filename=PARTSEG_ZIP, local_dir=dst, token=token)
    except Exception as e:
        print(f"[skip] ShapeNet-Part not fetched from '{PARTSEG_REPO}' (gated: accept "
              f"its terms + set HF_TOKEN, or set PARTSEG_HF_REPO to a mirror): {e}")
        return
    print("[..]   unzipping ShapeNet-Part (709 MB) ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dst)
    print("[ok]   ShapeNet-Part ->", os.path.join(dst, PARTSEG_DIR))


def fetch_derived():
    """Public derived benchmark -> DATA_ROOT/  (no token needed)."""
    try:
        snapshot_download(repo_id=DERIVED_REPO, repo_type="dataset", local_dir=DATA_ROOT)
        print("[ok]   derived benchmark ->", DATA_ROOT)
    except Exception as e:
        print(f"[skip] derived dataset '{DERIVED_REPO}' not fetched "
              f"(set HF_DATASET_REPO once it is published): {e}")


if __name__ == "__main__":
    os.makedirs(DATA_ROOT, exist_ok=True)
    print("DATA_ROOT =", DATA_ROOT)
    fetch_shapenet()
    fetch_partseg()
    fetch_derived()
