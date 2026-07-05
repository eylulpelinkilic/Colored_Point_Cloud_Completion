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

from huggingface_hub import snapshot_download  # noqa: E402

SHAPENET_REPO = "ShapeNet/ShapeNetCore"                       # gated, third-party
DERIVED_REPO  = os.environ.get("HF_DATASET_REPO", "<your-hf-username>/colored-shapenet-completion")
SYNSETS       = ["02691156", "02958343", "03001627"]         # airplane, car, chair


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
    fetch_derived()
