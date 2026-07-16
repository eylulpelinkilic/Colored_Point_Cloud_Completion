"""Publish the DERIVED benchmark (your contribution) to HuggingFace Datasets.

Uploads only derived outputs (colored GT + occluded partials) — NEVER the raw
ShapeNet meshes (redistribution is not permitted). Needs YOUR HF_TOKEN; the token
stays in your environment and is never written to the repo.

Usage:
    export PCC_DATA_ROOT=/path/to/data
    export HF_DATASET_REPO=<user>/colored-shapenet-completion
    export HF_TOKEN=hf_xxx
    python scripts/upload_dataset.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT  # noqa: E402

from huggingface_hub import HfApi, create_repo  # noqa: E402

# Derived subfolders to publish (skip whatever you don't have yet).
# These match the pipeline outputs: s2 dense GT, s3 dense + downsampled GT, partials.
DERIVED_SUBDIRS = ["data_s2", "data_s3", "gt_s3", "occluded_occ", "labeled_s3"]


def main():
    repo = os.environ.get("HF_DATASET_REPO")
    token = os.environ.get("HF_TOKEN")
    if not repo:
        sys.exit("Set HF_DATASET_REPO=<user>/<dataset-name>.")
    if not token:
        sys.exit("Set HF_TOKEN (your HuggingFace write token).")

    api = HfApi(token=token)
    # PRIVATE by default — flip to public deliberately at paper time (note: exist_ok
    # won't change an existing repo's visibility; set it in the HF UI if it already exists).
    create_repo(repo, repo_type="dataset", exist_ok=True, token=token, private=True)

    uploaded = []
    for sub in DERIVED_SUBDIRS:
        path = os.path.join(DATA_ROOT, sub)
        if os.path.isdir(path):
            api.upload_folder(folder_path=path, path_in_repo=sub,
                              repo_id=repo, repo_type="dataset")
            uploaded.append(sub)
            print("[ok]   uploaded", sub)
        else:
            print("[skip] not found:", path)
    print(f"\nDone. Published {uploaded} to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
