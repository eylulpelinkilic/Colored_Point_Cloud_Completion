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
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT  # noqa: E402

from huggingface_hub import HfApi, create_repo, HfFolder  # noqa: E402

# Derived subfolders to publish (skip whatever you don't have yet).
# These match the pipeline outputs: s2 dense GT, s3 dense + downsampled GT, partials.
DERIVED_SUBDIRS = ["data_s2", "data_s3", "gt_s3", "occluded_occ", "labeled_s3"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="",
                    help="folder inside the repo to publish under, e.g. 'pelin'. "
                         "Use this when SHARING a repo so contributions stay separated.")
    ap.add_argument("--subdirs", nargs="+", default=DERIVED_SUBDIRS)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what WOULD be uploaded and exit; touches nothing remote")
    ap.add_argument("--sleep", type=float, default=45,
                    help="seconds to pause between chunks (HF allows 1000 req / 5 min)")
    ap.add_argument("--retries", type=int, default=4,
                    help="retries per chunk, with exponential backoff")
    args = ap.parse_args()

    repo = os.environ.get("HF_DATASET_REPO")
    # HF_TOKEN, or whatever `huggingface-cli login` cached — no need to paste it twice.
    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not repo:
        sys.exit("Set HF_DATASET_REPO=<user>/<dataset-name>.")
    if not token and not args.dry_run:
        sys.exit("No token found. Run `huggingface-cli login`, or set HF_TOKEN.")

    plan = []
    for sub in args.subdirs:
        path = os.path.join(DATA_ROOT, sub)
        dest = f"{args.prefix.strip('/')}/{sub}" if args.prefix else sub
        if os.path.isdir(path):
            n = sum(len(f) for _, _, f in os.walk(path))
            mb = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(path) for f in fs) / 1e6
            plan.append((path, dest, n, mb))
        else:
            print("[skip] not found:", path)

    if not plan:
        sys.exit("Nothing to upload.")
    print(f"\nrepo: {repo}\n")
    for path, dest, n, mb in plan:
        print(f"  {path}\n    -> {dest}   ({n} files, {mb:.0f} MB)")
    print()
    if args.dry_run:
        print("dry-run: nothing was uploaded."); return

    api = HfApi(token=token)
    # Only create when the repo is genuinely missing. Calling create_repo on an
    # EXISTING shared repo risks touching settings that are not ours to change.
    try:
        api.dataset_info(repo, token=token)
        print("[info] repo exists — uploading into it, creating nothing.")
    except Exception:
        print("[info] repo not found — creating it PRIVATE.")
        create_repo(repo, repo_type="dataset", exist_ok=True, token=token, private=True)

    # HF rate-limits at 1000 API requests / 5 min and this client uploads roughly one
    # request per file. A few thousand small .ply files blow straight through that, so
    # upload leaf folder by leaf folder, pausing in between, and retry with backoff.
    def leaves(root):
        out = [(r, sorted(f)) for r, _, f in os.walk(root) if f]
        return sorted(out)

    def push(folder, dest, attempt=1):
        try:
            # additive by design: no delete_patterns, nothing already in the repo is removed
            api.upload_folder(folder_path=folder, path_in_repo=dest,
                              repo_id=repo, repo_type="dataset")
            return True
        except Exception as e:
            if attempt > args.retries:
                print(f"[FAIL] {dest}: {type(e).__name__}: {e}")
                return False
            wait = args.sleep * (2 ** (attempt - 1))
            print(f"[warn] {dest}: {type(e).__name__} — retry {attempt}/{args.retries} in {wait}s")
            time.sleep(wait)
            return push(folder, dest, attempt + 1)

    chunks = []
    for path, dest, _, _ in plan:
        for folder, files in leaves(path):
            rel = os.path.relpath(folder, path)
            chunks.append((folder, dest if rel == "." else f"{dest}/{rel}", len(files)))

    print(f"{len(chunks)} chunk, pausing {args.sleep}s between (rate-limit safety)\n")
    okc = 0
    for i, (folder, dest, n) in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {dest}  ({n} files)", flush=True)
        if push(folder, dest):
            okc += 1
        if i < len(chunks):
            time.sleep(args.sleep)
    print(f"\nDone. {okc}/{len(chunks)} chunks -> https://huggingface.co/datasets/{repo}")
    if okc < len(chunks):
        print("Re-run the same command: already-uploaded files are skipped.")


if __name__ == "__main__":
    main()
