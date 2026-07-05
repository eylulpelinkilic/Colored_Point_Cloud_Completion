"""Central data-root resolver for the Colored Point Cloud Completion project.

Data lives OUTSIDE git (HuggingFace Datasets / Google Drive) — this module tells
code where to find it, so the same notebooks/scripts run on Colab and locally.

Resolution order:
  1. $PCC_DATA_ROOT                        (set this locally / on a server / on Colab)
  2. ./data  next to the repo              (default)

Usage:
    from config import DATA_ROOT, subdir
    gt_dir = subdir("data_s2")
"""
import os

_LOCAL_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def resolve_data_root() -> str:
    return os.environ.get("PCC_DATA_ROOT") or _LOCAL_DEFAULT


DATA_ROOT = resolve_data_root()


def subdir(*parts: str) -> str:
    """Path to a subfolder of DATA_ROOT, e.g. subdir('data_s2', '02691156')."""
    return os.path.join(DATA_ROOT, *parts)


if __name__ == "__main__":
    print("DATA_ROOT =", DATA_ROOT)
    print("exists    =", os.path.isdir(DATA_ROOT))
