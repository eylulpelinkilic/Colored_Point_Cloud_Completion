# Colored Point Cloud Completion

Building a **colored point-cloud completion benchmark** from textured ShapeNet
meshes. The pipeline samples color + geometry from meshes, cleans double-surface
speckle, downsamples to fixed-size ground truth, and generates partial (occluded)
inputs for the completion task.

> **Code lives here on GitHub; data does not.** Point clouds and the raw ShapeNet
> zips are far too large for git — they live on HuggingFace / Google Drive and are
> pulled in at runtime (see [Data](#data)).

## Pipeline

| Notebook | Stage | Output |
|---|---|---|
| [`notebooks/sample_s1.ipynb`](notebooks/sample_s1.ipynb) | Naive CloudCompare mesh sampling (65k pts) | dense colored PLY, with double-surface speckle |
| [`notebooks/sample_s2.ipynb`](notebooks/sample_s2.ipynb) | + EPFL/MMSPG ambient-occlusion dual-face removal, 2²⁰ pts | de-speckled dense colored PLY (ground truth) |
| [`notebooks/sample_s3.ipynb`](notebooks/sample_s3.ipynb) | FPS downsample (8192 / 16384) with optional color-edge budget | fixed-size GT clouds |
| [`notebooks/occluded.ipynb`](notebooks/occluded.ipynb) | Partial/occluded generation (PoinTr `seprate_point_cloud`, ported + colored) | partial inputs at 25 / 50 / 75 % occlusion |
| [`notebooks/eval.ipynb`](notebooks/eval.ipynb) | Mesh-fidelity evaluation (naive vs EPFL): speckle, uniformity, p2f, coverage, color ΔE | `data/eval/fidelity.csv` |

## Repository layout

```
config.py            # DATA_ROOT resolver ($PCC_DATA_ROOT > ./data)
requirements.txt     # Python deps (CloudCompare is apt/brew, not pip)
notebooks/           # the pipeline
scripts/             # download_data.py, upload_dataset.py, eval_fidelity.py
docs/adr/            # architecture decision records
data/                # gitignored — where data lives locally
```

## Setup

```bash
pip install -r requirements.txt
# CloudCompare is a desktop app, not a pip package:
#   Colab / Ubuntu:  apt-get install -y cloudcompare xvfb
#   macOS:           brew install --cask cloudcompare
```

## Data

Code lives in git; **data never does.** Everything is keyed off one env var
(defaults to `./data`; see [`config.py`](config.py)):

```bash
export PCC_DATA_ROOT=/path/to/data     # local, server, or Colab — no Google Drive
```

There are two kinds of data, handled the way a published benchmark should:

- **Raw ShapeNet meshes** (airplane `02691156`, car `02958343`, chair `03001627`)
  are **third-party and gated** — *not redistributed here*. Accept the ShapeNet
  license on HuggingFace, then fetch them with **your own** token:
  ```bash
  export HF_TOKEN=hf_xxx                 # only for this step; never commit it
  python scripts/download_data.py        # -> $PCC_DATA_ROOT/shapenetcore/*.zip
  ```
- **Derived benchmark data** (colored GT + occluded partials) — this project's
  contribution — is published as a **public HuggingFace dataset** and downloads
  with **no token**. Publish your outputs with:
  ```bash
  export HF_DATASET_REPO=<user>/colored-shapenet-completion
  export HF_TOKEN=hf_xxx                 # your write token, kept in the env only
  python scripts/upload_dataset.py
  ```
  _TODO: add the dataset link + a Zenodo DOI once published._

## Status / TODO

- [ ] Copy ADR-0001…0003 from Drive into [`docs/adr/`](docs/adr/).
- [ ] Upload the derived dataset to HuggingFace and link it above.
- [ ] Add a `LICENSE` (code) and state the dataset license (derived-from-ShapeNet terms apply).
- [x] Mesh-fidelity eval harness — [`scripts/eval_fidelity.py`](scripts/eval_fidelity.py) + [`notebooks/eval.ipynb`](notebooks/eval.ipynb) (speckle, uniformity, p2f, coverage, color ΔE). _Next: scale to all 60 models; add PCQM at the completion stage._

## Acknowledgements

- Colored sampling & dual-face removal: Lazzarotto & Ebrahimi, EPFL/MMSPG
  ([arXiv:2201.06935](https://arxiv.org/abs/2201.06935)).
- Occlusion generation: PoinTr ([Yu et al.](https://github.com/yuxumin/PoinTr)).
