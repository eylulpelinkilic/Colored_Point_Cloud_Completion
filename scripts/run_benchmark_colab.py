#!/usr/bin/env python3
"""Multi_dataset_benchmark.ipynb'nin COLAB + DRIVE surumu — bayrak gerekmez.

Lightning surumunden farki: butun kalici cikti Google Drive'a yazilir, boylece
runtime kopsa/silinse bile egitilmis modeller, model sonuclari, indirilen veri
onbellegi ve rapor durur.

  DURUST UYARI: Colab'da tarayiciyi kapatinca runtime olur (ucretsiz ve Pro).
  Sadece Colab Pro+ "background execution" ile bilgisayari kapatabilirsin.
  Bu script bunu ASMAZ; ama koparsa kayip SIFIRA yakin olur ve tek komutla
  kaldigi yerden devam eder.

KULLANIM — Colab'da tek bir hucreye:

    !git clone -q https://github.com/eylulpelinkilic/Colored_Point_Cloud_Completion.git /content/pcc \
        && cd /content/pcc && git checkout -q Pelin
    !pip install -q wandb
    import wandb; wandb.login()                    # anahtari bir kez yapistir
    from google.colab import drive; drive.mount('/content/drive')

    # isi kernel'den KOPAR: hucre biterse / kernel yeniden baslarsa bile surer
    !cd /content/pcc && nohup python scripts/run_benchmark_colab.py \
        > /content/drive/MyDrive/pcc_out/bench.log 2>&1 &

Izleme (istedigin zaman, ayri hucre):

    !tail -40 /content/drive/MyDrive/pcc_out/bench.log
    !wc -l < /content/drive/MyDrive/pcc_out/results.jsonl

Kopma sonrasi: ayni hucreleri tekrar calistir. Egitilmis modeller ve hesaplanmis
model sonuclari atlanir, kaldigi yerden devam eder.

URETILMIS DOSYA — kalici degisiklikleri notebook'ta yap, sonra yeniden uret.
"""
import argparse, os, sys, subprocess

# ══════════════════════════ CONFIG — duzenlemek istedigin tek yer ══════════════════════════
CONFIG = dict(
    datasets        = ["ours"],   # Colab diski dar; densepoint 6.95 GB zip + ~12 GB acilmis ister
    models_per_cat  = 200,        # None = tamami
    eval_n          = None,       # None = BUTUN test seti
    ddpm_epochs     = 300,
    force_retrain   = False,
    skip_env        = False,      # ortam kuruluysa True -> ~10 dk kazanc
    drive_root      = "/content/drive/MyDrive",   # kalici cikti burada
)
# ═══════════════════════════════════════════════════════════════════════════════════════════

ap = argparse.ArgumentParser(description="Bayrak GEREKMEZ; CONFIG'i ezmek istersen kullan.")
ap.add_argument("--datasets", nargs="+")
ap.add_argument("--eval-n", type=int)
ap.add_argument("--models-per-cat", type=int)
ap.add_argument("--ddpm-epochs", type=int)
ap.add_argument("--force-retrain", action="store_true")
ap.add_argument("--skip-env", action="store_true")
_a = ap.parse_args()
for k, v in vars(_a).items():
    if v not in (None, False):
        CONFIG[k] = v
ARGS = argparse.Namespace(**CONFIG)
print("CONFIG:", CONFIG, flush=True)

# --- Drive: notebook'ta zaten mount edilmis olmali (bassiz surecte OAuth yapilamaz) ---
DRIVE = ARGS.drive_root
if not os.path.isdir(DRIVE):
    try:                                   # yine de dene: ayni VM'de mount edilmisse acilir
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        pass
if not os.path.isdir(DRIVE):
    sys.exit(f"Drive bagli degil ({DRIVE}). Bu script'i baslatmadan ONCE notebook'ta\n"
             "  from google.colab import drive; drive.mount('/content/drive')\n"
             "calistir — bassiz surec OAuth penceresi acamaz.")

def _wandb_ready():
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc
        return "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
    except Exception:
        return False

if not _wandb_ready():
    sys.exit("W&B kimligi yok. Bu script'ten ONCE notebook'ta `import wandb; wandb.login()` "
             "calistir (bassiz surec interaktif login yapamaz).")

# -- hucre 0 --
import os, sys, subprocess, glob

# Colab: WORK gecici (PoinTr klonu). Kopunca gider, ~1 dk'da yeniden kurulur.
# Kalici olan her sey DATA_ROOT/OUT_ROOT altinda, yani Drive'da.
WORK = "/content/pcc_work"
os.makedirs(WORK, exist_ok=True)
DATA_ROOT = os.path.join(DRIVE, "pcc_data")   # KALICI     # indirilen veri setleri
OUT_ROOT  = os.path.join(DRIVE, "pcc_out")    # KALICI      # checkpoint + sonuc + rapor
for d in (DATA_ROOT, OUT_ROOT): os.makedirs(d, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(DATA_ROOT, "hf_cache"))

print("WORK      :", WORK)
print("DATA_ROOT :", DATA_ROOT)
print("OUT_ROOT  :", OUT_ROOT)
print()
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True)
import shutil
print(f"bos disk  : {shutil.disk_usage(WORK).free/1e9:.0f} GB")
print("  (densepoint 6.95 GB zip + ~12 GB acilmis alan ister)")

# -- hucre 1 (ortam) --
if not ARGS.skip_env:
    subprocess.run(r"""pip install -q easydict h5py matplotlib opencv-python pyyaml scipy tensorboardX tqdm transforms3d einops timm open3d gdown huggingface_hub wandb pandas""", shell=True, check=False)
    import numpy as np, torch
    print("numpy", np.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())

# -- hucre 2 (ortam) --
if not ARGS.skip_env:
    # --- PoinTr fork'unu klonla ---
    POINTR = os.path.join(WORK, "PoinTr")
    if not os.path.isdir(POINTR):
        subprocess.run(f"git clone https://github.com/eylulpelinkilic/Pelin_Efe_PoinTr.git {POINTR}",
                       shell=True, check=True)
    os.chdir(POINTR); sys.path.insert(0, POINTR)
    subprocess.run("git rev-parse --short HEAD", shell=True)

    # --- CUDA build ortami (hardcode etme, otomatik tespit) ---
    cuda_home = "/usr/local/cuda" if os.path.isdir("/usr/local/cuda") else \
                (sorted(glob.glob("/usr/local/cuda*"))[-1] if glob.glob("/usr/local/cuda*") else "")
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ["PATH"]
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}"
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"), "| arch:", os.environ.get("TORCH_CUDA_ARCH_LIST"))

# -- hucre 3 --
# pointnet2_ops'u DERLEMEK yerine saf-PyTorch SHIM olarak enjekte ediyoruz.
# torch 2.11+cu128 gibi çok yeni stack'te eski CUDA repo'su derlenmiyor. PoinTr sadece
# şu 6 fonksiyonu kullanıyor; hepsi saf torch'la doğru (yerelde brute-force'a karşı test edildi).
# FPS saf-torch döngüsü biraz yavaş ama bu ölçekte (~8k nokta) sorun değil.
import sys, types, torch

def furthest_point_sample(xyz, npoint):        # xyz (B,N,3) -> idx (B,npoint) int32
    B, N, _ = xyz.shape; dev = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=dev)
    dist = torch.full((B, N), 1e10, device=dev, dtype=xyz.dtype)
    far = torch.zeros(B, dtype=torch.long, device=dev); ar = torch.arange(B, device=dev)
    for i in range(npoint):
        idx[:, i] = far
        dist = torch.minimum(dist, ((xyz - xyz[ar, far].unsqueeze(1)) ** 2).sum(-1))
        far = torch.max(dist, dim=1).indices
    return idx.int()

def gather_operation(features, idx):           # (B,C,N),(B,S) -> (B,C,S)
    B, C, N = features.shape; idx = idx.long()
    return torch.gather(features, 2, idx.unsqueeze(1).expand(B, C, idx.shape[1])).contiguous()

def three_nn(query, ref):                      # (B,N,3),(B,M,3) -> dist(B,N,3) öklid, idx(B,N,3)
    d = torch.cdist(query, ref)
    dist, idx = torch.topk(d, 3, dim=-1, largest=False, sorted=True)
    return dist.contiguous(), idx.int().contiguous()

def three_interpolate(features, idx, weight):  # (B,C,M),(B,N,3),(B,N,3) -> (B,C,N)
    B, C, M = features.shape; N = idx.shape[1]; idx = idx.long()
    g = torch.gather(features, 2, idx.reshape(B,1,N*3).expand(B,C,N*3)).reshape(B,C,N,3)
    return (g * weight.unsqueeze(1)).sum(-1).contiguous()

def grouping_operation(features, idx):         # (B,C,N),(B,S,K) -> (B,C,S,K)  (SnowFlakeNet için)
    B, C, N = features.shape; _, S, K = idx.shape; idx = idx.long()
    return torch.gather(features, 2, idx.reshape(B,1,S*K).expand(B,C,S*K)).reshape(B,C,S,K).contiguous()

def ball_query(radius, nsample, xyz, new_xyz): # (r,k,(B,N,3),(B,S,3)) -> idx(B,S,k)  (SnowFlakeNet için)
    B, N, _ = xyz.shape; S = new_xyz.shape[1]
    d = torch.cdist(new_xyz, xyz)
    idx = torch.arange(N, device=xyz.device).view(1,1,N).expand(B,S,N).contiguous()
    idx[d > radius] = N
    idx = idx.sort(dim=-1).values[:, :, :nsample]
    first = idx[:, :, 0:1].clone(); first[first == N] = 0
    idx = torch.where(idx == N, first.expand(-1,-1,nsample), idx)
    return idx.int()

_u = types.ModuleType("pointnet2_ops.pointnet2_utils")
for _f in [furthest_point_sample, gather_operation, three_nn, three_interpolate,
           grouping_operation, ball_query]:
    setattr(_u, _f.__name__, _f)
_p = types.ModuleType("pointnet2_ops"); _p.pointnet2_utils = _u
sys.modules["pointnet2_ops"] = _p
sys.modules["pointnet2_ops.pointnet2_utils"] = _u
from pointnet2_ops import pointnet2_utils
print("pointnet2_ops shim enjekte edildi:",
      [n for n in dir(pointnet2_utils) if not n.startswith("_")])

# -- hucre 4 (ortam) --
if not ARGS.skip_env:
    # --- CUDA extension'lari: DERLEME YOK, cikarim icin stub ---
    # PoinTr'i DONMUS halde sadece tahmin icin kullaniyoruz. Derlenmis extension'lar
    # yalnizca KAYIP fonksiyonlarinda geciyor (chamfer) ya da GRNet'in import zincirinde
    # (gridding / cubic_feature_sampling) — hicbiri cagrilmiyor. Colab'da bu derleme
    # hem 5-10 dk suruyor hem de torch/CUDA surumune gore sik sik patliyor.
    # Gercekten derlenmis olan varsa O kullanilir; yoksa cagirinca anlasilir hata veren
    # stub konur. PoinTr'i EGITECEKSEN BUILD_EXTENSIONS=True yap.
    BUILD_EXTENSIONS = False

    import types
    if BUILD_EXTENSIONS:
        for ext in ["chamfer_dist", "gridding", "gridding_loss", "cubic_feature_sampling"]:
            p = os.path.join(POINTR, "extensions", ext)
            if not os.path.isdir(p): continue
            r = subprocess.run("pip install -q --no-build-isolation .", shell=True, cwd=p,
                               capture_output=True, text=True)
            print(f"  build {ext}: {'ok' if r.returncode == 0 else 'FAIL'}")

    def _stub_ext(name, funcs=("forward", "backward")):
        m = types.ModuleType(name)
        def _die(*a, _n=name, **k):
            raise NotImplementedError(
                f"'{_n}' CUDA extension'i derlenmedi. Bu kurulum SADECE CIKARIM icin: "
                "donmus PoinTr tahmin yapar, kayip fonksiyonlari cagrilmaz. "
                "Egitim istiyorsan BUILD_EXTENSIONS=True yap.")
        for f in funcs: setattr(m, f, _die)
        sys.modules[name] = m

    _stubbed = []
    for _n in ("chamfer", "gridding", "gridding_distance", "cubic_feature_sampling"):
        if _n in sys.modules: continue
        try: __import__(_n)                      # gercek derlenmisse ONU kullan
        except ImportError: _stub_ext(_n); _stubbed.append(_n)
    print("stub'lanan extension:", _stubbed or "yok (hepsi derlenmis)")


# -- hucre 5 --
# --- checkpoint + smoke test ---
CKPT = os.path.join(POINTR, "ckpts", "PoinTr_ShapeNet55.pth")
os.makedirs(os.path.dirname(CKPT), exist_ok=True)
if not os.path.exists(CKPT) or os.path.getsize(CKPT) < 50e6:
    subprocess.run(f"gdown 1WzERLlbSwzGOBybzkjBrApwyVMTG00CJ -O {CKPT}", shell=True, check=True)
import chamfer
from pointnet2_ops import pointnet2_utils
from models.PoinTr import PoinTr, fps
print("checkpoint MB:", round(os.path.getsize(CKPT)/1e6, 1), "| importlar OK")

# -- hucre 6 --
import wandb
_k = os.environ.get("WANDB_API_KEY")
wandb.login(key=_k) if _k else wandb.login()   # onbellekten

WANDB_PROJECT = "colored-pc-completion"
WANDB_ENTITY  = None
print("wandb", wandb.__version__)

# -- hucre 7 --
# ═════════════ AYARLAR ═════════════
RUN_DATASETS = ARGS.datasets      # + "omniobject3d", "3dcompat" (erisim alinca)
DIFFICULTIES = ["simple", "moderate", "hard"]          # %25 / %50 / %75
N_MODELS_PER_CAT = ARGS.models_per_cat     # kategori basina (None = tamami)
N_PTS            = 2048
TEST_FRAC        = 0.25
EVAL_N           = ARGS.eval_n      # degerlendirilen test modeli (None = tamami)
N_SEEDS          = 1
FORCE_RETRAIN    = ARGS.force_retrain

HF_DATA = "eylulpelinkilic/Colored_Point_Clouds"
# 3DCoMPaT++: lisans formunu doldurup aldigin klasoru buraya yaz
COMPAT_ROOT = os.path.join(DATA_ROOT, "3dcompat")
# OmniObject3D: openxlab ile indirdigin ply klasoru
OMNI_ROOT   = os.path.join(DATA_ROOT, "omniobject3d")
# ═══════════════════════════════════

import numpy as np, json, glob
from scipy.spatial import cKDTree

def srgb_to_lab(rgb):
    rgb = np.clip(rgb, 0, 1)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[.4124,.3576,.1805],[.2126,.7152,.0722],[.0193,.1192,.9505]])
    xyz = (lin @ M.T) / np.array([.95047, 1.0, 1.08883]); d = 6/29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz/(3*d**2) + 4/29)
    return np.stack([116*f[:,1]-16, 500*(f[:,0]-f[:,1]), 200*(f[:,1]-f[:,2])], 1)
def deltaE(a, b): return np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=1)

def _crop_order(xyz, seed):
    """Rastgele bakis yonune gore sirala; yon SADECE seed'e bagli."""
    c = xyz.mean(0); n = (xyz - c) / (np.linalg.norm(xyz - c, axis=1).max() + 1e-9)
    r = np.random.default_rng(seed); v = r.standard_normal(3); v /= np.linalg.norm(v)
    return np.argsort(np.linalg.norm(n - v[None], axis=1))

def make_entry(xyz, rgb, part, mid, difficulty, seed):
    """Ortak son islem: normalize -> N_PTS -> viewpoint crop -> girdi sozlugu."""
    xyz = xyz - xyz.mean(0); xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)
    if len(xyz) != N_PTS:
        s = np.random.default_rng(seed).choice(len(xyz), N_PTS, replace=len(xyz) < N_PTS)
        xyz, rgb = xyz[s], rgb[s]
        part = None if part is None else part[s]
    gt = np.concatenate([xyz, np.clip(rgb, 0, 1)], 1).astype(np.float32)
    ratio = {"simple": .25, "moderate": .50, "hard": .75}[difficulty]
    order = _crop_order(gt[:, :3], seed)
    miss = np.zeros(len(gt), bool); miss[order[:int(round(len(gt)*ratio))]] = True
    return dict(gt=gt, partial=gt[~miss], miss=miss,
                gt_part=None if part is None else part.astype(np.int64),
                partial_dense=None, model_id=mid)
print("ortak yardimcilar hazir")

# -- hucre 8 --
# ═════════════ 1) OURS — HF, yayinlanmis occlusion ═════════════
from huggingface_hub import snapshot_download
OURS_CATS = {"airplane": "02691156", "car": "02958343", "chair": "03001627"}
OURS_PARTS = {"airplane": ["body","wing","tail","engine"],
              "car": ["roof","hood","wheel","body"],
              "chair": ["back","seat","leg","arm"]}

def load_ours(category, difficulty):
    import open3d as o3d
    syn = OURS_CATS[category]
    root = snapshot_download(HF_DATA, repo_type="dataset",
                             allow_patterns=[f"labeled_s3/{syn}/*", f"occluded_occ/*/{syn}/*"])
    files = sorted(glob.glob(os.path.join(root, "labeled_s3", syn, "*.npz")))
    if N_MODELS_PER_CAT: files = files[:N_MODELS_PER_CAT]
    out = []
    for j, f in enumerate(files):
        mid = os.path.splitext(os.path.basename(f))[0]
        pp = os.path.join(root, "occluded_occ", difficulty, syn, mid + ".ply")
        if not os.path.exists(pp): continue
        z = np.load(f)
        gxyz, grgb, gpart = z["xyz"].astype(np.float32), z["rgb"].astype(np.float32), z["part"].astype(np.int64)
        pxyz = np.asarray(o3d.io.read_point_cloud(pp).points, np.float32)
        d, idx = cKDTree(gxyz).query(pxyz, k=1)          # partial GT'nin ALT KUMESI
        vis = np.zeros(len(gxyz), bool); vis[idx[d < 1e-6]] = True
        r = np.random.default_rng(j)
        s = r.choice(len(gxyz), N_PTS, replace=len(gxyz) < N_PTS)
        gxyz, grgb, gpart, miss = gxyz[s], grgb[s], gpart[s], (~vis)[s]
        c = gxyz.mean(0); gxyz = (gxyz - c) / (np.linalg.norm(gxyz - c, axis=1).max() + 1e-9)
        gt = np.concatenate([gxyz, np.clip(grgb, 0, 1)], 1).astype(np.float32)
        if miss.sum() < 32 or (~miss).sum() < 32: continue
        out.append(dict(gt=gt, partial=gt[~miss], miss=miss, gt_part=gpart,
                        partial_dense=None, model_id=mid))
    return out, OURS_PARTS[category]
print("ours yukleyicisi hazir")

# -- hucre 9 --
# ═════════════ 2) DENSEPOINT — dogrudan zip, ShapeNet+ShapeNetPart turevi ═════════════
DP_URL  = "http://rwdc.nagao.nuie.nagoya-u.ac.jp/DensePoint/Download"
DP_ROOT = os.path.join(DATA_ROOT, "densepoint")

def fetch_densepoint():
    """6.95 GB zip -> DATA_ROOT/densepoint (bir kez)."""
    if glob.glob(os.path.join(DP_ROOT, "**", "*.ply"), recursive=True): return DP_ROOT
    os.makedirs(DP_ROOT, exist_ok=True)
    z = os.path.join(DATA_ROOT, "densepoint.zip")
    if not os.path.exists(z) or os.path.getsize(z) < 6e9:
        print("  DensePoint indiriliyor (~6.95 GB)...", flush=True)
        subprocess.run(f'wget -q --show-progress -O "{z}" "{DP_URL}"', shell=True, check=True)
    print("  aciliyor...", flush=True)
    subprocess.run(f'unzip -q -o "{z}" -d "{DP_ROOT}"', shell=True, check=True)
    return DP_ROOT

def _read_ply_fields(path):
    """DensePoint ply: xyz + rgb + normal + parca etiketi. Etiket ozelliginin adi
       surumden surume degisebiliyor -> ismi tahmin etmek yerine ARIYORUZ."""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    ccols = [c for c in ("red","green","blue") if c in names]
    rgb = (np.stack([v[c] for c in ccols], 1).astype(np.float32) / 255.0) if len(ccols) == 3 \
          else np.zeros_like(xyz)
    lab = next((c for c in names if c.lower() in
                ("label","part","seg","part_label","scalar_label","class")), None)
    part = v[lab].astype(np.int64) if lab else None
    return xyz, rgb, part, names

def load_densepoint(category, difficulty):
    root = fetch_densepoint()
    cands = [d for d in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if os.path.isdir(d) and os.path.basename(d).lower() == category.lower()]
    if not cands:                       # klasor adi farkliysa: icinde kategori gecen
        cands = [d for d in glob.glob(os.path.join(root, "*")) if category.lower() in d.lower()]
    assert cands, (f"DensePoint'te '{category}' klasoru yok. Mevcutlar: "
                   f"{sorted(os.path.basename(d) for d in glob.glob(os.path.join(root,'*')))[:20]}")
    plys = sorted(glob.glob(os.path.join(cands[0], "**", "*.ply"), recursive=True))
    if N_MODELS_PER_CAT: plys = plys[:N_MODELS_PER_CAT]
    out, base = [], None
    for j, p in enumerate(plys):
        xyz, rgb, part, _ = _read_ply_fields(p)
        if part is not None:
            base = part.min() if base is None else min(base, part.min())
            part = part - base                      # global 0..49 -> kategori-yerel
        out.append(make_entry(xyz, rgb, part, os.path.splitext(os.path.basename(p))[0],
                              difficulty, j))
    npart = int(max(d["gt_part"].max() for d in out)) + 1 if out and out[0]["gt_part"] is not None else 0
    return out, [f"part{i}" for i in range(npart)]
print("densepoint yukleyicisi hazir")

# -- hucre 10 (ortam) --
if not ARGS.skip_env:
    # ═════════════ 3) OMNIOBJECT3D — gercek tarama, PARCA ETIKETI YOK ═════════════
    # Kurulum (bir kez, terminalden):
    #   pip install openxlab && openxlab login          # AK/SK: repo README'sinde paylasilmis
    #   openxlab dataset download --dataset-repo omniobject3d/OmniObject3D-New \
    #            --source-path /raw/point_clouds/ply_files --target-path $DATA_ROOT/omniobject3d
    def load_omniobject3d(category, difficulty):
        import open3d as o3d
        cands = [d for d in glob.glob(os.path.join(OMNI_ROOT, "**", "*"), recursive=True)
                 if os.path.isdir(d) and category.lower() in os.path.basename(d).lower()]
        assert cands, (f"OmniObject3D'de '{category}' yok ({OMNI_ROOT}). "
                       "openxlab ile indirdin mi?")
        plys = sorted(sum([glob.glob(os.path.join(c, "**", "*.ply"), recursive=True) for c in cands], []))
        if N_MODELS_PER_CAT: plys = plys[:N_MODELS_PER_CAT]
        out = []
        for j, p in enumerate(plys):
            pc = o3d.io.read_point_cloud(p)
            xyz = np.asarray(pc.points, np.float32); rgb = np.asarray(pc.colors, np.float32)
            if len(rgb) != len(xyz) or len(xyz) < 512: continue
            out.append(make_entry(xyz, rgb, None,      # <-- parca YOK
                                  os.path.splitext(os.path.basename(p))[0], difficulty, j))
        return out, []
    print("omniobject3d yukleyicisi hazir (parca etiketi yok)")

# -- hucre 11 --
# ═════════════ 4) 3DCoMPaT++ — parca + malzeme ═════════════
# Kurulum (bir kez): lisans formunu doldur -> https://3dcompat-dataset.org/doc/dl-dataset.html
# Aldigin point-cloud arsivini COMPAT_ROOT altina ac.
def load_3dcompat(category, difficulty):
    import open3d as o3d
    hits = [d for d in glob.glob(os.path.join(COMPAT_ROOT, "**", "*"), recursive=True)
            if os.path.isdir(d) and category.lower() in os.path.basename(d).lower()]
    assert hits, (f"3DCoMPaT++'ta '{category}' yok ({COMPAT_ROOT}). "
                  "Lisans formunu doldurup veriyi buraya actin mi?")
    files = sorted(sum([glob.glob(os.path.join(h, "**", "*.npz"), recursive=True) +
                        glob.glob(os.path.join(h, "**", "*.ply"), recursive=True) for h in hits], []))
    if N_MODELS_PER_CAT: files = files[:N_MODELS_PER_CAT]
    out = []
    for j, f in enumerate(files):
        if f.endswith(".npz"):
            z = np.load(f)
            k_xyz = next((k for k in z.files if k.lower() in ("xyz","points","pos","v")), None)
            k_rgb = next((k for k in z.files if k.lower() in ("rgb","color","colors")), None)
            k_prt = next((k for k in z.files if "part" in k.lower() or k.lower() in ("seg","label")), None)
            if k_xyz is None or k_rgb is None: continue
            xyz, rgb = z[k_xyz].astype(np.float32), z[k_rgb].astype(np.float32)
            part = z[k_prt].astype(np.int64) if k_prt else None
        else:
            xyz, rgb, part, _ = _read_ply_fields(f)
        if rgb.max() > 1.5: rgb = rgb / 255.0
        out.append(make_entry(xyz, rgb, part, os.path.splitext(os.path.basename(f))[0],
                              difficulty, j))
    npart = int(max(d["gt_part"].max() for d in out)) + 1 if out and out[0]["gt_part"] is not None else 0
    return out, [f"part{i}" for i in range(npart)]

# ---- kayit defteri ----
REGISTRY = {
    "ours":         dict(load=load_ours,         cats=list(OURS_CATS)),
    "densepoint":   dict(load=load_densepoint,   cats=["airplane", "car", "chair"]),
    "omniobject3d": dict(load=load_omniobject3d, cats=["airplane", "car", "chair"]),
    "3dcompat":     dict(load=load_3dcompat,     cats=["airplane", "car", "chair"]),
}
for k in RUN_DATASETS: assert k in REGISTRY, f"bilinmeyen veri seti: {k}"
print("kayit defteri:", list(REGISTRY))

# -- hucre 12 --
# --- STEP 1: dondurulmuş orijinal PoinTr ---
import torch
from easydict import EasyDict
from scipy.spatial import cKDTree
from models.PoinTr import PoinTr, fps

DEV = "cuda"
cfg = EasyDict(trans_dim=384, knn_layer=1, num_pred=6144, num_query=96)
geo = PoinTr(cfg)
sd = torch.load(CKPT, map_location="cpu")
base = sd.get("base_model", sd.get("model", sd))
base = {k.replace("module.", ""): v for k, v in base.items()}
mi, ui = geo.load_state_dict(base, strict=False)
assert len(mi) == 0, f"orijinal PoinTr bekleniyordu, missing={mi[:4]} (0 olmalı)"
for p in geo.parameters(): p.requires_grad_(False)
geo.eval().to(DEV)

# ShapeNet-Part -> PoinTr(ShapeNet-55) frame hizası. Aşağıdaki hücre 48 işaretli
# permütasyonu Chamfer ile tarayıp bunları OTOMATİK ayarlıyor — elle dokunma.
AXIS_PERM, AXIS_SIGN = (2, 1, 0), (1, 1, 1)

@torch.no_grad()
def complete_geometry(partial, perm=None, sign=None):
    """RENKSİZ partial (xyz) -> PoinTr tamamlaması, doğru frame'e çevirip geri çevirerek."""
    perm = AXIS_PERM if perm is None else tuple(perm)
    sign = AXIS_SIGN if sign is None else tuple(sign)
    s = np.asarray(sign, np.float32)
    x = np.ascontiguousarray(partial[:, :3][:, list(perm)] * s)      # x'[:,i] = x[:,perm[i]]*s[i]
    p = torch.from_numpy(x).float().unsqueeze(0).to(DEV)
    fine = geo(p)[1][0, :geo.num_pred].cpu().numpy()
    inv = np.argsort(perm)                                          # ters çevir: x[:,j] = x'[:,inv[j]]*s[inv[j]]
    return np.ascontiguousarray(fine[:, inv] * s[inv])

def chamfer_l1(a, b):
    """Simetrik ortalama en-yakın-komşu mesafesi (düşük = iyi)."""
    d1, _ = cKDTree(b).query(a, k=1)
    d2, _ = cKDTree(a).query(b, k=1)
    return float(d1.mean() + d2.mean())

# PoinTr ShapeNet-55, 8192 noktalı bulutlardan kırpılmış 2048-6144 noktalı partial'larla
# eğitildi. Bizim seyrek partial'ımız (N_PTS=2048, CROP=0.5 -> 1024 nokta) o dağılımın
# ALTINDA kalıyor ve DGCNN grouper'ın kNN komşulukları ~2x geniş düşüyor. Bu yüzden
# PoinTr'a verilen girdiyi renk hattından AYIRIYORUZ: aşağıdaki hücre en iyisini ölçüyor.
POINTR_IN = 2048          # PoinTr'a verilecek nokta sayısı (yoğun partial varsa FPS ile)
USE_DENSE_FOR_POINTR = True

def _fps_np(xyz, n):
    """PoinTr'ın kendi fps'i (pointnet2_ops shim'i üzerinden)."""
    if len(xyz) <= n:
        return np.ascontiguousarray(xyz).astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(xyz[:, :3])).float().unsqueeze(0).to(DEV)
    return fps(t, n)[0].cpu().numpy().astype(np.float32)

def pointr_input(d):
    """Bu model için PoinTr'a verilecek xyz — mümkünse yoğun partial, POINTR_IN'e indirilmiş."""
    src = d.get("partial_dense") if USE_DENSE_FOR_POINTR else None
    src = d["partial"][:, :3] if src is None else src
    return _fps_np(src, POINTR_IN) if POINTR_IN and len(src) > POINTR_IN else src

def complete_of(d, perm=None, sign=None):
    """Bu modelin PoinTr tamamlaması. Boru hattının HER yeri bunu kullanmalı."""
    return complete_geometry(pointr_input(d), perm, sign)

def nn_color(partial, comp):                   # BASELINE 1
    _, i = cKDTree(partial[:, :3]).query(comp[:, :3], k=1)
    return partial[i, 3:6]
print(f"dondurulmuş PoinTr hazır | num_pred={geo.num_pred}")

# -- hucre 13 --
# --- STEP 2: PointNet part-seg (PoinTr_setup_colab.ipynb ile aynı model) ---
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class _TNet(nn.Module):
    def __init__(s, k):
        super().__init__(); s.k = k
        s.mlp = nn.Sequential(nn.Conv1d(k,64,1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64,128,1),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128,1024,1),nn.BatchNorm1d(1024),nn.ReLU())
        s.fc = nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,k*k))
    def forward(s, x):
        B=x.size(0); f=s.mlp(x).max(-1)[0]
        return s.fc(f).view(B,s.k,s.k) + torch.eye(s.k,device=x.device).unsqueeze(0)

class PointNetPartSeg(nn.Module):
    def __init__(s, P):
        super().__init__(); s.itn=_TNet(3)
        s.mlp1=nn.Sequential(nn.Conv1d(3,64,1),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,128,1),nn.BatchNorm1d(128),nn.ReLU())
        s.fstn=_TNet(128)
        s.mlp2=nn.Sequential(nn.Conv1d(128,128,1),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,1024,1),nn.BatchNorm1d(1024),nn.ReLU())
        s.seg=nn.Sequential(nn.Conv1d(1152,512,1),nn.BatchNorm1d(512),nn.ReLU(),
            nn.Conv1d(512,256,1),nn.BatchNorm1d(256),nn.ReLU(),nn.Conv1d(256,P,1))
    def forward(s, x):
        x=x.transpose(1,2); x=torch.bmm(s.itn(x),x)
        f=s.mlp1(x); f=torch.bmm(s.fstn(f),f); pf=f
        g=s.mlp2(f).max(-1,keepdim=True)[0].expand(-1,-1,f.size(-1))
        return s.seg(torch.cat([pf,g],1)).transpose(1,2)

def train_partseg(model, loader, epochs=30, lr=1e-3, device=DEV):
    model.to(device).train(); opt=torch.optim.Adam(model.parameters(),lr); lf=nn.CrossEntropyLoss()
    for ep in range(epochs):
        cor=seen=0; tot=0.0
        for xyz,lab in loader:
            xyz,lab=xyz.to(device),lab.to(device); opt.zero_grad()
            lo=model(xyz); loss=lf(lo.reshape(-1,lo.size(-1)),lab.reshape(-1))
            loss.backward(); opt.step()
            tot+=loss.item()*xyz.size(0); cor+=(lo.argmax(-1)==lab).sum().item(); seen+=lab.numel()
        if ep%10==0 or ep==epochs-1: print(f"  ep{ep:3d} loss {tot/len(loader.dataset):.4f} acc {cor/seen*100:.1f}%")
    return model

@torch.no_grad()
def segment(model, xyz, device=DEV):
    model.eval(); x=np.asarray(xyz,np.float32)[:,:3]
    c=x.mean(0); x=(x-c)/(np.linalg.norm(x-c,axis=1).max()+1e-9)
    return model(torch.from_numpy(x).float().unsqueeze(0).to(device))[0].argmax(-1).cpu().numpy()

class _GTPartDS(Dataset):
    def __init__(s, D): s.D = D
    def __len__(s): return len(s.D)
    def __getitem__(s, i):
        d = s.D[i]
        return torch.from_numpy(d["gt"][:, :3]).float(), torch.from_numpy(d["gt_part"]).long()

def segment_oracle(d, xyz):
    """En yakın GT noktasının parçası. Segmentasyon hatasını İZOLE etmek için —
       'renklendirme fikri iyi mi' ile 'segmenter yeterli mi' ayrı sorular."""
    x = np.asarray(xyz, np.float32)[:, :3]
    c = x.mean(0); x = (x - c) / (np.linalg.norm(x - c, axis=1).max() + 1e-9)
    g = d["gt"][:, :3]; gc = g.mean(0); g = (g - gc) / (np.linalg.norm(g - gc, axis=1).max() + 1e-9)
    _, j = cKDTree(g).query(x, k=1)
    return d["gt_part"][j]

# part-seg kalitesi — tabloyu yorumlamak için ŞART: parça koşullaması bu etikete dayanır,
# etiket kötüyse RePaint-part de düşer (bu gerçekten gözlendi).
def part_color_pointnet(partial, comp, model):   # BASELINE 2 — mevcut yöntemin
    vl = segment(model, partial[:, :3]); cl = segment(model, comp[:, :3])
    P = NUM_PARTS; vr = partial[:, 3:6]
    mean = np.tile(vr.mean(0), (P, 1))           # dayanaksız parça -> global ortalama
    for k in range(P):
        m = vl == k
        if m.any(): mean[k] = vr[m].mean(0)
    return mean[cl]


# -- hucre 14 --
# ---------------- D1 · DDPM şeması + RePaint zıplama şeması ----------------
import math

def cosine_betas(T, s=0.008):
    """Nichol & Dhariwal cosine schedule — düşük boyutlu sinyalde linear'dan iyi."""
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)

class Diffusion:
    """Düz DDPM (eps-tahmini), [-1,1] aralığındaki (N,3) renk alanı üzerinde."""
    def __init__(self, T=200, device=DEV):
        self.T, self.device = T, device
        b = cosine_betas(T).to(device); a = 1.0 - b
        abar = torch.cumprod(a, 0)
        abar_prev = torch.cat([torch.ones(1, device=device), abar[:-1]])
        self.betas, self.alphas, self.abar, self.abar_prev = b, a, abar, abar_prev
        self.sqrt_abar, self.sqrt_1mabar = abar.sqrt(), (1 - abar).sqrt()
        self.post_var = b * (1 - abar_prev) / (1 - abar)            # q(x_{t-1}|x_t,x_0)
        self.post_c0  = b * abar_prev.sqrt() / (1 - abar)
        self.post_ct  = (1 - abar_prev) * a.sqrt() / (1 - abar)

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        sa = self.sqrt_abar[t].view(-1, *([1] * (x0.dim() - 1)))
        sb = self.sqrt_1mabar[t].view(-1, *([1] * (x0.dim() - 1)))
        return sa * x0 + sb * noise

    def p_sample(self, eps, x_t, t, generator=None):
        x0 = ((x_t - self.sqrt_1mabar[t] * eps) / self.sqrt_abar[t]).clamp(-1, 1)
        mean = self.post_c0[t] * x0 + self.post_ct[t] * x_t
        if t == 0: return mean, x0
        z = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        return mean + self.post_var[t].sqrt() * z, x0

    def forward_jump(self, x, t, generator=None):        # RePaint time-travel: x_t -> x_{t+1}
        z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return self.alphas[t].sqrt() * x + self.betas[t].sqrt() * z

def get_schedule_jump(T, jump_length=10, jump_n_sample=5):
    """RePaint'in resampling ('time-travel') şeması — resmî repo ile aynı.

    Dönen listede ardışık AZALAN çift = ters difüzyon adımı, ARTAN çift = ileri zıplama.
    Zıplamalar üretilen bölgenin bilinen bölgeyle ANLAMCA uyumlanmasını sağlar; makalenin
    ana katkısı bu (zıplamasız versiyon sadece dokuca uyar)."""
    jumps = {j: jump_n_sample - 1 for j in range(0, T - jump_length, jump_length)}
    t, ts = T, []
    while t >= 1:
        t -= 1; ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1; ts.append(t)
    ts.append(-1)
    return ts

DIF = Diffusion(T=200)
print("T =", DIF.T, "| RePaint adım sayısı (j=10, U=3):", len(get_schedule_jump(DIF.T, 10, 3)))

# -- hucre 15 --
# ---------------- D2 · parça-koşullu denoiser ----------------
def knn_graph(xyz, k, chunk=4096):
    """(B,N,3) -> (B,N,k) komşu indisleri (kendisi hariç), sorgu üzerinden parçalı."""
    B, N, _ = xyz.shape
    out = torch.empty(B, N, k, dtype=torch.long, device=xyz.device)
    kk = min(k + 1, N)
    for s in range(0, N, chunk):
        d = torch.cdist(xyz[:, s:s + chunk], xyz)
        idx = d.topk(kk, dim=-1, largest=False).indices[:, :, 1:]
        if idx.shape[-1] < k:
            idx = idx[..., [i % idx.shape[-1] for i in range(k)]]
        out[:, s:s + chunk] = idx
    return out

def _gather_nb(h, idx):                        # h (B,N,C), idx (B,N,k) -> (B,N,k,C)
    B, N, C = h.shape; k = idx.shape[-1]
    off = (torch.arange(B, device=h.device) * N).view(B, 1, 1)
    return h.reshape(B * N, C)[(idx + off).reshape(-1)].reshape(B, N, k, C)

def part_pool(h, part, P):
    """h'nin PARÇA İÇİ ortalaması, her noktaya geri yayılır. İşte 'part-based' burada:
       bir parçanın (enjekte edilmiş) görünür renkleri kendi eksik noktalarını sürer."""
    B, N, C = h.shape
    flat = (part + torch.arange(B, device=h.device).view(B, 1) * P).reshape(-1)
    s = torch.zeros(B * P, C, device=h.device, dtype=h.dtype).index_add_(0, flat, h.reshape(-1, C))
    n = torch.zeros(B * P, 1, device=h.device, dtype=h.dtype).index_add_(
        0, flat, torch.ones(B * N, 1, device=h.device, dtype=h.dtype))
    return torch.gather((s / n.clamp(min=1.0)).reshape(B, P, C), 1, part.unsqueeze(-1).expand(B, N, C))

def timestep_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    a = t.float().view(-1, 1) * f.view(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)

class Block(nn.Module):
    """EdgeConv (yerel geometri) + parça havuzu + global havuz, t ile FiLM'lenir."""
    def __init__(self, w, part_cond=True):
        super().__init__(); self.part_cond = part_cond
        self.edge = nn.Sequential(nn.Linear(2 * w + 4, w), nn.GELU(), nn.Linear(w, w))
        ctx = w * (3 if part_cond else 2)
        self.fuse = nn.Sequential(nn.LayerNorm(ctx), nn.Linear(ctx, w), nn.GELU(), nn.Linear(w, w))
        self.film = nn.Linear(w, 2 * w)
    def forward(self, h, idx, rel, part, P, temb):
        hj = _gather_nb(h, idx); hi = h.unsqueeze(2).expand_as(hj)
        e = self.edge(torch.cat([hi, hj - hi, rel], -1)).max(2).values
        g = h.max(1, keepdim=True).values.expand_as(h)
        c = [e, g] + ([part_pool(h, part, P)] if self.part_cond else [])
        d = self.fuse(torch.cat(c, -1))
        sc, sh = self.film(temb).unsqueeze(1).chunk(2, -1)
        return h + d * (1 + sc) + sh

class PartColorDenoiser(nn.Module):
    """Nokta başına renk alanı için eps-tahmini; xyz (+ parça) ile koşullu.

    Permütasyona eşdeğişken ve N'den bağımsız: 2048 noktalı GT bulutlarında eğitilir,
    ~7k noktalı PoinTr birleşim bulutunda çalıştırılır.
    `part_cond=False` -> parça-kör (vanilla RePaint) ablasyonu.
    """
    def __init__(self, num_parts, width=128, k=16, n_blocks=3, part_cond=True):
        super().__init__()
        self.P, self.k, self.part_cond, self.width = num_parts, k, part_cond, width
        cin = 3 + 3 + (num_parts if part_cond else 0)          # xyz, c_t, parça one-hot
        self.inp = nn.Linear(cin, width)
        self.temb = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([Block(width, part_cond) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, 3))
    def build_ctx(self, xyz, part):
        """Geometri her difüzyon adımında SABİT -> kNN grafiği bir kez kurulur (büyük hızlanma)."""
        idx = knn_graph(xyz, self.k)
        rel = _gather_nb(xyz, idx) - xyz.unsqueeze(2)
        scale = rel.norm(dim=-1).mean(dim=(1, 2), keepdim=True).clamp(min=1e-6).unsqueeze(-1)
        rel = torch.cat([rel / scale, rel.norm(dim=-1, keepdim=True) / scale], -1)  # yoğunluktan bağımsız
        return dict(xyz=xyz, idx=idx, rel=rel, part=part,
                    onehot=torch.nn.functional.one_hot(part, self.P).float())
    def forward(self, c_t, t, ctx):
        B = c_t.shape[0]
        f = [ctx["xyz"], c_t] + ([ctx["onehot"]] if self.part_cond else [])
        h = self.inp(torch.cat(f, -1))
        temb = self.temb(timestep_embedding(t.expand(B) if t.dim() else t.repeat(B), self.width))
        for blk in self.blocks:
            h = blk(h, ctx["idx"], ctx["rel"], ctx["part"], self.P, temb)
        return self.out(h)


# -- hucre 16 --
# ---------------- D4 · RePaint çıkarımı (parça bazlı) ----------------
@torch.no_grad()
def repaint_colors(model, dif, xyz, part, known, known_rgb, *, jump_length=10,
                   jump_n_sample=3, adaptive_jumps=True, seed=0, device=DEV, return_diag=False):
    """Parça bazlı RePaint: doldurulan noktaların rengini inpaint eder.

    Lugmayr et al. (2022)'ye sadık: her ters adımda BİLİNEN renkler doğru gürültü
    seviyesinde geri enjekte edilir, şema yukarı zıplayarak üretilen bölgenin onlarla
    uyumlanmasını sağlar:

        x_{t-1} = m . q(x_bilinen, t-1)  +  (1-m) . p_theta(x_t)

    "Part-based" üç noktada:
      1. denoiser özellikleri PARÇA İÇİNDE havuzlar -> bir parçanın bilinen renkleri kendi
         eksik noktalarını sürer (parça-ortalaması kuralının öğrenilmiş hâli);
      2. dayanaklar parça bazlı sayılır — görünür noktası SIFIR olan parça *anchorless*
         işaretlenir ve komşusundan renk sızdırmak yerine öğrenilmiş önselden doldurulur;
      3. resampling bütçesi en kötü parçanın görünürlüğüne göre uyarlanır.

    xyz (N,3), part (N,), known (N,) bool, known_rgb (N,3) [0,1] -> (N,3) [0,1].
    """
    g = torch.Generator(device=device).manual_seed(seed)
    xyz_t = torch.as_tensor(xyz).float().unsqueeze(0).to(device)
    prt = torch.as_tensor(part).long().unsqueeze(0).to(device)
    m = torch.as_tensor(known).bool().view(1, -1, 1).to(device)
    c0 = (torch.as_tensor(known_rgb).float().unsqueeze(0).to(device) * 2 - 1) * m

    # --- parça bazlı dayanak muhasebesi ---
    pid = prt[0]
    vis = np.array([(m[0, :, 0][pid == p].float().mean().item() if (pid == p).any() else np.nan)
                    for p in range(model.P)])
    present = ~np.isnan(vis)
    anchorless = [p for p in range(model.P) if present[p] and vis[p] == 0.0]
    if adaptive_jumps and present.any():
        worst = float(np.nanmin(np.where(present, vis, np.nan)))
        jump_n_sample = int(np.clip(round(jump_n_sample * (1.5 - worst)), 1, 2 * jump_n_sample))

    model.eval().to(device)
    ctx = model.build_ctx(xyz_t, prt)
    x = torch.randn(1, xyz_t.shape[1], 3, device=device, generator=g)

    ts = get_schedule_jump(dif.T, jump_length, jump_n_sample)
    for t_cur, t_next in zip(ts[:-1], ts[1:]):
        if t_next < t_cur:                                          # --- ters adım
            eps = model(x, torch.tensor(t_cur, device=device), ctx)
            x_unknown, _ = dif.p_sample(eps, x, t_cur, generator=g)
            if t_cur > 0:
                noise = torch.randn(x.shape, device=device, generator=g)
                x_known = dif.q_sample(c0, torch.tensor([t_cur - 1], device=device), noise)
            else:
                x_known = c0
            x = torch.where(m, x_known, x_unknown)                  # maske = parça bazlı maskelerin birleşimi
        else:                                                       # --- ileri zıplama
            x = dif.forward_jump(x, t_cur, generator=g)

    out = ((x[0] + 1) / 2).clamp(0, 1).cpu().numpy()
    kn = np.asarray(known, bool)
    out[kn] = np.asarray(known_rgb, np.float32)[kn]                 # görünür renkler birebir korunur
    if return_diag:
        return out, dict(part_visibility=vis, anchorless_parts=anchorless,
                         jump_n_sample=jump_n_sample, n_steps=len(ts))
    return out


def make_repaint_input(d, comp, seg_model, drop_part=None, oracle_seg=False):
    """Birleşim bulutu = görünür partial (renk BİLİNEN) + PoinTr'ın doldurduğu noktalar (BİLİNMEYEN).

    oracle_seg=True -> parça etiketleri PointNet yerine en yakın GT'den (segmentasyon hatasız tavan).
    drop_part verilirse o parçanın bütün görünür noktaları da maskelenir -> *anchorless* senaryo.
    Düşürme HER ZAMAN GT etiketine göre yapılır: aksi hâlde segmenter o parçayı hiç tahmin
    etmediğinde 'hiçbir şey düşmez' ve deney sessizce anlamsızlaşır (bu tuzağa bir kez düşüldü).
    """
    partial = d["partial"]
    xyz = np.concatenate([partial[:, :3], comp], 0).astype(np.float32)
    c = xyz.mean(0); xyz = ((xyz - c) / (np.linalg.norm(xyz - c, axis=1).max() + 1e-9)).astype(np.float32)
    known = np.zeros(len(xyz), bool); known[:len(partial)] = True
    rgb = np.zeros((len(xyz), 3), np.float32); rgb[:len(partial)] = partial[:, 3:6]
    part = segment_oracle(d, xyz) if oracle_seg else segment(seg_model, xyz)
    if drop_part is not None:
        known &= (segment_oracle(d, xyz) != drop_part); rgb[~known] = 0
    return dict(xyz=xyz, rgb=rgb, known=known, part=part, n_vis=len(partial))

print("RePaint çıkarımı hazır")

# -- hucre 17 --
# --- oryantasyon araması, FONKSİYON olarak (döngüde kategori başına çağrılır) ---
# Doğru ShapeNet-Part -> ShapeNet-55 frame'i 6 permütasyon x 8 işaret = 48 aday
# arasından, GT'ye Chamfer ile ÖLÇÜLEREK seçilir. Göz kararı değil.
import itertools

def find_frame(DATA, IDX, n_probe=3):
    """-> (perm, sign, en_iyi_chamfer, ham_partial_referansi)"""
    probe = [DATA[i] for i in IDX[:n_probe]]
    ref = float(np.mean([chamfer_l1(d["partial"][:, :3], d["gt"][:, :3]) for d in probe]))
    sc = []
    for perm in itertools.permutations(range(3)):
        for sign in itertools.product((1, -1), repeat=3):
            c = [chamfer_l1(complete_geometry(pointr_input(d), perm, sign), d["gt"][:, :3])
                 for d in probe]
            sc.append((float(np.mean(c)), perm, sign))
    sc.sort()
    return sc[0][1], sc[0][2], sc[0][0], ref

print("find_frame hazır")

# -- hucre 18 --
# eğitim döngüsü (D3'ün fonksiyon kısmı — burada modelleri biz kuruyoruz)
def train_color_ddpm(model, dif, clouds, epochs=300, bs=8, lr=2e-4, device=DEV, log=100):
    """Koşulsuz DDPM eğitimi — occlusion maskesi HİÇ görülmez."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    n = len(clouds)
    for ep in range(epochs):
        perm = np.random.permutation(n); tot = 0.0
        for s in range(0, n, bs):
            b = [clouds[i] for i in perm[s:s + bs]]
            xyz = torch.stack([torch.as_tensor(d["xyz"]) for d in b]).float().to(device)
            rgb = torch.stack([torch.as_tensor(d["rgb"]) for d in b]).float().to(device)
            prt = torch.stack([torch.as_tensor(d["part"]) for d in b]).long().to(device)
            x0 = rgb * 2 - 1
            t = torch.randint(0, dif.T, (len(b),), device=device)
            noise = torch.randn_like(x0)
            loss = ((model(dif.q_sample(x0, t, noise), t, model.build_ctx(xyz, prt)) - noise) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if log and (ep % log == 0 or ep == epochs - 1):
            print(f"    ep{ep:4d} eps-MSE {tot/n:.4f}", flush=True)
    return tot / n
print("eğitim döngüsü hazır")

# -- hucre 19 --
# ---------------- kalici depolama + sonuc onbellegi ----------------
import time
RESULTS = os.path.join(OUT_ROOT, "results.jsonl")

def load_results():
    if not os.path.exists(RESULTS): return []
    out = []
    for ln in open(RESULTS):
        ln = ln.strip()
        if ln:
            try: out.append(json.loads(ln))
            except json.JSONDecodeError: pass
    return out

def append_result(rec):
    with open(RESULTS, "a") as f: f.write(json.dumps(rec) + "\n")

def ckpt_path(tag, name): return os.path.join(OUT_ROOT, f"{tag}_{name}.pt")

def save_model(tag, name, model, meta):
    torch.save({"sd": model.state_dict(), "meta": meta}, ckpt_path(tag, name))

def load_model(tag, name, model, meta):
    p = ckpt_path(tag, name)
    if FORCE_RETRAIN or not os.path.exists(p): return False
    try: z = torch.load(p, map_location=DEV, weights_only=False)
    except Exception: return False
    if z.get("meta") != meta: return False
    model.load_state_dict(z["sd"]); model.to(DEV); return True

_done = {(r["dataset"], r["category"], r["difficulty"], r["model"]) for r in load_results()}
print(f"onbellekte {len(_done)} model-sonucu | {RESULTS}")

# -- hucre 20 --
# ---------------- ANA DONGU ----------------
DDPM_EPOCHS, SEG_EPOCHS, DDPM_ARCH = ARGS.ddpm_epochs, 60, dict(width=128, k=16, n_blocks=3)
PART_KEYS = ["part-mean", "RePaint-part", "RePaint-part (oracle seg)", "oracle ceiling"]
KEYS = ["NN-copy", "part-mean", "RePaint-vanilla", "RePaint-part",
        "RePaint-part (oracle seg)", "oracle ceiling"]

def cloud3d(xyz, rgb):
    return wandb.Object3D(np.concatenate([xyz, np.clip(rgb, 0, 1) * 255], 1).astype(np.float32))

T0 = time.time()
for ds_name in RUN_DATASETS:
    spec = REGISTRY[ds_name]
    for category in spec["cats"]:
        tag = f"{ds_name}_{category}"
        print(f"\n{'='*64}\n  {ds_name.upper()} / {category}   ({(time.time()-T0)/60:.0f} dk)\n{'='*64}", flush=True)
        try:
            DATA, PARTS = spec["load"](category, DIFFICULTIES[0])
        except Exception as e:
            print(f"  ATLANDI: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
        if len(DATA) < 8:
            print(f"  ATLANDI: sadece {len(DATA)} model", flush=True); continue

        HAS_PARTS = DATA[0]["gt_part"] is not None
        NUM_PARTS = (int(max(d["gt_part"].max() for d in DATA)) + 1) if HAS_PARTS else 0
        n_test = max(1, int(len(DATA) * TEST_FRAC))
        TRAIN_IDX = list(range(len(DATA) - n_test))
        TEST_IDX  = list(range(len(DATA) - n_test, len(DATA)))
        print(f"  {len(DATA)} model | train {len(TRAIN_IDX)} / test {len(TEST_IDX)} | "
              f"parca: {NUM_PARTS if HAS_PARTS else 'YOK -> part-yontemleri atlanacak'}", flush=True)

        AXIS_PERM, AXIS_SIGN, best, ref = find_frame(DATA, TRAIN_IDX)
        verdict = "OK" if best < 0.6*ref else ("ZAYIF" if best < ref else "BOZUK")
        print(f"  frame {list(AXIS_PERM)}{list(AXIS_SIGN)} | chamfer {best:.4f} vs {ref:.4f} -> {verdict}",
              flush=True)

        META = dict(ds=ds_name, cat=category, n_train=len(TRAIN_IDX), n_pts=N_PTS,
                    parts=NUM_PARTS, ddpm_ep=DDPM_EPOCHS, seg_ep=SEG_EPOCHS, T=DIF.T, **DDPM_ARCH)
        seg_model, SEG_ACC = None, float("nan")
        if HAS_PARTS:
            seg_model = PointNetPartSeg(NUM_PARTS).to(DEV)
            if not load_model(tag, "seg", seg_model, META):
                print("  -- part-seg egitimi", flush=True)
                train_partseg(seg_model, DataLoader(_GTPartDS([DATA[i] for i in TRAIN_IDX]),
                                                    batch_size=16, shuffle=True), epochs=SEG_EPOCHS)
                save_model(tag, "seg", seg_model, META)
            SEG_ACC = float(np.mean([(segment(seg_model, DATA[i]["gt"][:, :3]) == DATA[i]["gt_part"]).mean()
                                     for i in TEST_IDX]))
            print(f"  part-seg TEST dogrulugu: {SEG_ACC*100:.1f}%", flush=True)

        TC = [dict(xyz=DATA[i]["gt"][:, :3], rgb=DATA[i]["gt"][:, 3:6],
                   part=(DATA[i]["gt_part"] if HAS_PARTS else np.zeros(N_PTS, np.int64)))
              for i in TRAIN_IDX]
        NP_EFF = max(NUM_PARTS, 1)
        ddpm_part = None
        if HAS_PARTS:
            ddpm_part = PartColorDenoiser(NP_EFF, part_cond=True, **DDPM_ARCH)
            if not load_model(tag, "ddpm_part", ddpm_part, META):
                print("  -- DDPM (part-conditioned)", flush=True)
                train_color_ddpm(ddpm_part, DIF, TC, epochs=DDPM_EPOCHS); save_model(tag, "ddpm_part", ddpm_part, META)
        ddpm_van = PartColorDenoiser(NP_EFF, part_cond=False, **DDPM_ARCH)
        if not load_model(tag, "ddpm_van", ddpm_van, META):
            print("  -- DDPM (part-blind)", flush=True)
            train_color_ddpm(ddpm_van, DIF, TC, epochs=DDPM_EPOCHS); save_model(tag, "ddpm_van", ddpm_van, META)

        for difficulty in DIFFICULTIES:
            DATA, _ = spec["load"](category, difficulty)
            idxs = TEST_IDX if EVAL_N is None else TEST_IDX[:EVAL_N]
            idxs = [i for i in idxs if i < len(DATA)]
            todo = [i for i in idxs if (ds_name, category, difficulty, DATA[i]["model_id"]) not in _done]
            print(f"\n  -- {difficulty}: {len(idxs)} test, {len(todo)} yapilacak", flush=True)

            run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                             id=f"{ds_name}-{category}-{difficulty}",
                             name=f"{ds_name}/{category}/{difficulty}",
                             group=ds_name, job_type="eval", resume="allow", reinit=True,
                             config=dict(dataset=ds_name, category=category, difficulty=difficulty,
                                         crop_ratio={"simple":.25,"moderate":.5,"hard":.75}[difficulty],
                                         has_parts=HAS_PARTS, n_parts=NUM_PARTS,
                                         n_models=len(DATA), n_test=len(idxs), n_pts=N_PTS,
                                         partseg_acc=SEG_ACC, axis_perm=list(AXIS_PERM),
                                         axis_sign=list(AXIS_SIGN), frame_verdict=verdict,
                                         ddpm_epochs=DDPM_EPOCHS, T=DIF.T, **DDPM_ARCH))
            if HAS_PARTS: run.log({"partseg/test_acc": SEG_ACC})

            t1 = time.time()
            for n, i in enumerate(todo):
                d = DATA[i]; gt, partial = d["gt"], d["partial"]
                comp = complete_of(d)
                _, gi = cKDTree(gt[:, :3]).query(comp, k=1)
                true_rgb = gt[gi, 3:6]; m = d["miss"][gi]
                if not m.any(): continue
                rec = {"dataset": ds_name, "category": category, "difficulty": difficulty,
                       "model": d["model_id"], "has_parts": HAS_PARTS,
                       "chamfer_in": chamfer_l1(partial[:, :3], gt[:, :3]),
                       "chamfer_out": chamfer_l1(comp, gt[:, :3])}
                rec["NN-copy"] = float(deltaE(nn_color(partial, comp)[m], true_rgb[m]).mean())
                inp = make_repaint_input(d, comp, seg_model) if HAS_PARTS else \
                      dict(**make_repaint_input(d, comp, None, oracle_seg=False)) if False else None
                if not HAS_PARTS:
                    # parcasiz: birlesim bulutu + hepsi tek "parca"
                    xyzu = np.concatenate([partial[:, :3], comp], 0).astype(np.float32)
                    c = xyzu.mean(0); xyzu = ((xyzu - c) / (np.linalg.norm(xyzu - c, axis=1).max() + 1e-9)).astype(np.float32)
                    kn = np.zeros(len(xyzu), bool); kn[:len(partial)] = True
                    rgbu = np.zeros((len(xyzu), 3), np.float32); rgbu[:len(partial)] = partial[:, 3:6]
                    inp = dict(xyz=xyzu, rgb=rgbu, known=kn,
                               part=np.zeros(len(xyzu), np.int64), n_vis=len(partial))
                sv = [deltaE(repaint_colors(ddpm_van, DIF, inp["xyz"], inp["part"], inp["known"],
                                            inp["rgb"], seed=1000*n+k)[inp["n_vis"]:][m],
                             true_rgb[m]).mean() for k in range(N_SEEDS)]
                rec["RePaint-vanilla"] = float(np.mean(sv))
                if HAS_PARTS:
                    rec["part-mean"] = float(deltaE(part_color_pointnet(partial, comp, seg_model)[m],
                                                    true_rgb[m]).mean())
                    pf = np.stack([gt[d["gt_part"] == p, 3:6].mean(0) if (d["gt_part"] == p).any()
                                   else gt[:, 3:6].mean(0) for p in range(NUM_PARTS)])
                    rec["oracle ceiling"] = float(deltaE(pf[d["gt_part"][gi]][m], true_rgb[m]).mean())
                    inp_or = make_repaint_input(d, comp, seg_model, oracle_seg=True)
                    for tg, ip in [("RePaint-part", inp), ("RePaint-part (oracle seg)", inp_or)]:
                        sv = [deltaE(repaint_colors(ddpm_part, DIF, ip["xyz"], ip["part"], ip["known"],
                                                    ip["rgb"], seed=1000*n+k)[ip["n_vis"]:][m],
                                     true_rgb[m]).mean() for k in range(N_SEEDS)]
                        rec[tg] = float(np.mean(sv))
                append_result(rec); _done.add((ds_name, category, difficulty, d["model_id"]))
                run.log({f"model/{k}": v for k, v in rec.items() if isinstance(v, float)})
                el = time.time() - t1
                pm = f"{rec.get('part-mean', float('nan')):5.2f}"
                rp = f"{rec.get('RePaint-part', float('nan')):5.2f}"
                print(f"    [{n+1}/{len(todo)}] {d['model_id'][:10]} NN {rec['NN-copy']:5.2f} | "
                      f"pm {pm} | RP {rp} | van {rec['RePaint-vanilla']:5.2f} ({el/(n+1):.0f}s)", flush=True)

            rs = [r for r in load_results() if r["dataset"] == ds_name
                  and r["category"] == category and r["difficulty"] == difficulty]
            if rs:
                agg = {"chamfer/partial_to_gt": float(np.mean([r["chamfer_in"] for r in rs])),
                       "chamfer/completion_to_gt": float(np.mean([r["chamfer_out"] for r in rs]))}
                agg["chamfer/improvement_x"] = agg["chamfer/partial_to_gt"]/max(agg["chamfer/completion_to_gt"],1e-9)
                for k in KEYS:
                    v = [r[k] for r in rs if k in r]
                    if v: agg[f"dE/{k}"] = float(np.mean(v))
                if "dE/RePaint-part" in agg:
                    agg["dE/repaint_part_vs_nn_x"] = agg["dE/NN-copy"]/max(agg["dE/RePaint-part"],1e-9)
                agg["n_evaluated"] = len(rs)
                run.log(agg); run.summary.update(agg)
                print(f"    => " + " | ".join(f"{k.split('/')[-1]} {v:.2f}"
                      for k, v in agg.items() if k.startswith("dE/")), flush=True)

            d0 = DATA[idxs[0]]; c0 = complete_of(d0)
            panels = {"cloud/gt": cloud3d(d0["gt"][:, :3], d0["gt"][:, 3:6]),
                      "cloud/partial": cloud3d(d0["partial"][:, :3], d0["partial"][:, 3:6]),
                      "cloud/nn_copy": cloud3d(c0, nn_color(d0["partial"], c0))}
            if HAS_PARTS:
                i0 = make_repaint_input(d0, c0, seg_model)
                r0 = repaint_colors(ddpm_part, DIF, i0["xyz"], i0["part"], i0["known"], i0["rgb"], seed=0)
                panels["cloud/repaint_part"] = cloud3d(c0, r0[i0["n_vis"]:])
            run.log(panels); run.finish()

print(f"\nBENCHMARK BITTI — {(time.time()-T0)/60:.0f} dk")

# -- hucre 21 --
import pandas as pd
rs = load_results()
assert rs, "henuz sonuc yok"
df = pd.DataFrame(rs)
present = [k for k in KEYS if k in df.columns]
g = (df.groupby(["dataset", "category", "difficulty"])
       .agg(n=("model", "count"), has_parts=("has_parts", "first"),
            chamfer_in=("chamfer_in", "mean"), chamfer_out=("chamfer_out", "mean"),
            **{k: (k, "mean") for k in present}).reset_index())
g["chamfer_x"] = g.chamfer_in / g.chamfer_out
g = g.sort_values(["dataset", "category", "difficulty"],
                  key=lambda s: s.map({"simple": 0, "moderate": 1, "hard": 2}).fillna(s))
pd.set_option("display.width", 260, "display.max_columns", 60)
print(g.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

summ = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="SUMMARY",
                  id="SUMMARY", resume="allow", job_type="summary", reinit=True)
summ.log({"grid": wandb.Table(dataframe=g)})
for metric in present:
    rows = [[f"{r.dataset}/{r.category}/{r.difficulty}", float(r[metric])]
            for _, r in g.iterrows() if pd.notna(r[metric])]
    if rows:
        summ.log({f"bar/{metric}": wandb.plot.bar(
            wandb.Table(data=rows, columns=["config", metric]), "config", metric,
            title=f"dE {metric}")})
summ.finish()

csv_p = os.path.join(OUT_ROOT, "benchmark_grid.csv"); g.to_csv(csv_p, index=False)
print("\n->", csv_p)

# -- hucre 22 --
# ---------------- HTML rapor ----------------
def _fmt(v):
    return "&ndash;" if pd.isna(v) else f"{v:.2f}"

rows_html = []
for _, r in g.iterrows():
    best = min([r[k] for k in present if pd.notna(r[k])], default=None)
    tds = "".join(
        f"<td class='num{' best' if (pd.notna(r[k]) and best is not None and abs(r[k]-best) < 1e-9) else ''}'>"
        f"{_fmt(r[k])}</td>" for k in present)
    rows_html.append(
        f"<tr><td>{r.dataset}</td><td>{r.category}</td><td>{r.difficulty}</td>"
        f"<td class='num'>{int(r.n)}</td><td class='num'>{r.chamfer_x:.2f}&times;</td>{tds}</tr>")

hdr = "".join(f"<th class='num'>{k}</th>" for k in present)
html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Renkli point-cloud completion — çok-veri-setli benchmark</title><style>
:root{{color-scheme:light;--bg:#eef2f5;--surface:#fcfcfb;--surface2:#f4f6f8;--ink:#0b0b0b;
--muted:#52514e;--line:#dfe4e9;--accent:#0e7490;--accent-soft:#d9eef2;--best:#1baf7a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;
--bg:#0e151d;--surface:#1a1a19;--surface2:#141c25;--ink:#fff;--muted:#c3c2b7;--line:#2a3540;
--accent:#2bb7cd;--accent-soft:#123640;--best:#41c081}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:clamp(18px,4vw,48px)}}
.sheet{{max-width:1250px;margin:0 auto}}
h1{{font-size:clamp(22px,3vw,34px);margin:0 0 6px;letter-spacing:-.02em}}
.sub{{color:var(--muted);margin:0 0 26px;line-height:1.55;max-width:74ch}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px;
margin-bottom:18px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;min-width:900px}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
background:var(--surface2);position:sticky;top:0}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.best{{color:var(--best);font-weight:700}}
.note{{font-size:12.5px;color:var(--muted);line-height:1.6}}
.note b{{color:var(--ink)}}
code{{background:var(--surface2);border:1px solid var(--line);border-radius:5px;padding:1px 5px;
font-family:ui-monospace,Menlo,monospace;font-size:.9em}}
</style></head><body><div class="sheet">
<h1>Renkli point-cloud completion — çok-veri-setli benchmark</h1>
<p class="sub">Eksik bölgede ortalama <b>&Delta;E(Lab)</b>, düşük = iyi. Her satırda en iyi
yöntem <span style="color:var(--best);font-weight:700">yeşil</span>. <code>&ndash;</code> =
o veri setinde parça etiketi yok, yöntem çalıştırılamadı.</p>
<div class="card"><table><thead><tr>
<th>veri seti</th><th>kategori</th><th>zorluk</th><th class="num">n</th>
<th class="num">PoinTr kazanç</th>{hdr}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>
<div class="card"><p class="note">
<b>PoinTr kazanç</b> = ham partial&rarr;GT Chamfer'ının tamamlama&rarr;GT'ye oranı; 1.0&times;
demek PoinTr hiçbir şey katmadı demektir.<br><br>
<b>oracle ceiling</b>, parça-başına-sabit herhangi bir kuralın <b>yapısal alt sınırı</b> —
parça-ortalaması bunun altına inemez. <b>RePaint-part</b>'ın oraya göre konumu katkının ölçüsü.<br><br>
<b>RePaint-part (oracle seg)</b> ile <b>RePaint-part</b> farkı tamamen
<b>part-seg hatasının bedeli</b>; büyükse bir sonraki iş difüzyon değil segmenter.<br><br>
<b>Uyarı:</b> &Delta;E bir distortion metriğidir ve koşullu ortalamayı ödüllendirir, yani
ortalama alan yöntemleri üretici olanlara karşı yapısal olarak kayırır
(distortion&ndash;perception ödünleşimi, Blau &amp; Michaeli 2018). Yanına dağılım metriği
koymadan tek başına yorumlama.</p></div>
</div></body></html>"""
rep = os.path.join(OUT_ROOT, "benchmark_report.html")
open(rep, "w").write(html)
print("->", rep)
try:
    summ2 = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="REPORT",
                       id="REPORT", resume="allow", job_type="report", reinit=True)
    summ2.log({"report": wandb.Html(html)}); summ2.finish()
    print("-> W&B'ye de yuklendi (REPORT run'i)")
except Exception as e:
    print("W&B rapor yuklenemedi:", type(e).__name__)