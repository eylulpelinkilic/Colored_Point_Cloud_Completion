"""PyTorch PointNet part-segmentation — same framework as PoinTr, so the whole
part-aware colored-completion pipeline runs in ONE environment (no TF/Kaggle split).

Pieces
------
* PointNetPartSeg : input (B,N,3) -> per-point part logits (B,N,P).
* ShapeNetPartDataset : loads a category's clouds+labels (.pts/.seg OR _normal .txt),
  samples a fixed N points.
* train_partseg / segment : train loop and inference (label any cloud, incl. PoinTr's
  completed output).

Wire-up: train on ShapeNet-Part clean clouds, then `segment(model, completed_xyz)`
labels PoinTr's completed points -> colour each by its part.

Self-test: `python scripts/pointnet_partseg.py` trains on a synthetic 3-part toy and
asserts it segments it (no data download needed to check the model/train loop).
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------ model
class TNet(nn.Module):
    """k×k input/feature alignment transform."""
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, k * k))

    def forward(self, x):                       # x: (B,k,N)
        B = x.size(0)
        f = self.mlp(x).max(-1)[0]
        m = self.fc(f).view(B, self.k, self.k)
        return m + torch.eye(self.k, device=x.device).unsqueeze(0)


class PointNetPartSeg(nn.Module):
    def __init__(self, num_parts, input_transform=True):
        super().__init__()
        self.itn = TNet(3) if input_transform else None
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.fstn = TNet(128)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.seg = nn.Sequential(
            nn.Conv1d(1024 + 128, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, num_parts, 1))

    def forward(self, x):                       # x: (B,N,3) -> (B,N,P)
        x = x.transpose(1, 2)                   # (B,3,N)
        if self.itn is not None:
            x = torch.bmm(self.itn(x), x)
        f = self.mlp1(x)                        # (B,128,N)
        f = torch.bmm(self.fstn(f), f)
        point_feat = f
        g = self.mlp2(f).max(-1, keepdim=True)[0].expand(-1, -1, f.size(-1))  # (B,1024,N)
        return self.seg(torch.cat([point_feat, g], 1)).transpose(1, 2)


# ------------------------------------------------------------------ data
def _read_pts(path):
    return np.loadtxt(path, dtype=np.float32)[:, :3]


def _read_seg(path):
    return np.loadtxt(path).astype(np.int64)


def _read_normal_txt(path):
    a = np.loadtxt(path).astype(np.float32)
    return a[:, :3], a[:, -1].astype(np.int64)


class ShapeNetPartDataset(torch.utils.data.Dataset):
    """One category. Supports either PartAnnotation (.pts + .seg in points/ + points_label/)
    or the `_normal .txt` layout (x y z nx ny nz label). Returns (xyz (N,3), label (N,))."""
    def __init__(self, part_dir, n_points=2048, split_ids=None, normalize=True):
        self.n = n_points; self.norm = normalize
        pts = sorted(glob.glob(os.path.join(part_dir, "points", "*.pts")))
        if pts:                                 # .pts/.seg layout
            self.items = [(p, os.path.join(part_dir, "points_label",
                          os.path.splitext(os.path.basename(p))[0] + ".seg")) for p in pts]
            self.kind = "pts"
        else:                                   # _normal .txt layout
            self.items = [(p, None) for p in sorted(glob.glob(os.path.join(part_dir, "*.txt")))]
            self.kind = "txt"
        if split_ids is not None:
            self.items = [it for it in self.items
                          if os.path.splitext(os.path.basename(it[0]))[0] in split_ids]
        # remap raw part labels -> contiguous 0..P-1
        labs = set()
        for i in range(min(len(self.items), 40)):
            labs.update(np.unique(self._raw(i)[1]).tolist())
        self.classes = sorted(labs)
        self.remap = {c: i for i, c in enumerate(self.classes)}

    def _raw(self, i):
        p, s = self.items[i]
        if self.kind == "pts":
            return _read_pts(p), _read_seg(s)
        return _read_normal_txt(p)

    def num_parts(self):
        return len(self.classes)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        xyz, lab = self._raw(i)
        lab = np.array([self.remap.get(int(v), 0) for v in lab], np.int64)
        idx = (np.random.choice(len(xyz), self.n, replace=len(xyz) < self.n))
        xyz, lab = xyz[idx], lab[idx]
        if self.norm:
            xyz = xyz - xyz.mean(0)
            xyz = xyz / (np.linalg.norm(xyz, axis=1).max() + 1e-9)
        return torch.from_numpy(xyz).float(), torch.from_numpy(lab).long()


# ------------------------------------------------------------------ train / infer
def train_partseg(model, loader, epochs=20, lr=1e-3, device="cpu", log_every=5):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tot, correct, seen = 0.0, 0, 0
        for xyz, lab in loader:
            xyz, lab = xyz.to(device), lab.to(device)
            opt.zero_grad()
            logits = model(xyz)                                 # (B,N,P)
            loss = lossf(logits.reshape(-1, logits.size(-1)), lab.reshape(-1))
            loss.backward(); opt.step()
            tot += loss.item() * xyz.size(0)
            correct += (logits.argmax(-1) == lab).sum().item(); seen += lab.numel()
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d}  loss {tot/len(loader.dataset):.4f}  point-acc {correct/seen*100:.1f}%")
    return model


@torch.no_grad()
def segment(model, xyz, device="cpu", normalize=True):
    """Label an arbitrary cloud (e.g. PoinTr's completed output). xyz (N,3) np -> part (N,) np."""
    model.to(device).eval()
    x = np.asarray(xyz, np.float32)[:, :3]
    if normalize:
        c = x.mean(0); x = (x - c) / (np.linalg.norm(x - c, axis=1).max() + 1e-9)
    t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    return model(t)[0].argmax(-1).cpu().numpy()


# ------------------------------------------------------------------ self-test (synthetic)
def _synthetic_loader(n_clouds=64, n=1024, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1.4, 0, 0], [-1.4, 0, 0], [0, 1.4, 0]])   # 3 parts
    X, Y = [], []
    for _ in range(n_clouds):
        per = n // 3
        xyz = np.concatenate([c + rng.normal(0, 0.25, (per, 3)) for c in centers])
        lab = np.concatenate([np.full(per, k) for k in range(3)])
        xyz = np.concatenate([xyz, xyz[:n - len(xyz)]]); lab = np.concatenate([lab, lab[:n - len(lab)]])
        p = rng.permutation(n)
        X.append(xyz[p].astype(np.float32)); Y.append(lab[p].astype(np.int64))
    X, Y = np.stack(X), np.stack(Y)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).long())
    return torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)


if __name__ == "__main__":
    print("self-test: training PointNet part-seg on a synthetic 3-part toy...")
    loader = _synthetic_loader()
    model = PointNetPartSeg(num_parts=3)
    train_partseg(model, loader, epochs=15, log_every=3)
    xb, yb = next(iter(loader))
    acc = (model(xb).argmax(-1) == yb).float().mean().item()
    print(f"final batch point-acc: {acc*100:.1f}%")
    assert acc > 0.9, "part-seg did not learn the toy"
    print("OK: model + train loop + segment() work ✅")
