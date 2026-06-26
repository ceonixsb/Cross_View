"""train_yroad_unet.py — U-Net: drone 91m BEV RGB -> 4-class traversability map.

Input  : data/cnn/rgb/NNNN.png      (the drone's whole-map nadir photo)
Target : data/cnn/label/NNNN.png    (GT class ids 0..3, world-frame aligned to the rgb)
Classes: 0=impassable 1=free 2=rough 3=tunnel    (--classes 3 folds rough->free)

Self-contained compact U-Net (no segmentation_models dep). Saves the best model (by val mIoU)
both as a state_dict and a TorchScript .pt so deploy_yroad can load it like the locomotion policy
and swap the planner's GT cost map for unet(rgb).

    conda activate isaacsim
    python train_yroad_unet.py --data data/cnn --epochs 60 --classes 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = {0: "impassable", 1: "free", 2: "rough", 3: "tunnel"}


# ── data ──────────────────────────────────────────────────────────────────────
class YRoadSeg(Dataset):
    def __init__(self, root: Path, ids, classes=4, train=True, depth=False):
        self.root, self.ids, self.classes, self.train, self.depth = root, ids, classes, train, depth

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        rgb = np.asarray(Image.open(self.root / "rgb" / f"{i:04d}.png").convert("RGB"), np.float32) / 255.0
        lab = np.asarray(Image.open(self.root / "label" / f"{i:04d}.png"), np.int64)
        if self.classes == 3:
            lab = lab.copy(); lab[lab == 2] = 1; lab[lab == 3] = 2          # impassable / free(+rough) / tunnel
        dep = (np.asarray(Image.open(self.root / "depth" / f"{i:04d}.png"), np.float32) / 65535.0
               if self.depth else None)                                     # nadir height map, [0,1]
        if self.train:                                                      # light appearance + flip aug
            if np.random.rand() < 0.5:
                rgb = rgb[:, ::-1].copy(); lab = lab[:, ::-1].copy()
                if dep is not None: dep = dep[:, ::-1].copy()
            if np.random.rand() < 0.5:
                rgb = rgb[::-1, :].copy(); lab = lab[::-1, :].copy()
                if dep is not None: dep = dep[::-1, :].copy()
            rgb = np.clip(rgb * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.06, 0.06), 0, 1)  # appearance: rgb only
        arr = np.concatenate([rgb, dep[..., None]], axis=-1) if dep is not None else rgb   # (H,W,4) RGB+D or (H,W,3)
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        y = torch.from_numpy(np.ascontiguousarray(lab)).long()
        return x, y


# ── compact U-Net ───────────────────────────────────────────────────────────
def _cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, n_cls=4, c=32, in_ch=3):
        super().__init__()
        self.d1 = _cbr(in_ch, c); self.d2 = _cbr(c, c * 2); self.d3 = _cbr(c * 2, c * 4); self.d4 = _cbr(c * 4, c * 8)
        self.bott = _cbr(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2); self.c4 = _cbr(c * 16, c * 8)
        self.u3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2); self.c3 = _cbr(c * 8, c * 4)
        self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2); self.c2 = _cbr(c * 4, c * 2)
        self.u1 = nn.ConvTranspose2d(c * 2, c, 2, 2); self.c1 = _cbr(c * 2, c)
        self.head = nn.Conv2d(c, n_cls, 1)

    def forward(self, x):
        e1 = self.d1(x); e2 = self.d2(self.pool(e1)); e3 = self.d3(self.pool(e2)); e4 = self.d4(self.pool(e3))
        b = self.bott(self.pool(e4))
        x = self.c4(torch.cat([self.u4(b), e4], 1)); x = self.c3(torch.cat([self.u3(x), e3], 1))
        x = self.c2(torch.cat([self.u2(x), e2], 1)); x = self.c1(torch.cat([self.u1(x), e1], 1))
        return self.head(x)


# ── metrics ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def per_class_iou(pred, tgt, n_cls):
    ious = []
    for k in range(n_cls):
        p, t = pred == k, tgt == k
        u = (p | t).sum().item()
        ious.append((p & t).sum().item() / u if u else float("nan"))
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cnn")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--classes", type=int, default=4, choices=[3, 4])
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--n_train", type=int, default=0, help="cap #training images (0=all) for few-shot")
    ap.add_argument("--depth", action="store_true", help="stack the nadir depth (height) as a 4th input channel (RGB+D)")
    ap.add_argument("--out", default="data/cnn/unet")
    a = ap.parse_args()
    in_ch = 4 if a.depth else 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).resolve().parent / a.data
    out = Path(__file__).resolve().parent / a.out; out.mkdir(parents=True, exist_ok=True)

    n = len([p for p in (root / "rgb").glob("*.png")])
    rng = np.random.default_rng(0); order = rng.permutation(n)
    nv = max(1, int(n * a.val_frac)); val_ids, tr_ids = order[:nv].tolist(), order[nv:].tolist()
    if a.n_train > 0:
        tr_ids = tr_ids[:a.n_train]                                # few-shot: identical val, smaller train
    print(f"[train] {n} pairs -> {len(tr_ids)} train / {len(val_ids)} val   classes={a.classes}  in_ch={in_ch}({'RGB+D' if a.depth else 'RGB'})  dev={dev}")

    tr = DataLoader(YRoadSeg(root, tr_ids, a.classes, True, a.depth), batch_size=a.bs, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(YRoadSeg(root, val_ids, a.classes, False, a.depth), batch_size=a.bs, shuffle=False, num_workers=4)

    # inverse-frequency class weights (capped) to counter the huge impassable background
    freq = np.zeros(a.classes)
    for i in tr_ids[: min(len(tr_ids), 120)]:
        lab = np.asarray(Image.open(root / "label" / f"{i:04d}.png"), np.int64)
        if a.classes == 3:
            lab = lab.copy(); lab[lab == 2] = 1; lab[lab == 3] = 2
        for k in range(a.classes):
            freq[k] += (lab == k).sum()
    w = np.clip((freq.sum() / (a.classes * np.maximum(freq, 1))), 0.3, 8.0)
    print(f"[train] class px frac = {(freq / freq.sum()).round(4).tolist()}   weights = {w.round(2).tolist()}")
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))

    net = UNet(a.classes, in_ch=in_ch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    best = -1.0
    for ep in range(a.epochs):
        net.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(net(x), y); loss.backward(); opt.step()
        sched.step()

        net.eval(); ious = np.zeros(a.classes); cnt = np.zeros(a.classes); acc = 0; npx = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                pred = net(x).argmax(1)
                acc += (pred == y).sum().item(); npx += y.numel()
                for k, v in enumerate(per_class_iou(pred, y, a.classes)):
                    if not np.isnan(v):
                        ious[k] += v; cnt[k] += 1
        miou = float(np.mean(ious / np.maximum(cnt, 1)))
        if ep % 5 == 0 or ep == a.epochs - 1:
            pc = (ious / np.maximum(cnt, 1)).round(3)
            print(f"  ep{ep:3d}  loss={loss.item():.3f}  val_acc={acc/npx:.3f}  mIoU={miou:.3f}  "
                  f"perclass={ {CLASS_NAMES[k]: float(pc[k]) for k in range(a.classes)} }")
        if miou > best:
            best = miou
            torch.save({"state_dict": net.state_dict(), "classes": a.classes, "in_ch": in_ch, "miou": best}, out / "best.pt")
            net_cpu = UNet(a.classes, in_ch=in_ch); net_cpu.load_state_dict(net.state_dict()); net_cpu.eval()
            torch.jit.save(torch.jit.trace(net_cpu, torch.randn(1, in_ch, 256, 256)), out / "best_jit.pt")

    json.dump({"best_miou": best, "classes": a.classes, "n": n}, open(out / "result.json", "w"), indent=2)
    print(f"[train] DONE. best mIoU={best:.3f}  -> {out}/best.pt (+best_jit.pt)")


if __name__ == "__main__":
    main()
