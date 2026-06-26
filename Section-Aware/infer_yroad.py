"""infer_yroad.py — visualize U-Net predictions on val samples (RGB+D model).

Left: BEV RGB | Centre: GT cost map | Right: prediction | Right+1: confidence

    python infer_yroad.py --ckpt data/BEV/best.pt --data data/cnn_hard --n 8 --out data/shots_bev
    xdg-open data/shots_bev/infer_000.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from train_yroad_unet import UNet

NAMES  = ["impassable", "free", "rough", "tunnel"]
COLORS = np.array([[40, 40, 48], [235, 235, 235], [240, 190, 90], [90, 150, 235]], np.uint8)


def label_to_rgb(lbl: np.ndarray, n_cls: int) -> np.ndarray:
    h, w = lbl.shape
    out = np.zeros((h, w, 3), np.uint8)
    for c in range(n_cls):
        out[lbl == c] = COLORS[c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/BEV/best.pt")
    ap.add_argument("--data", default="data/cnn_hard")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="data/shots_bev")
    ap.add_argument("--val_frac", type=float, default=0.15)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt, map_location=dev)
    n_cls = ckpt["classes"]; in_ch = ckpt.get("in_ch", 3)
    depth = in_ch == 4
    net = UNet(n_cls, in_ch=in_ch).to(dev)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    print(f"[infer] model: {n_cls} classes  in_ch={in_ch}  best_mIoU={ckpt.get('miou',0):.3f}")

    root = Path(a.data)
    ids = sorted(int(p.stem) for p in (root / "rgb").glob("*.png"))
    rng = np.random.default_rng(0); order = rng.permutation(len(ids))
    nv = max(1, int(len(ids) * a.val_frac))
    val_ids = [ids[i] for i in order[:nv]]
    step = max(1, nv // a.n)
    pick = val_ids[::step][: a.n]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for k, idx in enumerate(pick):
        rgb = np.asarray(Image.open(root / "rgb" / f"{idx:04d}.png").convert("RGB"), np.float32) / 255.0
        lab = np.asarray(Image.open(root / "label" / f"{idx:04d}.png"), np.int64)
        if depth:
            dep = np.asarray(Image.open(root / "depth" / f"{idx:04d}.png"), np.float32) / 65535.0
            arr = np.concatenate([rgb, dep[..., None]], axis=-1)
        else:
            arr = rgb
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().unsqueeze(0).to(dev)

        with torch.no_grad():
            logits = net(x)
            prob = logits.softmax(1).squeeze(0).cpu().numpy()   # (C,H,W)
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int32)

        rgb_disp = (rgb * 255).clip(0, 255).astype(np.uint8)
        gt_disp  = label_to_rgb(lab.astype(np.int32), n_cls)
        pr_disp  = label_to_rgb(pred, n_cls)
        conf     = prob.max(0)

        iou = {}
        for c in range(n_cls):
            tp = ((pred == c) & (lab == c)).sum()
            fp = ((pred == c) & (lab != c)).sum()
            fn = ((pred != c) & (lab == c)).sum()
            u = tp + fp + fn
            iou[NAMES[c]] = round(float(tp) / u, 3) if u else float("nan")  # absent class -> nan (excluded)
        present = [v for v in iou.values() if not np.isnan(v)]
        miou = float(np.mean(present)) if present else float("nan")

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        fig.suptitle(f"sample {idx:04d}   mIoU={miou:.3f}  (best_train={ckpt.get('miou',0):.3f})",
                     fontsize=13, fontweight="bold")
        axes[0].imshow(rgb_disp);  axes[0].set_title("BEV RGB (input)")
        axes[1].imshow(gt_disp);   axes[1].set_title("GT cost map")
        axes[2].imshow(pr_disp);   axes[2].set_title("U-Net prediction")
        im = axes[3].imshow(conf, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        axes[3].set_title("Confidence"); fig.colorbar(im, ax=axes[3], fraction=0.046)

        legend = [Patch(facecolor=COLORS[c]/255, label=f"{NAMES[c]} {iou[NAMES[c]]:.3f}")
                  for c in range(n_cls)]
        axes[2].legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.85, ncol=2)
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])

        fig.tight_layout()
        fout = out / f"infer_{k:03d}.png"
        fig.savefig(fout, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"  [{k:02d}] idx={idx:04d}  mIoU={miou:.3f}  -> {fout}")


if __name__ == "__main__":
    main()
