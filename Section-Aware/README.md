# CrossView — Y-Road Cost-Map Navigation (Go2)

Drone BEV → **U-Net 4-class cost map** → **Dijkstra path** → **Go2 walks it**
(a frozen `rl_lab` locomotion policy executes the velocity commands).

```
drone whole-map BEV (RGB+D)
        │  U-Net (train_yroad_unet.py)
        ▼
   4-class cost map  ──►  Dijkstra distance field  ──►  (vx, vy, vyaw)  ──►  Go2 (rl_lab locomotion)
   (planner.py)                                          deploy_yroad.py
```

## Pipeline

```bash
conda activate isaacsim

# 1) Capture the drone whole-map BEV (RGB+D) + GT 4-class labels
python collect_yroad.py --n 300 --out data/cnn

# 2) Train the U-Net (BEV -> cost map)  ->  data/BEV/best.pt (+ best_jit.pt)
python train_yroad_unet.py --data data/cnn --epochs 60 --classes 4

# 3) (optional) verify the U-Net (mIoU + 4-panel visualisations)
python infer_yroad.py --ckpt data/BEV/best.pt --data data/cnn --n 8 --out data/shots_bev

# 4) Go2 walks the planned path in Isaac Sim
python deploy_yroad.py --gui --steps 40000 --vx 1.5 --show --drone_ctrl
#   default      = GT planner (cost map rasterised from the known geometry)
#   add --cnn    = walk the U-Net-PREDICTED cost map (loads data/BEV/best_jit.pt)
#   with --cnn   = also pass --sec_len 4.0 (the U-Net was trained on the 4.0 m map)
```

## Cost-map classes

| id | class | cost | meaning |
|----|-------|------|---------|
| 0 | impassable | ∞ | void / wall / sunken / too-narrow — never traversed |
| 1 | free | 1.0 | open ground — cheapest, preferred |
| 2 | rough | 4.0 | bumpy terrain — most avoided, still passable |
| 3 | tunnel | 3.0 | roofed / BEV-occluded — avoided when an open route exists |

Only class 0 is `∞`; 1·2·3 are finite so a route is always found even if it must use rough/tunnel.

## Files

| file | role |
|------|------|
| `l_map.py` | Y-Road (ㄷ) map builder — forks/junctions + GT cost-map rasteriser |
| `collect_yroad.py` | render the drone nadir BEV (RGB+D) + GT labels → dataset |
| `train_yroad_unet.py` | compact 4-channel (RGB+D) U-Net trainer (no external seg deps) |
| `infer_yroad.py` | U-Net prediction visualisation (BEV \| GT \| pred \| confidence) |
| `planner.py` | cost map → Dijkstra distance field → carrot-pursuit (vx,vy,vyaw) |
| `deploy_yroad.py` | Go2 (rl_lab policy) walking the planned path in Isaac Sim |

## Requirements / external assets

Set these at the top of `deploy_yroad.py` to match your machine:

- **Isaac Sim 5.x + Isaac Lab**, conda env `isaacsim`
- **rl_lab locomotion policy** — `RL_LAB/logs/rsl_rl/unitree_go2_velocity/<run>/exported/policy.pt`
  (obs 45-D → 12-D joint targets, 50 Hz, PD kp25/kd0.5)
- **Go2 USD** — `GO2_USD` (joint order must match the policy)
- The trained U-Net (`data/BEV/best.pt`) is **not committed** — regenerate via steps 1–2, or ask the author.

## Drone control (`--drone_ctrl`)

`W/S` forward·back · `A/D` strafe · `←/→` yaw · `PgUp/PgDn` altitude.
The drone's downward camera is the BEV viewport.

## Notes

- Default map is `--sec_len 2.8` (≈30 % smaller than the original 4.0 → faster traversal). Pass `--sec_len 4.0` for the original size / to match the U-Net.
- Low step obstacles are off by default (`--n_obs 0`); bring them back with `--n_obs 16` (local-avoidance RL is future work).
- Reaching the goal auto-stops the run — except under `--drone_ctrl`, which holds the Go2 at the goal so you can keep flying.
