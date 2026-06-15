# Go2 Mover-Avoidance Navigation (Cross-View)

A Unitree **Go2** quadruped navigates from a random start to a random goal through a
dense obstacle field while **avoiding moving obstacles (movers)**, using a fused
cross-view perception signal (ground cone + drone BEV scout). The high-level RL policy
outputs a velocity command `(vx, vy, vyaw)`; a frozen **Walk-These-Ways (WTW)** low-level
policy turns it into joint targets on the real Go2 physics in Isaac Sim.

```
[PPO high-level policy]  -> velocity command (vx, vy, vyaw)
        |
[Walk-These-Ways low-level]  -> joint position targets (12D, frozen)
        |
[Actuator network]  -> torques  ->  Isaac Sim physics
```

Training runs in a fast **Isaac-free kinematic env** (the Go2 is a point tracking the
commanded velocity); the trained policy is then deployed onto the real WTW controller in
Isaac Sim (sim-to-sim).

---

## Repo layout

```
scripts/
  deploy_isaac.py            # sim-to-sim deploy: trained policy -> WTW -> Isaac Sim (GUI demo)
  train_go2.py               # PPO training of the mover-avoidance policy (kinematic, fast)
  eval_go2.py                # held-out eval + top-down GIF (no Isaac needed)
  marl_nav_env.py            # the "brain": kinematic env, policy obs, mover simulation
  go2_avoid_env.py           # shared constants + _Mover (A*-wandering moving obstacle)
  map_A.py                   # bounded lane + randomised obstacle layout (per episode)
  astar_nav.py               # A* path planner over the obstacle grid
  perception.py              # geometric visibility (Go2 cone w/ occlusion, drone BEV window)
  nav_go2_static.py          # no-RL waypoint-following baseline (Isaac)
  scene_map_A.py             # arena viewer (Isaac)
  go2_mqe_low_level_policy.py# WTW low-level wrapper (vendored)
assets/wtw/                  # bundled WTW policy + actuator net (~7 MB)
  body_latest.jit, adaptation_module_latest.jit, unitree_go1.pt
logs/go2_ppo_tuned/          # trained checkpoint (go2_ppo.zip + vecnorm.pkl)
data/eval_go2_tuned.gif      # demo: Go2 dodging movers (kinematic top-down)
```

---

## Setup

### A) Kinematic training / eval — no Isaac Sim
Any Python 3.10+ environment:
```bash
pip install -r requirements.txt
```

### B) Isaac Sim deploy — requires Isaac Sim + Isaac Lab
`deploy_isaac.py` / `nav_go2_static.py` / `scene_map_A.py` need **Isaac Sim 5.x** and
**Isaac Lab**, installed separately (NVIDIA — not pip). Run them inside that conda env
(here: `isaacsim`). The WTW low-level policy and actuator net are already bundled in
`assets/wtw/`, so no other external files are needed.

---

## Usage

### Train (fast, kinematic)
```bash
python scripts/train_go2.py --iters 200 --tag tuned      # ~3.3M steps -> logs/go2_ppo_tuned/
```

### Evaluate + render GIF (no Isaac)
```bash
python scripts/eval_go2.py --tag tuned --episodes 30     # -> data/map_A/eval_go2_tuned.gif
```

### Deploy in Isaac Sim (real WTW physics) — the GUI demo
```bash
# inside the Isaac Sim conda env:
python scripts/deploy_isaac.py --gui --episodes 5
python scripts/deploy_isaac.py --gui --cameras --episodes 5   # + drone BEV / Go2 camera views
```
Useful flags: `--map_seed 100` (held-out layout), `--movers 6`, `--tag tuned`,
`--drone_lead 6.0`, drop `--gui` for headless.

A trained checkpoint (`logs/go2_ppo_tuned/`) ships with the repo, so the deploy and eval
commands work out of the box without retraining.

---

## Notes
- Obstacle layouts are **resampled every episode** (no fixed path is memorisable → the
  policy must perceive the scene), and movers are slower than the Go2 so avoidance is feasible.
- Velocity limits match the deployed low-level controller: `vx ±1.0`, `vy ±0.4`, `vyaw ±1.0`.
- `data/*.png` and `logs/**/tb/` are git-ignored; the shipped checkpoint and demo GIF are kept.
