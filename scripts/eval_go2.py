"""eval_go2.py — load the Stage-1 Go2 checkpoint, measure success on HELD-OUT maps, render a GIF.

Loads logs/go2_ppo_<tag>/go2_ppo.zip + vecnorm.pkl, runs deterministic episodes on map_seeds the
policy never trained on (>=100), reports reach/collide/timeout, and saves a top-down animation of
one episode so you can SEE the Go2 follow the A* path and dodge the moving obstacles.

    python scripts/eval_go2.py                      # 30 eval episodes + GIF
    python scripts/eval_go2.py --episodes 50 --tag tuned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_go2 import Go2Single

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _load(tag):
    out = Path(__file__).resolve().parent.parent / "logs" / ("go2_ppo" + (f"_{tag}" if tag else ""))
    model = PPO.load(out / "go2_ppo.zip", device="cpu")
    return out, model


def _make_venv(map_seed, vecnorm_path):
    venv = DummyVecEnv([lambda: Go2Single(map_seed=map_seed, n_movers=6, seed=10_000 + map_seed)])
    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def evaluate(model, vecnorm_path, episodes, base_seed=100):
    reach = collide = timeout = 0
    for ep in range(episodes):
        venv = _make_venv(base_seed + ep, vecnorm_path)
        obs = venv.reset()
        done = [False]
        info = [{}]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(action)
        i = info[0]
        if i.get("reach"):
            reach += 1
        elif i.get("hit_static") or i.get("hit_mover"):
            collide += 1
        else:
            timeout += 1
        venv.close()
    n = episodes
    print(f"\n  HELD-OUT eval ({n} eps, map_seeds {base_seed}..{base_seed+n-1}):")
    print(f"    reach   {reach/n:5.1%}  ({reach})")
    print(f"    collide {collide/n:5.1%}  ({collide})")
    print(f"    timeout {timeout/n:5.1%}  ({timeout})")
    return reach / n, collide / n


def render_episode(model, vecnorm_path, map_seed, out_gif):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.animation import FuncAnimation, PillowWriter

    venv = _make_venv(map_seed, vecnorm_path)
    env = venv.venv.envs[0].env                                                    # the underlying MarlNavEnv
    obs = venv.reset()
    frames = []                                                                    # (go2, yaw, movers, drone, outcome)
    done = [False]
    outcome = "timeout"
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(action)
        frames.append((env.go2.copy(), env.yaw,
                       [mv.pos.copy() for mv in env.movers], env.drone.copy()))
        if done[0]:
            i = info[0]
            outcome = "REACH" if i.get("reach") else ("COLLIDE" if (i.get("hit_static") or i.get("hit_mover")) else "timeout")

    spec, obstacles, poly = env.spec, env.obstacles, env.poly
    goal = env._pt(env.path_len)
    fig, ax = plt.subplots(figsize=(12, 9))
    sub = frames[::2]                                                              # subsample for a lighter gif

    def draw(k):
        ax.clear()
        for ox, oy, sx, sy, _ in obstacles:
            ax.add_patch(Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, facecolor="#3f4b5b", edgecolor="k", lw=0.4))
        ax.plot(poly[:, 0], poly[:, 1], "-", color="#9ca3af", lw=1.5, label="A* path")
        ax.plot(*goal, "*", color="#16a34a", ms=22, label="goal")
        g, yaw, movers, drone = sub[k]
        ax.plot(*g, "o", color="#1d4ed8", ms=12)
        ax.arrow(g[0], g[1], 0.7 * np.cos(yaw), 0.7 * np.sin(yaw), head_width=0.3, color="#1d4ed8")
        for m in movers:
            ax.plot(*m, "o", color="#dc2626", ms=11)
        ax.add_patch(Rectangle((drone[0] - 3, drone[1] - 3), 6, 6, fill=False, edgecolor="#f59e0b", lw=1.5, ls="--"))
        ax.set_title(f"Stage-1 Go2 (blue) dodging movers (red) — drone window (orange) — [{outcome}]  step {k*2}")
        ax.set_xlim(0, spec.lane_len); ax.set_ylim(-spec.half_w, spec.half_w); ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)

    anim = FuncAnimation(fig, draw, frames=len(sub), interval=60)
    anim.save(out_gif, writer=PillowWriter(fps=15))
    plt.close(fig)
    venv.close()
    print(f"    [{outcome}] {len(frames)} steps  ->  {out_gif}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--render_seed", type=int, default=100)
    args = ap.parse_args()

    out, model = _load(args.tag)
    vn = out / "vecnorm.pkl"
    evaluate(model, vn, args.episodes)
    gif = Path(__file__).resolve().parent.parent / "data" / "map_A" / f"eval_go2_{args.tag}.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    render_episode(model, vn, args.render_seed, gif)
