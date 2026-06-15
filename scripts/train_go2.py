"""train_go2.py — STAGE 1: Go2-only PPO (drone SCRIPTED). De-risk before MARL.

Wraps MarlNavEnv (drone_mode='scripted', fixed lead) as a single-agent gymnasium env exposing only
the Go2's obs/action, and trains SB3 PPO over parallel envs. Goal: verify the Go2 learns to follow
the A* path while dodging movers, using the (drone-relayed) path-ahead strip. Stage 2 swaps the
scripted drone for an RL drone.

    python scripts/train_go2.py --smoke                 # quick wiring test (~20k steps)
    python scripts/train_go2.py --timesteps 3000000 --n_envs 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_nav_env import MarlNavEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback


class Go2Single(gym.Env):
    """Single-agent view of MarlNavEnv: drone is scripted, only the Go2 is controlled/observed."""

    def __init__(self, map_seed=0, n_movers=6, seed=0, drone_lead=6.0, randomize_map=False):
        self.env = MarlNavEnv(map_seed=map_seed, n_movers=n_movers, seed=seed,
                              drone_mode="scripted", drone_lead=drone_lead, randomize_map=randomize_map)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.env.go2_obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.env.go2_act_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        return self.env.reset()["go2"], {}

    def step(self, action):
        o, r, d, info = self.env.step({"go2": action})
        term = bool(d and (info["reach"] or info["hit_static"] or info["hit_mover"]))
        trunc = bool(d and not term)
        return o["go2"], r["go2"], term, trunc, info


class StatCb(BaseCallback):
    """Log reach / collision rate over recent episodes (ep_rew alone isn't interpretable)."""

    def __init__(self):
        super().__init__()
        self.reach, self.hit = [], []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.reach.append(1.0 if info.get("reach") else 0.0)
                self.hit.append(1.0 if (info.get("hit_static") or info.get("hit_mover")) else 0.0)
        return True

    def _on_rollout_end(self):
        if self.reach:
            self.logger.record("rollout/reach_rate", float(np.mean(self.reach[-200:])))
            self.logger.record("rollout/collide_rate", float(np.mean(self.hit[-200:])))


def make_env(rank, n_movers, drone_lead):
    def _init():
        return Go2Single(map_seed=rank, n_movers=n_movers, seed=1000 + rank,
                         drone_lead=drone_lead, randomize_map=True)               # fresh map/episode -> generalize
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--iters", type=int, default=None, help="train for N iterations (overrides --timesteps); 1 iter = n_envs*n_steps")
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--movers", type=int, default=6)
    ap.add_argument("--drone_lead", type=float, default=6.0)
    ap.add_argument("--tag", default="tuned", help="run name suffix -> logs/go2_ppo_<tag>/ (keeps prior runs)")
    ap.add_argument("--smoke", action="store_true", help="quick wiring test")
    args = ap.parse_args()
    if args.smoke:
        args.timesteps, args.n_envs = 20_000, 4
    elif args.iters is not None:
        args.timesteps = args.iters * args.n_envs * 1024                          # 1 iter = n_envs * n_steps(1024)

    name = "go2_ppo" + (f"_{args.tag}" if args.tag else "")
    out = Path(__file__).resolve().parent.parent / "logs" / name
    out.mkdir(parents=True, exist_ok=True)

    venv = SubprocVecEnv([make_env(i, args.movers, args.drone_lead) for i in range(args.n_envs)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)   # ~ valuenorm/adv_normalize

    model = PPO(
        "MlpPolicy", venv,
        n_steps=1024, batch_size=2048, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1, tensorboard_log=str(out / "tb"),
    )
    print(f"  obs={model.observation_space.shape} act={model.action_space.shape} "
          f"n_envs={args.n_envs} timesteps={args.timesteps} device={model.device}")
    model.learn(total_timesteps=args.timesteps, callback=StatCb(), progress_bar=False)

    model.save(out / "go2_ppo")
    venv.save(str(out / "vecnorm.pkl"))
    print(f"  saved -> {out}/go2_ppo.zip , vecnorm.pkl")
    venv.close()


if __name__ == "__main__":
    main()
