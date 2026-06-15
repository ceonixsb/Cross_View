"""go2_avoid_env.py — STEP: Go2 dynamic-obstacle AVOIDANCE RL env (Isaac-free, fast).

The high-level policy is trained on a fast KINEMATIC model (Go2 = a point that moves at the
commanded velocity — WTW reliably follows velocity, so this is a faithful abstraction), NOT
Isaac physics. This lets PPO run millions of steps quickly; the trained policy is later deployed
on the real WTW + movers in Isaac (nav_go2_static).

Perception is LIVE cross-view (no belief/memory): each step the obs is what the drone window
(top-down, no occlusion) + Go2 cone (occluded) SEE RIGHT NOW. Moving obstacles only matter live,
so the drone feeds the Go2 the current obstacle map directly — no accumulation.

  obs    = LIVE occupancy patch (drone window ∪ Go2 cone, this instant) + goal dir + velocity
  action = [vx, vy, vyaw]  (body frame, -1..1 -> m/s)
  reward = + progress to goal  - proximity to movers  -- collision(end)  + reach(end)  - time

    python scripts/go2_avoid_env.py        # sanity: random-action rollouts
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_A import MapASpec, sample_obstacles
import astar_nav
from perception import obstacle_grid, visible_go2, visible_drone

BODY_R = 0.22          # Go2 half-width
MOVER_R = 0.25
MAX_SPD = 1.0          # forward vx (m/s) — rl_lab Go2 limit
MAX_VY = 0.4           # lateral vy (m/s) — rl_lab Go2 limit (strafe is capped lower than forward)
MAX_YAW = 1.0          # rad/s


class _Mover:
    """Kinematic moving obstacle: A*-wanders to random goals (avoids static, turns corners)."""

    def __init__(self, grid, spec, rng, speed):
        self.grid, self.spec, self.rng, self.speed = grid, spec, rng, speed
        self.pos = np.array(self._rand_free())
        self.path, self.wi = [self.pos], 0
        self._new_goal()

    def _rand_free(self):
        for _ in range(300):
            x = self.rng.uniform(1.5, self.spec.lane_len - 1.5)
            y = self.rng.uniform(-(self.spec.half_w - 1.5), self.spec.half_w - 1.5)
            if self.grid.free(*self.grid.cell(x, y)):
                return [float(x), float(y)]
        return [self.spec.lane_len / 2, 0.0]

    def _new_goal(self):
        for _ in range(20):
            g = self._rand_free()
            if np.linalg.norm(np.array(g) - self.pos) < 2.5:
                continue
            p = astar_nav.astar(self.grid, self.grid.cell(*self.pos), self.grid.cell(*g))
            if p:
                self.path = [np.array(self.grid.world(*c)) for c in astar_nav.simplify(self.grid, p)]
                self.wi = 1
                return
        self.path, self.wi = [self.pos], 0

    def step(self, dt):
        if self.wi >= len(self.path):
            self._new_goal(); return
        d = self.path[self.wi] - self.pos
        dist = float(np.linalg.norm(d))
        if dist < 0.2:
            self.wi += 1
            if self.wi >= len(self.path):
                self._new_goal()
            return
        self.pos = self.pos + (d / dist) * min(self.speed * dt, dist)


class Go2AvoidEnv:
    """Single-env (gym-style reset/step) kinematic avoidance task. Vectorise later for training."""

    def __init__(self, map_seed=0, n_movers=6, seed=0, dt=0.1, max_steps=300, patch=16, patch_res=0.25):
        self.spec = MapASpec()
        self.obstacles = sample_obstacles(self.spec, map_seed)
        self.vgrid = obstacle_grid(self.spec, self.obstacles)                       # raw, for visibility
        self.pgrid = astar_nav.Grid(self.spec, self.obstacles, robot_radius=0.45)   # mover A* + free sampling
        self.cgrid = astar_nav.Grid(self.spec, self.obstacles, robot_radius=BODY_R) # Go2-body static collision
        self.rng = np.random.default_rng(seed)
        self.n_movers, self.dt, self.max_steps = n_movers, dt, max_steps
        self.patch, self.patch_res = patch, patch_res
        self.obs_dim = patch * patch + 4
        self.act_dim = 3

    def _rand_free(self):
        for _ in range(300):
            x = self.rng.uniform(1.5, self.spec.lane_len - 1.5)
            y = self.rng.uniform(-(self.spec.half_w - 1.5), self.spec.half_w - 1.5)
            if self.pgrid.free(*self.pgrid.cell(x, y)):
                return np.array([float(x), float(y)])
        return np.array([self.spec.lane_len / 2, 0.0])

    def reset(self):
        for _ in range(50):
            self.pos = self._rand_free(); self.goal = self._rand_free()
            if np.linalg.norm(self.pos - self.goal) >= 8.0:
                break
        self.yaw = float(self.rng.uniform(-math.pi, math.pi))
        self.vel = np.zeros(2)
        self.movers = [_Mover(self.pgrid, self.spec, self.rng, float(self.rng.uniform(0.4, 0.8)))
                       for _ in range(self.n_movers)]
        self.t = 0
        self.prev_dist = float(np.linalg.norm(self.pos - self.goal))
        return self._obs()

    def _live_occ(self):
        """What the team SEES this instant: drone window (no occlusion) ∪ Go2 cone (occluded).
        Returns (occupied, known) masks — static obstacles where seen + movers where seen. No memory."""
        vis = visible_go2(self.vgrid, self.pos, self.yaw) | visible_drone(self.vgrid, self.pos)
        occ = self.vgrid.blocked & vis                                              # static obstacles in view now
        for mv in self.movers:                                                      # moving obstacles in view now
            cx, cy = self.vgrid.cell(*mv.pos)
            if vis[cx, cy]:
                x0, x1 = max(0, cx - 2), min(self.vgrid.nx, cx + 3)
                y0, y1 = max(0, cy - 2), min(self.vgrid.ny, cy + 3)
                occ[x0:x1, y0:y1] = True
        return occ, vis

    def _obs(self):
        occ, known = self._live_occ()
        P, r = self.patch, self.patch_res
        ii, jj = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
        wx = self.pos[0] + (ii - P / 2) * r
        wy = self.pos[1] + (jj - P / 2) * r
        cx = np.clip((wx / self.vgrid.res).astype(int), 0, self.vgrid.nx - 1)
        cy = np.clip(((wy + self.vgrid.hw) / self.vgrid.res).astype(int), 0, self.vgrid.ny - 1)
        kn, oc = known[cx, cy], occ[cx, cy]
        patch = np.where(kn, np.where(oc, 1.0, 0.0), 0.5).astype(np.float32)        # 0 free / 1 occ / 0.5 unknown
        gr = self.goal - self.pos
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        gr_body = np.array([gr[0] * c + gr[1] * s, -gr[0] * s + gr[1] * c]) / 10.0  # body frame, scaled
        return np.concatenate([patch.flatten(), gr_body, self.vel]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        vx, vy, vyaw = a[0] * MAX_SPD, a[1] * MAX_VY, a[2] * MAX_YAW
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.pos = self.pos + np.array([vx * c - vy * s, vx * s + vy * c]) * self.dt
        self.yaw = (self.yaw + vyaw * self.dt + math.pi) % (2 * math.pi) - math.pi
        self.vel = np.array([vx, vy])
        for mv in self.movers:
            mv.step(self.dt)
        self.t += 1

        dist = float(np.linalg.norm(self.pos - self.goal))
        reach = dist < 1.0
        cx, cy = self.cgrid.cell(*self.pos)
        hit_static = (not (0 <= cx < self.cgrid.nx and 0 <= cy < self.cgrid.ny)) or bool(self.cgrid.blocked[cx, cy])
        dmin = min((float(np.linalg.norm(self.pos - mv.pos)) for mv in self.movers), default=99.0)
        hit_mover = dmin < BODY_R + MOVER_R

        rew = 3.0 * (self.prev_dist - dist) - 0.01                                  # progress (telescoping) - time
        if dmin < 1.0:                                                              # soft proximity to movers
            rew -= (1.0 - dmin) * 0.5
        done = False
        if hit_static or hit_mover:
            rew -= 10.0; done = True
        elif reach:
            rew += 10.0; done = True
        elif self.t >= self.max_steps:
            done = True
        self.prev_dist = dist
        info = {"reach": reach, "hit_static": hit_static, "hit_mover": hit_mover}
        return self._obs(), float(rew), done, info


if __name__ == "__main__":
    env = Go2AvoidEnv(map_seed=0, n_movers=4, seed=0)
    print(f"obs_dim={env.obs_dim}  act_dim={env.act_dim}")
    for ep in range(3):
        o = env.reset()
        ret = 0.0
        for _ in range(env.max_steps):
            o, r, d, info = env.step(env.rng.uniform(-1, 1, 3))                     # random actions
            ret += r
            if d:
                break
        print(f"  ep{ep}: steps={env.t:3d} return={ret:+6.1f}  reach={info['reach']} "
              f"hit_static={info['hit_static']} hit_mover={info['hit_mover']}  obs.shape={o.shape}")
    print("  env runs ✓ (random actions). next: vectorize + PPO.")
