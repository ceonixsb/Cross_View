"""marl_nav_env.py — 2-agent MARL nav env (Isaac-free, fast). Drone scouts AHEAD on the Go2's path.

Per episode: random start/goal -> A* path over STATIC obstacles -> waypoints (the "big picture").
The Go2 follows that path but must dodge MOVING obstacles live; the Drone flies ALONG the path,
positioned some lead-distance AHEAD of the Go2, and feeds early warning of movers the Go2 can't
yet see. The contribution is the Drone LEARNING where to look — not just a bigger obs window.

  Go2   obs = local patch (immediate) + path-ahead strip (drone relay) + look dir + remaining + vel
        act = [vx, vy, vyaw]  (body frame, -1..1)
        rew = + path progress  - mover proximity  -- collision(end)  + reach(end)  - time
  Drone obs = path-ahead strip + current lead + Go2 speed + Go2 progress
        act = [lead]  (-1..1 -> how far AHEAD on the path to position)
        rew = + path-ahead coverage  + scoop(movers only the drone sees, on the path)  + team  - far - time

  drone_mode: 'rl' (learned lead) | 'scripted' (fixed lead) | 'off' (no drone window, ablation)

    python scripts/marl_nav_env.py        # sanity: random-action rollouts (scripted + rl)
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
from go2_avoid_env import _Mover, BODY_R, MOVER_R, MAX_SPD, MAX_VY, MAX_YAW

# drone altitude: window half-size = alt/3 (higher -> wider) but above MOVER_SEE_ALT the drone is too
# high to resolve the small moving obstacles (sees static layout, MISSES movers) -> a real tradeoff.
H_MIN, H_MAX = 6.0, 18.0
MOVER_SEE_ALT = 12.0       # drone detects movers only at/below this altitude
ALT_RATE = 1.0             # max altitude change per step (m)
ALT_DEFAULT = 9.0          # scripted / start altitude -> half = 3.0 (reproduces the old fixed window)


def _densify_arc(pts, step=0.2):
    """Polyline of a waypoint list + cumulative arc length, for projecting/sampling along the path."""
    poly = []
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(float(np.linalg.norm(b - a)) / step))
        for k in range(n):
            poly.append(a + (b - a) * (k / n))
    poly.append(np.asarray(pts[-1], float))
    poly = np.array(poly)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    return poly, arc


class MarlNavEnv:
    def __init__(self, map_seed=0, n_movers=6, seed=0, dt=0.1, max_steps=400,
                 patch=16, patch_res=0.25, strip_n=12, strip_step=1.0,
                 drone_mode="scripted", drone_lead=6.0, randomize_map=False):
        self.spec = MapASpec()
        self.randomize_map = randomize_map                                         # True: fresh map every episode
        self._build_map(map_seed)
        self.rng = np.random.default_rng(seed)
        self.n_movers, self.dt, self.max_steps = n_movers, dt, max_steps
        self.patch, self.patch_res = patch, patch_res
        self.strip_n, self.strip_step = strip_n, strip_step
        self.lookahead = 1.5
        self.lead_min, self.lead_max, self.lead_rate, self.lead_soft = 2.0, 12.0, 1.0, 10.0
        self.drone_mode, self.drone_lead = drone_mode, drone_lead

        self.go2_obs_dim = patch * patch + strip_n + 2 + 1 + 2
        self.drone_obs_dim = strip_n + 1 + 1 + 1 + 1                                # + altitude
        self.go2_act_dim, self.drone_act_dim = 3, 2                                 # drone: [lead, altitude]

    def _build_map(self, map_seed):
        self.obstacles = sample_obstacles(self.spec, map_seed)
        self.vgrid = obstacle_grid(self.spec, self.obstacles)                       # raw, for visibility
        self.pgrid = astar_nav.Grid(self.spec, self.obstacles, robot_radius=0.45)   # A* path + free sampling
        self.cgrid = astar_nav.Grid(self.spec, self.obstacles, robot_radius=BODY_R) # Go2-body static collision

    # ---- path geometry -------------------------------------------------------
    def _rand_free(self):
        for _ in range(300):
            x = self.rng.uniform(1.5, self.spec.lane_len - 1.5)
            y = self.rng.uniform(-(self.spec.half_w - 1.5), self.spec.half_w - 1.5)
            if self.pgrid.free(*self.pgrid.cell(x, y)):
                return np.array([float(x), float(y)])
        return np.array([self.spec.lane_len / 2, 0.0])

    def _s_of(self, p):                                                             # project point -> arc length
        return float(self.arc[int(np.argmin(np.linalg.norm(self.poly - p, axis=1)))])

    def _pt(self, s):                                                              # arc length -> point on path
        s = float(np.clip(s, 0.0, self.path_len))
        i = min(int(np.searchsorted(self.arc, s)), len(self.poly) - 1)
        return self.poly[i]

    # ---- sensing -------------------------------------------------------------
    def _sense(self):
        """Live cross-view this instant. Drone window scales with altitude; above MOVER_SEE_ALT the
        drone is too high to resolve movers (sees static layout but MISSES the moving obstacles)."""
        vg = visible_go2(self.vgrid, self.go2, self.yaw)
        half = self.drone_alt / 3.0                                                 # higher -> wider window
        vd = visible_drone(self.vgrid, self.drone, half=half) if self.drone_on else np.zeros_like(vg)
        sees_movers = self.drone_alt <= MOVER_SEE_ALT                               # too high -> can't see movers
        vis = vg | vd
        mover = np.zeros_like(vis)
        for mv in self.movers:
            cx, cy = self.vgrid.cell(*mv.pos)
            if vg[cx, cy] or (vd[cx, cy] and sees_movers):                          # Go2 cone, or low-enough drone
                x0, x1 = max(0, cx - 2), min(self.vgrid.nx, cx + 3)
                y0, y1 = max(0, cy - 2), min(self.vgrid.ny, cy + 3)
                mover[x0:x1, y0:y1] = True
        self._vg, self._vd, self._vis = vg, vd, vis
        self._occ = (self.vgrid.blocked & vis) | mover
        self._mover = mover

    def _strip(self, s0):
        """Occupancy sampled along the path ahead of arc s0: 1 occupied / 0 free / 0.5 unseen."""
        vals = np.full(self.strip_n, 0.5, np.float32)
        cells = []
        for k in range(self.strip_n):
            cx, cy = self.vgrid.cell(*self._pt(s0 + (k + 1) * self.strip_step))
            cells.append((cx, cy))
            if self._vis[cx, cy]:
                vals[k] = 1.0 if self._occ[cx, cy] else 0.0
        self._strip_cells = cells
        return vals

    # ---- obs -----------------------------------------------------------------
    def _patch(self):
        P, r = self.patch, self.patch_res
        ii, jj = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
        wx = self.go2[0] + (ii - P / 2) * r
        wy = self.go2[1] + (jj - P / 2) * r
        cx = np.clip((wx / self.vgrid.res).astype(int), 0, self.vgrid.nx - 1)
        cy = np.clip(((wy + self.vgrid.hw) / self.vgrid.res).astype(int), 0, self.vgrid.ny - 1)
        vis, occ = self._vis[cx, cy], self._occ[cx, cy]
        return np.where(vis, np.where(occ, 1.0, 0.0), 0.5).astype(np.float32).flatten()

    def _obs(self):
        s = self.s_go2
        strip = self._strip(s)
        look_w = self._pt(s + self.lookahead) - self.go2
        c, sn = math.cos(self.yaw), math.sin(self.yaw)
        look_b = np.array([look_w[0] * c + look_w[1] * sn, -look_w[0] * sn + look_w[1] * c])
        nb = float(np.linalg.norm(look_b)); look_b = look_b / nb if nb > 1e-6 else np.array([1.0, 0.0])
        remain = np.array([(self.path_len - s) / 10.0], np.float32)
        go2 = np.concatenate([self._patch(), strip, look_b, remain, self.vel]).astype(np.float32)
        drone = np.concatenate([strip,
                                [self.lead / self.lead_max],
                                [float(np.linalg.norm(self.vel))],
                                [s / self.path_len],
                                [self.drone_alt / H_MAX]]).astype(np.float32)
        return {"go2": go2, "drone": drone}

    # ---- rollout -------------------------------------------------------------
    def reset(self):
        if self.randomize_map:                                                     # fresh layout every episode
            self._build_map(int(self.rng.integers(0, 1_000_000)))
        for _ in range(80):
            start, goal = self._rand_free(), self._rand_free()
            if np.linalg.norm(start - goal) < 8.0:
                continue
            cells = astar_nav.astar(self.pgrid, self.pgrid.cell(*start), self.pgrid.cell(*goal))
            if cells is None:
                continue
            wps = [self.pgrid.world(*c) for c in astar_nav.simplify(self.pgrid, cells)]
            self.poly, self.arc = _densify_arc(wps, 0.2)
            self.path_len = float(self.arc[-1])
            if self.path_len >= 8.0:
                break
        self.go2 = start.astype(float)
        look = self._pt(self.lookahead) - self.go2
        self.yaw = math.atan2(look[1], look[0])
        self.vel = np.zeros(2)
        self.drone_on = self.drone_mode != "off"
        self.lead = self.drone_lead if self.drone_mode == "scripted" else self.lead_min
        self.drone_alt = ALT_DEFAULT
        self.drone = self._pt(self.lead) if self.drone_on else self.go2.copy()
        self.movers = [_Mover(self.pgrid, self.spec, self.rng, float(self.rng.uniform(0.4, 0.8)))
                       for _ in range(self.n_movers)]                               # movers slower than Go2 (1.0)
        self.t = 0
        self.s_go2 = self._s_of(self.go2)
        self.prev_s = self.s_go2
        self._sense()
        return self._obs()

    def step(self, actions):
        # drone: set lead + altitude (learned / scripted), rate-limited
        if self.drone_mode == "rl":
            a = np.clip(np.asarray(actions["drone"], dtype=float), -1, 1)
            lead_cmd = self.lead_min + (a[0] + 1) / 2 * (self.lead_max - self.lead_min)
            alt_cmd = H_MIN + (a[1] + 1) / 2 * (H_MAX - H_MIN)
        else:
            lead_cmd, alt_cmd = self.drone_lead, ALT_DEFAULT
        self.lead += float(np.clip(lead_cmd - self.lead, -self.lead_rate, self.lead_rate))
        self.drone_alt += float(np.clip(alt_cmd - self.drone_alt, -ALT_RATE, ALT_RATE))

        # go2: kinematic velocity command
        ag = np.clip(np.asarray(actions["go2"], dtype=float), -1, 1)
        vx, vy, vyaw = ag[0] * MAX_SPD, ag[1] * MAX_VY, ag[2] * MAX_YAW
        c, sn = math.cos(self.yaw), math.sin(self.yaw)
        self.go2 = self.go2 + np.array([vx * c - vy * sn, vx * sn + vy * c]) * self.dt
        self.yaw = (self.yaw + vyaw * self.dt + math.pi) % (2 * math.pi) - math.pi
        self.vel = np.array([vx, vy])
        for mv in self.movers:
            mv.step(self.dt)
        self.t += 1

        # update progress + drone position on path, then sense
        self.s_go2 = self._s_of(self.go2)
        self.drone = self._pt(self.s_go2 + self.lead) if self.drone_on else self.go2.copy()
        self._sense()

        # outcomes
        dmin = min((float(np.linalg.norm(self.go2 - mv.pos)) for mv in self.movers), default=99.0)
        cx, cy = self.cgrid.cell(*self.go2)
        hit_static = (not (0 <= cx < self.cgrid.nx and 0 <= cy < self.cgrid.ny)) or bool(self.cgrid.blocked[cx, cy])
        hit_mover = dmin < BODY_R + MOVER_R
        reach = float(np.linalg.norm(self.go2 - self._pt(self.path_len))) < 1.0

        # go2 reward (individual: advance the path, avoid movers)
        rg = 3.0 * (self.s_go2 - self.prev_s) - 0.01
        if dmin < 2.0:                                                              # mover within 2m → penalty
            rg -= (2.0 - dmin) * 0.8
        self.prev_s = self.s_go2

        # drone reward (individual: illuminate path ahead, scoop movers Go2 can't see)
        strip_cells = self._strip_cells
        cover = float(np.mean([self._vis[cx, cy] for cx, cy in strip_cells]))
        scoop = sum(1 for cx, cy in strip_cells
                    if self._mover[cx, cy] and self._vd[cx, cy] and not self._vg[cx, cy])
        rd = 0.1 * cover + 0.5 * scoop - 0.01
        if self.lead > self.lead_soft:
            rd -= (self.lead - self.lead_soft) * 0.05

        done = False
        if hit_static or hit_mover:
            rg -= 15.0; rd -= 15.0; done = True                                    # team failure shared
        elif reach:
            rg += 10.0; rd += 10.0; done = True                                    # team success shared
        elif self.t >= self.max_steps:
            done = True

        info = {"reach": reach, "hit_static": hit_static, "hit_mover": hit_mover,
                "lead": self.lead, "alt": self.drone_alt, "cover": cover, "scoop": scoop,
                "progress": self.s_go2 / self.path_len}
        return self._obs(), {"go2": float(rg), "drone": float(rd)}, done, info


def _rollout(mode, seed=0):
    env = MarlNavEnv(map_seed=0, n_movers=6, seed=seed, drone_mode=mode)
    o = env.reset()
    rg = rd = 0.0
    for _ in range(env.max_steps):
        acts = {"go2": env.rng.uniform(-1, 1, env.go2_act_dim),
                "drone": env.rng.uniform(-1, 1, env.drone_act_dim)}
        o, r, d, info = env.step(acts)
        rg += r["go2"]; rd += r["drone"]
        if d:
            break
    print(f"  mode={mode:9s} steps={env.t:3d}  Rg={rg:+6.1f} Rd={rd:+6.1f}  "
          f"reach={info['reach']} hit={info['hit_static'] or info['hit_mover']} "
          f"prog={info['progress']:.0%} lead={info['lead']:.1f} alt={info.get('alt',0):.1f} cover={info['cover']:.0%}")


if __name__ == "__main__":
    env = MarlNavEnv()
    print(f"go2_obs={env.go2_obs_dim} go2_act={env.go2_act_dim}  "
          f"drone_obs={env.drone_obs_dim} drone_act={env.drone_act_dim}")
    for mode in ("scripted", "rl", "off"):
        for s in range(2):
            _rollout(mode, seed=s)
    print("  env runs ✓ (random actions). next: PPO/MAPPO.")
