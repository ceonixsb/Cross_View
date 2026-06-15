"""astar_nav.py — STEP 1b core: A* path planner over the dense map_A (Isaac-free, testable).

Rasterises map_A obstacles into an occupancy grid (dilated by the robot radius), runs grid
A* from start to goal, and shortcuts the cell path into a few straight-line WAYPOINTS that
the Go2 waypoint controller follows. No Isaac Sim needed — verify the path on a PNG first,
then nav_go2_static.py drives the Go2 along these waypoints.

    python scripts/astar_nav.py            # plan over a few seeds -> data/map_A/astar_*.png
"""
from __future__ import annotations

import heapq
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_A import MapASpec, sample_obstacles

ROBOT_RADIUS = 0.35           # Go2 half-width + margin; obstacles are dilated by this
RES = 0.10                    # grid cell size (m)


class Grid:
    """Occupancy grid over the lane: x in [0, L], y in [-hw, hw].  True = blocked."""

    def __init__(self, spec: MapASpec, obstacles, res=RES, robot_radius=ROBOT_RADIUS):
        self.spec, self.res, self.hw = spec, res, spec.half_w
        self.nx = int(round(spec.lane_len / res))
        self.ny = int(round(spec.lane_width / res))
        self.blocked = np.zeros((self.nx, self.ny), dtype=bool)
        pad = robot_radius
        for ox, oy, sx, sy, _ in obstacles:                       # dilate each footprint by the robot radius
            x0, x1 = ox - sx / 2 - pad, ox + sx / 2 + pad
            y0, y1 = oy - sy / 2 - pad, oy + sy / 2 + pad
            self.blocked[self._xr(x0):self._xr(x1) + 1, self._yr(y0):self._yr(y1) + 1] = True
        b = int(math.ceil(robot_radius / res))                    # walls: block a robot-radius band at the edges
        if b > 0:                                                 # (b==0 would make [-b:] == [0:] = the whole grid!)
            self.blocked[:b, :] = True; self.blocked[-b:, :] = True
            self.blocked[:, :b] = True; self.blocked[:, -b:] = True

    def _xr(self, x): return int(np.clip(x / self.res, 0, self.nx - 1))
    def _yr(self, y): return int(np.clip((y + self.hw) / self.res, 0, self.ny - 1))

    def cell(self, x, y): return (self._xr(x), self._yr(y))
    def world(self, ix, iy): return ((ix + 0.5) * self.res, (iy + 0.5) * self.res - self.hw)
    def free(self, ix, iy): return 0 <= ix < self.nx and 0 <= iy < self.ny and not self.blocked[ix, iy]

    def line_clear(self, a, b):                                   # supercover line-of-sight on the grid
        (x0, y0), (x1, y1) = a, b
        n = max(abs(x1 - x0), abs(y1 - y0))
        if n == 0:
            return self.free(x0, y0)
        for i in range(n + 1):
            x = int(round(x0 + (x1 - x0) * i / n))
            y = int(round(y0 + (y1 - y0) * i / n))
            if not self.free(x, y):
                return False
        return True


_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def astar(grid: Grid, start_cell, goal_cell):
    """8-connected A*; returns a list of cells start->goal, or None."""
    if not grid.free(*start_cell) or not grid.free(*goal_cell):
        return None
    sx, sy = start_cell
    gx, gy = goal_cell
    openq = [(0.0, start_cell)]
    came = {}
    g = {start_cell: 0.0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == goal_cell:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        cx, cy = cur
        for dx, dy in _NB:
            nx, ny = cx + dx, cy + dy
            if not grid.free(nx, ny):
                continue
            step = math.hypot(dx, dy)
            ng = g[cur] + step
            if ng < g.get((nx, ny), 1e18):
                g[(nx, ny)] = ng
                came[(nx, ny)] = cur
                h = math.hypot(nx - gx, ny - gy)
                heapq.heappush(openq, (ng + h, (nx, ny)))
    return None


def simplify(grid: Grid, path):
    """Shortcut the dense cell path into a few waypoints via greedy line-of-sight."""
    if not path:
        return path
    wp = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not grid.line_clear(path[i], path[j]):
            j -= 1
        wp.append(path[j]); i = j
    return wp


def plan(spec: MapASpec, seed: int, start_xy, goal_xy, res=RES, robot_radius=ROBOT_RADIUS):
    """Return (waypoints_world, grid, obstacles) where waypoints includes start..goal, or (None, grid, obs)."""
    obstacles = sample_obstacles(spec, seed)
    grid = Grid(spec, obstacles, res, robot_radius)
    path = astar(grid, grid.cell(*start_xy), grid.cell(*goal_xy))
    if path is None:
        return None, grid, obstacles
    cells = simplify(grid, path)
    wps = [list(start_xy)] + [list(grid.world(*c)) for c in cells[1:-1]] + [list(goal_xy)]
    return wps, grid, obstacles


def plan_random(spec: MapASpec, seed: int, start_xy, goal_xy, n_via=2, res=RES, robot_radius=ROBOT_RADIUS):
    """RANDOM (non-optimal) but collision-free path: A* through random free via-points.
    Wanders differently every episode (good for RL diversity) yet never clips a static obstacle."""
    obstacles = sample_obstacles(spec, seed)
    grid = Grid(spec, obstacles, res, robot_radius)
    rng = np.random.default_rng(seed * 1000 + 7)
    pts = [list(start_xy)]
    for _ in range(n_via):                                    # random free via-points between start and goal
        for _ in range(300):
            x = float(rng.uniform(1.0, spec.lane_len - 1.0))
            y = float(rng.uniform(-(spec.half_w - 1.0), spec.half_w - 1.0))
            if grid.free(*grid.cell(x, y)):
                pts.append([x, y]); break
    pts.append(list(goal_xy))
    full = [list(start_xy)]
    for a, b in zip(pts[:-1], pts[1:]):                       # A* each segment, concatenate the waypoints
        path = astar(grid, grid.cell(*a), grid.cell(*b))
        if path is None:
            return None, grid, obstacles
        seg = simplify(grid, path)
        full += [list(grid.world(*c)) for c in seg[1:]]
    return full, grid, obstacles


def random_path_on_grid(grid: Grid, spec: MapASpec, start_xy, goal_xy, n_via, rng):
    """Random collision-free path on a PRE-BUILT grid (same map, different route per call)."""
    pts = [list(start_xy)]
    for _ in range(n_via):
        for _ in range(300):
            x = float(rng.uniform(1.0, spec.lane_len - 1.0))
            y = float(rng.uniform(-(spec.half_w - 1.0), spec.half_w - 1.0))
            if grid.free(*grid.cell(x, y)):
                pts.append([x, y]); break
    pts.append(list(goal_xy))
    full = [list(start_xy)]
    for a, b in zip(pts[:-1], pts[1:]):
        path = astar(grid, grid.cell(*a), grid.cell(*b))
        if path is None:
            return None
        full += [list(grid.world(*c)) for c in simplify(grid, path)[1:]]
    return full


def render(spec, obstacles, grid, wps, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(grid.blocked.T, origin="lower", extent=[0, spec.lane_len, -spec.half_w, spec.half_w],
              cmap="Greys", alpha=0.35)
    for ox, oy, sx, sy, _ in obstacles:
        ax.add_patch(Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, facecolor="#3f4b5b", edgecolor="k", lw=0.5))
    if wps:
        xs = [w[0] for w in wps]; ys = [w[1] for w in wps]
        ax.plot(xs, ys, "-o", color="#e11d48", lw=2, ms=5, label="A* waypoints")
        ax.plot(xs[0], ys[0], "o", color="#2563eb", ms=12, label="start")
        ax.plot(xs[-1], ys[-1], "*", color="#16a34a", ms=18, label="goal")
        ax.legend(loc="upper right")
    else:
        ax.set_title("NO PATH", color="red")
    ax.set_xlim(0, spec.lane_len); ax.set_ylim(-spec.half_w, spec.half_w); ax.set_aspect("equal")
    ax.set_title(title or "A* over map_A")
    fig.tight_layout(); fig.savefig(out_png, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    spec = MapASpec()
    start = (spec.start_x, 0.0)
    goal = (spec.target_x, 0.0)
    out = Path(__file__).resolve().parent.parent / "data" / "map_A"
    out.mkdir(parents=True, exist_ok=True)
    print(f"A* over map_A  start={start}  goal={goal}  res={RES}  robot_r={ROBOT_RADIUS}")
    for seed in range(4):
        wps, grid, obstacles = plan(spec, seed, start, goal)
        if wps is None:
            print(f"  seed{seed}: NO PATH")
        else:
            length = sum(math.dist(wps[i], wps[i + 1]) for i in range(len(wps) - 1))
            print(f"  seed{seed}: {len(wps)} waypoints, path length {length:.1f} m")
        render(spec, obstacles, grid, wps, out / f"astar_seed{seed}.png",
               title=f"A* over map_A (seed {seed})")
    print(f"  rendered -> {out}/astar_seed*.png")
