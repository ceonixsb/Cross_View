"""perception.py — STEP 3-(1): geometric VISIBILITY (Isaac-free, testable).

The cross-view perception core. Given the obstacle grid:
  * visible_go2 : forward sector (radius R, +/- theta) with OCCLUSION (rays stop at obstacles
                  -> shadows behind walls). This is "the ground robot can't see behind things".
  * visible_drone: a small top-down WINDOW around the drone (no occlusion, but LIMITED area).
                  The drone is a scout you position -> in step 4 an RL policy moves this window.

Run it to render Go2 cone (+shadows) and drone window on map_A and eyeball the occlusion:

    python scripts/perception.py            # -> data/map_A/visibility.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_A import MapASpec, sample_obstacles
import astar_nav

# perception uses the RAW obstacle occupancy (no robot-radius dilation) — occlusion is about the
# real obstacle geometry, not the robot's planning clearance.
GO2_R = 4.0          # Go2 forward sensor range (m)
GO2_THETA = 50.0     # Go2 half field-of-view (deg)
DRONE_HALF = 3.0     # drone window half-size (m) -> a 6x6 m patch


def obstacle_grid(spec, obstacles, res=astar_nav.RES):
    return astar_nav.Grid(spec, obstacles, res=res, robot_radius=0.0)


def visible_go2(grid, pos, heading, R=GO2_R, theta_deg=GO2_THETA, ang_step_deg=1.0):
    """Boolean mask of cells the Go2 sees: forward sector, rays stop at obstacles (occlusion).
    Vectorised over rays — all angles marched together, rays die when they hit an obstacle / exit."""
    mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    angs = heading + np.radians(np.arange(-theta_deg, theta_deg + 1e-6, ang_step_deg))
    dx, dy = np.cos(angs), np.sin(angs)
    alive = np.ones(angs.shape[0], dtype=bool)
    step = grid.res * 0.5
    r = 0.0
    while r <= R and alive.any():
        cx = ((pos[0] + dx * r) / grid.res).astype(int)
        cy = ((pos[1] + dy * r + grid.hw) / grid.res).astype(int)
        inb = (cx >= 0) & (cx < grid.nx) & (cy >= 0) & (cy < grid.ny)
        act = alive & inb                                     # rays still marching and in-bounds
        idx = np.where(act)[0]
        ccx, ccy = cx[idx], cy[idx]
        mask[ccx, ccy] = True                                 # see this cell (incl. an obstacle's face)
        alive[idx[grid.blocked[ccx, ccy]]] = False            # rays that hit an obstacle stop (shadow behind)
        alive &= inb                                          # rays that left the grid stop
        r += step
    return mask


def visible_drone(grid, pos, half=DRONE_HALF):
    """Boolean mask of cells in the drone's top-down window (no occlusion, limited area)."""
    mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    x0, y0 = grid.cell(pos[0] - half, pos[1] - half)
    x1, y1 = grid.cell(pos[0] + half, pos[1] + half)
    mask[max(0, x0):min(grid.nx, x1 + 1), max(0, y0):min(grid.ny, y1 + 1)] = True
    return mask


def render(spec, grid, go2_pos, go2_heading, drone_pos, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    vg = visible_go2(grid, go2_pos, go2_heading)
    vd = visible_drone(grid, drone_pos)
    blocked = grid.blocked

    img = np.full((grid.nx, grid.ny, 3), 1.0)                  # white = unknown/unseen
    img[vd & ~blocked] = [1.0, 0.82, 0.45]                     # drone-seen = orange
    img[vg & ~blocked] = [0.55, 0.78, 1.0]                     # Go2-seen = blue
    img[vg & vd & ~blocked] = [0.55, 0.9, 0.55]               # both = green (fusion)
    img[blocked] = [0.25, 0.28, 0.34]                          # obstacles = dark gray

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower",
              extent=[0, spec.lane_len, -spec.half_w, spec.half_w])
    # Go2 marker + heading arrow
    ax.plot(*go2_pos, "o", color="#1d4ed8", ms=12)
    ax.arrow(go2_pos[0], go2_pos[1], 1.2 * math.cos(go2_heading), 1.2 * math.sin(go2_heading),
             head_width=0.4, color="#1d4ed8")
    # drone marker + window box
    ax.plot(*drone_pos, "s", color="#dc2626", ms=12)
    ax.add_patch(Rectangle((drone_pos[0] - DRONE_HALF, drone_pos[1] - DRONE_HALF),
                           2 * DRONE_HALF, 2 * DRONE_HALF, fill=False, edgecolor="#dc2626", lw=2))
    ax.set_title("visibility — blue=Go2 (w/ shadows), orange=drone window, green=both, gray=obstacle")
    ax.set_xlim(0, spec.lane_len); ax.set_ylim(-spec.half_w, spec.half_w); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    n_g, n_d = int((vg & ~blocked).sum()), int((vd & ~blocked).sum())
    print(f"  Go2 sees {n_g} cells (cone+shadows), drone sees {n_d} cells (window)")


class Belief:
    """Accumulated 'what we've seen' grid: a STATIC layer (occupied/free, remembered once seen)
    + a DYNAMIC layer (mover cells with a timestamp that decays). Built from visibility masks."""

    def __init__(self, grid):
        self.grid = grid
        self.seen = np.zeros((grid.nx, grid.ny), dtype=bool)        # ever observed -> static known
        self.static_occ = np.zeros((grid.nx, grid.ny), dtype=bool)  # static obstacle, where seen
        self.dyn_t = np.full((grid.nx, grid.ny), -1e9)              # last time a mover was seen here

    def update(self, visible, mover_cells, t):
        self.seen |= visible
        self.static_occ[visible] = self.grid.blocked[visible]       # record static occupancy where seen
        for cx, cy in mover_cells:                                  # a mover counts only if it's in view now
            if visible[cx, cy]:
                self.dyn_t[cx, cy] = t

    def occupancy(self, t=1e9, decay=1.5):
        dyn_fresh = (t - self.dyn_t) <= decay
        occupied = (self.seen & self.static_occ) | dyn_fresh
        known = self.seen | dyn_fresh
        return occupied, known


def _densify(path, step=0.4):
    pts = []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.array(a, float), np.array(b, float)
        n = max(1, int(float(np.linalg.norm(b - a)) / step))
        for k in range(n):
            pts.append(a + (b - a) * (k / n))
    pts.append(np.array(path[-1], float))
    return pts


def simulate_belief(spec, seed, n_via=2):
    """Walk a Go2 along a random A* path (drone hovering above it) and accumulate the FUSED belief."""
    obstacles = sample_obstacles(spec, seed)
    grid = obstacle_grid(spec, obstacles)
    pgrid = astar_nav.Grid(spec, obstacles, robot_radius=0.45)
    rng = np.random.default_rng(seed)
    path = astar_nav.random_path_on_grid(pgrid, spec, (spec.start_x, 0.0), (spec.target_x, 0.0), n_via, rng)
    pts = _densify(path, 0.4)
    belief = Belief(grid)
    for i, p in enumerate(pts):
        nxt = pts[min(i + 1, len(pts) - 1)]
        heading = math.atan2(nxt[1] - p[1], nxt[0] - p[0]) if not np.allclose(nxt, p) else 0.0
        vg = visible_go2(grid, p, heading)
        vd = visible_drone(grid, p)                                 # drone directly above the Go2 (for now)
        belief.update(vg | vd, [], t=i * 0.3)
    return belief, grid, obstacles, pts


def render_belief(spec, grid, obstacles, pts, belief, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    occ, known = belief.occupancy()
    img = np.full((grid.nx, grid.ny, 3), 0.62)                      # gray = unknown
    img[known & ~occ] = [1.0, 1.0, 1.0]                            # free = white
    img[occ] = [0.1, 0.1, 0.1]                                     # occupied = black

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower",
              extent=[0, spec.lane_len, -spec.half_w, spec.half_w])
    for ox, oy, sx, sy, _ in obstacles:                            # GT obstacle outlines (dashed) for comparison
        ax.add_patch(Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, fill=False, edgecolor="#e11d48", lw=0.8, ls="--"))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="#2563eb", lw=2, label="Go2 path")
    ax.set_title("belief — white=free, black=occupied(seen), gray=UNKNOWN, red dash=GT obstacle")
    ax.legend(loc="upper right")
    ax.set_xlim(0, spec.lane_len); ax.set_ylim(-spec.half_w, spec.half_w); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    print(f"  belief: {float(known.mean())*100:.0f}% of cells known (rest gray=unknown) after the walk")


if __name__ == "__main__":
    spec = MapASpec()
    obstacles = sample_obstacles(spec, 0)
    grid = obstacle_grid(spec, obstacles)
    out = Path(__file__).resolve().parent.parent / "data" / "map_A"
    out.mkdir(parents=True, exist_ok=True)
    render(spec, grid, go2_pos=(6.0, 0.0), go2_heading=0.0, drone_pos=(13.0, 4.0),
           out_png=out / "visibility.png")
    print(f"  rendered -> {out}/visibility.png")
    belief, bgrid, bobs, pts = simulate_belief(spec, 0)
    render_belief(spec, bgrid, bobs, pts, belief, out / "belief.png")
    print(f"  rendered -> {out}/belief.png")
