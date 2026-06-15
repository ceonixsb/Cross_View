"""map_A.py — Baseline map "A" for cross-view cooperative pushing.

A wide, *bounded* LANE (not a narrow corridor — the robots need room to surround
and push a ~1 m box) with a few obstacles that the team must avoid while pushing
the box from start (-x) to target (+x).

WHY RANDOM OBSTACLES (the whole point):
  A *fixed* obstacle layout is memorizable — a policy just learns "hug the bottom
  wall" and never needs to look ahead, so a drone adds nothing. Like MAPush, we
  therefore **resample obstacle positions every episode**. Now no fixed path works:
  the team must *perceive* the layout. That is exactly what makes the global view
  valuable:
    * local-only MARL (box occludes forward view) -> discovers obstacles late = baseline
    * MAPush oracle   (given global obstacle map)  -> avoids them              = upper bound
    * drone BEV cross-view (perceives the map)     -> our method

Simulator-agnostic: defines the 2D layout (metres) + a top-down render so we can
tune difficulty BEFORE committing to Isaac Gym (MAPush) or Isaac Sim. `spec_dict()`
+ `sample_obstacles()` export everything for whichever sim instantiates it.

Run:
    python scripts/map_A.py                  # 2x2 grid of random samples
    python scripts/map_A.py --n 5            # 5 obstacles per episode
Then:
    xdg-open data/map_A/map_A_samples.png
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class MapASpec:
    lane_len: float = 21.0          # x in [0, lane_len]; 1.5x the MAPush corridor (was 14)
    lane_width: float = 22.5        # y in [-w/2, +w/2]; 1.5x the MAPush width (was 15)
    wall_thick: float = 0.2

    start_x: float = 2.25           # box + robots spawn band (near the init room)
    target_x: float = 18.75        # object goal centre -> ~16.5 m push (scaled with the map)
    target_tol: float = 1.0        # success radius (box within 1 m of target)

    box_size: float = 1.5          # footprint marker (m); MAPush cuboid 1.5x1.0x0.5 / T-block ~1.5 span
    n_agents: int = 2

    n_obstacles: int = 28          # FIXED composition: 13 square + 15 elongated (positions/height random/episode)
    n_cubes: int = 13              # n_cubes obstacles have a SQUARE footprint; the rest are elongated (직육면체)
    cube_frac: float = 1.0                  # (legacy; sample_obstacles now uses n_cubes for a fixed mix)
    cube_side_range: tuple = (0.8, 1.0)     # square footprint side (m)
    cuboid_long_range: tuple = (1.3, 1.9)   # elongated: long axis — modest barrier (not too large)
    cuboid_short_range: tuple = (0.5, 0.7)  # short axis
    obstacle_height_range: tuple = (0.3, 2.5)  # ALL obstacles: WIDE random height -> clear BEV depth contrast
    n_large: int = 0                        # no enlarged blockers
    large_scale: float = 1.4
    obstacle_x_range: tuple = (3.0, 19.5)  # along the push axis; reaches near the back/front walls (corners)
    obstacle_y_band: float = 10.0  # obstacles spread out to +/-10 of the 22.5 m corridor -> fills the corners/edges
    wall_margin: float = 0.8       # min gap from obstacle centre to a wall
    min_spacing: float = 1.9       # min centre-to-centre distance (denser; still room for the 1.2 m box)
    # extra SMALL filler obstacles dropped into the empty gaps between the main ones
    n_small: int = 16
    small_side_range: tuple = (0.30, 0.55)   # small square footprint (m)
    small_height_range: tuple = (0.30, 1.30) # smaller random height too
    small_gap: float = 0.25        # min clearance (m) from any existing obstacle footprint
    # DENSE packing: pack as many obstacles as fit while keeping a box-width passage everywhere
    dense: bool = True             # if True, sample_obstacles ignores the binned counts and packs densely
    passage_gap: float = 1.50      # ~3% less dense than 1.46 (back to 1.5 baseline; still passable)
    pack_attempts: int = 7000      # rejection-sampling attempts
    max_obstacles: int = 110       # cap
    start_safe_x: float = 3.4      # keep x<this & |y|<start_safe_y clear (box+robots spawn)
    start_safe_y: float = 1.7
    goal_clear: float = 1.7        # keep this radius around the goal clear (box must reach it)
    # perimeter boundary WALL (fence) enclosing the whole arena
    boundary_wall: bool = True     # 4-piece fence just outside the lane (collider; drone/A* see it)
    wall_thick_int: float = 0.3
    wall_height_int: float = 0.9   # tall enough to block a ground robot's view (occlusion = cross-view motivation)

    @property
    def half_w(self) -> float:
        return self.lane_width / 2.0

    def spec_dict(self) -> dict:
        d = asdict(self)
        d["start"] = [self.start_x, 0.0]
        d["target"] = [self.target_x, 0.0]
        d["agents_xy"] = [[self.start_x - 0.7, +0.45], [self.start_x - 0.7, -0.45]][: self.n_agents]
        return d


def sample_obstacles(spec: MapASpec, seed: int) -> list[tuple]:
    """Resample obstacles (x, y, sx, sy, sz) for one episode.

    x is binned (one per bin + jitter) so blocks are spread along the whole push;
    y is uniform within the central band so the path is genuinely blocked yet varies
    per episode (no fixed path is safe -> perception needed). Each obstacle is a cube
    (정육면체) or an ELONGATED cuboid (직육면체, long+short axis, random orientation) with
    its own dimensions; `n_large` of them are scaled up into big barrier blockers.
    """
    rng = np.random.default_rng(seed)
    x_lo, x_hi = spec.obstacle_x_range
    y_lim = min(spec.obstacle_y_band, spec.half_w - spec.wall_margin)

    if spec.dense:
        # DENSE rejection-packing: keep a box-width passage (edge gap >= passage_gap) between
        # EVERY pair, so a 1.2 m box can always weave through (start->goal path always exists).
        gap = spec.passage_gap
        hw = spec.half_w
        obs: list[tuple] = []

        # boundary WALL enclosing the whole arena (a perimeter fence). 4 pieces just outside the
        # lane so the play area is the full lane; the fence is a collider + the drone/A* see it.
        if spec.boundary_wall:
            t, h, L, w = spec.wall_thick_int, spec.wall_height_int, spec.lane_len, spec.lane_width
            walls = [(L / 2, hw + t / 2, L + 2 * t, t, h),         # top
                     (L / 2, -hw - t / 2, L + 2 * t, t, h),        # bottom
                     (-t / 2, 0.0, t, w + 2 * t, h),               # left
                     (L + t / 2, 0.0, t, w + 2 * t, h)]            # right
        else:
            walls = []
        n_walls = len(walls)                                       # keep fence pieces out of inter-obstacle gap check
        obs.extend(walls)

        def _fits(x, y, sx, sy):
            for ox, oy, osx, osy, _ in obs[n_walls:]:              # gap only vs scattered obstacles, not the fence
                if abs(x - ox) < (sx + osx) / 2 + gap and abs(y - oy) < (sy + osy) / 2 + gap:
                    return False
            if x - sx / 2 < spec.start_safe_x and abs(y) < spec.start_safe_y:   # start (box+robots) clear
                return False
            if (x - spec.target_x) ** 2 + y ** 2 < spec.goal_clear ** 2:        # goal reachable
                return False
            return True

        for _ in range(spec.pack_attempts):
            if len(obs) >= spec.max_obstacles:
                break
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(-y_lim, y_lim))
            r = rng.random()
            if r < 0.45:                                  # small square
                s = float(rng.uniform(*spec.small_side_range)); sx = sy = s
            elif r < 0.78:                                # medium square
                s = float(rng.uniform(*spec.cube_side_range)); sx = sy = s
            else:                                         # elongated
                lo = float(rng.uniform(*spec.cuboid_long_range))
                sh = float(rng.uniform(*spec.cuboid_short_range))
                sx, sy = (lo, sh) if rng.random() < 0.5 else (sh, lo)
            if _fits(x, y, sx, sy):
                sz = float(rng.uniform(*spec.obstacle_height_range))
                obs.append((x, y, sx, sy, sz))
        return obs

    edges = np.linspace(x_lo, x_hi, spec.n_obstacles + 1)
    n_large = min(spec.n_large, spec.n_obstacles)
    large_idx = set(rng.choice(spec.n_obstacles, size=n_large, replace=False).tolist()) if n_large else set()
    # FIXED composition: exactly n_cubes cubes + the rest cuboids; which indices are cubes is random/episode
    n_cubes = min(spec.n_cubes, spec.n_obstacles)
    cube_idx = set(rng.choice(spec.n_obstacles, size=n_cubes, replace=False).tolist())

    obs: list[tuple] = []
    for i in range(spec.n_obstacles):
        x, y = float(edges[i]), 0.0
        for _ in range(80):  # rejection sample to honour min spacing (more tries for 15 obstacles)
            x = float(rng.uniform(edges[i] + 0.2, edges[i + 1] - 0.2))
            y = float(rng.uniform(-y_lim, y_lim))
            if all((x - ox) ** 2 + (y - oy) ** 2 >= spec.min_spacing ** 2 for ox, oy, *_ in obs):
                break
        scale = spec.large_scale if i in large_idx else 1.0
        sz = float(rng.uniform(*spec.obstacle_height_range)) * scale   # ALL obstacles: random height
        if i in cube_idx:                                 # square footprint (정육면체-계열)
            side = float(rng.uniform(*spec.cube_side_range)) * scale
            sx = sy = side
        else:                                             # elongated footprint (직육면체) — long + short axis
            lo = float(rng.uniform(*spec.cuboid_long_range)) * scale
            sh = float(rng.uniform(*spec.cuboid_short_range)) * scale
            sx, sy = (lo, sh) if rng.random() < 0.5 else (sh, lo)   # random orientation
        obs.append((x, y, sx, sy, sz))

    # ── SMALL filler obstacles: drop into the empty GAPS (reject if they touch an existing one) ──
    for _ in range(spec.n_small):
        for _ in range(200):
            x = float(rng.uniform(x_lo + 0.3, x_hi))
            y = float(rng.uniform(-y_lim, y_lim))
            side = float(rng.uniform(*spec.small_side_range))
            free = all(abs(x - ox) >= (side + osx) / 2 + spec.small_gap or
                       abs(y - oy) >= (side + osy) / 2 + spec.small_gap
                       for ox, oy, osx, osy, _ in obs)
            if free:
                sz = float(rng.uniform(*spec.small_height_range))
                obs.append((x, y, side, side, sz))
                break
        # (if no gap found in 200 tries, that one is skipped — the arena is locally full)
    return obs


# ── Top-down render ───────────────────────────────────────────────────────────
def _draw(ax, spec: MapASpec, obstacles, title: str) -> None:
    from matplotlib.patches import Rectangle, Circle
    L, W, hw = spec.lane_len, spec.lane_width, spec.half_w

    ax.add_patch(Rectangle((0, -hw), L, W, facecolor="#eef2f7", edgecolor="none", zorder=0))
    for y0 in (-hw - spec.wall_thick, hw):
        ax.add_patch(Rectangle((0, y0), L, spec.wall_thick, facecolor="#334155", zorder=3))

    for i, (ox, oy, sx, sy, sz) in enumerate(obstacles):
        is_cube = abs(sx - sy) < 1e-6              # square footprint (height is now random)
        fc = "#64748b" if is_cube else "#3f4b5b"   # square lighter, elongated darker
        ax.add_patch(Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, facecolor=fc,
                               edgecolor="#1e293b", lw=1.2, zorder=4))
        ax.text(ox, oy, str(i), color="white", ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=5)

    b = spec.box_size
    ax.add_patch(Rectangle((spec.start_x - b / 2, -b / 2), b, b, facecolor="#f59e0b",
                           edgecolor="#92400e", lw=2, zorder=4))
    for (axx, ayy) in spec.spec_dict()["agents_xy"]:
        ax.add_patch(Circle((axx, ayy), 0.18, facecolor="#2563eb", edgecolor="white", lw=1, zorder=5))

    ax.add_patch(Circle((spec.target_x, 0), spec.target_tol, facecolor="#86efac",
                        edgecolor="#16a34a", lw=2, alpha=0.5, zorder=2))
    ax.plot(spec.target_x, 0, marker="*", color="#16a34a", markersize=16, zorder=5)

    ax.set_xlim(-1, L + 1)
    ax.set_ylim(-hw - 1.0, hw + 1.0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)


def render_samples(spec: MapASpec, out_png: Path, seeds=(0, 1, 2, 3)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, seed in zip(axes.flat, seeds):
        _draw(ax, spec, sample_obstacles(spec, seed), title=f"episode seed {seed}")
    fig.suptitle(f"Map A — wide lane {spec.lane_len:.0f}×{spec.lane_width:.0f} m · "
                 f"{spec.n_obstacles} RANDOM obstacles/episode · {spec.n_agents}x Go2 push box → target",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, "orange=box  blue=Go2  gray=obstacle(resampled each episode)  "
             "green★=target(1 m tol).  Random layout ⇒ no fixed path works ⇒ must perceive ⇒ drone BEV helps.",
             ha="center", fontsize=9, color="#475569")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Render baseline map A (random obstacles, top-down)")
    p.add_argument("--out", type=str, default="data/map_A")
    p.add_argument("--n", type=int, default=None, help="obstacles per episode (override)")
    args = p.parse_args()

    spec = MapASpec()
    if args.n is not None:
        spec.n_obstacles = args.n
    out_dir = Path(args.out)
    png = out_dir / "map_A_samples.png"
    render_samples(spec, png)
    (out_dir / "map_A_spec.json").write_text(json.dumps(spec.spec_dict(), indent=2))

    print(f"[map_A] lane {spec.lane_len:.0f}×{spec.lane_width:.0f} m, "
          f"{spec.n_obstacles} random obstacles/episode, {spec.n_agents}x Go2")
    print(f"[map_A] start=({spec.start_x},0) target=({spec.target_x},0) tol={spec.target_tol}")
    print(f"[map_A] render -> {png}")
    print(f"[map_A] spec   -> {out_dir / 'map_A_spec.json'}")


if __name__ == "__main__":
    main()
