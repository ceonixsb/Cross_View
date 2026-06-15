"""scene_map_A.py — Map A (wide lane + random obstacles) as an Isaac Sim scene.

Builds the *static arena* for the cross-view cooperative-pushing baseline, straight
from `map_A.MapASpec` (single source of truth for the 2D layout):
  * ground plane
  * bounding walls (lane edges)              -> keeps the box/robots in the lane
  * N random 1 m static obstacle blocks      -> resampled per --seed (memorization-proof)
  * target marker (visual disk, no collision)

This is ONLY the arena. Next layers (separate files): T-shape box + 2x Go2 +
MARL high-level + the 7 metrics.  Obstacles are STATIC (immovable pillars to avoid).

Conventions copied from scene_sections.py (proven in this repo):
  * no isaaclab/pxr imports at module top — they live after AppLauncher boots.
  * standalone __main__ boots AppLauncher itself.

Usage:
    conda activate isaacsim
    python scripts/scene_map_A.py --seed 0                 # GUI, look at the lane
    python scripts/scene_map_A.py --seed 2 --steps 0       # GUI until closed
    python scripts/scene_map_A.py --headless --steps 60    # smoke test (no window)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Map A arena in Isaac Sim")
parser.add_argument("--seed", type=int, default=0, help="obstacle layout seed (per-episode resample)")
parser.add_argument("--steps", type=int, default=0, help="0 = run until window closed")
parser.add_argument("--n-obstacles", type=int, default=None, help="override obstacle count")
parser.add_argument("--warehouse", action="store_true", help="use warehouse props instead of varied boxes")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for map_A
import math

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from map_A import MapASpec, sample_obstacles

# Warehouse clutter palette — probed as EXISTING in Isaac 5.1 Simple_Warehouse/Props.
# (The 4.2 tutorial's "WareHousePile_A04" is NOT at this path in 5.1; SM_RackPile_03/04
#  are the actual warehouse pile assets.) Mixed for non-monotonous obstacles.
_WH_BASE = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props"
WAREHOUSE_PROPS = [
    "SM_RackPile_03", "SM_RackPile_04",      # warehouse piles (tall, blocking)
    "SM_CratePlastic_A_01", "SM_BarelPlastic_A_01", "S_TrafficCone",
]


# ── Colors ───────────────────────────────────────────────────────────────────
C_GROUND = (0.92, 0.94, 0.97)
C_WALL = (0.20, 0.25, 0.33)
C_OBST = (0.28, 0.33, 0.41)
C_TARGET = (0.30, 0.85, 0.45)


def _spawn_box(path: str, size, pos, color, *, collision: bool):
    """Spawn one static cuboid (collision optional) with a flat preview material."""
    cfg = sim_utils.CuboidCfg(
        size=tuple(size),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0, roughness=0.8),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
    )
    cfg.func(path, cfg, translation=tuple(pos))


def _yaw_quat(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))   # (w,x,y,z) about +z


def _spawn_warehouse(path: str, asset: str, pos, yaw: float):
    """Spawn a warehouse prop USD as a STATIC obstacle (collision on, no rigid body)."""
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{_WH_BASE}/{asset}.usd",
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    cfg.func(path, cfg, translation=tuple(pos), orientation=_yaw_quat(yaw))


def build_map_A(spec: MapASpec, seed: int, use_warehouse: bool = True) -> list:
    """Spawn ground + walls + random obstacles + target. Returns obstacle list."""
    L, hw = spec.lane_len, spec.half_w
    wall_h, wall_t = 0.6, spec.wall_thick

    # ground plane (collision)
    sim_utils.GroundPlaneCfg(color=C_GROUND, size=(200.0, 200.0)).func("/World/ground",
        sim_utils.GroundPlaneCfg(color=C_GROUND, size=(200.0, 200.0)))

    # lights
    sim_utils.DomeLightCfg(intensity=2200.0, color=(1.0, 1.0, 1.0)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=2200.0, color=(1.0, 1.0, 1.0)))
    sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.88), angle=0.5).func(
        "/World/SunLight", sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.88), angle=0.5))

    # bounding walls: top/bottom (along x) + back/front (along y), enclosing the lane
    _spawn_box("/World/walls/top",    (L, wall_t, wall_h), (L / 2,  hw + wall_t / 2, wall_h / 2), C_WALL, collision=True)
    _spawn_box("/World/walls/bottom", (L, wall_t, wall_h), (L / 2, -hw - wall_t / 2, wall_h / 2), C_WALL, collision=True)
    _spawn_box("/World/walls/back",   (wall_t, 2 * hw, wall_h), (-wall_t / 2, 0, wall_h / 2), C_WALL, collision=True)
    _spawn_box("/World/walls/front",  (wall_t, 2 * hw, wall_h), (L + wall_t / 2, 0, wall_h / 2), C_WALL, collision=True)

    # obstacles, resampled for this seed — warehouse props (varied) or plain boxes
    obstacles = sample_obstacles(spec, seed)
    if use_warehouse:
        rng = np.random.default_rng(seed + 1000)   # separate stream for asset/yaw
        for i, (ox, oy, *_) in enumerate(obstacles):
            asset = WAREHOUSE_PROPS[int(rng.integers(len(WAREHOUSE_PROPS)))]
            yaw = float(rng.uniform(0, 2 * math.pi))
            _spawn_warehouse(f"/World/obstacles/obs_{i}", asset, (ox, oy, 0.0), yaw)
            print(f"    obs_{i}: {asset:22s} @ ({ox:5.2f},{oy:+5.2f})")
    else:
        crng = np.random.default_rng(seed + 2000)  # box colour variety
        shades = [(0.30, 0.34, 0.42), (0.38, 0.42, 0.50), (0.46, 0.40, 0.36), (0.34, 0.40, 0.46)]
        for i, (ox, oy, sx, sy, sz) in enumerate(obstacles):
            col = shades[int(crng.integers(len(shades)))]
            _spawn_box(f"/World/obstacles/obs_{i}", (sx, sy, sz), (ox, oy, sz / 2), col, collision=True)
            kind = "cube  " if abs(sx - sy) < 1e-6 and abs(sx - sz) < 1e-6 else "cuboid"
            print(f"    obs_{i}: {kind} {sx:.2f}x{sy:.2f}x{sz:.2f} @ ({ox:5.2f},{oy:+5.2f})")

    # target marker — flat green disk, VISUAL ONLY (no collision)
    tcfg = sim_utils.CylinderCfg(
        radius=spec.target_tol, height=0.02, axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=C_TARGET, roughness=0.4),
    )
    tcfg.func("/World/target", tcfg, translation=(spec.target_x, 0.0, 0.011))

    return obstacles


def main() -> None:
    spec = MapASpec()
    if args_cli.n_obstacles is not None:
        spec.n_obstacles = args_cli.n_obstacles

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=0.005))

    mode = "warehouse props" if args_cli.warehouse else "varied boxes (cube+cuboid)"
    print(f"[map_A/sim] lane {spec.lane_len:.0f}x{spec.lane_width:.0f}m, target x={spec.target_x}, obstacles = {mode}")
    obstacles = build_map_A(spec, args_cli.seed, use_warehouse=args_cli.warehouse)
    print(f"[map_A/sim] {len(obstacles)} obstacles (seed {args_cli.seed})")

    sim.reset()
    # oblique view of the whole lane
    sim.set_camera_view([spec.lane_len / 2, -spec.lane_width * 1.6, 9.0],
                        [spec.lane_len / 2, 0.0, 0.0])
    print("[map_A/sim] built. (press the window's stop / close to exit)")

    step = 0
    max_steps = args_cli.steps if args_cli.steps > 0 else int(1e9)
    while simulation_app.is_running() and step < max_steps:
        sim.step()
        step += 1
    print(f"[map_A/sim] done at step {step}.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
