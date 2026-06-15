"""nav_go2_static.py — STEP 1b: one Go2 follows a random A* path through the dense STATIC map.

Spawns map_A obstacles as colliders, plans a random (non-optimal but collision-free) path from
start to goal with astar_nav, and drives the Go2 along the waypoints with the proven ①-a
controller. Per episode: reports reach / static-collision / fall. This is the STATIC navigation
baseline that step ② (dynamic obstacles) then breaks.

    conda activate isaacsim
    python scripts/nav_go2_static.py --episodes 10            # headless
    python scripts/nav_go2_static.py --episodes 5 --gui       # watch it weave
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 follows a random A* path through the static dense map")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--map_seed", type=int, default=0, help="map_A obstacle layout (fixed across episodes)")
parser.add_argument("--n_via", type=int, default=2, help="random via-points -> how wavy the path is")
parser.add_argument("--movers", type=int, default=0, help="moving obstacles (0 = static baseline, 3 = dynamic)")
parser.add_argument("--wp_tol", type=float, default=0.35, help="waypoint reach radius (m)")
parser.add_argument("--speed", type=float, default=0.8)
parser.add_argument("--turn_gain", type=float, default=1.5)
parser.add_argument("--ep_timeout_s", type=float, default=60.0)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--cameras", action="store_true",
                    help="attach drone BEV + Go2 D455 cameras and show both views in a cv2 window (demo)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = not args_cli.gui
args_cli.enable_cameras = args_cli.cameras          # only render cameras when asked (RL training stays fast)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ──────────────────────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

sys.path.insert(0, str(Path(__file__).resolve().parent))   # go2_mqe_low_level_policy is vendored here
from go2_mqe_low_level_policy import MQEGo2LowLevelPolicy
from map_A import MapASpec, sample_obstacles
import astar_nav

# WTW low-level policy + actuator net are bundled in the repo under assets/wtw/ (see README)
_REPO = Path(__file__).resolve().parent.parent
ACTUATOR_NET = str(_REPO / "assets" / "wtw" / "unitree_go1.pt")
WTW_DIR = str(_REPO / "assets" / "wtw")
ISAAC_TO_MQE = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
MQE_TO_ISAAC = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
DECIMATION = 4
PHYS_DT = 0.005
WTW_DT = PHYS_DT * DECIMATION
PLAN_RADIUS = 0.45            # A* obstacle dilation (clearance) — margin for corner-cutting
BODY_RADIUS = 0.22            # actual Go2 half-width, for realistic collision detection


def _go2_cfg(name, pos):
    cfg = UNITREE_GO2_CFG.copy()
    cfg.actuators = {
        "base_legs": ActuatorNetMLPCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            network_file=ACTUATOR_NET, pos_scale=-1.0, vel_scale=1.0, torque_scale=1.0,
            input_order="pos_vel", input_idx=[0, 1, 2],
            effort_limit=23.7, velocity_limit=30.0, saturation_effort=23.7),
    }
    cfg = cfg.replace(prim_path="{ENV_REGEX_NS}/" + name)
    cfg.init_state = cfg.init_state.replace(pos=pos)
    return cfg


@configclass
class NavSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)))
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(1.0, 1.0, 1.0)))
    robot: ArticulationCfg = _go2_cfg("Go2", (0.0, 0.0, 0.40))


def _yaw_from_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _drive(robot, ll, default_jp, i2m, m2i, cmd):
    jp_rel = (robot.data.joint_pos - default_jp)[:, i2m]
    jv = robot.data.joint_vel[:, i2m]
    low = torch.clamp(ll.step(robot.data.projected_gravity_b, jp_rel, jv, cmd), -10.0, 10.0)
    scaled = 0.25 * low
    scaled[:, [0, 3, 6, 9]] *= 0.5
    robot.set_joint_position_target(default_jp + scaled[:, m2i])


def _waypoint_cmd(robot, goal, speed, turn_gain):
    """holonomic steering toward `goal` (needs vy to translate; limit hard reverse). From ①-a."""
    pos = robot.data.root_pos_w[0, :2]
    yaw = _yaw_from_quat(robot.data.root_quat_w[0])
    to_goal = goal - pos
    dist = torch.norm(to_goal)
    d = to_goal / dist.clamp_min(1e-6)
    c, s = torch.cos(yaw), torch.sin(yaw)
    fwd = float(d[0] * c + d[1] * s)
    lat = float(-d[0] * s + d[1] * c)
    v = min(speed, float(dist))
    vx = float(np.clip(fwd * v, -0.2, speed))
    vy = float(np.clip(lat * v, -speed, speed))
    des_yaw = torch.atan2(to_goal[1], to_goal[0])
    err = torch.atan2(torch.sin(des_yaw - yaw), torch.cos(des_yaw - yaw))
    vyaw = float(torch.clamp(err * turn_gain, -1.0, 1.0))
    return torch.tensor([[vx, vy, vyaw]], device=robot.data.root_pos_w.device), float(dist)


def _spawn_obstacles(obstacles):
    """Static colliders for the dense map (so the Go2 physically cannot pass through them)."""
    for i, (ox, oy, sx, sy, sz) in enumerate(obstacles):
        c = sim_utils.CuboidCfg(
            size=(float(sx), float(sy), float(sz)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.34, 0.42), roughness=0.8))
        c.func(f"/World/mapA/obs_{i}", c, translation=(float(ox), float(oy), float(sz) / 2))


def _rand_free(grid, spec, rng, margin=1.2):
    for _ in range(500):
        x = float(rng.uniform(margin, spec.lane_len - margin))
        y = float(rng.uniform(-(spec.half_w - margin), spec.half_w - margin))
        if grid.free(*grid.cell(x, y)):
            return (x, y)
    return None


class Mover:
    """A kinematic moving obstacle that navigates itself via A* to random goals (avoids static
    obstacles + walls, turns corners). Reaches a goal -> picks a new one. Crosses the Go2's path."""

    def __init__(self, grid, spec, rng, speed):
        self.grid, self.spec, self.rng, self.speed = grid, spec, rng, speed
        self.pos = np.array(_rand_free(grid, spec, rng) or (spec.lane_len / 2, 0.0))
        self.path, self.wi = [self.pos], 0
        self._new_goal()

    def _new_goal(self):
        for _ in range(20):
            g = _rand_free(self.grid, self.spec, self.rng)
            if g is None or np.linalg.norm(np.array(g) - self.pos) < 2.5:
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


def main():
    spec = MapASpec()
    obstacles = sample_obstacles(spec, args_cli.map_seed)
    grid = astar_nav.Grid(spec, obstacles, robot_radius=PLAN_RADIUS)

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=PHYS_DT))
    scene = InteractiveScene(NavSceneCfg(num_envs=1, env_spacing=4.0))
    _spawn_obstacles(obstacles)

    # ── (optional) drone BEV + Go2 D455 cameras, for SEEING both views in Isaac (not the RL input)
    bev_cam = go2_cam = drone_op = None
    DRONE_ALT = 12.0
    if args_cli.cameras:
        import omni.usd
        from pxr import UsdGeom, Gf as _Gf
        globals()["_Gf"] = _Gf
        stg = omni.usd.get_context().get_stage()
        dparent = UsdGeom.Xform.Define(stg, "/World/Drone")           # drone marker we move (cam follows it)
        drone_op = dparent.AddTranslateOp(); drone_op.Set(_Gf.Vec3d(spec.lane_len / 2, 0.0, DRONE_ALT))
        body = sim_utils.CuboidCfg(size=(0.4, 0.4, 0.1),
                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)))
        body.func("/World/Drone/body", body, translation=(0.0, 0.0, 0.0))
        bev_cam = Camera(CameraCfg(                                   # downward window BEV (follows drone)
            prim_path="/World/Drone/bev_cam", update_period=0.0, height=480, width=480, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=30.0, horizontal_aperture=20.955, clipping_range=(0.1, 40.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.70711, 0.0, 0.70711, 0.0), convention="world")))
        go2_cam = Camera(CameraCfg(                                   # Go2 forward D455 (follows the base)
            prim_path="/World/envs/env_0/Go2/base/d455_cam", update_period=0.0, height=300, width=420,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.30, 0.0, 0.06), rot=(0.99619, 0.0, 0.08716, 0.0), convention="world")))

    sim.reset()
    robot = scene["robot"]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.write_data_to_sim()

    if args_cli.cameras:                                 # NATIVE Isaac viewports (fast, no cv2 copy)
        from omni.kit.viewport.utility import create_viewport_window
        import omni.ui as ui
        vp1 = create_viewport_window("DRONE BEV (aerial)", camera_path="/World/Drone/bev_cam", width=520, height=520)
        vp2 = create_viewport_window("Go2 D455 (local)", camera_path="/World/envs/env_0/Go2/base/d455_cam",
                                     width=520, height=520)
        try:                                             # dock side-by-side -> both views in ONE window
            vp2.dock_in(vp1, ui.DockPosition.RIGHT, 0.5)
        except Exception as e:
            print(f"[cam] auto-dock failed ({e}); drag the 'Go2 D455' tab next to 'DRONE BEV' to combine")
        print("[cam] opened Isaac viewports (DRONE BEV | Go2 D455, docked side-by-side)")

    dev = str(sim.device)
    ll = MQEGo2LowLevelPolicy(num_envs=1, device=dev, dt=WTW_DT, locomotion_policy_dir=WTW_DIR)
    ll.reset(torch.arange(1, device=dev))
    i2m = torch.tensor(ISAAC_TO_MQE, device=dev)
    m2i = torch.tensor(MQE_TO_ISAAC, device=dev)
    default_jp = robot.data.default_joint_pos.clone()
    obs_xy = torch.tensor([[o[0], o[1]] for o in obstacles], device=dev)
    obs_half = torch.tensor([[o[2] / 2, o[3] / 2] for o in obstacles], device=dev)

    # moving obstacles: red kinematic cylinders, each A*-navigates itself to random goals
    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    MOVER_R = 0.25
    _mover_ops = []                                  # one translate op per mover (reliable kinematic update)
    for i in range(args_cli.movers):
        parent = UsdGeom.Xform.Define(stage, f"/World/movers/m{i}")
        op = parent.AddTranslateOp()
        op.Set(Gf.Vec3d(spec.lane_len / 2, 0.0, 0.4))
        _mover_ops.append(op)
        cyl = sim_utils.CylinderCfg(radius=MOVER_R, height=0.8, axis="Z",
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)))
        cyl.func(f"/World/movers/m{i}/cyl", cyl, translation=(0.0, 0.0, 0.0))

    def _set_mover(i, pos):
        _mover_ops[i].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), 0.4))

    def _warmup(n=50):
        z = torch.zeros((1, 3), device=dev)
        for k in range(n * DECIMATION):
            if k % DECIMATION == 0:
                _drive(robot, ll, default_jp, i2m, m2i, z)
            scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())

    def _reset_to_start(start_xy):
        rs = robot.data.default_root_state.clone()
        rs[0, 0], rs[0, 1], rs[0, 2] = start_xy[0], start_xy[1], 0.40
        rs[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        rs[0, 7:] = 0.0
        robot.write_root_pose_to_sim(rs[:, :7]); robot.write_root_velocity_to_sim(rs[:, 7:])
        robot.write_joint_state_to_sim(default_jp, robot.data.default_joint_vel.clone())
        ll.reset(torch.arange(1, device=dev))
        _warmup(40)

    def _hit_obstacle():
        p = (robot.data.root_pos_w[0, :2]).unsqueeze(0)
        db = torch.abs(p - obs_xy)
        return bool(((db < (obs_half + BODY_RADIUS)).all(dim=1)).any())   # realistic body overlap


    sim.set_camera_view([spec.lane_len / 2, -spec.lane_width, spec.lane_width * 0.7],
                        [spec.lane_len / 2, 0.0, 0.3])
    timeout_ticks = int(args_cli.ep_timeout_s / WTW_DT)

    reached = collided = falls = mover_collided = 0
    for ep in range(args_cli.episodes):
        rng = np.random.default_rng(10_000 + ep)
        rng_m = np.random.default_rng(20_000 + ep)
        movers = [Mover(grid, spec, rng_m, float(rng_m.uniform(0.6, 1.2))) for _ in range(args_cli.movers)]
        start = goal = wps = None
        for _ in range(40):                                   # random start + goal (free, >=8 m apart) + path
            s, g = _rand_free(grid, spec, rng), _rand_free(grid, spec, rng)
            if s is None or g is None or math.dist(s, g) < 8.0:
                continue
            w = astar_nav.random_path_on_grid(grid, spec, s, g, args_cli.n_via, rng)
            if w is not None:
                start, goal, wps = s, g, w; break
        if wps is None:
            print(f"[ep {ep}] no start/goal/path"); continue
        wps_t = [torch.tensor(w, device=dev, dtype=torch.float32) for w in wps]
        _reset_to_start(start)
        print(f"[ep {ep+1}/{args_cli.episodes}] start=({start[0]:.1f},{start[1]:.1f}) "
              f"goal=({goal[0]:.1f},{goal[1]:.1f})  {len(wps)} waypoints")

        for mi, mv in enumerate(movers):
            _set_mover(mi, mv.pos)
        wi = 1
        ep_collided = ep_mover_hit = False
        tick = 0
        done = False
        step = 0
        while simulation_app.is_running() and not done:
            if step % DECIMATION == 0:
                carrot = wps_t[min(wi, len(wps_t) - 1)]
                cmd, dist = _waypoint_cmd(robot, carrot, args_cli.speed, args_cli.turn_gain)
                tick += 1
                for mi, mv in enumerate(movers):                 # advance the moving obstacles
                    mv.step(WTW_DT)
                    _set_mover(mi, mv.pos)
                if dist < args_cli.wp_tol:
                    wi += 1
                    if wi >= len(wps_t):                         # passed the final goal
                        reached += 1; done = True
                        print(f"        reached goal in {tick} ticks"); break
                if not ep_collided and _hit_obstacle():
                    ep_collided = True                           # count once, keep going
                if not ep_mover_hit and movers:                  # ★ Go2 vs a moving obstacle (A* ignores these)
                    gp = robot.data.root_pos_w[0, :2].detach().cpu().numpy()
                    if any(float(np.linalg.norm(gp - mv.pos)) < BODY_RADIUS + MOVER_R for mv in movers):
                        ep_mover_hit = True
                if robot.data.root_pos_w[0, 2] < 0.18:
                    falls += 1; done = True; print("        FELL"); break
                if tick >= timeout_ticks:
                    print(f"        timeout (wp {wi}/{len(wps_t)}, dist {dist:.2f})"); done = True; break
                if args_cli.cameras:                             # move the drone; viewports auto-render the cams
                    gp = robot.data.root_pos_w[0]
                    drone_op.Set(_Gf.Vec3d(float(gp[0]), float(gp[1]), DRONE_ALT))
                _drive(robot, ll, default_jp, i2m, m2i, cmd)
            scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
            step += 1
        if ep_collided:
            collided += 1; print("        (touched a static obstacle)")
        if ep_mover_hit:
            mover_collided += 1; print("        ★ HIT a moving obstacle (A* didn't avoid it)")

    n = args_cli.episodes
    mode = f"DYNAMIC ({args_cli.movers} movers)" if args_cli.movers else "STATIC"
    print("\n" + "=" * 56)
    print(f"  {mode} nav (map seed {args_cli.map_seed}):  {n} episodes")
    print(f"    reached goal     : {reached}/{n}")
    print(f"    static collisions: {collided}/{n}   (A* keeps clearance -> ~0)")
    if args_cli.movers:
        print(f"    ★ MOVER hits     : {mover_collided}/{n}   (A* can't avoid moving obstacles -> needs RL)")
    print(f"    falls            : {falls}/{n}")
    print("=" * 56)

    if args_cli.gui:
        zero = torch.zeros((1, 3), device=dev); g = 0
        while simulation_app.is_running():
            if g % DECIMATION == 0:
                _drive(robot, ll, default_jp, i2m, m2i, zero)
            scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt()); g += 1


if __name__ == "__main__":
    import os
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
    finally:
        os._exit(0)
