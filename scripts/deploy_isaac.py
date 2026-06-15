"""deploy_isaac.py — run the trained Stage-1 Go2 policy in Isaac Sim (sim-to-sim).

The policy was trained in the fast kinematic MarlNavEnv. Here the SAME geometric obs is built each
high-level tick (0.1 s) from the Go2's ACTUAL Isaac pose, fed to the loaded PPO policy -> velocity
command -> WTW low-level -> real walking. MarlNavEnv stays the "brain" (map, A* path, kinematic
movers, scripted drone, obs); Isaac provides the Go2 physics + visualization. This shows whether
the velocity policy transfers onto the real locomotion controller.

    conda activate isaacsim
    python scripts/deploy_isaac.py --gui --episodes 5                 # watch it
    python scripts/deploy_isaac.py --gui --cameras --episodes 5       # + drone BEV / Go2 D455 views
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy the trained Go2 avoidance policy in Isaac Sim")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--map_seed", type=int, default=100, help="held-out map layout (not seen in training)")
parser.add_argument("--movers", type=int, default=6)
parser.add_argument("--drone_lead", type=float, default=6.0)
parser.add_argument("--tag", default="tuned", help="checkpoint logs/go2_ppo_<tag>/")
parser.add_argument("--ep_timeout_s", type=float, default=60.0)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--cameras", action="store_true", help="drone BEV + Go2 D455 views in Isaac")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = not args_cli.gui
args_cli.enable_cameras = args_cli.cameras
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
from marl_nav_env import MarlNavEnv
from go2_avoid_env import MAX_SPD, MAX_VY, MAX_YAW, BODY_R, MOVER_R

import pickle
from stable_baselines3 import PPO

# WTW low-level policy + actuator net are bundled in the repo under assets/wtw/ (see README)
_REPO = Path(__file__).resolve().parent.parent
ACTUATOR_NET = str(_REPO / "assets" / "wtw" / "unitree_go1.pt")
WTW_DIR = str(_REPO / "assets" / "wtw")
ISAAC_TO_MQE = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
MQE_TO_ISAAC = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
DECIMATION = 4
PHYS_DT = 0.005
WTW_DT = PHYS_DT * DECIMATION
HL_EVERY = 5                      # WTW ticks per high-level RL decision (5 * 0.02 = 0.1 s = env.dt)
DRONE_ALT = 12.0


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


def _spawn_obstacles(obstacles):
    for i, (ox, oy, sx, sy, sz) in enumerate(obstacles):
        c = sim_utils.CuboidCfg(
            size=(float(sx), float(sy), float(sz)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.34, 0.42), roughness=0.8))
        c.func(f"/World/mapA/obs_{i}", c, translation=(float(ox), float(oy), float(sz) / 2))


def main():
    # ── load policy + normalization stats ────────────────────────────────────
    out = Path(__file__).resolve().parent.parent / "logs" / ("go2_ppo_" + args_cli.tag)
    model = PPO.load(out / "go2_ppo.zip", device="cpu")
    with open(out / "vecnorm.pkl", "rb") as f:
        vn = pickle.load(f)
    o_mean, o_var = vn.obs_rms.mean.astype(np.float32), vn.obs_rms.var.astype(np.float32)
    o_clip, o_eps = float(vn.clip_obs), float(vn.epsilon)

    def norm_obs(o):
        return np.clip((o - o_mean) / np.sqrt(o_var + o_eps), -o_clip, o_clip).astype(np.float32)

    # ── the kinematic env is the BRAIN (map, path, movers, drone, obs) ───────
    env = MarlNavEnv(map_seed=args_cli.map_seed, n_movers=args_cli.movers,
                     drone_mode="scripted", drone_lead=args_cli.drone_lead, randomize_map=False)

    # ── Isaac scene ──────────────────────────────────────────────────────────
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=PHYS_DT))
    scene = InteractiveScene(NavSceneCfg(num_envs=1, env_spacing=4.0))
    _spawn_obstacles(env.obstacles)

    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()

    drone_op = None
    if args_cli.cameras:
        dparent = UsdGeom.Xform.Define(stage, "/World/Drone")
        drone_op = dparent.AddTranslateOp(); drone_op.Set(Gf.Vec3d(env.spec.lane_len / 2, 0.0, DRONE_ALT))
        body = sim_utils.CuboidCfg(size=(0.4, 0.4, 0.1),
                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)))
        body.func("/World/Drone/body", body, translation=(0.0, 0.0, 0.0))
        Camera(CameraCfg(prim_path="/World/Drone/bev_cam", update_period=0.0, height=480, width=480, data_types=["rgb"],
                         spawn=sim_utils.PinholeCameraCfg(focal_length=30.0, horizontal_aperture=20.955, clipping_range=(0.1, 40.0)),
                         offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.70711, 0.0, 0.70711, 0.0), convention="world")))
        Camera(CameraCfg(prim_path="/World/envs/env_0/Go2/base/d455_cam", update_period=0.0, height=300, width=420, data_types=["rgb"],
                         spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
                         offset=CameraCfg.OffsetCfg(pos=(0.30, 0.0, 0.06), rot=(0.99619, 0.0, 0.08716, 0.0), convention="world")))

    sim.reset()
    robot = scene["robot"]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.write_data_to_sim()

    if args_cli.cameras:
        from omni.kit.viewport.utility import create_viewport_window
        import omni.ui as ui
        vp1 = create_viewport_window("DRONE BEV (aerial)", camera_path="/World/Drone/bev_cam", width=520, height=520)
        vp2 = create_viewport_window("Go2 D455 (local)", camera_path="/World/envs/env_0/Go2/base/d455_cam", width=520, height=520)
        try:
            vp2.dock_in(vp1, ui.DockPosition.RIGHT, 0.5)
        except Exception as e:
            print(f"[cam] auto-dock failed ({e})")

    dev = str(sim.device)
    ll = MQEGo2LowLevelPolicy(num_envs=1, device=dev, dt=WTW_DT, locomotion_policy_dir=WTW_DIR)
    ll.reset(torch.arange(1, device=dev))
    i2m = torch.tensor(ISAAC_TO_MQE, device=dev)
    m2i = torch.tensor(MQE_TO_ISAAC, device=dev)
    default_jp = robot.data.default_joint_pos.clone()

    # goal marker (green) + mover markers (red), moved per episode/tick via translate ops
    gp = UsdGeom.Xform.Define(stage, "/World/goal"); goal_op = gp.AddTranslateOp(); goal_op.Set(Gf.Vec3d(0, 0, 0.05))
    gs = sim_utils.CylinderCfg(radius=0.4, height=0.1, axis="Z",
                               visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)))
    gs.func("/World/goal/cyl", gs, translation=(0.0, 0.0, 0.0))
    mover_ops = []
    for i in range(args_cli.movers):
        mp = UsdGeom.Xform.Define(stage, f"/World/movers/m{i}"); op = mp.AddTranslateOp(); op.Set(Gf.Vec3d(0, 0, 0.4))
        mover_ops.append(op)
        cyl = sim_utils.CylinderCfg(radius=MOVER_R, height=0.8, axis="Z",
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)))
        cyl.func(f"/World/movers/m{i}/cyl", cyl, translation=(0.0, 0.0, 0.0))

    def _warmup(n=40):
        z = torch.zeros((1, 3), device=dev)
        for k in range(n * DECIMATION):
            if k % DECIMATION == 0:
                _drive(robot, ll, default_jp, i2m, m2i, z)
            scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())

    def _reset_to(xy, yaw):
        rs = robot.data.default_root_state.clone()
        rs[0, 0], rs[0, 1], rs[0, 2] = float(xy[0]), float(xy[1]), 0.40
        rs[0, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=dev)
        rs[0, 7:] = 0.0
        robot.write_root_pose_to_sim(rs[:, :7]); robot.write_root_velocity_to_sim(rs[:, 7:])
        robot.write_joint_state_to_sim(default_jp, robot.data.default_joint_vel.clone())
        ll.reset(torch.arange(1, device=dev))
        _warmup(40)

    obs_xy = torch.tensor([[o[0], o[1]] for o in env.obstacles], device=dev)
    obs_half = torch.tensor([[o[2] / 2, o[3] / 2] for o in env.obstacles], device=dev)

    def _hit_static():
        p = robot.data.root_pos_w[0, :2].unsqueeze(0)
        return bool((((torch.abs(p - obs_xy)) < (obs_half + BODY_RADIUS)).all(dim=1)).any())

    sim.set_camera_view([env.spec.lane_len / 2, -env.spec.lane_width, env.spec.lane_width * 0.7],
                        [env.spec.lane_len / 2, 0.0, 0.3])
    timeout_ticks = int(args_cli.ep_timeout_s / WTW_DT)
    BODY_RADIUS = BODY_R

    reached = static_hit = mover_hit = falls = 0
    for ep in range(args_cli.episodes):
        env.reset()                                                   # new start/goal/movers/path (same map)
        goal = env._pt(env.path_len)
        goal_op.Set(Gf.Vec3d(float(goal[0]), float(goal[1]), 0.05))
        _reset_to(env.go2, env.yaw)
        env.vel = np.zeros(2)
        print(f"[ep {ep+1}/{args_cli.episodes}] start=({env.go2[0]:.1f},{env.go2[1]:.1f}) "
              f"goal=({goal[0]:.1f},{goal[1]:.1f}) path={env.path_len:.1f}m")

        cmd = torch.zeros((1, 3), device=dev)
        wtw_tick = 0
        done = ep_static = ep_mover = False
        outcome = "timeout"
        step = 0
        while simulation_app.is_running() and not done:
            if step % DECIMATION == 0:                                # WTW tick (50 Hz)
                if wtw_tick % HL_EVERY == 0:                          # high-level RL decision (10 Hz)
                    gxy = robot.data.root_pos_w[0, :2].detach().cpu().numpy()
                    yaw = float(_yaw_from_quat(robot.data.root_quat_w[0]))
                    env.go2 = gxy.astype(float); env.yaw = yaw
                    env.s_go2 = env._s_of(env.go2)
                    env.drone = env._pt(env.s_go2 + env.drone_lead)
                    env._sense()
                    obs = env._obs()["go2"]
                    a = model.predict(norm_obs(obs), deterministic=True)[0]
                    vx, vy, vyaw = float(a[0]) * MAX_SPD, float(a[1]) * MAX_VY, float(a[2]) * MAX_YAW
                    env.vel = np.array([vx, vy])
                    cmd = torch.tensor([[vx, vy, vyaw]], device=dev)
                    for mi, mv in enumerate(env.movers):              # advance kinematic movers by env.dt
                        mv.step(env.dt)
                        mover_ops[mi].Set(Gf.Vec3d(float(mv.pos[0]), float(mv.pos[1]), 0.4))
                    if drone_op is not None:
                        drone_op.Set(Gf.Vec3d(float(env.drone[0]), float(env.drone[1]), DRONE_ALT))
                    # outcomes
                    if float(np.linalg.norm(gxy - goal)) < 1.0:
                        reached += 1; outcome = "REACH"; done = True
                    elif _hit_static():
                        ep_static = True; outcome = "static-hit"; done = True
                    elif any(float(np.linalg.norm(gxy - mv.pos)) < BODY_RADIUS + MOVER_R for mv in env.movers):
                        ep_mover = True; outcome = "MOVER-hit"; done = True
                    elif robot.data.root_pos_w[0, 2] < 0.18:
                        falls += 1; outcome = "FELL"; done = True
                    elif wtw_tick >= timeout_ticks:
                        outcome = "timeout"; done = True
                wtw_tick += 1
                if not done:
                    _drive(robot, ll, default_jp, i2m, m2i, cmd)
            scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
            step += 1
        if ep_static:
            static_hit += 1
        if ep_mover:
            mover_hit += 1
        print(f"        -> {outcome}")

    n = args_cli.episodes
    print("\n" + "=" * 56)
    print(f"  Isaac deploy (held-out map {args_cli.map_seed}, {args_cli.movers} movers): {n} eps")
    print(f"    reached    : {reached}/{n}")
    print(f"    MOVER hits : {mover_hit}/{n}")
    print(f"    static hits: {static_hit}/{n}")
    print(f"    falls      : {falls}/{n}")
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
