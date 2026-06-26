"""deploy_yroad.py — Go2 (unitree_rl_lab velocity policy) walking on the Y-Road ㄷ map.

MILESTONE 1: spawn the Go2 on the elevated plateau at the start, drive it with the rl_lab
locomotion policy and a FIXED velocity command, to verify the policy / obs / gains / joint
order work on our map. Later milestones replace the command with planner guidance from the
drone-view CNN cost map.

rl_lab velocity policy (exported jit): obs 45-D, action 12-D joint targets.
  obs = [ ang_vel_b*0.2 (3) | proj_grav_b (3) | vel_cmd (3) |
          joint_pos_rel (12) | joint_vel_rel*0.05 (12) | last_action (12) ]
  joint_target = default_joint_pos + 0.35 * action      (PD: kp 25, kd 0.5, 50 Hz)

    conda activate isaacsim
    python deploy_yroad.py --gui --steps 1500 --vx 0.6
    python deploy_yroad.py --headless --steps 400        # smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l_map
import planner as planner_mod

parser = argparse.ArgumentParser(description="Go2 rl_lab policy on the Y-Road map")
l_map.add_map_args(parser)
# DEPLOY-ONLY smaller map: section length 4.0 -> 2.8 m shrinks the whole walk ~30% (path length ∝ sec_len),
# structure (forks/layout) unchanged. NOTE: the U-Net was trained on the 4.0 m map, so for --cnn pass
# `--sec_len 4.0` to match (or retrain). The GT-planner demo rasterises geometry directly -> no retrain needed.
parser.set_defaults(sec_len=2.8)
parser.set_defaults(n_obs=0)                 # low step obstacles OFF for now (clean walk). Bring back with --n_obs 16 when training the avoidance RL.
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--vx", type=float, default=1.0, help="max forward speed for planner guidance (M2) / fixed cmd (M1)")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--vyaw", type=float, default=0.0)
parser.add_argument("--no_guide", action="store_true",
                    help="disable GT-costmap path guidance; drive the fixed --vx/--vy/--vyaw instead (M1)")
parser.add_argument("--dino_loc", action="store_true",
                    help="use DINOv3 cross-view localization (Go2 D455 -> estimated world pos) instead of GT pos")
parser.add_argument("--dino_deploy", type=str,
                    default="data/xview_dense/deploy",
                    help="path to exported DINOv3 assets (front_encoder.pt, bev_db.pt, bev_poses.npy)")
parser.add_argument("--avoid", action="store_true",
                    help="rule-based local obstacle avoidance: D455 depth -> steer around obstacles")
parser.add_argument("--n_avoid_obs", type=int, default=4, help="# obstacles to place along the first arm")
parser.add_argument("--avoid_thresh", type=float, default=1.6, help="depth (m) below which to steer away")
parser.add_argument("--show", action="store_true",
                    help="live OpenCV window of the Go2 D455 (front) + whole-map BEV feeds")
parser.add_argument("--docked", action="store_true",
                    help="show the camera feeds as in-Isaac docked viewports (like before) instead of cv2 windows. "
                         "Heavier (re-renders the scene per viewport) but now affordable thanks to the 50 Hz render decouple.")
parser.add_argument("--drone_ctrl", action="store_true",
                    help="fly the drone with WASD (move) + PgUp/PgDn (altitude); its downward cam is the BEV viewport")
parser.add_argument("--cnn", action="store_true",
                    help="drive from the U-Net prediction of the 91m BEV instead of the GT cost map (M3)")
parser.add_argument("--unet", type=str, default="data/BEV/best_jit.pt",
                    help="TorchScript U-Net used by --cnn (relative to Section-Aware). RGB+D 4ch model.")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--cameras", action="store_true", help="activate D455 (Go2) + BEV (drone) camera sensors")
parser.add_argument("--drone_alt", type=float, default=2.5, help="visible drone hover height above plateau (m)")
parser.add_argument("--drone_scale", type=float, default=8.0, help="enlarge the tiny Crazyflie so it is visible")
parser.add_argument("--drone_spin", type=float, default=0.0, help="propeller spin speed (deg/step); 0 = still (default: off — pivot is the body origin, so spinning makes the 4 props orbit instead of spin in place)")
parser.add_argument("--drone_orbit", type=float, default=0.0, help="gentle hover-orbit radius (m); 0 = stationary (default)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = not args_cli.gui
if args_cli.docked:
    args_cli.show = True                                         # --docked is a display variant of --show
args_cli.enable_cameras = (args_cli.cameras or args_cli.cnn or args_cli.show
                           or args_cli.dino_loc or args_cli.avoid or args_cli.drone_ctrl)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch ──
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

D455_USD = f"{ISAAC_NUCLEUS_DIR}/Sensors/Intel/RealSense/rsd455.usd"
CRAZYFLIE_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd"

RL_LAB = Path("/home/hdc/Desktop/Unitree/YSB_Labtask/My_go2_locomotion/Walking/unitree_rl_lab")
POLICY = RL_LAB / "logs/rsl_rl/unitree_go2_velocity/2026-03-25_22-11-57/exported/policy.pt"
# Cross_View's composed Go2 USD — base + physics + SENSOR(D455) already attached
GO2_USD = Path("/home/hdc/Desktop/Unitree/YSB_Labtask/Cross_View/assets/Go2/usd/go2.usd")
DECIMATION = 4
ACT_SCALE = 0.35


def go2_cfg(pos):
    cfg = UNITREE_GO2_CFG.copy()
    cfg = cfg.replace(prim_path="{ENV_REGEX_NS}/Go2")
    if GO2_USD.exists():
        cfg.spawn = cfg.spawn.replace(usd_path=str(GO2_USD))     # match training USD (joint order)
    cfg.actuators = {                                            # match deploy.yaml PD gains
        "legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=25.0, damping=0.5,
                                    effort_limit_sim=23.5),
    }
    # EXACT rl_lab training default pose (rear thigh 1.0, calf -1.8) — must match or obs/stance is off
    cfg.init_state = cfg.init_state.replace(
        pos=pos,
        joint_pos={".*R_hip_joint": -0.1, ".*L_hip_joint": 0.1,
                   "F[LR]_thigh_joint": 0.8, "R[LR]_thigh_joint": 1.0,
                   ".*_calf_joint": -1.8},
    )
    return cfg


@configclass
class YRoadSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = go2_cfg((0.0, 0.0, 0.5))
    drone = AssetBaseCfg(                                        # Crazyflie BEV scout (kinematic — we move it)
        prim_path="/World/Drone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CRAZYFLIE_USD, scale=(8.0, 8.0, 8.0),        # cf2x is ~9 cm -> enlarge so it is visible
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)))


def main():
    a = args_cli
    if a.show or a.dino_loc:
        import numpy as np
        from PIL import Image
    if a.show:
        import cv2 as _cv2
    sim = SimulationContext(sim_utils.SimulationCfg(device=a.device, dt=0.005))

    info = l_map.build_L(a)                                      # terrain into /World (plateau + floor + lights)
    sx, sy = info["start"]; H = info["H"]
    gx, gy = info["goal"][0], info["goal"][1]
    print(f"[yroad] map built. start=({sx:.1f},{sy:.1f}) goal=({gx:.1f},{gy:.1f}) plateau H={H:.1f}")


    # ── cost-map planner: GT (rasterised known map) now, or the U-Net prediction (--cnn, built post-reset) ──
    planner = None
    if not a.no_guide and not a.cnn:
        planner = planner_mod.GTPlanner(info["elements"], info["bbox"], (gx, gy), polys=info.get("polys"))
    if planner is not None:
        reach = planner.reachable(sx, sy)
        sr, sc = planner.world_to_cell(sx, sy)
        print(f"[yroad] GT planner: grid {planner.Hp}x{planner.Wp} cell={planner.cell}m  "
              f"start->goal reachable={reach}  cost={float(planner.dist[sr, sc]):.1f}")
        try:                                                     # save the cost map + planned path to eyeball
            from PIL import Image
            out = Path(__file__).resolve().parent / "data" / "shots"; out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(planner.render((sx, sy), (gx, gy))).save(out / "deploy_costmap.png")
            print(f"[yroad] cost map + path -> {out}/deploy_costmap.png")
        except Exception as e:                                   # noqa: BLE001
            print(f"[yroad] cost-map render skipped: {e}")

    import math
    cx, cy = info["center"]; msize = info["size"]
    bev_alt = msize * 1.15                                        # whole-map overhead BEV (matches CNN training)
    bev_focal = 20.955 / (2.0 * math.tan(math.atan(msize * 0.58 / bev_alt)))
    scene_cfg = YRoadSceneCfg(num_envs=1, env_spacing=8.0)
    scene_cfg.robot.init_state.pos = (float(sx), float(sy), float(H + 0.45))
    # visible enlarged Crazyflie hovering just above & ahead of the Go2 (an actual prop in third-person)
    scene_cfg.drone.spawn.scale = (a.drone_scale, a.drone_scale, a.drone_scale)
    drone_x, drone_y, drone_z = float(sx + 1.2), float(sy), float(H + a.drone_alt)
    scene_cfg.drone.init_state.pos = (drone_x, drone_y, drone_z)
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]

    # mount the VISIBLE Intel RealSense D455 model on the Go2 head front-top, visual only.
    # go2.usd defines a depth/rgb sensor but ships no visible camera body -> add the model here.
    # transform copied from the proven Cross_View attach_d455_rig: base/<rig> @ (0.32, 0, 0.08), identity.
    D455_MOUNT = (0.32, 0.0, 0.08)
    import omni.usd
    from pxr import Usd, UsdPhysics
    d455_cfg = sim_utils.UsdFileCfg(usd_path=D455_USD)
    d455_cfg.func("/World/envs/env_0/Go2/base/d455", d455_cfg, translation=D455_MOUNT)
    _stage = omni.usd.get_context().get_stage()
    for _p in Usd.PrimRange(_stage.GetPrimAtPath("/World/envs/env_0/Go2/base/d455")):
        for _api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.ArticulationRootAPI):
            if _p.HasAPI(_api):
                _p.RemoveAPI(_api)                                # strip physics -> silence nested-rigid-body error
    if a.drone_ctrl:                                             # make the drone pure-visual so USD moves reach the viewport
        for _p in Usd.PrimRange(_stage.GetPrimAtPath("/World/Drone")):
            for _api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.ArticulationRootAPI):
                if _p.HasAPI(_api):
                    _p.RemoveAPI(_api)
    print(f"[yroad] Go2 + visible D455 @ base{D455_MOUNT} (front-top)  |  visible drone (x{a.drone_scale:g}) @ "
          f"({drone_x:.0f},{drone_y:.0f},{drone_z:.0f})  |  whole-map BEV cam @ {H + bev_alt:.0f}m")

    # --- make the drone look alive: spin the 4 propellers + gentle hover-orbit (kinematic, visual only).
    # cf2x joints fail to create (static bodies) so the props never move on their own; we drive the prim
    # transforms each step -- same proven USD-translate trick the Cross_View collection uses for viewport sync.
    import re
    from pxr import UsdGeom, Gf
    drone_root_prim = _stage.GetPrimAtPath("/World/Drone")
    prop_ops = []                                                # (rotateXYZ op, base euler Vec3, spin sign)
    for _p in Usd.PrimRange(drone_root_prim):
        nm = _p.GetName().lower()
        if re.fullmatch(r"m[1-4]_prop", nm) and _p.IsA(UsdGeom.Xformable):   # the 4 rotor frames (cf2x)
            xf = UsdGeom.Xformable(_p)
            rop = next((op for op in xf.GetOrderedXformOps()
                        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None) or xf.AddRotateXYZOp()
            base = rop.Get(); base = Gf.Vec3f(base) if base is not None else Gf.Vec3f(0.0, 0.0, 0.0)
            sign = 1.0 if int(nm[1]) % 2 == 1 else -1.0          # m1/m3 ccw, m2/m4 cw (counter-rotating)
            prop_ops.append((rop, base, sign))
    _drone_xf = UsdGeom.Xformable(drone_root_prim)
    drone_tr = next((op for op in _drone_xf.GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if drone_tr is None:
        drone_tr = _drone_xf.AddTranslateOp()
    print(f"[yroad] drone propellers found: {len(prop_ops)}  (spin={a.drone_spin:g} deg/step, "
          f"orbit={a.drone_orbit:g} m)")

    go2_cam = drone_cam = scene_cam = closeup_cam = avoid_cam = drone_down_cam = None
    if a.drone_ctrl:                                             # downward camera the user-flown drone carries
        drone_down_cam = Camera(CameraCfg(
            prim_path="/World/DroneCam", update_period=0.0, height=256, width=256, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, horizontal_aperture=20.955,
                                             clipping_range=(0.5, 250.0))))
    if a.avoid:                                                  # D455 front depth for local avoidance
        avoid_cam = Camera(CameraCfg(
            prim_path="/World/envs/env_0/Go2/base/d455_depth", update_period=0.0, height=48, width=64,
            data_types=["distance_to_image_plane"], spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.05, 6.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.36, 0.0, 0.085), rot=(1.0, 0.0, 0.0, 0.0), convention="world")))   # straight ahead (no down-tilt) so the ground isn't read as an obstacle
    if a.cameras or a.cnn or a.show:                             # whole-map nadir BEV = the RGB the CNN consumes
        drone_cam = Camera(CameraCfg(
            prim_path="/World/BEV_cam", update_period=0.0, height=256, width=256,
            data_types=["rgb", "distance_to_image_plane"], spawn=sim_utils.PinholeCameraCfg(  # +depth -> nadir height (RGB+D U-Net)
                focal_length=bev_focal, horizontal_aperture=20.955, clipping_range=(0.1, bev_alt * 3.0)),
            offset=CameraCfg.OffsetCfg(pos=(float(cx), float(cy), float(H + bev_alt)),
                                       rot=(0.70711, 0.0, 0.70711, 0.0), convention="world")))
    if a.cameras or a.show or a.dino_loc or a.drone_ctrl:        # Go2 D455 front view (also needed for localization / drone-ctrl viewport)
        go2_cam = Camera(CameraCfg(
            prim_path="/World/envs/env_0/Go2/base/d455_cam", update_period=0.0, height=240, width=320,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.36, 0.0, 0.085), rot=(0.99619, 0.0, 0.08716, 0.0), convention="world")))
    if a.cameras:
        # third-person verification shots (set via look-at after reset)
        scene_cam = Camera(CameraCfg(
            prim_path="/World/Scene_cam", update_period=0.0, height=480, width=720,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
                focal_length=11.0, horizontal_aperture=20.955, clipping_range=(0.1, 300.0))))
        closeup_cam = Camera(CameraCfg(
            prim_path="/World/Closeup_cam", update_period=0.0, height=480, width=640,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
                focal_length=28.0, horizontal_aperture=20.955, clipping_range=(0.05, 50.0))))
        print("[yroad] cameras: Go2 D455 (front) + whole-map BEV (nadir) + scene + closeup")

    sim.reset()

    # place Go2 on the plateau at the start, facing +X
    root = robot.data.default_root_state.clone()
    root[0, 0], root[0, 1], root[0, 2] = float(sx), float(sy), float(H + 0.45)
    robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(root[:, 7:])
    default_jp = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(default_jp, robot.data.default_joint_vel.clone())
    scene.write_data_to_sim()

    if scene_cam is not None:                                     # frame BOTH the Go2 (ground) and the drone (up)
        scene_cam.set_world_poses_from_view(
            torch.tensor([[sx - 5.0, sy - 6.0, H + 4.0]], device=a.device),
            torch.tensor([[drone_x - 0.4, sy, H + 1.4]], device=a.device))
        closeup_cam.set_world_poses_from_view(                    # tight on the Go2 head to show the D455
            torch.tensor([[sx + 1.6, sy - 1.2, H + 0.9]], device=a.device),
            torch.tensor([[sx + 0.32, sy, H + 0.5]], device=a.device))

    if a.cnn:                                                     # M3: build the planner from the U-Net BEV reading
        import numpy as np
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())   # hold pose while the BEV renders
        unet_path = a.unet if Path(a.unet).is_absolute() else str(Path(__file__).resolve().parent / a.unet)
        unet = torch.jit.load(unet_path).to(a.device).eval()
        for _ in range(8):
            scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
            drone_cam.update(sim.get_physics_dt())
        rgb = drone_cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        rgb_w = np.ascontiguousarray(np.rot90(rgb, 3))           # BEV cam axes -> world/raster frame (as in collect)
        dep = drone_cam.data.output["distance_to_image_plane"][0].detach().cpu().numpy()   # (H,W) metres
        if dep.ndim == 3:
            dep = dep[..., 0]
        cam_alt = float(H + bev_alt); Z_NORM = float(H + 2.0)    # EXACT collect_yroad depth normalisation
        hnorm = np.clip((cam_alt - dep) / Z_NORM, 0.0, 1.0)      # surface height above canyon floor, [0,1]
        hnorm_w = np.ascontiguousarray(np.rot90(hnorm, 3))       # SAME rot as rgb -> pixel-aligned
        arr = np.concatenate([rgb_w.astype(np.float32) / 255.0,
                              hnorm_w[..., None].astype(np.float32)], axis=-1)   # (H,W,4) RGB+D, matches training
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].to(a.device)
        with torch.no_grad():
            pred = unet(x).argmax(1)[0].cpu().numpy().astype(np.int16)   # 256x256 class grid (world frame)
        from scipy.ndimage import binary_closing             # heal spurious CNN cracks that pinch the corridor
        _walk = binary_closing(np.isin(pred, (1, 2, 3)), structure=np.ones((3, 3)), iterations=2)
        pred = np.where(_walk, np.where(pred == 0, 1, pred), 0).astype(np.int16)
        half = 0.58 * msize
        window = (cx - half, cx + half, cy - half, cy + half)    # MATCHES the BEV footprint -> aligned to world
        planner = planner_mod.GTPlanner.from_class_grid(pred, window, (gx, gy), edge_clear=1.0)
        sr, sc = planner.world_to_cell(sx, sy)
        print(f"[yroad] CNN planner: U-Net={Path(unet_path).name}  grid {planner.Hp}x{planner.Wp} "
              f"cell={planner.cell:.2f}m  start->goal reachable={planner.reachable(sx, sy)}  "
              f"cost={float(planner.dist[sr, sc]):.1f}")
        try:                                                     # save BEV + CNN cost-map(+path) + GT-vs-CNN compare
            from PIL import Image
            out = Path(__file__).resolve().parent / "data" / "shots"; out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb_w).save(out / "deploy_cnn_bev.png")
            Image.fromarray(planner.render((sx, sy), (gx, gy))).save(out / "deploy_costmap_cnn.png")
            gt = planner_mod.GTPlanner(info["elements"], info["bbox"], (gx, gy), polys=info.get("polys"))   # GT for visual comparison only
            gi = Image.fromarray(gt.render((sx, sy), (gx, gy)))
            ci = Image.fromarray(planner.render((sx, sy), (gx, gy))).resize(gi.size, Image.NEAREST)
            sep = np.full((gi.size[1], 6, 3), 255, np.uint8)
            Image.fromarray(np.concatenate([np.asarray(gi), sep, np.asarray(ci)], 1)).save(out / "deploy_costmap_compare.png")
            print("[yroad] saved: deploy_cnn_bev / deploy_costmap_cnn / deploy_costmap_compare .png (GT | CNN)")
        except Exception as e:                                   # noqa: BLE001
            print(f"[yroad] CNN cost-map render skipped: {e}")

    # ── DINOv3 cross-view localization assets ──────────────────────────────────
    dino_enc = dino_db = dino_poses = dino_meta = None
    if a.dino_loc:
        import numpy as np
        dp = Path(__file__).resolve().parent / a.dino_deploy
        dino_enc   = torch.jit.load(str(dp / "front_encoder.pt")).to(a.device).eval()
        dino_db    = torch.load(str(dp / "bev_db.pt"), map_location=a.device)   # (N,128)
        dino_poses = np.load(str(dp / "bev_poses.npy"))                         # (N,3) x,y,yaw
        import json as _json; dino_meta = _json.load(open(dp / "meta.json"))
        res = dino_meta["res"]
        print(f"[yroad] DINOv3 loc loaded: DB={dino_db.shape}  res={res}  deploy={dp.name}")
        if not a.no_guide:
            planner = planner_mod.GTPlanner(info["elements"], info["bbox"], (gx, gy), polys=info.get("polys"))
            print(f"[yroad] planner (GT) ready for DINOv3-estimated positions")

    policy = torch.jit.load(str(POLICY)).to(a.device).eval()
    print(f"[yroad] policy loaded: {POLICY.name}  (45→12)")

    if a.gui:
        sim.set_camera_view([sx - 5, sy - 6, H + 4], [drone_x - 0.4, sy, H + 1.4])
        # Docked in-Isaac viewports re-render the whole 200 m scene per viewport. With the 50 Hz render
        # decouple that's now affordable, so --docked (and interactive --drone_ctrl) use them. Plain --show
        # defaults to cheap cv2 windows (sensor buffer blit, ~free).
        if a.drone_ctrl or a.docked:                             # docked camera viewports INSIDE Isaac Sim (not cv2)
            try:
                from omni.kit.viewport.utility import create_viewport_window
                import omni.ui as _ui
                def _dock(name, cam_path, w, h, px, py):
                    win = create_viewport_window(name, width=w, height=h, position_x=px, position_y=py)
                    vp = win.viewport_api
                    if hasattr(vp, "set_active_camera"):
                        vp.set_active_camera(cam_path)
                    else:
                        vp.camera_path = cam_path
                    try:                                          # float free of the main viewport (no tab-stacking)
                        win.undock(); win.position_x = px; win.position_y = py
                    except Exception:
                        pass
                    return win
                n_win = 0
                if go2_cam is not None:                           # top-left
                    _dock("Go2 D455 (front)", "/World/envs/env_0/Go2/base/d455_cam", 460, 345, 20, 40); n_win += 1
                if drone_down_cam is not None:                    # top-right: the flown drone's downward view
                    _dock("Drone DOWN cam (fly: WASD / PgUp-PgDn)", "/World/DroneCam", 460, 460, 500, 40); n_win += 1
                elif drone_cam is not None:
                    _dock("Drone BEV (whole map)", "/World/BEV_cam", 460, 460, 500, 40); n_win += 1
                print(f"[yroad] in-Isaac viewports created ({n_win} windows, side by side)")
            except Exception as e:                               # noqa: BLE001
                print(f"[yroad] viewport create skipped: {e}")

    cmd = torch.tensor([[a.vx, a.vy, a.vyaw]], device=a.device, dtype=torch.float32)
    last_action = torch.zeros(1, 12, device=a.device)

    fell = goal_done = False
    traj = []
    stuck_hist = []; recover = 0
    est_pos = None                                               # DINOv3-estimated position (x, y, yaw)
    avoid_phase = 0; avoid_sign = 1.0                            # local-avoidance commit state
    AVOID_COMMIT = 50

    drone_ctrl_pos = None; key_state = {}; _carb = None; drone_yaw_op = None; drone_ctrl_yaw = 0.0
    if a.drone_ctrl:
        import carb as _carb
        import omni.appwindow
        drone_ctrl_pos = [float(sx), float(sy), float(H + 8.0)]   # start near Go2, low enough to be visible
        _dxf = UsdGeom.Xformable(_stage.GetPrimAtPath("/World/Drone"))   # drone heading rotateZ op
        drone_yaw_op = next((op for op in _dxf.GetOrderedXformOps()
                             if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ), None) or _dxf.AddRotateZOp()
        _iface = _carb.input.acquire_input_interface()
        _kb = omni.appwindow.get_default_app_window().get_keyboard()
        def _on_key(e, *args):
            if e.type == _carb.input.KeyboardEventType.KEY_PRESS:   key_state[e.input] = True
            elif e.type == _carb.input.KeyboardEventType.KEY_RELEASE: key_state[e.input] = False
            return True
        _kb_sub = _iface.subscribe_to_keyboard_events(_kb, _on_key)
        print("[yroad] DRONE CONTROL: W/S=fwd/back  A/D=strafe  Left/Right=rotate(yaw)  PgUp/PgDn=altitude  (downward cam = BEV viewport)")
    LOC_EVERY = 20                                               # update localization every N decimation steps
    loc_step = 0

    for t in range(a.steps):
        if t % DECIMATION == 0:
            p = robot.data.root_pos_w[0]
            traj.append((float(p[0]), float(p[1])))

            # ── DINOv3 localization: encode D455 image → nearest BEV patch → est pos ──
            if dino_enc is not None and go2_cam is not None and loc_step % LOC_EVERY == 0:
                f_raw = go2_cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
                f_raw = f_raw if f_raw.dtype == np.uint8 else (np.clip(f_raw, 0, 1) * 255).astype(np.uint8)
                res = dino_meta["res"]
                f_img = np.asarray(Image.fromarray(f_raw).resize((res, res)), np.float32) / 255.0
                f_t = torch.from_numpy(f_img).permute(2, 0, 1)[None].to(a.device)
                with torch.no_grad():
                    q = dino_enc(f_t)                            # (1, 128)
                nn_i = (q @ dino_db.t()).argmax(1).item()
                ep = dino_poses[nn_i]                            # (x, y, yaw)
                new_est = (float(ep[0]), float(ep[1]), float(ep[2]))
                # outlier guard: reject if jump > 5m from GT (use GT as anchor, not prev est)
                gt_xy = (float(p[0]), float(p[1]))
                if math.hypot(new_est[0]-gt_xy[0], new_est[1]-gt_xy[1]) < 5.0:
                    est_pos = new_est                            # accept
                else:
                    est_pos = (gt_xy[0], gt_xy[1], float(ep[2]))  # fallback to GT xy, keep yaw
                    print(f"  [loc] t={t:4d} OUTLIER rejected (err={math.hypot(new_est[0]-gt_xy[0], new_est[1]-gt_xy[1]):.1f}m) -> GT fallback")
                if t % (LOC_EVERY * DECIMATION * 5) == 0:
                    gt_xy = (float(p[0]), float(p[1]))
                    err = math.hypot(est_pos[0]-gt_xy[0], est_pos[1]-gt_xy[1])
                    print(f"  [loc] t={t:4d}  GT=({gt_xy[0]:.1f},{gt_xy[1]:.1f})  "
                          f"est=({est_pos[0]:.1f},{est_pos[1]:.1f})  err={err:.1f}m")
            loc_step += 1

            if planner is not None:
                if dino_enc is not None and est_pos is not None:
                    # use DINOv3-estimated position instead of GT
                    ex, ey, eyaw = est_pos
                else:
                    qw, qx, qy, qz = robot.data.root_quat_w[0].tolist()
                    ex, ey = float(p[0]), float(p[1])
                    eyaw = math.atan2(2.0*(qw*qz+qx*qy), 1.0-2.0*(qy*qy+qz*qz))
                qw, qx, qy, qz = robot.data.root_quat_w[0].tolist()
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                vx, vy, vyaw, reached = planner.command(ex, ey, yaw, (gx, gy), vx_max=a.vx)
                if t % (DECIMATION * 20) == 0:          # debug: print every 20 policy steps
                    nx, ny = planner.lookahead_target(ex, ey, ahead_m=0.9)
                    desired = math.atan2(ny - ey, nx - ex)
                    err = (desired - yaw + math.pi) % (2 * math.pi) - math.pi
                    print(f"  [plan] t={t:4d}  pos=({ex:.1f},{ey:.1f})  yaw={math.degrees(yaw):.0f}°  "
                          f"carrot=({nx:.1f},{ny:.1f})  desired={math.degrees(desired):.0f}°  "
                          f"err={math.degrees(err):.0f}°  vx={vx:.2f}  vyaw={vyaw:.2f}")
                # ── rule-based local avoidance: steer toward the clearer side if blocked ahead ──
                if avoid_cam is not None:
                    dep = avoid_cam.data.output["distance_to_image_plane"][0]
                    dep = dep[..., 0] if dep.dim() == 3 else dep            # (Hd, Wd) metres
                    Hd, Wd = dep.shape
                    band = dep[Hd//3:2*Hd//3]                               # central horizontal band
                    valid = band > 0.05
                    centre = band[:, Wd//4:3*Wd//4]; cv = centre > 0.05
                    min_d = float(centre[cv].min()) if cv.any() else 99.0
                    left = band[:, :Wd//2]; right = band[:, Wd//2:]
                    lc = float(left[left > 0.05].mean()) if (left > 0.05).any() else 6.0
                    rc = float(right[right > 0.05].mean()) if (right > 0.05).any() else 6.0
                    if min_d < a.avoid_thresh and avoid_phase <= 0:         # new obstacle -> commit to a side
                        avoid_sign = 1.0 if lc > rc else -1.0              # +1 = turn left, -1 = turn right
                        avoid_phase = AVOID_COMMIT
                        print(f"  [avoid] t={t} obstacle min_d={min_d:.2f}m -> commit {'L' if avoid_sign>0 else 'R'}")
                    if avoid_phase > 0:                                    # committed: override planner
                        avoid_phase -= 1
                        if min_d < a.avoid_thresh:
                            vyaw = 0.9 * avoid_sign; vx = max(0.45, a.vx * 0.55)   # blocked: turn toward clear
                        else:
                            vyaw = 0.0; vx = a.vx * 0.85                   # clear ahead: drive straight past
                cmd = torch.tensor([[vx, vy, vyaw]], device=a.device, dtype=torch.float32)
                if reached and not goal_done:
                    goal_done = True
                    if a.drone_ctrl:                             # keep the sim running so the user can keep flying
                        print(f"  [goal] reached at t={t} (Go2 holds at goal; keep flying the drone)")
                    else:
                        print(f"  [goal] reached at t={t} -> stopping"); break
                if not a.avoid:                                  # stuck-recovery off when avoidance handles obstacles
                    stuck_hist.append((float(p[0]), float(p[1])))
                    if len(stuck_hist) > 60:
                        stuck_hist.pop(0)
                    if recover > 0:                              # mid-escape: back up + turn
                        recover -= 1
                        cmd = torch.tensor([[-0.4, 0.0, 0.9]], device=a.device, dtype=torch.float32)
                    elif len(stuck_hist) == 60 and not goal_done:
                        d2 = (stuck_hist[-1][0]-stuck_hist[0][0])**2 + (stuck_hist[-1][1]-stuck_hist[0][1])**2
                        if d2 < 0.15 ** 2:
                            recover = 40; stuck_hist.clear()
                            print(f"  [recover] stuck at ({float(p[0]):.1f},{float(p[1]):.1f}) -> back up & turn")
            ang  = robot.data.root_ang_vel_b
            grav = robot.data.projected_gravity_b
            jp_rel = robot.data.joint_pos - default_jp
            jv = robot.data.joint_vel
            obs = torch.cat([ang * 0.2, grav, cmd, jp_rel, jv * 0.05, last_action], dim=1)
            with torch.no_grad():
                action = policy(obs)
            last_action = action
            robot.set_joint_position_target(default_jp + ACT_SCALE * action)
        # Decouple RENDER from the 200 Hz physics: step physics WITHOUT rendering, then render only every
        # RENDER_EVERY steps (~50 Hz). The 200 Hz main-viewport render was the real FPS killer -> this is ~4x faster.
        scene.write_data_to_sim(); sim.step(render=False); scene.update(sim.get_physics_dt())

        if prop_ops:                                             # spin props + gentle hover-orbit (visual)
            spin = (t * a.drone_spin) % 360.0
            for rop, base, sign in prop_ops:                     # animate the existing rotateXYZ Z-component
                rop.Set(Gf.Vec3f(base[0], base[1], base[2] + sign * spin))
            if a.drone_orbit > 0.0 and not a.drone_ctrl:
                th = t * 0.03
                drone_tr.Set(Gf.Vec3d(drone_x + a.drone_orbit * math.cos(th),
                                      drone_y + a.drone_orbit * math.sin(th),
                                      drone_z + 0.08 * math.sin(t * 0.05)))

        if a.drone_ctrl and drone_ctrl_pos is not None:          # WASD move + Left/Right yaw + PgUp/PgDn altitude
            KI = _carb.input.KeyboardInput
            dt = sim.get_physics_dt(); SPD = 8.0; VSPD = 6.0; YAWSPD = 1.4   # m/s, m/s, rad/s
            if key_state.get(KI.LEFT):  drone_ctrl_yaw += YAWSPD * dt        # rotate left (CCW)
            if key_state.get(KI.RIGHT): drone_ctrl_yaw -= YAWSPD * dt        # rotate right (CW)
            cyaw, syaw = math.cos(drone_ctrl_yaw), math.sin(drone_ctrl_yaw)
            fx, fy = cyaw, syaw                                  # heading-relative forward / strafe-left axes
            lx, ly = -syaw, cyaw
            if key_state.get(KI.W): drone_ctrl_pos[0] += SPD*dt*fx; drone_ctrl_pos[1] += SPD*dt*fy
            if key_state.get(KI.S): drone_ctrl_pos[0] -= SPD*dt*fx; drone_ctrl_pos[1] -= SPD*dt*fy
            if key_state.get(KI.A): drone_ctrl_pos[0] += SPD*dt*lx; drone_ctrl_pos[1] += SPD*dt*ly
            if key_state.get(KI.D): drone_ctrl_pos[0] -= SPD*dt*lx; drone_ctrl_pos[1] -= SPD*dt*ly
            if key_state.get(KI.PAGE_UP):   drone_ctrl_pos[2] += VSPD * dt
            if key_state.get(KI.PAGE_DOWN): drone_ctrl_pos[2] = max(2.0, drone_ctrl_pos[2] - VSPD * dt)
            drone_tr.Set(Gf.Vec3d(*drone_ctrl_pos))              # move the visible (pure-visual) drone
            if drone_yaw_op is not None:
                drone_yaw_op.Set(math.degrees(drone_ctrl_yaw))  # rotate the visible drone model with heading
            if drone_down_cam is not None:                       # cam follows pos + rotates with heading (nadir+yaw)
                hw = 0.7071068
                q = torch.tensor([[hw*math.cos(drone_ctrl_yaw/2), -hw*math.sin(drone_ctrl_yaw/2),
                                   hw*math.cos(drone_ctrl_yaw/2),  hw*math.sin(drone_ctrl_yaw/2)]],
                                 device=a.device, dtype=torch.float32)
                drone_down_cam.set_world_poses(
                    torch.tensor([drone_ctrl_pos], device=a.device, dtype=torch.float32), q, convention="world")

        # ONE render serves both the main viewport (~50 Hz) and the camera render-products. avoid_cam drives
        # steering so it stays fresh; the view cameras refresh ~17 Hz for the cv2 feeds + final frame dump.
        RENDER_EVERY = 1 if a.headless else 4                    # main viewport ~50 Hz instead of 200 Hz
        cam_view_every = 2 if a.headless else 12                 # camera feeds ~17 Hz
        want_avoid = avoid_cam is not None
        want_cam = (t % cam_view_every == 0)
        if (t % RENDER_EVERY == 0) or want_avoid or want_cam:
            sim.render()                                         # refreshes main viewport AND all camera render-products
        if want_avoid:
            avoid_cam.update(sim.get_physics_dt())
        if want_cam:
            for c in (go2_cam, drone_cam, scene_cam, closeup_cam, drone_down_cam):
                if c is not None:
                    c.update(sim.get_physics_dt())
        # cv2 feeds only for plain --show. When --docked or --drone_ctrl is on, the in-Isaac docked viewports
        # are the display (responsive for flying), so skip the redundant/laggy cv2 windows.
        if a.show and not a.docked and not a.drone_ctrl and go2_cam is not None and t % cam_view_every == 0:
            _f = go2_cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
            _f = _f if _f.dtype == np.uint8 else (np.clip(_f, 0, 1) * 255).astype(np.uint8)
            _cv2.imshow("Go2 D455 (front)", _cv2.cvtColor(_f, _cv2.COLOR_RGB2BGR))
            if drone_cam is not None:
                _b = drone_cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
                _b = _b if _b.dtype == np.uint8 else (np.clip(_b, 0, 1) * 255).astype(np.uint8)
                _cv2.imshow("Drone BEV (whole map)", _cv2.cvtColor(_b, _cv2.COLOR_RGB2BGR))
            if _cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if t % 100 == 0:
            p = robot.data.root_pos_w[0]
            dg = math.hypot(gx - float(p[0]), gy - float(p[1]))
            meas_vx = float(robot.data.root_lin_vel_b[0, 0])     # actual forward speed (m/s)
            path_left = ""                                       # remaining distance ALONG the ㄷ path (truly decreases)
            if planner is not None:
                rr, cc = planner.world_to_cell(float(p[0]), float(p[1]))
                d = float(planner.dist[rr, cc])
                if math.isfinite(d):
                    path_left = f"  path_left={d:.1f}m"
            print(f"  t={t:4d}  pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})  "
                  f"d2goal={dg:.1f}m{path_left}  cmd_vx={cmd[0,0]:.2f}  meas_vx={meas_vx:.2f}m/s")
            if p[2] < H - 0.5:                                   # fell off the plateau
                fell = True; print("  [!] Go2 fell off the plateau"); break

    p = robot.data.root_pos_w[0]
    dg = math.hypot(gx - float(p[0]), gy - float(p[1]))
    path_left = float("nan")
    if planner is not None:                                      # how far ALONG the path it still had to go
        rr, cc = planner.world_to_cell(float(p[0]), float(p[1]))
        path_left = float(planner.dist[rr, cc])
    status = "GOAL" if goal_done else ("FELL" if fell else "STOPPED short")
    print(f"[yroad] done. final=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})  d2goal={dg:.1f}m  "
          f"path_left={path_left:.1f}m  [{status}]")
    if not goal_done and not fell:
        print(f"[yroad] !! stopped before goal at ({float(p[0]):.1f},{float(p[1]):.1f}) with "
              f"{path_left:.1f}m of path remaining — check deploy_trajectory_gt.png for where the cyan trail ends.")

    if planner is not None and traj:                             # overlay the ACTUAL Go2 path on the cost map
        try:
            import numpy as np
            from PIL import Image
            out = Path(__file__).resolve().parent / "data" / "shots"; out.mkdir(parents=True, exist_ok=True)
            img = planner.render((sx, sy), (gx, gy))             # cost map + planned path (magenta) + start/goal
            for tx, ty in traj:                                  # actual trajectory in cyan
                r, c = planner.world_to_cell(tx, ty)
                img[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = (0, 255, 255)
            tag = "cnn" if a.cnn else "gt"
            Image.fromarray(img).save(out / f"deploy_trajectory_{tag}.png")
            print(f"[yroad] trajectory ({len(traj)} pts, min d2goal seen) -> {out}/deploy_trajectory_{tag}.png "
                  f"(cyan=Go2 actual, magenta=planned)")
        except Exception as e:                                   # noqa: BLE001
            print(f"[yroad] trajectory render skipped: {e}")

    if go2_cam is not None:                                      # save sample frames to verify cameras
        import numpy as np
        from PIL import Image
        out = Path(__file__).resolve().parent / "data" / "shots"; out.mkdir(parents=True, exist_ok=True)
        for cam, name in [(drone_cam, "deploy_bev"), (go2_cam, "deploy_d455"),
                          (scene_cam, "deploy_scene"), (closeup_cam, "deploy_go2_closeup")]:
            if cam is None:
                continue
            cam.update(sim.get_physics_dt())                     # force a fresh frame (sensor was throttled during the run)
            img = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(out / f"{name}.png")
        print(f"[yroad] saved frames -> {out}/ : deploy_bev, deploy_d455, deploy_scene, deploy_go2_closeup .png")

    if a.gui:                                                    # keep the window LIVE after the run (no freeze, no auto-close)
        print("[yroad] run finished — viewer stays open. Close the window or press Ctrl+C to exit.")
        try:
            while simulation_app.is_running():
                sim.render()                                     # render only (no physics) so the GUI stays responsive
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
