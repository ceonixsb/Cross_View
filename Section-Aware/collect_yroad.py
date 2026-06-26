"""collect_yroad.py — CNN/U-Net dataset: drone 91 m whole-map BEV RGB  +  GT 4-class label.

NO robot. One Isaac session; rebuild the ㄷ map N times (vary obstacle layout / fork-C arrangement /
sun direction) and for each variant:
    - render the nadir BEV from the SAME 91 m camera deploy uses   → U-Net INPUT  (rgb)
    - rasterise the GT 4-class map from the known geometry          → U-Net TARGET (label)
The map extent is held fixed (fixed arm lengths) so every sample shares ONE world frame — the BEV
pixel (u,v) and the label pixel (u,v) map to the same world cell.

Classes: 0=impassable(void/wall/sunken/narrow) 1=free 2=rough 3=tunnel.

    conda activate isaacsim
    python collect_yroad.py --n 4 --out data/cnn          # small alignment test
    python collect_yroad.py --n 300 --out data/cnn        # full dataset
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l_map

parser = argparse.ArgumentParser(description="Y-Road CNN dataset: 91m BEV RGB + GT label")
l_map.add_map_args(parser)
parser.add_argument("--n", type=int, default=4, help="number of map variants to render")
parser.add_argument("--out", type=str, default="data/cnn", help="output dir (under Section-Aware)")
parser.add_argument("--res", type=int, default=256, help="BEV/label resolution (square)")
parser.add_argument("--seed0", type=int, default=0, help="base seed for variant generation")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--hold", action="store_true",
                    help="with --gui: keep the viewport open after rendering so you can inspect the map in 3D")
parser.add_argument("--tilt_views", type=int, default=0,
                    help="number of TILTED orbit views to capture per map (0 = nadir only). 4 = aligned to the arms")
parser.add_argument("--tilt_elev", type=float, default=25.0,
                    help="tilted camera elevation above horizontal (deg). low(~25) peeks under tunnel roofs")
parser.add_argument("--tilt_radius_mul", type=float, default=0.5,
                    help="tilted orbit radius = mul * msize (smaller = closer, features clearer)")
parser.add_argument("--tilt_sections", action="store_true",
                    help="capture per-FORK tilted views (drone visits each junction up close) instead of a whole-map orbit")
parser.add_argument("--tilt_dirs", type=int, default=4,
                    help="tilted directions per fork (4 = 사방). used with --tilt_sections")
parser.add_argument("--only_fork", type=int, default=-1,
                    help="with --tilt_sections: capture ONLY this fork index (for testing one section). -1 = all")
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args()
a.headless = not a.gui
a.enable_cameras = True

app_launcher = AppLauncher(a)
simulation_app = app_launcher.app

# ── post-launch ──
import numpy as np
import torch
import omni.usd
from pxr import Usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sensors import Camera, CameraCfg
from PIL import Image

RES = a.res
# cliff-heavy pool: the entrance-looks-walkable-then-drops lane is the hardest case for a nadir model.
# one 'safe' is force-inserted per variant -> always exactly one solvable lane.
_FORK5_POOL = ["blocked", "narrow", "cliff", "tunnel", "cliff"]


def _del(stage, path):
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)


def clear_terrain(stage):
    for p in ("/World/Lmap", "/World/floor", "/World/start", "/World/goal",
              "/World/DomeLight", "/World/SunLight"):
        _del(stage, p)


def variant_args(base, i):
    """Return a shallow-copied args namespace with this variant's randomisation."""
    rng = np.random.default_rng(base.seed0 + i)
    v = argparse.Namespace(**vars(base))
    v.obs_seed = int(base.seed0 + i)                       # different obstacle layout
    v.n_obs = int(rng.integers(8, 16))                     # harder: denser step obstacles
    v.obs_hmax = max(getattr(base, "obs_hmax", 0.35), 0.40)
    v.narrow_jitter = max(getattr(base, "narrow_jitter", 0.0), 0.08)   # borderline-looking narrow lanes
    v.blocked_wall_frac = float(rng.uniform(0.55, 0.75))   # push the wall deeper, varies per variant
    pool = _FORK5_POOL.copy(); rng.shuffle(pool)
    safe_at = int(rng.integers(0, 5))
    types = pool[:]; types[safe_at] = "safe"               # exactly one guaranteed-safe lane
    v.fork5_types = ",".join(types)
    # 3-way detour fork B: center = blocked (going straight fails) -> the free lane sits to one side
    side_hazard = "cliff" if rng.random() < 0.5 else "narrow"
    sides = ["safe", side_hazard]; rng.shuffle(sides)
    v.fork3_types = ",".join([sides[0], "blocked", sides[1]])
    return v, rng


def randomize_sun(stage, rng):
    """Respawn the sun with a random azimuth/elevation so shadow direction varies (key RGB hazard cue)."""
    _del(stage, "/World/SunLight")
    az = float(rng.uniform(0, 2 * math.pi))
    el = float(rng.uniform(math.radians(25), math.radians(60)))   # low-ish diagonal sun
    # quaternion for a distant light pointing along (cos el*cos az, cos el*sin az, -sin el)
    cz, sz = math.cos(az / 2), math.sin(az / 2)
    cx, sx = math.cos(-el / 2), math.sin(-el / 2)
    quat = (cz * cx, cz * sx, sz * sx, sz * cx)            # approx ZX euler -> wxyz (visual variety only)
    inten = float(rng.uniform(1300.0, 2000.0))
    cfg = sim_utils.DistantLightCfg(intensity=inten, color=(1.0, 1.0, 1.0), angle=0.35)   # neutral white -> gray map
    cfg.func("/World/SunLight", cfg, orientation=quat)


def main():
    out = Path(__file__).resolve().parent / a.out
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "label").mkdir(parents=True, exist_ok=True)
    (out / "label_vis").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)        # nadir height map (16-bit, normalised)
    (out / "depth_vis").mkdir(parents=True, exist_ok=True)    # 8-bit grayscale for eyeballing

    sim = SimulationContext(sim_utils.SimulationCfg(device=a.device, dt=0.01))
    stage = omni.usd.get_context().get_stage()

    # build variant 0 to lock the world frame (fixed extent across all variants)
    v0, _ = variant_args(a, 0)
    info = l_map.build_L(v0)
    cx, cy = info["center"]; msize = info["size"]; H = info["H"]
    bev_alt = msize * 1.15
    bev_focal = 20.955 / (2.0 * math.tan(math.atan(msize * 0.58 / bev_alt)))
    half = 0.58 * msize                                    # ground half-coverage at the plateau plane
    window = (cx - half, cx + half, cy - half, cy + half)  # MATCHES the camera footprint -> label aligned
    print(f"[collect] world frame: center=({cx:.1f},{cy:.1f}) msize={msize:.1f} "
          f"BEV alt={H + bev_alt:.0f}m  window={tuple(round(w,1) for w in window)}  res={RES}")

    cam = Camera(CameraCfg(
        prim_path="/World/BEV_cam", update_period=0.0, height=RES, width=RES,
        data_types=["rgb", "distance_to_image_plane"],     # +depth -> nadir height map
        spawn=sim_utils.PinholeCameraCfg(focal_length=bev_focal, horizontal_aperture=20.955,
                                         clipping_range=(0.1, bev_alt * 3.0)),
        offset=CameraCfg.OffsetCfg(pos=(float(cx), float(cy), float(H + bev_alt)),
                                   rot=(0.70711, 0.0, 0.70711, 0.0), convention="world")))
    cam_alt = float(H + bev_alt)                            # camera height -> surface_z = cam_alt - depth
    Z_NORM = float(H + 2.0)                                 # normalise height to [0,1]: 0=canyon floor, plateau~0.5

    # --- tilted orbit camera (the drone's oblique eye) — repositioned per view via set_world_poses_from_view ---
    tcam, tilt_eyes, tilt_dist = None, [], 0.0
    if a.tilt_views > 0:
        tilt_R = a.tilt_radius_mul * msize
        tilt_dist = tilt_R / math.cos(math.radians(a.tilt_elev))
        tilt_focal = 20.955 * tilt_dist / (1.15 * msize)    # tight framing -> map fills the frame, features clear
        tcam = Camera(CameraCfg(
            prim_path="/World/Tilt_cam", update_period=0.0, height=RES, width=RES,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(tilt_focal), horizontal_aperture=20.955,
                                             clipping_range=(0.1, float(tilt_dist * 2.5))),
            offset=CameraCfg.OffsetCfg(pos=(float(cx), float(cy), float(H + tilt_dist)),
                                       rot=(0.70711, 0.0, 0.70711, 0.0), convention="world")))
        for k in range(a.tilt_views):                        # azimuths aligned to the arms (k=0 -> +X)
            az = 2.0 * math.pi * k / a.tilt_views
            tilt_eyes.append((float(cx + tilt_R * math.cos(az)), float(cy + tilt_R * math.sin(az)),
                              float(H + tilt_R * math.tan(math.radians(a.tilt_elev)))))
        (out / "tilted").mkdir(parents=True, exist_ok=True)
        (out / "tilted_vis").mkdir(parents=True, exist_ok=True)
        print(f"[collect] tilted: {a.tilt_views} views  elev={a.tilt_elev:.0f}deg  R={tilt_R:.0f}m  focal={tilt_focal:.1f}mm")

    # --- per-section tilted camera: the drone visits each FORK up close (radius set by fork size, not msize) ---
    scam, sec_R = None, 0.0
    if a.tilt_sections:
        fork_len = (a.n_branch + 2) * a.sec_len                 # one fork's length along its arm (~32 m)
        sec_R = 1.1 * fork_len                                  # horizontal distance from the fork
        sec_dist = sec_R / math.cos(math.radians(a.tilt_elev))
        sec_focal = 20.955 * sec_dist / (1.2 * fork_len)        # frame ONE fork tightly
        scam = Camera(CameraCfg(
            prim_path="/World/Sec_cam", update_period=0.0, height=RES, width=RES,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(sec_focal), horizontal_aperture=20.955,
                                             clipping_range=(0.1, float(sec_dist * 3.0))),
            offset=CameraCfg.OffsetCfg(pos=(float(cx), float(cy), float(H + sec_dist)),
                                       rot=(0.70711, 0.0, 0.70711, 0.0), convention="world")))
        (out / "tilted_sec").mkdir(parents=True, exist_ok=True)
        print(f"[collect] per-section tilt: {a.tilt_dirs} dirs/fork  R={sec_R:.0f}m  elev={a.tilt_elev:.0f}deg  focal={sec_focal:.1f}mm")
    sim.reset()

    manifest = {"window": window, "res": RES, "classes": l_map.COST_CLASSES,
                "bev_alt_m": float(H + bev_alt), "center": [float(cx), float(cy)],
                "msize": float(msize), "n": a.n,
                "depth": {"cam_alt_m": cam_alt, "z_norm_m": Z_NORM,
                          "encoding": "uint16 PNG, value/65535*z_norm_m = surface height above canyon floor"},
                "samples": []}

    for i in range(a.n):
        v, rng = variant_args(a, i)
        clear_terrain(stage)
        info = l_map.build_L(v)
        randomize_sun(stage, rng)
        for _ in range(6):                                 # let the renderer pick up the new prims
            sim.step(); cam.update(sim.get_physics_dt())

        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(np.rot90(rgb, 3))                  # BEV cam axes -> world/raster frame

        dep = cam.data.output["distance_to_image_plane"][0].detach().cpu().numpy()   # (H,W) metres
        if dep.ndim == 3:
            dep = dep[..., 0]
        height = cam_alt - dep                                        # surface height above canyon floor (z)
        hnorm = np.clip(height / Z_NORM, 0.0, 1.0)                    # 0=void floor, plateau~0.5, wall/roof higher
        hnorm = np.ascontiguousarray(np.rot90(hnorm, 3))             # SAME rot as rgb -> pixel-aligned

        label = l_map.rasterize_cost(info["elements"], window, (RES, RES),
                                     polys=info.get("polys"))                  # 0..3 class ids (world frame)
        vis = np.zeros((RES, RES, 3), np.uint8)
        for k, col in l_map.COST_RGB.items():
            vis[label == k] = col

        Image.fromarray(rgb).save(out / "rgb" / f"{i:04d}.png")
        Image.fromarray((hnorm * 65535).astype(np.uint16)).save(out / "depth" / f"{i:04d}.png")
        Image.fromarray((hnorm * 255).astype(np.uint8)).save(out / "depth_vis" / f"{i:04d}.png")
        Image.fromarray(label, mode="L").save(out / "label" / f"{i:04d}.png")
        Image.fromarray(vis).save(out / "label_vis" / f"{i:04d}.png")

        if tcam is not None:                                # capture K tilted orbit views (rgb + raw depth viz)
            tgt = torch.tensor([[float(cx), float(cy), float(H)]], device=a.device, dtype=torch.float32)
            for k, eye in enumerate(tilt_eyes):
                tcam.set_world_poses_from_view(
                    torch.tensor([list(eye)], device=a.device, dtype=torch.float32), tgt)
                for _ in range(4):
                    sim.step(); tcam.update(sim.get_physics_dt())
                trgb = tcam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
                if trgb.dtype != np.uint8:
                    trgb = (np.clip(trgb, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(np.ascontiguousarray(trgb)).save(out / "tilted" / f"{i:04d}_v{k}.png")
                td = tcam.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
                if td.ndim == 3:
                    td = td[..., 0]
                tdn = np.clip(td / (tilt_dist * 1.5), 0.0, 1.0)        # raw depth viz (reprojection/fusion = next step)
                Image.fromarray((tdn * 255).astype(np.uint8)).save(out / "tilted_vis" / f"{i:04d}_v{k}.png")

        if scam is not None:                                # per-section: visit each fork up close, tilt_dirs around it
            ez = float(H + sec_R * math.tan(math.radians(a.tilt_elev)))
            for fk, fdict in enumerate(info["forks"]):
                if a.only_fork >= 0 and fk != a.only_fork:
                    continue
                sp = np.asarray(fdict["split"], dtype=float); fv = np.asarray(fdict["dir"], dtype=float)
                fc = sp + fv * ((a.n_branch + 1) * a.sec_len / 2.0)   # fork center (xy)
                tgt = torch.tensor([[float(fc[0]), float(fc[1]), float(H)]], device=a.device, dtype=torch.float32)
                for dk in range(a.tilt_dirs):
                    az = 2.0 * math.pi * dk / a.tilt_dirs
                    eye = [float(fc[0] + sec_R * math.cos(az)), float(fc[1] + sec_R * math.sin(az)), ez]
                    scam.set_world_poses_from_view(
                        torch.tensor([eye], device=a.device, dtype=torch.float32), tgt)
                    for _ in range(4):
                        sim.step(); scam.update(sim.get_physics_dt())
                    srgb = scam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
                    if srgb.dtype != np.uint8:
                        srgb = (np.clip(srgb, 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(np.ascontiguousarray(srgb)).save(out / "tilted_sec" / f"{i:04d}_f{fk}_d{dk}.png")
        frac = {c: float((label == k).mean()) for c, k in l_map.COST_CLASSES.items()}
        manifest["samples"].append({"id": i, "fork5_types": v.fork5_types, "fork3_types": v.fork3_types,
                                    "n_obs": v.n_obs, "class_frac": frac})
        print(f"  [{i+1}/{a.n}] saved rgb+label  forkB={v.fork3_types}  forkC={v.fork5_types}  "
              f"free={frac['free']:.2f} rough={frac['rough']:.2f} tunnel={frac['tunnel']:.2f}")

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[collect] DONE. {a.n} pairs -> {out}/  (rgb/ label/ label_vis/ manifest.json)")

    if a.gui and getattr(a, "hold", False):
        print("[hold] GUI viewport open — rotate/zoom to inspect. Close the window to exit.")
        while simulation_app.is_running():
            sim.step(); cam.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
