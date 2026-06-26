"""l_map.py — elevated ㄷ (bracket) walkway with multiple Y forks + low obstacles.

Importable map builder shared by the viewer (scene_L_map.py) and renderers. No isaaclab
import at module top — `build_L` imports it lazily — so `add_map_args` is callable before
AppLauncher boots, while `build_L` runs after.

Layout (a ㄷ bracket = three arms, built with a direction-aware cursor):
  arm 1: top, +X        — lead-in, Fork A, lead-out
  corner → arm 2: -Y    — lead-in, Fork B, lead-out
  corner → arm 3: -X    — lead-in, Fork C, lead-out → goal
Each FORK splits into a SAFE lane (flat, full length) and a CLIFF lane (a few solid
entrance sections, then a void drop), with an occluder pillar at the mouth. The flat
single-lane sections are sprinkled with LOW cuboid step-obstacles. The whole path is an
elevated plateau (height H); the base ground plane is the canyon floor.
"""
from __future__ import annotations

C_GROUND = (0.86, 0.88, 0.92)
C_SEC_A = (0.22, 0.22, 0.24)        # dark gray (alternating shades so joins stay visible)
C_SEC_B = (0.30, 0.30, 0.32)
C_START = (0.20, 0.70, 0.35)
C_GOAL = (0.85, 0.30, 0.30)
C_OBST = (0.45, 0.42, 0.40)         # occluder pillars
C_STEP = (0.52, 0.40, 0.30)         # low step obstacles on the flat path
C_DMG = (0.30, 0.22, 0.20)          # collapsed (sunken) road section
C_BLOCK = (0.60, 0.22, 0.20)        # insurmountable wall block (fork C)
C_ROOF = (0.38, 0.40, 0.46)         # tunnel roof (fork C) — hides Go2 from BEV


# GT cost-map classes (for U-Net labels). Everything not explicitly free/rough/tunnel is impassable.
COST_CLASSES = {"impassable": 0, "free": 1, "rough": 2, "tunnel": 3}
COST_RGB = {0: (40, 40, 48), 1: (235, 235, 235), 2: (240, 190, 90), 3: (90, 150, 235)}  # viz colours


def rasterize_cost(elements, window, hw, polys=None):
    """Top-down GT cost-map raster (class ids) over a fixed world window, aligned to a nadir
    image: +X → column right, +Y → row up. window=(x0,x1,y0,y1), hw=(H,W).
    `polys` = optional [(klass, [4 world (x,y) corners])] for rotated (diagonal) lanes."""
    import numpy as np
    x0, x1, y0, y1 = window
    Hp, Wp = hw
    g = np.zeros((Hp, Wp), np.uint8)                      # 0 = impassable (void / wall / sunken / narrow)
    order = ["free", "rough", "wall", "tunnel"]           # later overrides earlier
    paint_val = {"free": 1, "rough": 2, "wall": 0, "tunnel": 3}
    buckets = {k: [] for k in order}
    for (klass, cx, cy, lx, ly) in elements:
        if klass in buckets:
            buckets[klass].append((cx, cy, lx, ly))
    for klass in order:
        v = paint_val[klass]
        for (cx, cy, lx, ly) in buckets[klass]:
            c0 = int((cx - lx / 2 - x0) / (x1 - x0) * Wp); c1 = int((cx + lx / 2 - x0) / (x1 - x0) * Wp)
            r0 = int((y1 - (cy + ly / 2)) / (y1 - y0) * Hp); r1 = int((y1 - (cy - ly / 2)) / (y1 - y0) * Hp)
            g[max(0, r0):max(0, min(Hp, r1)), max(0, c0):max(0, min(Wp, c1))] = v
    if polys:                                             # rotated diagonal lanes -> polygon fill
        from PIL import Image, ImageDraw
        im = Image.fromarray(g); dr = ImageDraw.Draw(im)
        for (klass, corners) in polys:
            pv = paint_val.get(klass, 0)
            pix = [((px - x0) / (x1 - x0) * Wp, (y1 - py) / (y1 - y0) * Hp) for (px, py) in corners]
            dr.polygon(pix, fill=int(pv))
        g = np.array(im, dtype=np.uint8)
    return g


def add_map_args(parser) -> None:
    """Add the map parameters to an argparse parser (safe to call pre-launch)."""
    parser.add_argument("--sec_len", type=float, default=4.0, help="section length along its arm (m)")
    parser.add_argument("--sec_w", type=float, default=4.25, help="path width (m)")
    parser.add_argument("--lead", type=int, default=2, help="lead-in/out single-lane sections around each fork")
    parser.add_argument("--n_branch", type=int, default=6, help="fork lane length in sections")
    parser.add_argument("--cliff_solid", type=int, default=4, help="solid entrance sections on the cliff lane before the drop")
    parser.add_argument("--dmg_solid", type=int, default=2, help="fork B damaged road: solid entrance sections before the collapse")
    parser.add_argument("--dmg_gap", type=int, default=2, help="fork B damaged road: sunken (collapsed) middle sections")
    parser.add_argument("--sunk_h", type=float, default=0.3, help="fork B collapsed-section top height (m, << path_h = impassable dip)")
    parser.add_argument("--arm1", type=int, default=0, help="extra sections on arm 1 (top)")
    parser.add_argument("--arm2", type=int, default=2, help="extra sections on arm 2 (vertical)")
    parser.add_argument("--arm3", type=int, default=2, help="extra sections on arm 3 (bottom)")
    parser.add_argument("--path_h", type=float, default=2.0, help="plateau height = cliff depth (m)")
    parser.add_argument("--median", type=float, default=2.0, help="gap between the two fork lanes (m)")
    # fork C = a 5-way junction: 1 safe / 2 blocked-by-wall / 1 too-narrow / 1 tunnel (hides Go2 from BEV)
    parser.add_argument("--fork5_gap", type=float, default=0.8, help="gap between the 5 lanes of fork C")
    parser.add_argument("--fork5_types", type=str, default="blocked,narrow,safe,tunnel,blocked",
                        help="comma list of 5 lane kinds (safe/blocked/narrow/tunnel/cliff)")
    parser.add_argument("--narrow_w", type=float, default=0.35, help="width of the too-narrow lane (< Go2 ~0.44 m)")
    parser.add_argument("--narrow_jitter", type=float, default=0.0,
                        help="per-lane random widening of narrow lanes, m (kept < 0.43 so GT-impassable stays honest)")
    parser.add_argument("--blocked_wall_frac", type=float, default=0.5,
                        help="where the blocked-lane wall sits along the lane (0=entrance, 1=far end). Deeper = harder for nadir")
    parser.add_argument("--hard", action="store_true",
                        help="harder preset: widen-ambiguous narrow lanes, deeper walls, more/taller step obstacles")
    parser.add_argument("--fork3_types", type=str, default="safe,blocked,cliff",
                        help="comma list of 3 lane kinds for fork B (detour junction). Center should be a hazard "
                             "so the free lane sits to a side -> the safe route detours around")
    parser.add_argument("--ravine_floor", type=float, default=0.3,
                        help="cliff fork: top-height at the bottom of the down-up ravine (m). Small = deep/steep dip")
    parser.add_argument("--wall_h", type=float, default=1.2, help="insurmountable wall height (m)")
    parser.add_argument("--tunnel_clear", type=float, default=1.0, help="tunnel roof clearance above the plateau (m)")
    parser.add_argument("--tunnel_len", type=int, default=3, help="tunnel roof length in sections")
    parser.add_argument("--no_occluder", action="store_true", help="disable the fork occluder pillars")
    parser.add_argument("--occ_r", type=float, default=0.6, help="occluder pillar radius (m)")
    parser.add_argument("--occ_h", type=float, default=1.8, help="occluder pillar height (m)")
    parser.add_argument("--no_obstacles", action="store_true", help="disable the low step obstacles")
    parser.add_argument("--n_obs", type=int, default=8, help="number of low step obstacles (random placement)")
    parser.add_argument("--obs_seed", type=int, default=0, help="obstacle layout seed (change for a new arrangement)")
    parser.add_argument("--obs_hmin", type=float, default=0.12, help="min step obstacle height (m)")
    parser.add_argument("--obs_hmax", type=float, default=0.35, help="max step obstacle height (m)")
    parser.add_argument("--debug_colors", action="store_true",
                        help="colour-code hazards (for eyeballing). Default = realistic uniform concrete "
                             "so an RGB model must use geometry/shadow, not colour.")


def build_L(a) -> dict:
    """Spawn the elevated ㄷ walkway with forks + low obstacles. Returns geometry info.
    Must be called AFTER AppLauncher boots (imports isaaclab.sim lazily)."""
    import isaaclab.sim as sim_utils
    import numpy as np

    L, W, H = a.sec_len, a.sec_w, a.path_h
    d = (W + a.median) / 2.0
    span = 2.0 * (d + W / 2.0)
    rng = np.random.default_rng(a.obs_seed)

    if getattr(a, "hard", False):                          # harder preset (only bumps if left at defaults)
        if getattr(a, "narrow_jitter", 0.0) == 0.0:
            a.narrow_jitter = 0.08
        a.blocked_wall_frac = max(getattr(a, "blocked_wall_frac", 0.5), 0.66)
        a.n_obs = max(a.n_obs, 12)
        a.obs_hmax = max(a.obs_hmax, 0.40)

    # REALISTIC by default: every surface is the same concrete with small per-piece jitter, so the
    # RGB model can't read hazard from category colour — it must use geometry / shadow / texture.
    realistic = not getattr(a, "debug_colors", False)
    _CAT = {"secA": C_SEC_A, "secB": C_SEC_B, "dmg": C_DMG, "wall": C_BLOCK,
            "roof": C_ROOF, "obst": C_OBST, "step": C_STEP}
    def terr(cat):
        if not realistic:
            return _CAT.get(cat, (0.5, 0.5, 0.5))
        j = float(rng.uniform(-0.03, 0.03))
        b = 0.10 if cat == "floor" else 0.24               # uniform GRAY: dark-gray walkway, near-black canyon pit
        return (b + j, b + j, b + j)                       # pure neutral gray (no colour cue at all)
    ground_col = (0.30, 0.29, 0.28) if realistic else C_GROUND

    def _box(path, size, pos, color):
        cfg = sim_utils.CuboidCfg(
            size=tuple(size),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0, roughness=0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        cfg.func(path, cfg, translation=tuple(pos))

    def _ramp(p0, z0, p1, z1, width, color, path):
        """A single TILTED slab: a real inclined slope (not steps) running horizontally from p0(xy)
        to p1(xy) while rising from z0 to z1. Tilt is a rotation about the lane-perpendicular axis."""
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        d = p1 - p0
        lh = float(np.hypot(d[0], d[1]))                 # horizontal run
        dz = float(z1 - z0)
        slope_len = float(np.hypot(lh, dz))              # length along the incline
        vdir = d / (lh + 1e-9)                            # horizontal unit direction
        alpha = float(np.arctan2(-dz, lh))               # tilt about perp=(-vy,vx,0)
        horiz = abs(vdir[0]) >= abs(vdir[1])
        size = (slope_len, width, 0.3) if horiz else (width, slope_len, 0.3)
        cx, cy = (p0 + p1) / 2.0
        s, cq = np.sin(alpha / 2.0), np.cos(alpha / 2.0)
        quat = (cq, -vdir[1] * s, vdir[0] * s, 0.0)      # (w,x,y,z) rotate alpha about perp axis
        cfg = sim_utils.CuboidCfg(
            size=tuple(float(x) for x in size),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0, roughness=0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        cfg.func(path, cfg, translation=(float(cx), float(cy), float((z0 + z1) / 2.0)),
                 orientation=tuple(float(q) for q in quat))

    def _disk(path, pos, color):
        cfg = sim_utils.CylinderCfg(radius=0.5, height=0.03, axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.4))
        cfg.func(path, cfg, translation=tuple(pos))

    def _pillar(path, pos, r, h, color):
        cfg = sim_utils.CylinderCfg(radius=r, height=h, axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0, roughness=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        cfg.func(path, cfg, translation=tuple(pos))

    # canyon floor: a large concrete slab (top at z=0) — replaces the default blue-grid ground plane
    _box("/World/floor", (600.0, 600.0, 0.5), (0.0, 0.0, -0.25), terr("floor"))
    # lights
    # LOW ambient fill + a strong LOW-ANGLE sun → cliffs / walls / pits cast shadows that a nadir
    # RGB camera can read (the only honest hazard cue once colours are uniform).
    sim_utils.DomeLightCfg(intensity=400.0, color=(1.0, 1.0, 1.0)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=400.0, color=(1.0, 1.0, 1.0)))
    # NEUTRAL WHITE sun (was warm) + lower intensity -> dark-gray albedo reads as gray, not washed-out tan.
    # still low-angle so cliffs/ravines/walls cast the shadows that are the only honest hazard cue.
    _sun = sim_utils.DistantLightCfg(intensity=1600.0, color=(1.0, 1.0, 1.0), angle=0.35)
    _sun.func("/World/SunLight", _sun, orientation=(0.906, 0.254, 0.338, 0.0))   # ~50° low diagonal sun

    state = {"k": 0, "pos": np.array([L / 2.0, 0.0]), "dir": np.array([1.0, 0.0]),
             "bb": [1e9, -1e9, 1e9, -1e9]}            # xmin, xmax, ymin, ymax
    flat = []            # (center, lx, ly) of flat single-lane sections (obstacle candidates)
    forks = []
    elements = []        # (klass, cx, cy, lx, ly) for GT cost-map rasterisation
                         #   free=walkable plateau / rough=step / wall=blocker / tunnel=roofed
                         #   (everything unrecorded stays impassable: void, sunken, narrow ledge)
    polys = []           # (klass, [4 world corners]) for ROTATED (diagonal) lanes -> polygon raster

    def _slab(center, lx, ly, klass="free"):
        col = terr("secA" if state["k"] % 2 == 0 else "secB")
        _box(f"/World/Lmap/s_{state['k']:03d}", (lx, ly, H), (center[0], center[1], H / 2.0), col)
        if klass != "impassable":               # impassable = leave as 0 (default), don't record
            elements.append((klass, float(center[0]), float(center[1]), lx, ly))
        bb = state["bb"]
        bb[0] = min(bb[0], center[0] - lx / 2); bb[1] = max(bb[1], center[0] + lx / 2)
        bb[2] = min(bb[2], center[1] - ly / 2); bb[3] = max(bb[3], center[1] + ly / 2)
        state["k"] += 1

    def _slab_yaw(center, length, width, yaw, klass, name):
        """A flat plateau slab rotated about Z by `yaw` (rad) — a DIAGONAL lane segment.
        Records a rotated-rectangle polygon (4 world corners) for the GT rasteriser if walkable."""
        cz, sz = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        cfg = sim_utils.CuboidCfg(
            size=(float(length), float(width), float(H)),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=terr("secB"), metallic=0.0, roughness=0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        cfg.func(name, cfg, translation=(float(center[0]), float(center[1]), float(H / 2.0)),
                 orientation=(float(cz), 0.0, 0.0, float(sz)))
        state["k"] += 1
        fwd = np.array([np.cos(yaw), np.sin(yaw)]); sd = np.array([-np.sin(yaw), np.cos(yaw)])
        hl, hw = length / 2.0, width / 2.0
        corners = [center + fwd * hl + sd * hw, center + fwd * hl - sd * hw,
                   center - fwd * hl - sd * hw, center - fwd * hl + sd * hw]
        bb = state["bb"]
        for cpt in corners:
            bb[0] = min(bb[0], cpt[0]); bb[1] = max(bb[1], cpt[0])
            bb[2] = min(bb[2], cpt[1]); bb[3] = max(bb[3], cpt[1])
        if klass != "impassable":
            polys.append((klass, [(float(p[0]), float(p[1])) for p in corners]))

    def _rough_mesh(center, lx, ly, name):
        """Procedural ROUGH heightfield (IsaacLab random_rough: 2-10 cm random bumps) laid on the plateau top.
        Reuses IsaacLab's height-field->mesh conversion; spawned as a UsdGeom.Mesh."""
        from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
        from pxr import UsdGeom, Gf
        import omni.usd
        hs = 0.10                                              # horizontal scale (m / cell)
        nr = max(3, int(lx / hs)); nc = max(3, int(ly / hs))
        levels = np.arange(0.02, 0.10 + 1e-9, 0.02)            # IsaacLab random_rough: noise_range/step
        hf = levels[rng.integers(0, len(levels), size=(nr, nc))].astype(np.float32)   # per-cell height (m)
        verts, tris = convert_height_field_to_mesh(hf, hs, 1.0, None)
        ox = float(center[0] - (nr - 1) * hs / 2.0); oy = float(center[1] - (nc - 1) * hs / 2.0)
        stage = omni.usd.get_context().get_stage()
        m = UsdGeom.Mesh.Define(stage, name)
        m.CreatePointsAttr([Gf.Vec3f(float(v[0] + ox), float(v[1] + oy), float(v[2] + H)) for v in verts])
        m.CreateFaceVertexCountsAttr([3] * len(tris))
        m.CreateFaceVertexIndicesAttr([int(i) for t in tris for i in t])
        m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)   # raw faceted heightfield (no smoothing)
        col = terr("step")
        m.CreateDisplayColorAttr([Gf.Vec3f(float(col[0]), float(col[1]), float(col[2]))])
        state["k"] += 1

    def _is_h(v):
        return abs(v[0]) > abs(v[1])

    def _lane_size(v):
        return (L, W) if _is_h(v) else (W, L)

    def run(n, collect=True):
        v = state["dir"]; lx, ly = _lane_size(v)
        for _ in range(n):
            _slab(state["pos"], lx, ly)
            if collect:
                flat.append((state["pos"].copy(), lx, ly))
            state["pos"] = state["pos"] + v * L

    def corner(new_dir):
        _slab(state["pos"], W, W)                     # square corner pad
        state["dir"] = np.array(new_dir, dtype=float)
        state["pos"] = state["pos"] + state["dir"] * L

    def _sunk(center, lx, ly):                        # collapsed road section: low slab => deep impassable dip
        _box(f"/World/Lmap/dmg_{state['k']:03d}", (lx, ly, a.sunk_h),
             (center[0], center[1], a.sunk_h / 2.0), terr("dmg"))
        state["k"] += 1

    def fork(tag, kind="cliff"):
        v = state["dir"]; perp = np.array([-v[1], v[0]])
        pad_lx, pad_ly = (L, span) if _is_h(v) else (span, L)
        split_c = state["pos"].copy()
        _slab(split_c, pad_lx, pad_ly)                # split pad
        if not a.no_occluder:
            _pillar(f"/World/Lmap/occ_{tag}", (split_c[0], split_c[1], H + a.occ_h / 2.0),
                    a.occ_r, a.occ_h, terr("obst"))
        base = split_c + v * L
        lx, ly = _lane_size(v)
        g0, g1 = a.dmg_solid, a.dmg_solid + a.dmg_gap     # collapsed span (damaged kind)
        ent_cliff = max(1, a.cliff_solid // 3)            # flat entrance length before the ravine slope
        for j in range(a.n_branch):
            c = base + v * (L * j)
            _slab(c + perp * d, lx, ly)               # SAFE lane (+perp, left) — full length, flat
            rc = c - perp * d                         # RIGHT lane (-perp)
            if kind == "cliff":
                # deceptive ravine: a short flat entrance, then a SMOOTH down-up slope built after the loop.
                if j < ent_cliff:
                    _slab(rc, lx, ly, klass="impassable")          # flat entrance ledge (looks walkable from above)
            elif kind == "damaged":                   # whole right lane impassable (sunken middle blocks route)
                (_sunk if g0 <= j < g1 else lambda c, lx, ly: _slab(c, lx, ly, klass="impassable"))(rc, lx, ly)
            elif kind == "rough":                     # right lane walkable but BUMPY (hard terrain) -> GT rough
                _slab(rc, lx, ly, klass="rough")      # solid base lane = rough class (passable, 3x cost)
                _rough_mesh(rc, lx, ly, f"/World/Lmap/rough_{state['k']:03d}")   # IsaacLab random_rough heightfield on top
        merge_c = base + v * (L * a.n_branch)
        _slab(merge_c, pad_lx, pad_ly)                # merge pad
        if kind == "cliff":
            # smooth down-up ravine (a real slope, not steps): plateau -> valley floor -> plateau.
            # GT impassable (too steep to traverse); only the shadow in the dip betrays it from nadir.
            rfloor = getattr(a, "ravine_floor", 0.3)
            s0 = base - perp * d + v * (L * (ent_cliff - 0.5))      # ravine mouth = entrance-ledge EDGE (touches it)
            s1 = base - perp * d + v * (L * (a.n_branch - 0.5))     # ravine exit  = merge-pad EDGE (touches it)
            smid = (s0 + s1) / 2.0                          # valley bottom
            col = terr("dmg")
            _ramp(s0, H, smid, rfloor, W, col, f"/World/Lmap/ramp_dn_{state['k']:03d}"); state["k"] += 1
            _ramp(smid, rfloor, s1, H, W, col, f"/World/Lmap/ramp_up_{state['k']:03d}"); state["k"] += 1
        hazard_centre = base - perp * d + v * (L * a.n_branch / 2.0)
        forks.append(dict(split=split_c.tolist(), dir=v.tolist(), kind=kind,
                          safe=(perp * d).tolist(), cliff_centre=hazard_centre.tolist()))
        state["pos"] = merge_c + v * L

    def forkN(tag, types):
        """N-way junction (N = len(types)). Lane kinds: safe/blocked/narrow/tunnel/cliff.
        Parallel lanes share one split pad + one merge pad; choosing the safe lane = a lateral detour."""
        n = len(types)
        v = state["dir"]; perp = np.array([-v[1], v[0]]); horiz = _is_h(v)
        u = W + a.fork5_gap
        offs = [(i - (n - 1) / 2.0) * u for i in range(n)]
        full = (n - 1) * u + W
        pad_lx, pad_ly = (L, full) if horiz else (full, L)
        split_c = state["pos"].copy()
        _slab(split_c, pad_lx, pad_ly)                # wide split pad (spans all 5 lanes)
        if not a.no_occluder:
            _pillar(f"/World/Lmap/occ_{tag}", (split_c[0], split_c[1], H + a.occ_h / 2.0),
                    a.occ_r, a.occ_h, terr("obst"))
        base = split_c + v * L
        lx, ly = _lane_size(v)
        nb = a.n_branch
        safe_off = None
        # GT class per fork5 lane type
        _LANE_KLASS = {"safe": "free", "blocked": "impassable",
                       "tunnel": "tunnel", "narrow": "impassable", "cliff": "impassable"}
        wall_j = int(round((nb - 1) * getattr(a, "blocked_wall_frac", 0.5)))   # deeper = harder for nadir
        for li, off in enumerate(offs):
            t = types[li] if li < len(types) else "blocked"
            lane0 = base + perp * off
            if t == "safe":
                safe_off = off
            lane_klass = _LANE_KLASS.get(t, "impassable")
            # per-lane narrow width: jittered but always < Go2 ~0.44 so the GT-impassable label stays honest
            nw = a.narrow_w
            if t == "narrow" and getattr(a, "narrow_jitter", 0.0) > 0.0:
                nw = float(min(0.43, a.narrow_w + rng.uniform(0.0, a.narrow_jitter)))
            for j in range(nb):
                c = lane0 + v * (L * j)
                if t == "narrow":                     # thin ledge — Go2 can't fit, stays impassable (0)
                    nlx, nly = (L, nw) if horiz else (nw, L)
                    _box(f"/World/Lmap/c5_{state['k']:03d}", (nlx, nly, H), (c[0], c[1], H / 2.0), terr("secB"))
                    state["k"] += 1
                elif t == "cliff":                    # deceptive: solid entrance ledge, then a drop into the void
                    if j < a.cliff_solid:
                        _slab(c, lx, ly, klass="impassable")   # looks walkable from above; GT impassable
                    # else: no slab -> void (impassable default), only a shadow betrays the drop
                else:
                    _slab(c, lx, ly, klass=lane_klass)  # correct GT class per lane
                if t == "blocked" and j == wall_j:    # insurmountable wall across the lane
                    wlx, wly = (0.5, W * 0.95) if horiz else (W * 0.95, 0.5)
                    _box(f"/World/Lmap/wall_{state['k']:03d}", (wlx, wly, a.wall_h),
                         (c[0], c[1], H + a.wall_h / 2.0), terr("wall")); state["k"] += 1
                    elements.append(("wall", float(c[0]), float(c[1]), wlx, wly))
                if t == "tunnel":                     # roof over the middle — hides Go2 from top-down BEV
                    t0 = (nb - a.tunnel_len) // 2
                    if t0 <= j < t0 + a.tunnel_len:
                        rlx, rly = (L, W) if horiz else (W, L)
                        _box(f"/World/Lmap/roof_{state['k']:03d}", (rlx, rly, 0.2),
                             (c[0], c[1], H + a.tunnel_clear), terr("roof")); state["k"] += 1
                        elements.append(("tunnel", float(c[0]), float(c[1]), rlx, rly))
        merge_c = base + v * (L * nb)
        _slab(merge_c, pad_lx, pad_ly)                # wide merge pad
        forks.append(dict(split=split_c.tolist(), dir=v.tolist(), kind=f"fork{n}", types=types,
                          safe=(perp * (safe_off or 0)).tolist(), cliff_centre=merge_c.tolist()))
        state["pos"] = merge_c + v * L

    def fork_detour(tag):
        """3-way DETOUR junction. straight(center)=blocked wall, the FREE route is a '>'-shaped
        detour that bulges out to one side and returns, the opposite side = a hazard lane."""
        v = state["dir"]; perp = np.array([-v[1], v[0]]); horiz = _is_h(v)
        nb = a.n_branch
        lx, ly = _lane_size(v)
        gap = a.fork5_gap
        sgn = 1.0                                          # bulge OUTWARD (+perp) — away from the ㄷ interior
        hazard = "narrow" if (a.obs_seed % 2 == 0) else "cliff"
        bulge = (W + gap) * sgn                            # lateral peak of the '>'
        span = 2.0 * (abs(bulge) + W)
        pad_lx, pad_ly = (L, span) if horiz else (span, L)
        split_c = state["pos"].copy()
        _slab(split_c, pad_lx, pad_ly)                    # split pad (spans the whole junction)
        if not a.no_occluder:
            _pillar(f"/World/Lmap/occ_{tag}", (split_c[0], split_c[1], H + a.occ_h / 2.0),
                    a.occ_r, a.occ_h, terr("obst"))
        base = split_c + v * L
        merge_c = base + v * (L * nb)
        # --- '>' FREE detour: two diagonal slabs  split-edge -> side peak -> merge-edge ---
        A = split_c + v * (L * 0.5)                        # touches the split pad
        Bpt = merge_c - v * (L * 0.5)                      # touches the merge pad
        P = (A + Bpt) / 2.0 + perp * bulge                 # the '>' peak, out to the side
        for ki, (p0, p1) in enumerate([(A, P), (P, Bpt)]):
            mid = (p0 + p1) / 2.0
            dd = p1 - p0; seg_len = float(np.hypot(dd[0], dd[1])); yaw = float(np.arctan2(dd[1], dd[0]))
            _slab_yaw(mid, seg_len, W * 0.9, yaw, "free", f"/World/Lmap/det{ki}_{tag}_{state['k']:03d}")
        # --- straight CENTER lane: looks walkable but a wall blocks it (GT impassable) ---
        for j in range(nb):
            c = base + v * (L * j)
            _slab(c, lx, ly, klass="impassable")
            if j == nb // 2:
                wlx, wly = (0.5, W * 0.95) if horiz else (W * 0.95, 0.5)
                _box(f"/World/Lmap/wallB_{state['k']:03d}", (wlx, wly, a.wall_h),
                     (c[0], c[1], H + a.wall_h / 2.0), terr("wall")); state["k"] += 1
                elements.append(("wall", float(c[0]), float(c[1]), wlx, wly))
        # --- opposite-side HAZARD lane (narrow ledge or cliff drop) ---
        hz0 = base - perp * ((W + gap) * sgn)
        for j in range(nb):
            c = hz0 + v * (L * j)
            if hazard == "narrow":
                nlx, nly = (L, a.narrow_w) if horiz else (a.narrow_w, L)
                _box(f"/World/Lmap/hzn_{state['k']:03d}", (nlx, nly, H), (c[0], c[1], H / 2.0), terr("secB"))
                state["k"] += 1
            elif j < max(1, a.cliff_solid // 2):           # cliff: short solid entrance, then void
                _slab(c, lx, ly, klass="impassable")
        _slab(merge_c, pad_lx, pad_ly)                    # merge pad
        forks.append(dict(split=split_c.tolist(), dir=v.tolist(), kind="detour3",
                          safe=(perp * bulge).tolist(), cliff_centre=merge_c.tolist()))
        state["pos"] = merge_c + v * L

    # ── build the ㄷ ────────────────────────────────────────────────────────
    start = state["pos"].copy()
    run(a.lead); fork("R", kind="rough"); run(a.lead)   # easy-vs-ROUGH terrain fork at the very start (2-way)
    run(a.arm1); fork("A"); run(a.lead)                 # arm 1 (top, +X): cliff/ravine fork (2-way)
    corner((0.0, -1.0))
    run(a.lead + a.arm2)                                # arm 2 (vertical, -Y): 3-way '>' detour junction
    fork_detour("B"); run(a.lead)
    corner((-1.0, 0.0))
    run(a.lead + a.arm3)                                # arm 3 (bottom, -X): 5-way junction
    forkN("C", [t.strip() for t in a.fork5_types.split(",")]); run(a.lead)
    goal = state["pos"] - state["dir"] * L

    _disk("/World/start", (start[0], start[1], H + 0.04), C_START)
    _disk("/World/goal", (goal[0], goal[1], H + 0.04), C_GOAL)

    # ── N low step obstacles, randomly placed on the flat single-lane sections ──
    n_obs = 0
    if not a.no_obstacles and a.n_obs > 0 and flat:
        import math as _math
        far_flat = [f for f in flat
                    if _math.hypot(f[0][0] - start[0], f[0][1] - start[1]) > 8.0]  # skip start-zone
        candidates = far_flat if far_flat else flat
        order = list(range(len(candidates)))
        rng.shuffle(order)
        pick = [order[i % len(order)] for i in range(a.n_obs)]      # allow repeats if N > #flat
        for k_i, i in enumerate(pick):
            c, lx, ly = candidates[i]
            ox = float(rng.uniform(-0.30, 0.30)) * lx                # off-centre, stays inside the section
            oy = float(rng.uniform(-0.30, 0.30)) * ly
            sx = float(rng.uniform(0.4, 0.8)); sy = float(rng.uniform(0.4, 0.8))
            h = float(rng.uniform(a.obs_hmin, a.obs_hmax))
            _box(f"/World/Lmap/obs_{k_i:03d}", (sx, sy, h), (c[0] + ox, c[1] + oy, H + h / 2.0), terr("step"))
            elements.append(("rough", float(c[0] + ox), float(c[1] + oy), sx, sy))
            n_obs += 1

    bb = state["bb"]
    center = ((bb[0] + bb[1]) / 2.0, (bb[2] + bb[3]) / 2.0)
    size = max(bb[1] - bb[0], bb[3] - bb[2])
    f0 = forks[0]
    return dict(n_sections=state["k"], n_forks=len(forks), n_obstacles=n_obs,
                start=start.tolist(), goal=goal.tolist(), H=H, sec_w=W, forks=forks,
                bbox=bb, center=center, size=size, elements=elements, polys=polys,
                # first-fork keys kept for downstream renderers
                safe_y=f0["safe"][1], cliff_y=-f0["safe"][1],
                fork_center=f0["split"], cliff_void_x=(f0["cliff_centre"][0] - L, f0["cliff_centre"][0] + L))
