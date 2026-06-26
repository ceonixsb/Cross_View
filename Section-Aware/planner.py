"""planner.py — GT cost-map → distance-field → velocity guidance for the Y-Road deploy.

NO learning here. We KNOW the map (l_map `elements`), so we rasterise the ground-truth 4-class
cost map, run Dijkstra from the GOAL to get a distance-to-goal field over walkable cells, and
each control step turn the field's steepest-descent direction at the robot into a body-frame
(vx, vy, vyaw) command for the rl_lab locomotion policy. Cliff edges / walls / voids are
impassable; an edge penalty keeps the path clear of drop-offs.

Later milestone: swap `rasterize_cost(GT elements)` for the U-Net prediction — nothing else changes.

    from planner import GTPlanner
    pl = GTPlanner(info["elements"], info["bbox"], info["goal"][:2])
    vx, vy, vyaw, reached = pl.command(rx, ry, yaw, info["goal"][:2])
"""
from __future__ import annotations

import heapq
import math

import numpy as np

# class id (l_map.COST_CLASSES) -> traversal cost. None = impassable.
#   ONLY open ground (free) is cheap. rough is bumpy, tunnel is BEV-occluded/closed-in -> both avoided
#   when an open detour exists, but still WALKABLE (finite cost) so they're used when they're the only route.
CLASS_COST = {0: None, 1: 1.0, 2: 4.0, 3: 3.0}   # free · tunnel costly · rough costliest (rough is bumpiest)
#   both stay finite (not impassable) so they never push the path onto a cliff edge; just dispreferred.

# 8-connectivity: (drow, dcol, step_factor)
_NEI = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.41421), (-1, 1, 1.41421), (1, -1, 1.41421), (1, 1, 1.41421)]


class GTPlanner:
    def __init__(self, elements, bbox, goal_xy, cell=0.15, margin=2.0, edge_clear=0.7, polys=None):
        import l_map
        x0, x1 = bbox[0] - margin, bbox[1] + margin
        y0, y1 = bbox[2] - margin, bbox[3] + margin
        self.window = (x0, x1, y0, y1)
        self.cell = float(cell)
        self.Wp = max(2, int(round((x1 - x0) / cell)))
        self.Hp = max(2, int(round((y1 - y0) / cell)))
        # MUST pass polys: rotated diagonal lanes (e.g. fork_detour's '>' free path) live in polys,
        # not elements. Omitting them blanks out the only walkable route through a detour -> graph split.
        self.cls = l_map.rasterize_cost(elements, self.window, (self.Hp, self.Wp), polys=polys).astype(np.int16)

        cost = np.full((self.Hp, self.Wp), np.inf, np.float32)        # base cost per class
        for k, c in CLASS_COST.items():
            if c is not None:
                cost[self.cls == k] = c
        self._add_edge_penalty(cost, edge_clear)
        self.cost = cost

        self.goal_cell = self.world_to_cell(*goal_xy)
        self.dist = self._dijkstra(self.goal_cell)

    @classmethod
    def from_class_grid(cls, cls_grid, window, goal_xy, edge_clear=0.7):
        """Build a planner directly from a class-id grid (e.g. the U-Net prediction of the BEV)
        instead of GT elements. `cls_grid` is HxW with ids 0..3 over `window`=(x0,x1,y0,y1) in the
        rasterize_cost frame (+X->col right, +Y->row up). Everything downstream is identical to GT."""
        self = cls.__new__(cls)
        self.window = tuple(float(w) for w in window)
        self.Hp, self.Wp = cls_grid.shape
        self.cell = (self.window[1] - self.window[0]) / self.Wp
        self.cls = np.asarray(cls_grid, dtype=np.int16)
        cost = np.full(self.cls.shape, np.inf, np.float32)
        for k, c in CLASS_COST.items():
            if c is not None:
                cost[self.cls == k] = c
        self._add_edge_penalty(cost, edge_clear)
        self.cost = cost
        self.goal_cell = self.world_to_cell(*goal_xy)
        self.dist = self._dijkstra(self.goal_cell)
        return self

    # ── cost shaping ──────────────────────────────────────────────────────────
    def _add_edge_penalty(self, cost, edge_clear):
        """Add a soft cost ramp near impassable cells so the route keeps clearance from cliffs/walls."""
        impass = ~np.isfinite(cost)
        try:
            from scipy.ndimage import distance_transform_edt
            d_m = distance_transform_edt(~impass) * self.cell         # metres to nearest impassable
            pen = np.clip((edge_clear - d_m) / max(edge_clear, 1e-6), 0.0, 1.0) * 12.0   # push the path off edges
            cost += np.where(np.isfinite(cost), pen.astype(np.float32), 0.0)
        except Exception:                                             # numpy-only dilation fallback
            rings = max(1, int(round(edge_clear / self.cell)))
            grow = impass.copy()
            for _ in range(rings):
                g = np.zeros_like(grow)
                g[1:, :] |= grow[:-1, :]; g[:-1, :] |= grow[1:, :]
                g[:, 1:] |= grow[:, :-1]; g[:, :-1] |= grow[:, 1:]
                grow = g
            band = grow & ~impass
            cost[band] += 5.0

    # ── coordinate transforms (nadir-aligned: +X→col right, +Y→row up) ────────
    def world_to_cell(self, x, y):
        x0, x1, y0, y1 = self.window
        c = int((x - x0) / (x1 - x0) * self.Wp)
        r = int((y1 - y) / (y1 - y0) * self.Hp)
        return (min(max(r, 0), self.Hp - 1), min(max(c, 0), self.Wp - 1))

    def cell_to_world(self, r, c):
        x0, x1, y0, y1 = self.window
        x = x0 + (c + 0.5) / self.Wp * (x1 - x0)
        y = y1 - (r + 0.5) / self.Hp * (y1 - y0)
        return (x, y)

    # ── planning ──────────────────────────────────────────────────────────────
    def _dijkstra(self, src):
        """Vectorised C-level Dijkstra over the grid (scipy.csgraph). Edge weight between adjacent
        cells = step * mean(cost) ; undirected. Pure-Python heapq was O(100s) at 0.15 m / ~150k cells."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra as _csd
        Hp, Wp = self.Hp, self.Wp
        cost = self.cost
        finite = np.isfinite(cost)
        idx = np.arange(Hp * Wp, dtype=np.int64).reshape(Hp, Wp)
        rows, cols, wts = [], [], []
        for dr, dc, w in _NEI:
            r0, r1 = max(0, -dr), Hp - max(0, dr)
            c0, c1 = max(0, -dc), Wp - max(0, dc)
            if r1 <= r0 or c1 <= c0:
                continue
            s_fin = finite[r0:r1, c0:c1]
            t_fin = finite[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            m = s_fin & t_fin
            if not m.any():
                continue
            s_cost = cost[r0:r1, c0:c1][m]
            t_cost = cost[r0 + dr:r1 + dr, c0 + dc:c1 + dc][m]
            rows.append(idx[r0:r1, c0:c1][m])
            cols.append(idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][m])
            wts.append(w * self.cell * 0.5 * (s_cost + t_cost))
        n = Hp * Wp
        if rows:
            g = csr_matrix((np.concatenate(wts),
                            (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        else:
            g = csr_matrix((n, n))
        sr, sc = src
        if not finite[sr, sc]:
            sr, sc = self._nearest_finite(sr, sc, key=self.cost)
        d = _csd(g, directed=False, indices=int(sr * Wp + sc))
        return d.reshape(Hp, Wp).astype(np.float32)

    def _nearest_finite(self, r, c, rad=30, key=None):
        arr = self.dist if key is None else key
        best, bd = None, 1e18
        r0, r1 = max(0, r - rad), min(self.Hp, r + rad + 1)
        c0, c1 = max(0, c - rad), min(self.Wp, c + rad + 1)
        sub = arr[r0:r1, c0:c1]
        ys, xs = np.nonzero(np.isfinite(sub))
        for yy, xx in zip(ys, xs):
            dd = (yy + r0 - r) ** 2 + (xx + c0 - c) ** 2
            if dd < bd:
                bd, best = dd, (yy + r0, xx + c0)
        return best if best else (r, c)

    def reachable(self, x, y):
        r, c = self.world_to_cell(x, y)
        return bool(np.isfinite(self.dist[r, c]))

    def lookahead_target(self, x, y, ahead_m=1.3):
        """Descend the distance field ~ahead_m metres → a smooth carrot point (world xy)."""
        r, c = self.world_to_cell(x, y)
        if not np.isfinite(self.dist[r, c]):
            r, c = self._nearest_finite(r, c)
        for _ in range(max(1, int(ahead_m / self.cell))):
            best, nr, nc = self.dist[r, c], r, c
            for dr, dc, _w in _NEI:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.Hp and 0 <= cc < self.Wp and self.dist[rr, cc] < best:
                    best, nr, nc = self.dist[rr, cc], rr, cc
            if (nr, nc) == (r, c):
                break
            r, c = nr, nc
        return self.cell_to_world(r, c)

    def command(self, x, y, yaw, goal_xy, vx_max=1.0, vyaw_max=1.0, goal_tol=0.6):
        """Body-frame velocity toward the safe-path carrot, slowing BEFORE corners so the Go2 does not
        overshoot a sharp turn off a plateau edge. Returns (vx, vy, vyaw, reached)."""
        gx, gy = goal_xy
        if math.hypot(gx - x, gy - y) < goal_tol:
            return 0.0, 0.0, 0.0, True
        nx, ny = self.lookahead_target(x, y, ahead_m=0.9)             # near carrot to steer toward
        fx, fy = self.lookahead_target(x, y, ahead_m=3.5)             # far carrot reveals the upcoming bend
        desired = math.atan2(ny - y, nx - x)
        err = (desired - yaw + math.pi) % (2 * math.pi) - math.pi
        a_far = math.atan2(fy - ny, fx - nx)
        bend = abs((a_far - desired + math.pi) % (2 * math.pi) - math.pi)   # path curvature ahead
        vyaw = max(-vyaw_max, min(vyaw_max, 1.6 * err))
        face = max(0.0, math.cos(err))                                # slow if not facing the carrot
        bend_slow = 1.0 - 0.6 * min(1.0, bend / (math.pi / 2))        # slow approaching a corner (keep momentum)
        vx = vx_max * face * bend_slow
        return float(vx), 0.0, float(vyaw), False

    # ── visualisation ─────────────────────────────────────────────────────────
    def render(self, start_xy=None, goal_xy=None):
        """RGB image of the GT cost map (class colours) with the descent path + start/goal marked."""
        from l_map import COST_RGB
        img = np.zeros((self.Hp, self.Wp, 3), np.uint8)
        for k, rgb in COST_RGB.items():
            img[self.cls == k] = rgb
        if start_xy is not None:                                      # magenta steepest-descent path
            r, c = self.world_to_cell(*start_xy)
            if not np.isfinite(self.dist[r, c]):
                r, c = self._nearest_finite(r, c)
            for _ in range(self.Hp * self.Wp):
                img[r, c] = (255, 0, 255)
                best, nr, nc = self.dist[r, c], r, c
                for dr, dc, _w in _NEI:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.Hp and 0 <= cc < self.Wp and self.dist[rr, cc] < best:
                        best, nr, nc = self.dist[rr, cc], rr, cc
                if (nr, nc) == (r, c):
                    break
                r, c = nr, nc

        def _mark(xy, col):
            if xy is None:
                return
            r, c = self.world_to_cell(*xy)
            img[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = col
        _mark(start_xy, (0, 255, 0))
        _mark(goal_xy, (255, 60, 60))
        return img
