"""
Shared greenhouse layout: row/aisle positions, the connected rail-track network,
and the robot's route through it. Used by build_scene.py (to draw the rails)
and view_scene.py (to drive the cart along them).

Track network (top view, x to the right):

        headland cross-track              aisles (pipe rails between plant rows)
            |
   row 3 ===|=============================================== (plants)
            ╭───────────────────────────────────────────────  aisle 2
   row 2 ===|===============================================
            ├───────────────────────────────────────────────  aisle 1
   row 1 ===|===============================================
            ╰───────────────────────────────────────────────  aisle 0
   row 0 ===================================================
"""
from __future__ import annotations

import math

import numpy as np

CONFIG = dict(
    n_rows=4,               # plant rows (tabletop gutters); aisles = n_rows - 1
    row_length=12.0,
    row_spacing=1.5,
    gutter_half_width=0.15,
    gutter_top=1.00,
    plant_spacing=0.25,
    ripe_fraction=0.70,     # ~70 % ripe (red), ~30 % unripe (green / pale)
    fruit_on_outer_sides=False,  # outer faces of the edge rows face the wall, no track there
    robot_start_x=3.0,      # pose used for the still-image cameras
    seed=11,
)

RAIL_GAUGE = 0.55        # distance between the two pipes
RAIL_Z = 0.10            # pipe centre height
RAIL_R = 0.025           # pipe radius
TURN_R = 0.75            # centre-line radius of the curves (= half the row spacing)
X_CROSS = -1.25          # x of the headland cross-track centre line
X_AISLE0 = X_CROSS + TURN_R   # where the straight aisle rails start
X_END_MARGIN = 0.45      # how close to the far end the cart may drive


def row_y(i):
    return i * CONFIG["row_spacing"]


def aisle_y(i):
    return (row_y(i) + row_y(i + 1)) / 2


def n_aisles():
    return CONFIG["n_rows"] - 1


def greenhouse_bounds():
    """(x_min, x_max, y_min, y_max) of the greenhouse walls."""
    L = CONFIG["row_length"]
    return -2.1, L + 1.0, row_y(0) - 1.4, row_y(CONFIG["n_rows"] - 1) + 1.4


def full_crate_spots():
    """Where full crates are stacked at the headland after a crate swap (x, y, z), 2 layers x 3."""
    x_min = greenhouse_bounds()[0]
    return [(x_min + 0.25, aisle_y(0) + 0.7 + 0.45 * (k % 3), 0.131 * (k // 3)) for k in range(6)]


# ----------------------------------------------------------------------------
# track geometry (centre lines) — each piece is a polyline of (x, y) points
# ----------------------------------------------------------------------------
def _arc(cx, cy, a0, a1, n=24):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + TURN_R * np.cos(a), cy + TURN_R * np.sin(a)], axis=1)


def curve_aisle_to_cross(a, direction):
    """Curve from aisle `a` (at x=X_AISLE0) onto the cross-track, heading +y (direction=+1) or -y (-1)."""
    ya = aisle_y(a)
    if direction > 0:
        return _arc(X_AISLE0, ya + TURN_R, -math.pi / 2, -math.pi)
    return _arc(X_AISLE0, ya - TURN_R, math.pi / 2, math.pi)


def track_pieces():
    """All centre-line polylines of the network (for drawing rails)."""
    L = CONFIG["row_length"]
    pieces = []
    for a in range(n_aisles()):
        pieces.append(np.array([[X_AISLE0, aisle_y(a)], [L, aisle_y(a)]]))
        if a < n_aisles() - 1:
            pieces.append(curve_aisle_to_cross(a, +1))
        if a > 0:
            pieces.append(curve_aisle_to_cross(a, -1))
    if n_aisles() > 1:
        y0, y1 = aisle_y(0) + TURN_R, aisle_y(n_aisles() - 1) - TURN_R
        if y1 - y0 > 1e-6:
            pieces.append(np.array([[X_CROSS, y0], [X_CROSS, y1]]))
    return pieces


def offset_polyline(p, d):
    """Offset a polyline sideways by distance d (left of travel direction positive)."""
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return p + d * nrm


# ----------------------------------------------------------------------------
# route: dense list of (x, y, yaw) poses along the track
# ----------------------------------------------------------------------------
def _densify(poly, step=0.01):
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(out)


def _transfer(a_from, a_to):
    """Headland -> curve -> cross-track -> curve -> into next aisle (ends at X_AISLE0 of a_to)."""
    d = 1 if a_to > a_from else -1
    c1 = curve_aisle_to_cross(a_from, d)
    c2 = curve_aisle_to_cross(a_to, -d)[::-1]
    return np.vstack([c1, c2])


class Route:
    """The harvesting route: for each aisle drive in, harvest, back out, transfer to the next aisle."""

    def __init__(self, stop_spacing=0.9):
        L = CONFIG["row_length"]
        x_far = L - X_END_MARGIN
        legs = []        # list of polylines
        self.stops = []  # (s, aisle) at which the robot harvests
        for a in range(n_aisles()):
            ya = aisle_y(a)
            legs.append(("in", a, np.array([[X_AISLE0, ya], [x_far, ya]])))
            legs.append(("out", a, np.array([[x_far, ya], [X_AISLE0, ya]])))
            nxt = a + 1 if a < n_aisles() - 1 else 0
            if n_aisles() > 1:
                legs.append(("transfer", a, _transfer(a, nxt)))

        pts, starts = [], []
        for kind, a, poly in legs:
            starts.append(len(pts))
            for p in _densify(poly):
                if not pts or np.linalg.norm(p - pts[-1]) > 1e-6:
                    pts.append(p)
        self.xy = np.array(pts)
        self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))])
        self.length = self.s[-1]
        self.legs = []
        for i, (kind, a, _) in enumerate(legs):
            s0 = self.s[min(starts[i], len(self.s) - 1)]
            s1 = self.s[starts[i + 1]] if i + 1 < len(legs) else self.length
            self.legs.append((kind, a, s0, s1))
            if kind == "in":
                for x in np.arange(0.6, x_far + 1e-6, stop_spacing):
                    self.stops.append((s0 + (x - X_AISLE0), a))

        # yaw = track tangent, unwrapped modulo pi (cart behaves like a train: it
        # reverses rather than turning around, so yaw never jumps by pi)
        t = np.gradient(self.xy, axis=0)
        raw = np.arctan2(t[:, 1], t[:, 0])
        yaw = np.zeros_like(raw)
        prev = 0.0
        for i, r in enumerate(raw):
            k = np.round((prev - r) / math.pi)
            prev = r + k * math.pi
            yaw[i] = prev
        self.yaw = yaw

        # motion heading (direction of increasing s) — used to know which bogie leads
        self.heading = np.unwrap(raw)
        self._build_chains()

    def pose(self, s):
        """Point on the track centre line at route position s, with train-style yaw."""
        s = float(np.clip(s, 0, self.length))
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        yaw = np.interp(s, self.s, self.yaw)
        return x, y, yaw

    # ------------------------------------------------------------------
    # two-bogie vehicle on the track
    # ------------------------------------------------------------------
    def _build_chains(self):
        """For each leg: the physical track around it (neighbouring legs joined unless the
        cart reverses there), parameterised with the same s as the route."""
        self.chains = []
        n = len(self.legs)
        for i, (kind, a, s0, s1) in enumerate(self.legs):
            m = (self.s >= s0 - 1e-9) & (self.s <= s1 + 1e-9)
            parts_xy, parts_s = [self.xy[m]], [self.s[m]]
            prev_kind = self.legs[(i - 1) % n][0]
            next_kind = self.legs[(i + 1) % n][0]
            if not (prev_kind == "in" and kind == "out"):       # no reversal at the start
                pk, _, ps0, ps1 = self.legs[(i - 1) % n]
                pm = (self.s >= max(ps0, ps1 - 1.5)) & (self.s < ps1 - 1e-9)
                off = s0 - ps1 if i > 0 else -ps1                 # wrap-around for the first leg
                parts_xy.insert(0, self.xy[pm]); parts_s.insert(0, self.s[pm] + off)
            if not (kind == "in" and next_kind == "out"):       # no reversal at the end
                _, _, ns0, ns1 = self.legs[(i + 1) % n]
                nm = (self.s > ns0 + 1e-9) & (self.s <= min(ns1, ns0 + 1.5))
                off = s1 - ns0 if i < n - 1 else self.length - ns0 + (s1 - self.length)
                parts_xy.append(self.xy[nm]); parts_s.append(self.s[nm] + off)
            self.chains.append((s0, s1, np.vstack(parts_xy), np.concatenate(parts_s)))

    @staticmethod
    def _lookup(cxy, cs, s):
        """Point on a chain at parameter s, extrapolating straight past its ends."""
        if s < cs[0]:
            t = cxy[1] - cxy[0]; t /= np.linalg.norm(t)
            return cxy[0] + t * (s - cs[0])
        if s > cs[-1]:
            t = cxy[-1] - cxy[-2]; t /= np.linalg.norm(t)
            return cxy[-1] + t * (s - cs[-1])
        return np.array([np.interp(s, cs, cxy[:, 0]), np.interp(s, cs, cxy[:, 1])])

    def vehicle_pose(self, s, half_wheelbase):
        """Pose of a two-bogie rail vehicle whose bogie pivots sit on the track.
        Returns (x, y, yaw, front_bogie_rel_yaw, rear_bogie_rel_yaw)."""
        s = float(np.clip(s, 0, self.length))
        k = next((i for i, c in enumerate(self.chains) if c[0] - 1e-9 <= s <= c[1] + 1e-9), len(self.chains) - 1)
        _, _, cxy, cs = self.chains[k]
        _, _, yaw0 = self.pose(s)
        heading = np.interp(s, self.s, self.heading)
        sigma = 1.0 if math.cos(heading - yaw0) >= 0 else -1.0   # +1: cart +x points along increasing s
        e = 0.02
        pf = self._lookup(cxy, cs, s + sigma * half_wheelbase)
        pr = self._lookup(cxy, cs, s - sigma * half_wheelbase)
        c = (pf + pr) / 2
        yaw = math.atan2(*(pf - pr)[::-1])
        yaw += 2 * math.pi * round((yaw0 - yaw) / (2 * math.pi))

        def track_yaw(sq):
            d = self._lookup(cxy, cs, sq + sigma * e) - self._lookup(cxy, cs, sq - sigma * e)
            ty = math.atan2(d[1], d[0])
            return ty + math.pi * round((yaw - ty) / math.pi)

        yf = track_yaw(s + sigma * half_wheelbase) - yaw
        yr = track_yaw(s - sigma * half_wheelbase) - yaw
        return c[0], c[1], yaw, yf, yr
