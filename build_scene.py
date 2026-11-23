"""
Procedural greenhouse + harvesting-robot scene generator for MuJoCo.

Writes scene/greenhouse.xml (+ meshes/textures in scene/assets).
Layout values (rows, lengths, ripeness ratio, ...) live in layout.py -> CONFIG.

    python build_scene.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import meshes
import robot
from layout import (CONFIG, full_crate_spots, RAIL_GAUGE, RAIL_R, RAIL_Z, X_AISLE0, X_CROSS, aisle_y, greenhouse_bounds,
                    n_aisles, offset_polyline, row_y, track_pieces)

OUT = Path(__file__).parent / "scene" / "greenhouse.xml"

# fruit / leaf mesh sizes (metres)
BERRY_SIZES = {"L": 0.044, "M": 0.038, "S": 0.029, "XS": 0.023}   # fruit length
LEAFLET_SIZES = {"s": 0.050, "m": 0.062, "l": 0.074}              # leaflet length
BERRY_CENTER, BERRY_COL_R = 0.42, 0.44    # fruit centre below the top / contact sphere radius (x length)
META = Path(__file__).parent / "scene" / "berries.json"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def f(*v) -> str:
    out = []
    for x in v:
        s = f"{x:.4f}".rstrip("0").rstrip(".")
        out.append("0" if s in ("-0", "") else s)
    return " ".join(out)


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def q_axis(axis, ang):
    s = math.sin(ang / 2)
    return (math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def q_ypr(yaw, pitch, roll=0.0):
    """Rz(yaw) * Ry(pitch) * Rx(roll); positive pitch tips +x downward."""
    return qmul(qmul(q_axis((0, 0, 1), yaw), q_axis((0, 1, 0), pitch)), q_axis((1, 0, 0), roll))


def q_hang(d, spin):
    """Rotate local -z onto direction d, then spin about the local z axis."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    down = np.array([0.0, 0.0, -1.0])
    ax = np.cross(down, d)
    s = np.linalg.norm(ax)
    ang = math.atan2(s, float(np.dot(down, d)))
    q = q_axis(tuple(ax / s), ang) if s > 1e-9 else (1, 0, 0, 0)
    return qmul(q, q_axis((0, 0, 1), spin))


def look_at_xyaxes(pos, target, up=(0, 0, 1)):
    p, t, u = map(np.asarray, (pos, target, up))
    fwd = (t - p) / np.linalg.norm(t - p)
    x = np.cross(fwd, u); x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return f(*x, *y)


def bezier(pts, n):
    pts = np.asarray(pts, float)
    t = np.linspace(0, 1, n)[:, None]
    if len(pts) == 3:
        return (1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t ** 2 * pts[2]
    return ((1 - t) ** 3 * pts[0] + 3 * (1 - t) ** 2 * t * pts[1]
            + 3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3])


def chain(points, radius, material):
    return [f'<geom type="capsule" fromto="{f(*a, *b)}" size="{f(radius)}" material="{material}"/>'
            for a, b in zip(points[:-1], points[1:])]


# ----------------------------------------------------------------------------
# assets
# ----------------------------------------------------------------------------
def build_assets(gable_w, gable_h, plant_chunks):
    berry_meshes = "\n".join(
        f'    <mesh name="berry_{k}" file="berry.obj" scale="{f(s, s, s)}" inertia="shell"/>'
        for k, s in BERRY_SIZES.items())
    leaf_meshes = "\n".join(
        f'    <mesh name="{name}" file="{name}.obj" inertia="shell"/>' for name in plant_chunks)
    hw = gable_w / 2
    gable = f(-0.006, -hw, 0, 0.006, -hw, 0, -0.006, hw, 0, 0.006, hw, 0, -0.006, 0, gable_h, 0.006, 0, gable_h)
    return f"""
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.62 0.78 0.95" rgb2="0.95 0.97 1" width="512" height="512"/>
    <texture name="groundsheet" type="2d" builtin="checker" rgb1="0.88 0.88 0.85" rgb2="0.84 0.84 0.81" width="512" height="512"/>
    <texture name="concrete" type="2d" builtin="flat" rgb1="0.62 0.62 0.6" rgb2="0.5 0.5 0.48" width="64" height="64" random="0.25"/>
    <material name="floor" texture="groundsheet" texrepeat="40 40" reflectance="0.04"/>
    <material name="walkway" texture="concrete" texrepeat="6 20"/>

{berry_meshes}
{leaf_meshes}
    <mesh name="gable" vertex="{gable}"/>
    <mesh name="rails" file="rails.obj" inertia="shell"/>

    <texture name="tex_ripe" type="2d" file="berry_ripe.png"/>
    <texture name="tex_ripe_dark" type="2d" file="berry_ripe_dark.png"/>
    <texture name="tex_unripe" type="2d" file="berry_unripe.png"/>
    <texture name="tex_pale" type="2d" file="berry_pale.png"/>
    <texture name="tex_plant" type="2d" file="plant_atlas.png"/>
    <material name="berry_ripe"      texture="tex_ripe"      specular="0.7" shininess="0.75"/>
    <material name="berry_ripe_dark" texture="tex_ripe_dark" specular="0.7" shininess="0.75"/>
    <material name="berry_unripe"    texture="tex_unripe"    specular="0.5" shininess="0.6"/>
    <material name="berry_pale"      texture="tex_pale"      specular="0.5" shininess="0.6"/>
    <material name="plant" texture="tex_plant" specular="0.25" shininess="0.35"/>

    <material name="gutter" rgba="0.72 0.74 0.76 1" specular="0.6" shininess="0.5"/>
    <material name="substrate" rgba="0.94 0.94 0.91 1" specular="0.1"/>
    <material name="steel" rgba="0.55 0.57 0.6 1" specular="0.7" shininess="0.6"/>
    <material name="frame" rgba="0.82 0.83 0.85 1" specular="0.5"/>
    <material name="kneewall" rgba="0.78 0.79 0.78 1"/>
    <material name="film" rgba="0.90 0.93 0.92 0.93" specular="0.2" emission="0.45"/>
    <material name="roof" rgba="0.92 0.95 0.97 0.85" specular="0.2" emission="0.6"/>
    <material name="door" rgba="0.35 0.45 0.4 1"/>
    <material name="pipe" rgba="0.80 0.80 0.82 1" specular="0.9" shininess="0.8"/>
    <material name="dripline" rgba="0.08 0.08 0.08 1"/>
    <material name="lamp" rgba="0.2 0.2 0.22 1"/>
    <material name="lamp_glow" rgba="1 0.95 0.8 1" emission="1"/>
    <material name="crate_stack" rgba="0.12 0.42 0.75 1"/>

{robot.robot_assets()}
    <material name="robot_white"  rgba="0.93 0.93 0.94 1" specular="0.5" shininess="0.5"/>
    <material name="robot_orange" rgba="0.95 0.45 0.08 1" specular="0.5" shininess="0.5"/>
    <material name="robot_dark"   rgba="0.15 0.16 0.18 1" specular="0.4"/>
    <material name="crate"        rgba="0.12 0.42 0.75 1" specular="0.3"/>
    <material name="status_led"   rgba="0.1 1 0.3 1" emission="1"/>
  </asset>
"""


# ----------------------------------------------------------------------------
# greenhouse structure, walls, lamps
# ----------------------------------------------------------------------------
EAVE_Z, RIDGE_Z = 3.6, 4.4


def build_structure(cfg):
    L, n = cfg["row_length"], cfg["n_rows"]
    x_min, x_max, y_min, y_max = greenhouse_bounds()
    y_mid, xc, xl = (y_min + y_max) / 2, (x_min + x_max) / 2, (x_max - x_min) / 2
    yl = (y_max - y_min) / 2
    spans = [(y_min, y_mid), (y_mid, y_max)]
    g, cover = [], []   # cover = walls/roof (geom group 2, toggle with key "2" in the viewer)

    g.append(f'<geom name="floor" type="plane" pos="{f(xc, y_mid, 0)}" size="{f(xl+3, yl+3, 0.1)}" material="floor"/>')
    g.append(f'<geom type="box" pos="{f((x_min-0.1)/2, y_mid, 0.004)}" size="{f(abs(x_min+0.1)/2, yl, 0.004)}" material="walkway"/>')

    # frame: posts, trusses, eaves, ridges, wall rails
    for x in np.arange(x_min, x_max + 0.01, 3.0):
        for y in (y_min, y_max):
            g.append(f'<geom type="box" pos="{f(x, y, EAVE_Z/2)}" size="{f(0.04, 0.04, EAVE_Z/2)}" material="frame"/>')
        g.append(f'<geom type="box" pos="{f(x, y_mid, EAVE_Z)}" size="{f(0.03, yl, 0.04)}" material="frame"/>')
        for ya, yb in spans:
            yr = (ya + yb) / 2
            g += chain([(x, ya, EAVE_Z), (x, yr, RIDGE_Z), (x, yb, EAVE_Z)], 0.02, "frame")
            for yy in ((ya + yr) / 2, (yr + yb) / 2):
                g.append(f'<geom type="capsule" fromto="{f(x, yy, EAVE_Z, x, yy, (EAVE_Z+RIDGE_Z)/2)}" size="0.012" material="frame"/>')
    for y in (y_min, y_mid, y_max):
        g.append(f'<geom type="box" pos="{f(xc, y, EAVE_Z+0.05)}" size="{f(xl, 0.08, 0.05)}" material="frame"/>')
    for ya, yb in spans:
        g.append(f'<geom type="box" pos="{f(xc, (ya+yb)/2, RIDGE_Z)}" size="{f(xl, 0.03, 0.03)}" material="frame"/>')
    for y in (y_min + 0.03, y_max - 0.03):
        for z in (1.8, 2.7):
            g.append(f'<geom type="box" pos="{f(xc, y, z)}" size="{f(xl, 0.02, 0.02)}" material="frame"/>')

    # walls: opaque knee wall + light-diffusing film up to the eave
    for y in (y_min, y_max):
        g.append(f'<geom type="box" pos="{f(xc, y, 0.3)}" size="{f(xl, 0.035, 0.3)}" material="kneewall"/>')
        cover.append(f'<geom type="box" pos="{f(xc, y, (0.6+EAVE_Z)/2)}" size="{f(xl, 0.008, (EAVE_Z-0.6)/2)}" material="film"/>')
    for x in (x_min, x_max):
        g.append(f'<geom type="box" pos="{f(x, y_mid, 0.3)}" size="{f(0.035, yl, 0.3)}" material="kneewall"/>')
        cover.append(f'<geom type="box" pos="{f(x, y_mid, (0.6+EAVE_Z)/2)}" size="{f(0.008, yl, (EAVE_Z-0.6)/2)}" material="film"/>')
        for ya, yb in spans:
            cover.append(f'<geom type="mesh" mesh="gable" pos="{f(x, (ya+yb)/2, EAVE_Z)}" material="film"/>')
        for y in np.arange(y_min + 1.2, y_max - 0.5, 1.2):  # vertical glazing bars on the end walls
            g.append(f'<geom type="box" pos="{f(x, y, EAVE_Z/2)}" size="{f(0.02, 0.02, EAVE_Z/2)}" material="frame"/>')
    # roof panels
    for ya, yb in spans:
        half = (yb - ya) / 2
        ang = math.atan2(RIDGE_Z - EAVE_Z, half)
        ln = math.hypot(half, RIDGE_Z - EAVE_Z) / 2
        for sgn, yc in ((1, ya + half / 2), (-1, yb - half / 2)):
            cover.append(f'<geom type="box" pos="{f(xc, yc, (EAVE_Z+RIDGE_Z)/2+0.02)}" size="{f(xl, ln, 0.006)}" '
                         f'quat="{f(*q_axis((1, 0, 0), sgn*ang))}" material="roof"/>')
    # door in the headland end wall
    yd = y_max - 1.1
    g.append(f'<geom type="box" pos="{f(x_min+0.01, yd, 1.1)}" size="{f(0.03, 0.6, 1.1)}" material="door"/>')
    g.append(f'<geom type="box" pos="{f(x_min+0.045, yd-0.45, 1.05)}" size="0.01 0.04 0.012" material="steel"/>')

    # tabletop gutters with substrate bags, supports, drip lines
    hw, top = cfg["gutter_half_width"], cfg["gutter_top"]
    for i in range(n):
        y = row_y(i)
        g.append(f'<geom name="gutter_{i}" type="box" pos="{f(L/2, y, top-0.06)}" size="{f(L/2, hw, 0.06)}" material="gutter"/>')
        g.append(f'<geom type="box" pos="{f(L/2, y, top+0.035)}" size="{f(L/2-0.05, hw-0.05, 0.035)}" material="substrate"/>')
        g.append(f'<geom type="capsule" fromto="{f(0.05, y, top+0.075, L-0.05, y, top+0.075)}" size="0.005" material="dripline"/>')
        for x in np.arange(0.3, L, 2.0):
            for dy in (-hw + 0.03, hw - 0.03):
                g.append(f'<geom type="cylinder" fromto="{f(x, y+dy, 0, x, y+dy, top-0.12)}" size="0.018" material="steel"/>')
            g.append(f'<geom type="capsule" fromto="{f(x, y-hw+0.03, top-0.2, x, y+hw-0.03, top-0.2)}" size="0.012" material="steel"/>')

    # hanging lamp fixtures (the real light sources are added in build())
    for x in lamp_xs(cfg):
        for y in (aisle_y(0), aisle_y(n_aisles() - 1)) if n_aisles() > 1 else (aisle_y(0),):
            g.append(f'<geom type="box" pos="{f(x, y, 3.38)}" size="0.25 0.08 0.03" material="lamp"/>')
            g.append(f'<geom type="box" pos="{f(x, y, 3.348)}" size="0.22 0.06 0.003" material="lamp_glow"/>')
            g.append(f'<geom type="capsule" fromto="{f(x, y, 3.41, x, y, EAVE_Z)}" size="0.004" material="steel"/>')

    # headland props: charging dock and a stack of empty crates
    yd0 = aisle_y(0)
    g.append(f'<geom type="box" pos="{f(x_min+0.3, yd0, 0.45)}" size="0.12 0.25 0.45" material="robot_dark"/>')
    g.append(f'<geom type="box" pos="{f(x_min+0.43, yd0, 0.75)}" size="0.005 0.15 0.08" material="robot_orange"/>')
    g.append(f'<geom type="sphere" pos="{f(x_min+0.43, yd0+0.18, 0.8)}" size="0.02" material="status_led"/>')
    for k in range(4):
        g.append(f'<geom type="box" pos="{f(x_min+0.4, y_min+0.6, 0.065+0.13*k)}" size="0.2 0.3 0.06" material="crate_stack"/>')
    # full crates handed off at the headland (mocap; parked below the floor until used)
    for k in range(len(full_crate_spots())):
        pile = "".join(
            f'<geom type="mesh" mesh="berry_M" pos="{f(-0.09 + 0.06 * i, -0.15 + 0.06 * j, 0.115)}" '
            f'quat="{f(*q_ypr(0.7 * i + j, 1.45, 0))}" material="{("berry_ripe", "berry_ripe_dark")[(i + j) % 2]}"/>'
            for i in range(4) for j in range(6))
        g.append(f'<body name="full_crate_{k}" mocap="true" pos="0 0 -3">'
                 f'<geom type="box" pos="0 0 0.005" size="0.15 0.2 0.005" material="crate_stack"/>'
                 f'<geom type="box" pos="0 0 0.09" size="0.142 0.192 0.02" material="foam"/>'
                 f'<geom type="box" pos="0.15 0 0.06" size="0.006 0.2 0.06" material="crate_stack"/>'
                 f'<geom type="box" pos="-0.15 0 0.06" size="0.006 0.2 0.06" material="crate_stack"/>'
                 f'<geom type="box" pos="0 0.2 0.06" size="0.15 0.006 0.06" material="crate_stack"/>'
                 f'<geom type="box" pos="0 -0.2 0.06" size="0.15 0.006 0.06" material="crate_stack"/>{pile}</body>')
    return g, cover


def lamp_xs(cfg):
    return np.linspace(1.5, cfg["row_length"] - 1.5, 3)


def build_rails():
    """Pipe rails merged into one mesh (fast to render) + support blocks."""
    mb, g = meshes.MeshBuilder(), []
    for piece in track_pieces():
        for side in (-RAIL_GAUGE / 2, RAIL_GAUGE / 2):
            p = offset_polyline(piece, side)
            d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
            if len(p) == 2:   # straight: resample so parallel transport is stable
                p = np.stack([np.interp(np.linspace(0, d[-1], 8), d, p[:, k]) for k in range(2)], axis=1)
            mb.add_tube([(x, y, RAIL_Z) for x, y in p], RAIL_R, (0.5, 0.5), sides=10)
            dd = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
            for s in np.arange(0.3, dd[-1], 1.2):
                x, y = np.interp(s, dd, p[:, 0]), np.interp(s, dd, p[:, 1])
                g.append(f'<geom type="box" pos="{f(x, y, 0.04)}" size="0.025 0.025 0.04" material="steel"/>')
    meshes.ASSETS.mkdir(parents=True, exist_ok=True)
    mb.write(meshes.ASSETS / "rails.obj")
    g.append('<geom name="rails" type="mesh" mesh="rails" material="pipe"/>')
    return g


# ----------------------------------------------------------------------------
# plants
# ----------------------------------------------------------------------------
LEAF_TPL = meshes.leaflet_template()
CROWN_TPL = meshes.ellipsoid_template()


def _leaf_uv(band):
    v0, v1 = meshes.LEAF_BANDS[band]
    return lambda uv: np.stack([uv[:, 0], v0 + uv[:, 1] * (v1 - v0)], axis=1)


LEAF_UV = [_leaf_uv(b) for b in range(3)]


def _solid_uv(name):
    return lambda uv: np.tile(meshes.SOLID_UV[name], (len(uv), 1))


def build_plant(rng, mb, x, y0, cy, cfg, sides, stats):
    """One strawberry plant added to MeshBuilder `mb` (leaves, stems) + returns berry geoms."""
    top, hw = cfg["gutter_top"], cfg["gutter_half_width"]
    crown = np.array([x, cy, top + 0.075])
    mb.add_template(CROWN_TPL, crown, (1, 0, 0, 0), np.array([0.028, 0.028, 0.022]), _solid_uv("crown"))
    berries = []

    # --- leaves: arching petiole + three serrated leaflets ----------------------
    n_leaves = rng.integers(9, 12)
    for j in range(n_leaves):
        yaw = 2 * math.pi * j / n_leaves + rng.uniform(-0.35, 0.35)
        tau = rng.uniform(0, 1)                    # 0 = young upright leaf, 1 = old outer leaf
        reach = 0.05 + 0.13 * tau + rng.uniform(-0.01, 0.02)
        height = 0.24 - 0.13 * tau + rng.uniform(-0.02, 0.02)
        d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        p1 = crown + d * reach * 0.15 + [0, 0, height * 0.85]
        p2 = crown + d * reach + [0, 0, height]
        mb.add_tube(bezier([crown, p1, p2], 4), 0.0022, meshes.SOLID_UV["petiole"], sides=3, taper=0.7)
        droop = 0.15 + 0.55 * tau + rng.uniform(-0.1, 0.1)
        band = int(rng.integers(0, 3))
        size_c = LEAFLET_SIZES["l"] if rng.random() < 0.55 else LEAFLET_SIZES["m"]
        size_s = size_c * 0.85
        mb.add_template(LEAF_TPL, p2, q_ypr(yaw, droop, rng.uniform(-0.15, 0.15)), size_c, LEAF_UV[band])
        for s in (-1, 1):
            sy = yaw + s * rng.uniform(0.95, 1.2)
            mb.add_template(LEAF_TPL, p2, q_ypr(sy, droop + 0.12, s * rng.uniform(0.2, 0.45)), size_s, LEAF_UV[band])

    # --- fruit trusses over the aisle-side gutter edges ------------------------
    for side in sides:
        for _ in range(1 if rng.random() < 0.65 else 2):
            dx = rng.uniform(-0.07, 0.07)
            yw = y0 + side * hw                                     # gutter side wall
            p0 = crown + [dx * 0.2, side * 0.02, 0.005]
            p1 = np.array([x + dx * 0.7, y0 + side * (hw - 0.01), top + 0.075])
            p2 = np.array([x + dx, yw + side * 0.035, top + 0.025])
            p3 = np.array([x + dx + rng.uniform(-0.03, 0.03), yw + side * rng.uniform(0.035, 0.05), top - rng.uniform(0.0, 0.02)])
            truss = bezier([p0, p1, p2, p3], 10)
            mb.add_tube(truss, 0.0028, meshes.SOLID_UV["stem"], sides=5, taper=0.6)

            n_fruit = rng.integers(3, 5)
            offsets = np.linspace(-0.05, 0.05, n_fruit) + rng.uniform(-0.008, 0.008, n_fruit)
            rng.shuffle(offsets)
            for k in range(n_fruit):
                ripe = rng.random() < cfg["ripe_fraction"]
                if ripe:
                    mat = "berry_ripe" if rng.random() < 0.7 else "berry_ripe_dark"
                    size = "L" if rng.random() < 0.6 else "M"
                else:
                    mat = "berry_unripe" if rng.random() < 0.65 else "berry_pale"
                    size = ("XS" if rng.random() < 0.6 else "S") if mat == "berry_unripe" else "M"
                # pedicels branch off at staggered points along the end of the truss; fruit never
                # overlap (re-sample the hanging point until there is >= 4 mm between neighbours)
                L = BERRY_SIZES[size]
                for _try in range(25):
                    attach = truss[int(rng.integers(6, 10))]
                    drop = rng.uniform(0.016, 0.03)
                    off = offsets[k] if _try == 0 else rng.uniform(-0.07, 0.07)
                    btop = attach + [off, side * rng.uniform(0.0, 0.03), -drop]
                    hang = 0.3 * (btop - attach) / np.linalg.norm(btop - attach) + 0.7 * np.array([0, 0, -1.0])
                    hang /= np.linalg.norm(hang)
                    centre = btop + hang * BERRY_CENTER * L
                    near = [c for c in stats["centres"] if abs(c[0] - centre[0]) < 0.1]
                    if all(np.linalg.norm(centre - c[:3]) > BERRY_COL_R * L + c[3] + 0.004 for c in near):
                        break
                else:
                    continue
                stats["centres"].append(np.r_[centre, BERRY_COL_R * L])
                mid = attach + (btop - attach) * 0.5 + [0, side * 0.012, 0.012]
                pedicel = bezier([attach, mid, btop + [0, 0, 0.004]], 4)
                mb.add_tube(pedicel, 0.0016, meshes.SOLID_UV["stem"], sides=3)
                q = q_hang(hang, rng.uniform(0, 2 * math.pi))
                tag = "R" if ripe else "U"
                i = stats["n"]
                # each fruit is a mocap body: it hangs still until the robot cuts it, then the
                # harvester moves it (in the gripper, falling, in the crate). The invisible sphere
                # (group 3) is the contact shape the soft fingers press on.
                berries.append(
                    f'<body name="berry_{i}" mocap="true" pos="{f(*btop)}" quat="{f(*q)}">'
                    f'<geom name="berry_{i}_{tag}" type="mesh" mesh="berry_{size}" material="{mat}"/>'
                    f'<geom name="berry_{i}_col" type="sphere" pos="0 0 {f(-BERRY_CENTER * L)}" size="{f(BERRY_COL_R * L)}" '
                    f'contype="2" conaffinity="0" group="3" rgba="1 0 0 0.3"/></body>')
                stats["meta"].append(dict(
                    id=i, ripe=bool(ripe), material=mat, size=size, length=L,
                    top=[round(v, 5) for v in btop], axis=[round(v, 5) for v in hang],
                    center=[round(v, 5) for v in btop + hang * BERRY_CENTER * L],
                    radius=round(BERRY_COL_R * L, 5), side=int(side),
                    pedicel=[[round(v, 5) for v in pt] for pt in bezier([attach, mid, btop + [0, 0, 0.004]], 8)]))
                stats["n"] += 1
                stats["ripe" if ripe else "unripe"] += 1
    return berries


PLANT_CHUNK_LEN = 3.0   # metres of row merged into one plant mesh


def build_plants(cfg, rng):
    """Returns (berry geoms, plant-mesh geoms, chunk mesh names, stats); writes the chunk OBJs."""
    berries, stats, chunks = [], {"n": 0, "ripe": 0, "unripe": 0, "meta": [], "centres": []}, {}
    n = cfg["n_rows"]
    for i in range(n):
        y0 = row_y(i)
        sides = []
        if i > 0 or cfg["fruit_on_outer_sides"]:
            sides.append(-1)
        if i < n - 1 or cfg["fruit_on_outer_sides"]:
            sides.append(1)
        for k, x in enumerate(np.arange(cfg["plant_spacing"] / 2, cfg["row_length"], cfg["plant_spacing"])):
            key = f"plants_r{i}_c{int(x // PLANT_CHUNK_LEN)}"
            mb = chunks.setdefault(key, meshes.MeshBuilder())
            x += rng.uniform(-0.025, 0.025)
            cy = y0 + (0.025 if k % 2 else -0.025)
            berries += build_plant(rng, mb, x, y0, cy, cfg, sides, stats)
    meshes.ASSETS.mkdir(parents=True, exist_ok=True)
    for old in meshes.ASSETS.glob("plants_r*.obj"):
        old.unlink()
    geoms = []
    for name, mb in chunks.items():
        mb.write(meshes.ASSETS / f"{name}.obj")
        geoms.append(f'<geom name="{name}" type="mesh" mesh="{name}" material="plant"/>')
    return berries, geoms, list(chunks), stats


# ----------------------------------------------------------------------------
# robot
# ----------------------------------------------------------------------------
def build_cameras(cfg):
    L = cfg["row_length"]
    x_min, x_max, y_min, y_max = greenhouse_bounds()
    ym = (y_min + y_max) / 2
    ya = aisle_y(0)
    rx = cfg["robot_start_x"]
    cams = {
        "overview":        ((x_min + 0.4, y_min + 0.4, 3.3), (L * 0.55, ym, 0.6), 65),
        "headland_view":   ((2.2, ym - 0.4, 2.9), (X_CROSS + 0.2, ym, 0.3), 70),
        "aisle_view":      ((-0.9, ya, 1.55), (L * 0.6, ya, 0.9), 55),
        "berries_closeup": ((rx + 1.0, ya + 0.15, 0.97), (rx + 1.25, row_y(1) - 0.2, 0.9), 50),
        "robot_side":      ((rx + 1.2, ya - 0.35, 1.45), (rx + 0.1, ya + 0.15, 0.85), 70),
        "cutaway":         ((-4.0, y_min - 4.0, 7.0), (L * 0.45, ym, 0.4), 50),
    }
    out = [f'<camera name="{n}" pos="{f(*p)}" xyaxes="{look_at_xyaxes(p, t)}" fovy="{fv}"/>'
           for n, (p, t, fv) in cams.items()]
    out.append(f'<camera name="top_down" pos="{f(L/2-0.5, ym, 16)}" xyaxes="1 0 0 0 1 0" fovy="45"/>')
    return out


# ----------------------------------------------------------------------------
def build(cfg=CONFIG):
    meshes.build_assets()
    rng = np.random.default_rng(cfg["seed"])
    x_min, x_max, y_min, y_max = greenhouse_bounds()
    structure, cover = build_structure(cfg)
    rails = build_rails()
    berries, plant_geoms, plant_chunks, stats = build_plants(cfg, rng)
    cams = build_cameras(cfg)
    ym = (y_min + y_max) / 2

    lights = [
        '<light name="daylight" directional="true" pos="6 2 20" dir="-0.3 0.2 -1" diffuse="0.42 0.42 0.4" specular="0.1 0.1 0.1" castshadow="false"/>',
        '<light name="skyfill" directional="true" pos="6 2 20" dir="0.35 -0.3 -1" diffuse="0.18 0.18 0.2" specular="0 0 0" castshadow="false"/>',
    ]
    for k, x in enumerate(lamp_xs(cfg)[[0, 2]]):
        lights.append(f'<light name="lamp_{k}" mode="fixed" pos="{f(x, ym, 3.3)}" dir="0.05 0.03 -1" cutoff="80" exponent="1" '
                      f'diffuse="0.42 0.41 0.38" specular="0.15 0.15 0.15" castshadow="true"/>')

    rx = cfg["robot_start_x"]
    ind = lambda items: "\n".join("    " + s for s in items)

    xml = f"""<mujoco model="strawberry_greenhouse">
  <compiler angle="radian" autolimits="true" meshdir="assets" texturedir="assets"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <size memory="128M"/>
  <statistic center="{f(cfg['row_length']/2, ym, 1.0)}" extent="9"/>
  <visual>
    <global offwidth="1920" offheight="1080" azimuth="30" elevation="-25"/>
    <quality shadowsize="4096" offsamples="4" numslices="14" numstacks="7"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.22 0.22 0.22" specular="0.05 0.05 0.05"/>
    <map znear="0.005" zfar="200"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0"/>
    <default class="cover">
      <geom group="2"/>
    </default>
  </default>
{build_assets(gable_w=(y_max - y_min) / 2, gable_h=RIDGE_Z - EAVE_Z, plant_chunks=plant_chunks)}
  <worldbody>
{ind(lights)}
{ind(cams)}
    <!-- ===================== GREENHOUSE STRUCTURE ===================== -->
{ind(structure)}
    <!-- walls + roof cover (geom group 2: press "2" in the viewer to hide/show) -->
{ind(s.replace('<geom ', '<geom class="cover" ', 1) for s in cover)}
    <!-- ===================== RAIL TRACK NETWORK ===================== -->
{ind(rails)}
    <!-- ===================== STRAWBERRY PLANTS (merged meshes) ===================== -->
{ind(plant_geoms)}
    <!-- ===================== STRAWBERRIES (mocap body berry_<id>; visual geom berry_<id>_R = ripe / _U = unripe) ===================== -->
{ind(berries)}
{robot.robot_body(look_at_xyaxes((-1.6, -0.05, 1.85), (0.3, 0.25, 0.85)))}
  </worldbody>

  <actuator>{robot.robot_actuators()}  </actuator>

</mujoco>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(xml, encoding="utf-8")
    META.write_text(json.dumps(stats["meta"]), encoding="utf-8")
    (OUT.parent / "robot_only.xml").write_text(robot.ik_model_xml(), encoding="utf-8")
    pct = 100 * stats["ripe"] / max(1, stats["n"])
    print(f"Wrote {OUT}")
    print(f"  rows: {cfg['n_rows']} x {cfg['row_length']} m, aisles with track: {n_aisles()}, "
          f"plants: {cfg['n_rows'] * int(round(cfg['row_length'] / cfg['plant_spacing']))}")
    print(f"  strawberries: {stats['n']}  (ripe {stats['ripe']} = {pct:.0f} %, unripe {stats['unripe']})")
    return OUT


if __name__ == "__main__":
    build()
