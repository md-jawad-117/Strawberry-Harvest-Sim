"""
Procedural meshes (OBJ) and textures (PNG) for plants and fruit.

To keep rendering fast, all *static* plant parts (leaves, petioles, crowns, fruit
trusses, pedicels) are merged into a few large meshes per row section using
`MeshBuilder`; they share one texture atlas (`plant_atlas.png`).
Each strawberry is one mesh (fruit + calyx) so it can later be picked individually.

Texture conventions: v = 1 is the top row of the PNG.
  plant_atlas.png   v 0.75-1.0 leaf A | 0.50-0.75 leaf B | 0.25-0.50 leaf C | 0-0.25 solid colours
  berry_*.png       v 0.12-1.0 skin with seeds | 0-0.10 calyx green
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).parent / "scene" / "assets"

# atlas regions: (u, v) centres of the solid colour patches, and leaf bands (v0, v1)
SOLID_UV = {"stem": (0.125, 0.125), "petiole": (0.375, 0.125), "crown": (0.625, 0.125), "calyx": (0.875, 0.125)}
LEAF_BANDS = [(0.76, 0.99), (0.51, 0.74), (0.26, 0.49)]


# ----------------------------------------------------------------------------
# geometry utilities
# ----------------------------------------------------------------------------
def face_normals_to_vertex(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def double_sided(v, uv, f, offset):
    n = face_normals_to_vertex(v, f)
    vb = v - offset * n
    fb = f[:, [0, 2, 1]] + len(v)
    return np.vstack([v, vb]), np.vstack([uv, uv]), np.vstack([f, fb])


def write_obj(path, v, n, uv, f):
    with open(path, "w") as fh:
        fh.write("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in v))
        fh.write("".join(f"vt {a:.4f} {b:.4f}\n" for a, b in uv))
        fh.write("".join(f"vn {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in n))
        g = f + 1
        fh.write("".join(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" for a, b, c in g))


class MeshBuilder:
    """Accumulates transformed parts into one big mesh."""

    def __init__(self):
        self.v, self.n, self.uv, self.f = [], [], [], []
        self.count = 0

    def add(self, v, n, uv, f):
        self.v.append(v); self.n.append(n); self.uv.append(uv)
        self.f.append(f + self.count)
        self.count += len(v)

    def add_template(self, tpl, pos, quat, scale, uv_map=None):
        v, n, uv, f = tpl
        R = quat_to_mat(quat)
        uv2 = uv_map(uv) if uv_map else uv
        self.add(v * scale @ R.T + pos, n @ R.T, uv2, f)

    def add_tube(self, pts, radius, uv, sides=5, taper=1.0):
        pts = np.asarray(pts, float)
        k = len(pts)
        t = np.gradient(pts, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        ref = np.array([0, 0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0, 0])
        verts, norms = [], []
        a = np.cross(t[0], ref); a /= np.linalg.norm(a)
        for i in range(k):
            a = a - np.dot(a, t[i]) * t[i]; a /= np.linalg.norm(a)   # parallel transport
            b = np.cross(t[i], a)
            r = radius * (1 - (1 - taper) * i / max(1, k - 1))
            for s in range(sides):
                th = 2 * math.pi * s / sides
                d = math.cos(th) * a + math.sin(th) * b
                verts.append(pts[i] + r * d); norms.append(d)
        f = []
        for i in range(k - 1):
            for s in range(sides):
                p, q = i * sides + s, i * sides + (s + 1) % sides
                f += [(p, q, q + sides), (p, q + sides, p + sides)]   # outward-facing
        f = np.array(f)
        v = np.array(verts)
        self.add(v, np.array(norms), np.tile(uv, (len(v), 1)), f)

    def write(self, path):
        write_obj(path, np.vstack(self.v), np.vstack(self.n), np.vstack(self.uv), np.vstack(self.f))


# ----------------------------------------------------------------------------
# templates  -> (v, n, uv, f)
# ----------------------------------------------------------------------------
def leaflet_template(n=8):
    """Obovate serrated leaflet: base at origin, tip at x=1, folded up along the midrib. uv in [0,1]^2."""
    W = 0.48
    cols = (-1.0, 0.0, 1.0)
    v, uv = [], []
    for i in range(n + 1):
        u = i / n
        w = W * math.sin(math.pi * u ** 0.62) ** 0.6
        tooth = 0.84 if (i % 2 and 0.15 < u < 0.97) else 1.0      # serrated margin
        for c in cols:
            y = c * w * (tooth if abs(c) == 1.0 else 1.0)
            z = 0.22 * abs(y) - 0.10 * u ** 2 + 0.015 * math.sin(9 * u) * abs(c)
            v.append((u, y, z)); uv.append((u, 0.5 + y / (2 * W)))
    m = len(cols)
    f = []
    for i in range(n):
        for j in range(m - 1):
            a = i * m + j
            f += [(a, a + m, a + m + 1), (a, a + m + 1, a + 1)]
    v, uv, f = double_sided(np.array(v, float), np.array(uv, float), np.array(f), offset=0.006)
    return v, face_normals_to_vertex(v, f), uv, f


def ellipsoid_template(slices=7, stacks=4):
    v, f = [], []
    for i in range(stacks + 1):
        ph = math.pi * i / stacks
        for j in range(slices):
            th = 2 * math.pi * j / slices
            v.append((math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)))
    for i in range(stacks):
        for j in range(slices):
            a, b = i * slices + j, i * slices + (j + 1) % slices
            f += [(a, a + slices, b + slices), (a, b + slices, b)]
    v = np.array(v, float)
    return v, v.copy(), np.zeros((len(v), 2)), np.array(f)


def strawberry_body(rings=12, seg=12):
    """Unit fruit: height 1 (z 0 -> -1), max radius ~0.47; origin at the top.
    Rounded heart shape: full shoulders, gentle taper, blunt rounded bottom (no sharp tip)."""
    def radius(u):
        shoulder = math.sin(min(u / 0.3, 1.0) * math.pi / 2) ** 0.55
        dome = (1 - u ** 2.2) ** 0.5          # rounds off the bottom instead of a point
        return (0.35 + 0.65 * shoulder) * dome * (1 - 0.32 * u)

    t = np.linspace(0, 1, rings)
    us = 0.5 - 0.5 * np.cos(math.pi * t)       # more rings near the top and the rounded bottom
    rs = np.array([radius(u) for u in us]); rs *= 0.47 / rs.max()
    v, uv, f = [(0, 0, 0.04)], [(0.5, 0.99)], []
    for i, u in enumerate(us[:-1]):
        for j in range(seg + 1):
            th = 2 * math.pi * j / seg
            r = rs[i] * (1 + 0.035 * math.sin(3 * th + 1.3) + 0.02 * math.sin(5 * th))
            v.append((r * math.cos(th), r * math.sin(th), -u))
            uv.append((j / seg * 2, 1.0 - 0.88 * u))
    tip = len(v)
    v.append((0, 0, -1.0)); uv.append((0.5, 0.12))
    row = seg + 1
    for j in range(seg):
        f.append((0, 1 + j, 2 + j))
    for i in range(rings - 2):
        for j in range(seg):
            a = 1 + i * row + j
            f += [(a, a + row + 1, a + 1), (a, a + row, a + row + 1)]
    last = 1 + (rings - 2) * row
    for j in range(seg):
        f.append((last + j, tip, last + j + 1))
    return np.array(v, float), np.array(uv, float), np.array(f)


def calyx_part(n_sepals=8):
    """Star of drooping sepals, unit radius."""
    v, f = [(0, 0, 0.05)], []
    for k in range(n_sepals):
        th = 2 * math.pi * k / n_sepals + 0.15 * math.sin(k * 2.3)
        ln = 0.85 + 0.15 * math.sin(k * 1.7)
        pts = []
        for t, w in ((0.0, 0.12), (0.35, 0.16), (0.7, 0.09), (1.0, 0.0)):
            r = t * ln
            z = -0.32 * r ** 1.6 + 0.08 * math.sin(k * 3.1) * r
            for side in ((-1, 1) if w > 0 else (0,)):
                a = th + side * w
                rr = r if t else 0.25
                pts.append((rr * math.cos(a), rr * math.sin(a), z))
        b = len(v)
        v.extend(pts)
        l0, r0, l1, r1, l2, r2, tp = range(b, b + 7)
        f += [(0, l0, r0), (l0, r1, r0), (l0, l1, r1), (l1, r2, r1), (l1, l2, r2), (l2, tp, r2)]
    v = np.array(v, float)
    uv = np.tile((0.5, 0.05), (len(v), 1))
    return double_sided(v, uv, np.array(f), offset=0.03)


def berry_mesh():
    """Fruit + calyx in one mesh (unit length)."""
    vb, uvb, fb = strawberry_body()
    vc, uvc, fc = calyx_part()
    vc = vc * 0.6 + [0, 0, 0.02]
    v = np.vstack([vb, vc]); uv = np.vstack([uvb, uvc]); f = np.vstack([fb, fc + len(vb)])
    n = np.vstack([face_normals_to_vertex(vb, fb), face_normals_to_vertex(vc, fc)])
    # smooth the UV seam of the body (duplicated vertices at theta = 0 / 2pi)
    return v, n, uv, f


# ----------------------------------------------------------------------------
# textures
# ----------------------------------------------------------------------------
def _leaf_tile(w, h, tint, seed):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (70, 140, 55))
    px = np.array(img, float)
    yy, xx = np.mgrid[0:h, 0:w]
    xx, yy = xx / w, yy / h
    px *= (0.85 + 0.25 * xx)[..., None]
    px *= (1 - 0.15 * np.abs(yy - 0.5))[..., None]
    img = Image.fromarray(px.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    vein = (120, 175, 90)
    d.line([(0, h / 2), (w, h / 2)], fill=vein, width=3)
    for k in range(1, 10):
        x0 = k * w / 10.5
        for s in (-1, 1):
            d.line([(x0, h / 2), (x0 + w * 0.16, h / 2 + s * h * 0.45)], fill=vein, width=2)
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    px = np.array(img, float) * np.array(tint) * (1 + 0.06 * rng.normal(0, 1, (h, w, 1)))
    return Image.fromarray(px.clip(0, 255).astype(np.uint8))


def plant_atlas(path, size=512):
    img = Image.new("RGB", (size, size))
    q = size // 4
    for k, tint in enumerate([(1.0, 1.0, 1.0), (0.85, 0.95, 0.85), (0.72, 0.84, 0.72)]):
        img.paste(_leaf_tile(size, q, tint, k), (0, k * q))   # rows from top = v 1.0 downwards
    d = ImageDraw.Draw(img)
    solids = [(107, 143, 64), (102, 133, 61), (77, 97, 46), (61, 128, 41)]   # stem, petiole, crown, calyx
    for k, c in enumerate(solids):
        d.rectangle([k * q, 3 * q, (k + 1) * q, size], fill=c)
    img.save(path)


def berry_texture(path, base, seed_col, dimple, size=256, seed=0):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), tuple(int(255 * c) for c in base))
    px = np.array(img, float)
    noise = rng.normal(0, 1, (size // 16, size // 16))
    noise = np.array(Image.fromarray(((noise + 3) * 40).clip(0, 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), float) / 255 - 0.47
    px *= (1 + 0.18 * noise[..., None])
    img = Image.fromarray(px.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    calyx_rows = int(size * 0.11)          # bottom rows of the image = v < 0.11 = calyx green
    rows, cols = 13, 14
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5 * (r % 2) + rng.uniform(-0.15, 0.15)) * size / cols
            y = (r + rng.uniform(-0.15, 0.15)) * (size - calyx_rows) / rows
            for ox in (-size, 0, size):
                cx = x + ox
                d.ellipse([cx - 5, y - 6, cx + 5, y + 6], fill=tuple(int(255 * k) for k in dimple))
                d.ellipse([cx - 2.2, y - 3, cx + 2.2, y + 3], fill=tuple(int(255 * k) for k in seed_col))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, size - calyx_rows, size, size], fill=(58, 125, 38))
    img.save(path)


def build_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    write_obj(ASSETS / "berry.obj", *berry_mesh())
    plant_atlas(ASSETS / "plant_atlas.png")
    berry_texture(ASSETS / "berry_ripe.png", (0.78, 0.05, 0.07), (0.95, 0.82, 0.35), (0.55, 0.02, 0.04), seed=1)
    berry_texture(ASSETS / "berry_ripe_dark.png", (0.62, 0.03, 0.06), (0.92, 0.75, 0.3), (0.42, 0.01, 0.03), seed=2)
    berry_texture(ASSETS / "berry_unripe.png", (0.56, 0.74, 0.30), (0.85, 0.88, 0.45), (0.42, 0.6, 0.22), seed=3)
    berry_texture(ASSETS / "berry_pale.png", (0.86, 0.88, 0.62), (0.9, 0.85, 0.4), (0.7, 0.75, 0.45), seed=4)
    for old in ("strawberry.obj", "calyx.obj", "leaflet.obj", "leaf.png"):
        (ASSETS / old).unlink(missing_ok=True)


if __name__ == "__main__":
    build_assets()
    print("assets written to", ASSETS)
