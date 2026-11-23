"""
Camera-based fruit detection (step 4).

The robot does not get fruit positions from the scene file. It finds them the way a real one would:

1. The two side-looking mast cameras (scan_left / scan_right) render a colour + depth image.
2. Ripe pixels are picked by colour (strong red, little green/blue). Green and white fruit,
   leaves, stems and the robot's own orange parts fail the test.
3. Pixels are split into blobs. Depth jumps separate overlapping fruit, and oversized blobs
   (two fruit touching at the same depth) are split with k-means.
4. Each blob is back-projected with the depth image. The fruit centre is the median surface
   point pushed one fruit radius further along the viewing ray.
5. Detections from many views are fused in a world-frame FruitMap.
6. Just before the grasp, the Gen3 wrist camera re-measures the chosen fruit from close range.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy import ndimage

FRUIT_R = 0.018          # typical fruit radius (m)
CENTRE_PUSH = 0.011      # visible-surface median -> centre: about 0.6 radius further along the ray


def ripe_mask(rgb):
    c = rgb.astype(np.int16)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return (r > 70) & (r > 2.3 * g) & (r > 2.3 * b)


class Camera:
    """Renders colour + depth from a named MuJoCo camera and back-projects pixels to world points."""

    def __init__(self, model, renderer, name):
        self.m, self.r, self.name = model, renderer, name
        self.id = model.camera(name).id
        self.h, self.w = renderer.height, renderer.width
        self.f = 0.5 * self.h / math.tan(math.radians(model.cam_fovy[self.id]) / 2)

    def grab(self, data, opt):
        self.r.update_scene(data, camera=self.name, scene_option=opt)
        rgb = self.r.render().copy()
        self.r.enable_depth_rendering()
        self.r.update_scene(data, camera=self.name, scene_option=opt)
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        self.pos = data.cam_xpos[self.id].copy()
        self.mat = data.cam_xmat[self.id].reshape(3, 3).copy()
        return rgb, depth

    def rays(self, us, vs):
        """Unit view rays (world) through pixel coordinates."""
        x = (us + 0.5 - self.w / 2) / self.f
        y = -(vs + 0.5 - self.h / 2) / self.f
        d = np.stack([x, y, -np.ones_like(x)], axis=-1)
        d = d @ self.mat.T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def backproject(self, us, vs, depth):
        """World points for pixels; depth is the distance along the camera's viewing axis."""
        x = (us + 0.5 - self.w / 2) / self.f * depth
        y = -(vs + 0.5 - self.h / 2) / self.f * depth
        p = np.stack([x, y, -depth], axis=-1)
        return self.pos + p @ self.mat.T


def detect(cam, rgb, depth, max_range=1.6):
    """Ripe-fruit detections in one image: list of dicts(center, pixels, depth)."""
    mask = ripe_mask(rgb) & (depth < max_range)
    if not mask.any():
        return []
    # cut the mask where depth jumps (one fruit in front of another)
    gy, gx = np.gradient(depth)
    mask &= (np.abs(gx) < 0.006) & (np.abs(gy) < 0.006)
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        vs, us = np.nonzero(lab[sl] == k)
        vs = vs + sl[0].start
        us = us + sl[1].start
        d = depth[vs, us]
        dm = float(np.median(d))
        expect = math.pi * (cam.f * FRUIT_R / dm) ** 2          # pixel area of one fruit at that range
        if len(us) < 0.15 * expect:
            continue
        groups = [(us, vs, d)]
        k_split = int(round(len(us) / (1.15 * expect)))
        if k_split >= 2:                                          # several touching fruit -> k-means split
            pts = np.stack([us, vs], 1).astype(float)
            cent = pts[np.linspace(0, len(pts) - 1, k_split).astype(int)]
            for _ in range(12):
                a = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), 1)
                cent = np.array([pts[a == j].mean(0) if (a == j).any() else cent[j] for j in range(k_split)])
            groups = [(us[a == j], vs[a == j], d[a == j]) for j in range(k_split) if (a == j).sum() > 0.15 * expect]
        for gu, gv, gd in groups:
            surf = cam.backproject(gu.astype(float), gv.astype(float), gd)
            sp = np.median(surf, axis=0)
            ray = sp - cam.pos
            ray /= np.linalg.norm(ray)
            out.append(dict(center=sp + CENTRE_PUSH * ray, pixels=int(len(gu)), range=float(np.median(gd)),
                            uv=(float(gu.mean()), float(gv.mean())), r_px=float(math.sqrt(len(gu) / math.pi))))
    return out


class FruitMap:
    """World-frame list of detected ripe fruit, fused over many views."""

    def __init__(self, merge_dist=0.021):
        self.items = []          # dict(pos, n, status, id)
        self.merge = merge_dist
        self._next = 0

    def add(self, dets):
        new = 0
        for d in dets:
            p = d["center"]
            best, bd = None, self.merge
            for it in self.items:            # already attempted fruit are matched too, not re-added
                dist = np.linalg.norm(it["pos"] - p)
                if dist < bd:
                    best, bd = it, dist
            if best is None:
                self.items.append(dict(pos=p.copy(), n=1, status="new", id=self._next))
                self._next += 1
                new += 1
            elif best["status"] == "new":
                best["pos"] = (best["pos"] * best["n"] + p) / (best["n"] + 1)
                best["n"] += 1
        return new

    def candidates(self, x_lo, x_hi, y_lo, y_hi, min_views=1):
        return [it for it in self.items if it["status"] == "new" and it["n"] >= min_views
                and x_lo <= it["pos"][0] <= x_hi and y_lo <= it["pos"][1] <= y_hi]


class Perception:
    def __init__(self, model, scan_size=(384, 512), wrist_size=(240, 320)):
        self.m = model
        self.scan_r = mujoco.Renderer(model, *scan_size, max_geom=20000)
        self.wrist_r = mujoco.Renderer(model, *wrist_size, max_geom=20000)
        self.cams = {n: Camera(model, self.scan_r, n) for n in ("scan_left", "scan_right")}
        self.wrist = Camera(model, self.wrist_r, "gen3_wrist_cam")
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[2] = 0          # cameras are inside the greenhouse: walls/roof never in view
        self.opt.geomgroup[3] = 0          # invisible contact spheres
        self.map = FruitMap()
        self.n_images = 0

    def scan(self, data, cams=("scan_left", "scan_right")):
        dets = []
        for c in cams:
            rgb, depth = self.cams[c].grab(data, self.opt)
            self.n_images += 1
            dets += detect(self.cams[c], rgb, depth)
        return self.map.add(dets), len(dets)

    def refine(self, data, predicted, max_shift=0.03):
        """Re-measure the fruit nearest `predicted` with the wrist camera. Returns new centre or None."""
        rgb, depth = self.wrist.grab(data, self.opt)
        self.n_images += 1
        dets = detect(self.wrist, rgb, depth, max_range=0.6)
        if not dets:
            return None
        d = min(dets, key=lambda e: np.linalg.norm(e["center"] - predicted))
        if np.linalg.norm(d["center"] - predicted) > max_shift:
            return None
        return d["center"]

    def close(self):
        self.scan_r.close()
        self.wrist_r.close()


def annotate(rgb, dets, mask=True):
    """Debug picture: ripe-pixel mask tinted, one circle per detected fruit."""
    from PIL import Image, ImageDraw
    img = rgb.copy()
    if mask:
        m = ripe_mask(rgb)
        img[m] = (0.5 * img[m] + [0, 0, 127]).astype(np.uint8)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for d in dets:
        u, v = d["uv"]
        r = d["r_px"] * 1.6 + 3
        dr.ellipse([u - r, v - r, u + r, v + r], outline=(255, 255, 0), width=2)
    dr.text((8, 6), f"{len(dets)} ripe fruit detected", fill=(255, 255, 255))
    return im
