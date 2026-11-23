"""
Harvest controller (steps 2-4 of docs/DESIGN.md).

    Harvester(model, data, mode="vision")   # fruit found by the cameras (default)
    Harvester(model, data, mode="oracle")   # fruit positions read from scene/berries.json (step-2 baseline)
    h.tick()                                 # call once per physics step, then mujoco.mj_step

The mission runs as a Python generator that yields once per physics step:
  for each aisle: drive in, stopping every 0.9 m
      at each stop: scan with the mast cameras -> plan -> pick every ripe fruit in reach
      (approach, wrist-camera re-measure, soft grip, cut stem, retract, lower into crate, release)
      -> wrist-camera second look for hidden fruit
  drive back to the headland, hand off full crates, switch track to the next aisle.

The fruit are mocap bodies. They hang still until the knife cuts their stem, then the
harvester carries them in the gripper, lets them drop into a crate slot, and keeps them in
the crate as the cart moves. Everything that matters is measured, not scripted:
  * grasp:  contact between the finger pads and the fruit's contact sphere
  * cut:    the fruit's own stem (pedicel polyline) must cross the knife's cut window;
            any other stem in the window is cut too and that fruit falls to the floor
  * bruise: peak pad contact force on each fruit vs. that fruit's firmness
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

import robot
from layout import CONFIG, Route, aisle_y, full_crate_spots

ROOT = Path(__file__).parent
META = ROOT / "scene" / "berries.json"
IK_MODEL = ROOT / "scene" / "robot_only.xml"

AISLE_SPEED, TRANSFER_SPEED, ACCEL = 0.35, 0.25, 0.4
REACH_X = 0.47                 # pick window along the row around the arm base (stops are 0.9 m apart)
GRIP_FORCE = 1.0               # N per finger (force-limited soft grip)
FINGER_PREOPEN = 0.024         # finger opening for the approach (pads ~1.2 cm clear of an average fruit)
K_SWING = 8.0                  # N/m: a hanging fruit pushed sideways swings on its stem (m g / stem length)
SWING_TAU = 0.35               # s: pushed fruit settle back
ARM_VMAX = np.array([1.39, 1.39, 1.39, 1.39, 1.22, 1.22, 1.22])   # Gen3 joint speed limits (rad/s)
PRE_DIST = 0.09                # pre-grasp stand-off along the approach (m)
YAW_OPTIONS = (0.0, 0.35, -0.35, 0.65, -0.65)   # approach yaw candidates (rad, about vertical)
TILT = 0.3                     # approach comes slightly from below

# crate slots (crate-local): 4 layers of 8 x 6
SLOT_PITCH = 0.046
CRATE_SLOTS = [np.array([(-3.5 + i) * SLOT_PITCH, (-2.5 + j) * SLOT_PITCH, 0.036 + 0.031 * k])
               for k in range(4) for j in range(6) for i in range(8)]


# ----------------------------------------------------------------------------------------
# small maths helpers
# ----------------------------------------------------------------------------------------
def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def qmul_batch(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def nlerp(a, b, t):
    if np.dot(a, b) < 0:
        b = -b
    q = (1 - t) * a + t * b
    return q / np.linalg.norm(q)


def min_jerk(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def tool_mat(approach, up=(0, 0, 1.0)):
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    y = np.asarray(up, float) - z * np.dot(up, z)
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


def lying_quat(rng):
    """Fruit resting on its side (local -z horizontal), random heading."""
    yaw = rng.uniform(0, 2 * math.pi)
    tilt = math.pi / 2 + rng.uniform(-0.3, 0.3)
    qa = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    qb = np.array([math.cos(tilt / 2), math.sin(tilt / 2), 0, 0])
    return qmul_batch(qa, qb)


# ----------------------------------------------------------------------------------------
# fruit bookkeeping
# ----------------------------------------------------------------------------------------
class Berries:
    """State of every strawberry (hanging / held / falling / crate / floor / shipped)."""

    def __init__(self, model, data, seed=5):
        self.meta = json.loads(META.read_text())
        n = len(self.meta)
        self.n = n
        self.mocap = np.array([model.body_mocapid[model.body(f"berry_{i}").id] for i in range(n)])
        self.col = np.array([model.geom(f"berry_{i}_col").id for i in range(n)])
        self.geom2berry = {g: i for i, g in enumerate(self.col)}
        self.ripe = np.array([b["ripe"] for b in self.meta])
        self.center = np.array([b["center"] for b in self.meta])
        self.ped = [np.array(b["pedicel"]) for b in self.meta]
        self.radius = np.array([b["radius"] for b in self.meta])
        D = np.linalg.norm(self.center[:, None] - self.center[None], axis=2)
        self.neigh = [np.where((D[i] < 0.07) & (D[i] > 0))[0] for i in range(n)]
        self.home_pos = data.mocap_pos[self.mocap].copy()
        self.home_quat = data.mocap_quat[self.mocap].copy()
        rng = np.random.default_rng(seed)
        self.rng = rng
        # firmness: max pad force (N) before visible bruising. Dark (fully ripe) fruit are softer.
        tol = np.where([b["material"] == "berry_ripe_dark" for b in self.meta],
                       rng.normal(2.3, 0.4, n), rng.normal(3.0, 0.5, n))
        tol = np.where(self.ripe, tol, rng.normal(5.0, 0.8, n))
        self.tolerance = np.maximum(tol, 1.3)
        self.peak = np.zeros(n)
        self.reset_state(data)

    def reset_state(self, data):
        self.state = np.array(["hanging"] * self.n, dtype=object)
        data.mocap_pos[self.mocap] = self.home_pos
        data.mocap_quat[self.mocap] = self.home_quat
        self.peak[:] = 0
        self.offset = {}                  # hanging fruit pushed aside: berry -> displacement (m)
        self.held = None                  # (berry, rel_pos, rel_quat) in tcp frame
        self.falling = []                 # dicts
        self.crate_items = {0: [], 1: []}  # crate -> list of (berry, local pos, local quat)

    def pedicel(self, i):
        """Stem polyline, bent along when the fruit has been pushed aside."""
        o = self.offset.get(i)
        if o is None:
            return self.ped[i]
        w = np.linspace(0, 1, len(self.ped[i]))[:, None]
        return self.ped[i] + w * o

    def pos(self, data, i):
        """Current fruit centre (world)."""
        R = quat2mat(data.mocap_quat[self.mocap[i]])
        return data.mocap_pos[self.mocap[i]] + R @ np.array([0, 0, -0.42 * self.meta[i]["length"]])


# ----------------------------------------------------------------------------------------
# the harvester
# ----------------------------------------------------------------------------------------
class Harvester:
    def __init__(self, model, data, mode="vision", aisles=None, log_path=None, second_look=True):
        self.m, self.d = model, data
        self.mode = mode
        self.io = robot.RobotIO(model)
        self.route = Route()
        self.aisles = set(range(len(set(a for _, a in self.route.stops)))) if aisles is None else set(aisles)
        self.second_look = second_look and mode == "vision"
        self.berries = Berries(model, data)
        self.crate_bodies = [model.body("crate_a").id, model.body("crate_b").id]
        self.full_crates = [model.body_mocapid[model.body(f"full_crate_{k}").id] for k in range(len(full_crate_spots()))]
        self.pad_geoms = {model.geom("finger_l_pad").id: 0, model.geom("finger_r_pad").id: 1}
        self.finger_acts = [self.io.act["finger_l"], self.io.act["finger_r"]]

        # small robot-only model for fast IK in the cart frame
        self.ik_m = mujoco.MjModel.from_xml_path(str(IK_MODEL))
        self.ik_d = mujoco.MjData(self.ik_m)
        self.ik_io = robot.RobotIO(self.ik_m)
        self.ik_io.place_cart(self.ik_d, 0, 0, 0, 0, 0, 0)
        self.ik_io.set_lift(self.ik_d, robot.LIFT_HOME)
        self.ik_io.set_tool(self.ik_d, robot.FINGER_OPEN, robot.BLADE_OPEN, qpos_too=True)
        self.templ = robot.harvest_poses(self.ik_m)
        self.lo = np.array([self.ik_m.jnt_range[self.ik_m.joint(j).id][0] if self.ik_m.jnt_limited[self.ik_m.joint(j).id] else -9 for j in robot.ARM_JOINTS])
        self.hi = np.array([self.ik_m.jnt_range[self.ik_m.joint(j).id][1] if self.ik_m.jnt_limited[self.ik_m.joint(j).id] else 9 for j in robot.ARM_JOINTS])
        self.crate_pose_cache = {}
        # crate slots the arm can reach straight down (filled in this order)
        self.slots = {}
        for c in (0, 1):
            self.slots[c] = []
            for k in range(len(CRATE_SLOTS)):
                self._crate_pose(c, k)
                if self._crate_reach_ok:
                    self.slots[c].append(k)

        self.perception = None
        if mode == "vision":
            import perception
            self.perception = perception.Perception(model)

        self.log_path = log_path
        self.reset()

    # ------------------------------------------------------------------ state / reset
    def reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        self.berries.reset_state(d)
        for k, mid in enumerate(self.full_crates):
            d.mocap_pos[mid] = [0, 0, -3]
        if self.perception:
            self.perception.map = type(self.perception.map)()
        self.s = 0.0
        self._placed_s = None
        self.travelled, self._prev_xy = 0.0, None
        self.q_ref = np.array(robot.GEN3_STOW)
        self.fingers, self.blade = robot.FINGER_OPEN, robot.BLADE_OPEN
        self._place_cart()
        self.io.set_lift(d, robot.LIFT_HOME)
        self.io.set_arm(d, self.q_ref)
        self.io.set_tool(d, self.fingers, self.blade, qpos_too=True)
        self._set_grip_limit(4.0)
        mujoco.mj_forward(m, d)
        self.t = 0.0
        self.status = "Starting"
        self.done = False
        self.crates_shipped = 0
        self.stats = dict(attempts=0, picked=0, damaged=0, grasp_miss=0, cut_miss=0, unreachable=0,
                          collateral=0, collateral_ripe=0, unripe_picked=0, crate_full=0,
                          detections=0, images=0, not_confirmed=0, regrips=0, cycle_times=[], refine_shift_mm=[])
        self.events = []
        self.log = []
        self._prog = self._mission()
        self._t_wall0 = time.perf_counter()

    # ------------------------------------------------------------------ per-step update
    def tick(self):
        """Advance the mission by one physics step (call before mujoco.mj_step)."""
        if not self.done:
            try:
                next(self._prog)
            except StopIteration:
                self.done = True
                self.status = "Harvest complete"
                self._finish()
        self._place_cart()
        d = self.d
        for j, v in zip(robot.ARM_JOINTS, self.q_ref):
            d.ctrl[self.io.act[j]] = v
        self.io.set_tool(d, self.fingers, self.blade)
        d.ctrl[self.io.act["lift"]] = robot.LIFT_HOME
        self._update_berries()
        self._watch_contacts()
        self.t += self.m.opt.timestep

    def _place_cart(self):
        if getattr(self, "_placed_s", None) != self.s:      # pose only changes while driving
            self._placed_s = self.s
            x, y, yaw, yf, yr = self.route.vehicle_pose(self.s, robot.BOGIE_X)
            if self._prev_xy is not None:
                self.travelled += (x - self._prev_xy[0]) * math.cos(yaw) + (y - self._prev_xy[1]) * math.sin(yaw)
            self._prev_xy = (x, y)
            self._pose = (x, y, yaw, yf, yr)
        x, y, yaw, yf, yr = self._pose
        self.io.place_cart(self.d, x, y, yaw, yf, yr, self.travelled)
        self.cart = (x, y, yaw)

    def _set_grip_limit(self, f):
        for a in self.finger_acts:
            self.m.actuator_forcerange[a] = (-f, f)

    def _tcp(self):
        d = self.d
        return d.site_xpos[self.io.tcp].copy(), d.site_xmat[self.io.tcp].reshape(3, 3).copy()

    def _update_berries(self):
        B, d = self.berries, self.d
        if B.held is not None:
            i, rp, rq = B.held
            p, R = self._tcp()
            d.mocap_pos[B.mocap[i]] = p + R @ rp
            d.mocap_quat[B.mocap[i]] = qmul_batch(mat2quat(R), rq)
        dt = self.m.opt.timestep
        still = []
        for fobj in B.falling:
            fobj["t"] += dt
            a = min(fobj["t"] / fobj["T"], 1.0)
            p = fobj["p0"] + (fobj["p1"] - fobj["p0"]) * np.array([a, a, a * a])
            q = nlerp(fobj["q0"], fobj["q1"], a)
            if fobj["crate"] is not None:
                c = self.crate_bodies[fobj["crate"]]
                Rc = d.xmat[c].reshape(3, 3)
                d.mocap_pos[B.mocap[fobj["i"]]] = d.xpos[c] + Rc @ p
                d.mocap_quat[B.mocap[fobj["i"]]] = qmul_batch(d.xquat[c], q)
            else:
                d.mocap_pos[B.mocap[fobj["i"]]] = p
                d.mocap_quat[B.mocap[fobj["i"]]] = q
            if a >= 1.0:
                if fobj["crate"] is not None:
                    B.crate_items[fobj["crate"]].append((fobj["i"], fobj["p1"], fobj["q1"]))
                    B.state[fobj["i"]] = "crate"
                else:
                    B.state[fobj["i"]] = "floor"
            else:
                still.append(fobj)
        B.falling = still
        for k, items in B.crate_items.items():
            if not items:
                continue
            c = self.crate_bodies[k]
            idx = np.array([it[0] for it in items])
            lp = np.array([it[1] for it in items])
            lq = np.array([it[2] for it in items])
            d.mocap_pos[B.mocap[idx]] = d.xpos[c] + lp @ d.xmat[c].reshape(3, 3).T
            d.mocap_quat[B.mocap[idx]] = qmul_batch(np.broadcast_to(d.xquat[c], lq.shape), lq)

    def _watch_contacts(self):
        """Pad/fruit contacts. A fruit squeezed by both pads reports the contact force (grip).
        A fruit touched by one pad only is pushed aside and swings on its stem (soft force)."""
        d, B = self.d, self.berries
        dt = self.m.opt.timestep
        pushed = {}
        if d.ncon:
            f6 = np.zeros(6)
            self._touch = getattr(self, "_touch", {})
            per = {}
            for k in range(d.ncon):
                c = d.contact[k]
                g1, g2 = c.geom1, c.geom2
                if g1 in self.pad_geoms and g2 in B.geom2berry:
                    pad, b, sgn = self.pad_geoms[g1], B.geom2berry[g2], 1.0
                elif g2 in self.pad_geoms and g1 in B.geom2berry:
                    pad, b, sgn = self.pad_geoms[g2], B.geom2berry[g1], -1.0
                else:
                    continue
                mujoco.mj_contactForce(self.m, d, k, f6)
                per.setdefault(b, []).append((pad, abs(f6[0]), sgn * c.frame[:3].copy(), c.dist))
                self._touch.setdefault(b, set()).add(pad)
            for b, cl in per.items():
                if B.state[b] != "hanging":
                    continue                       # held fruit: squeeze is limited by the finger drive
                pads = {c[0] for c in cl}
                if len(pads) == 2:
                    B.peak[b] = max(B.peak[b], max(c[1] for c in cl))
                else:
                    # move the fruit just clear of the pad (it swings away freely; soft)
                    pushed[b] = sum(n * (max(-dist, 0.0) + 0.0004) for _, _, n, dist in cl)
        # pushed fruit move away; the others swing back
        for b in set(B.offset) | set(pushed):
            o = B.offset.get(b, np.zeros(3))
            if b in pushed:
                o = o + pushed[b]
                o[2] = max(o[2], -0.01)
                # a pushed fruit slides around the fruit next to it instead of passing through
                for j in B.neigh[b]:
                    if B.state[j] != "hanging":
                        continue
                    v = (B.center[b] + o) - (B.center[j] + B.offset.get(j, 0.0))
                    dist = np.linalg.norm(v)
                    gap = B.radius[b] + B.radius[j] - dist
                    if gap > 0 and dist > 1e-6:
                        o = o + v / dist * gap
                B.peak[b] = max(B.peak[b], K_SWING * np.linalg.norm(o))
            else:
                o = o * (1 - dt / SWING_TAU)
            if B.state[b] != "hanging" or np.linalg.norm(o) < 1e-4:
                B.offset.pop(b, None)
                if B.state[b] == "hanging":
                    d.mocap_pos[B.mocap[b]] = B.home_pos[b]
                continue
            B.offset[b] = o
            d.mocap_pos[B.mocap[b]] = B.home_pos[b] + o

    # ------------------------------------------------------------------ motion primitives
    def _wait(self, seconds):
        n = max(1, int(round(seconds / self.m.opt.timestep)))
        for _ in range(n):
            yield

    def _move_joints(self, q_goal, min_time=0.3, path=None):
        """Minimum-jerk joint-space move (optionally through intermediate waypoints)."""
        q0 = self.q_ref.copy()
        pts = [q0] + ([robot.unwrap_like(p, q0) for p in path] if path else []) + [robot.unwrap_like(q_goal, q0)]
        for k in range(1, len(pts)):
            pts[k] = robot.unwrap_like(pts[k], pts[k - 1])
        seg = np.array([np.max(np.abs(b - a) / ARM_VMAX) for a, b in zip(pts[:-1], pts[1:])])
        total = float(seg.sum())
        T = max(min_time, 1.875 * total)          # minimum-jerk peak speed = 1.875 x average
        cum = np.concatenate([[0], np.cumsum(seg)]) / max(total, 1e-9)
        n = max(1, int(T / self.m.opt.timestep))
        for k in range(1, n + 1):
            u = min_jerk(k / n)
            j = min(np.searchsorted(cum, u, side="right") - 1, len(pts) - 2)
            a = (u - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
            self.q_ref = pts[j] + (pts[j + 1] - pts[j]) * a
            yield
        self.q_ref = pts[-1]
        # settle: wait for the arm to catch up with the reference
        for _ in range(int(0.6 / self.m.opt.timestep)):
            if np.max(np.abs(self.d.qpos[self.io.arm_q] - self.q_ref)) < 0.006:
                break
            yield

    def _drive_to(self, s_goal, vmax, label, scans=()):
        s0 = self.s
        dist = s_goal - s0
        L = abs(dist)
        if L < 1e-6:
            return
        ta = vmax / ACCEL
        if L < vmax * ta:
            ta = math.sqrt(L / ACCEL)
            vmax = ACCEL * ta
        T = 2 * ta + (L - vmax * ta) / vmax
        dt = self.m.opt.timestep
        scans = sorted(scans)
        self.status = label
        t = 0.0
        while t < T:
            t += dt
            if t < ta:
                x = 0.5 * ACCEL * t * t
            elif t > T - ta:
                x = L - 0.5 * ACCEL * (T - t) ** 2
            else:
                x = 0.5 * ACCEL * ta * ta + vmax * (t - ta)
            self.s = s0 + math.copysign(min(x, L), dist)
            while scans and abs(self.s - s0) >= scans[0] * L:
                scans.pop(0)
                self._scan()
            yield
        self.s = s_goal

    # ------------------------------------------------------------------ perception / targets
    def _scan(self):
        if self.perception is None:
            return
        mujoco.mj_forward(self.m, self.d)
        new, n = self.perception.scan(self.d)
        self.stats["detections"] += n
        self.stats["images"] = self.perception.n_images

    def _targets(self, x_arm, aisle):
        """Ripe fruit to pick at this stop: list of (key, centre, berry-or-None)."""
        ya = aisle_y(aisle)
        if self.mode == "oracle":
            B = self.berries
            sel = [i for i in range(B.n) if B.ripe[i] and B.state[i] == "hanging"
                   and abs(B.center[i, 0] - x_arm) <= REACH_X and abs(B.center[i, 1] - ya) < 0.75]
            return [(("gt", i), B.center[i].copy()) for i in sel]
        items = self.perception.map.candidates(x_arm - REACH_X, x_arm + REACH_X, ya - 0.75, ya + 0.75)
        return [(("map", it["id"]), it["pos"].copy()) for it in items]

    def _mark(self, key, status):
        if key[0] == "map":
            for it in self.perception.map.items:
                if it["id"] == key[1]:
                    it["status"] = status

    # ------------------------------------------------------------------ planning
    def _to_cart(self, p):
        x, y, yaw = self.cart
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p[0] - x, p[1] - y
        return np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])

    def _to_world_dir(self, v):
        _, _, yaw = self.cart
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])

    def _ik(self, pos, mat, seed):
        q, ep, er = robot.solve_ik(self.ik_m, self.ik_d, self.ik_io, pos, mat, seed, iters=120)
        return q, ep < 0.002 and er < 0.03

    def _plan_pick(self, target_w, others_w, avoid_psi=None):
        """Choose the approach and solve IK for pre-grasp, approach path and grasp (cart frame)."""
        p = self._to_cart(target_w)
        side = 1.0 if p[1] > 0 else -1.0
        seed = self.templ["pre_L" if side > 0 else "pre_R"]
        others = [self._to_cart(o) for o in others_w]
        best = None
        for psi in YAW_OPTIONS:
            if avoid_psi is not None and abs(psi - avoid_psi) < 0.5:
                continue
            h = np.array([math.sin(psi), side * math.cos(psi), 0.0])
            app = h + [0, 0, TILT]
            app /= np.linalg.norm(app)
            R = tool_mat(app)
            # neighbours inside the gripper's sweep (between the fingers / in the knife window)
            crowd = 0.0
            for o in others:
                v = R.T @ (o - p)
                if abs(v[0]) < 0.04 and -0.03 < v[1] < 0.045 and -0.06 < v[2] < 0.05:
                    crowd += 1.0
            q_pre, ok1 = self._ik(p - PRE_DIST * app, R, seed)
            if not ok1:
                continue
            q_pick, ok2 = self._ik(p, R, q_pre)
            if not ok2:
                continue
            margin = float(np.min(np.minimum(q_pick - self.lo, self.hi - q_pick)))
            score = crowd * 2.0 + abs(psi) * 0.3 - min(margin, 0.5)
            if best is None or score < best[0]:
                best = (score, psi, app, R, q_pre, q_pick)
        if best is None:
            return None
        _, psi, app, R, q_pre, q_pick = best
        path = []
        q = q_pre
        for a in (0.33, 0.66):
            q, _ = self._ik(p - PRE_DIST * (1 - a) * app, R, q)
            path.append(q)
        return dict(p=p, app=app, R=R, q_pre=q_pre, q_pick=q_pick, path=path, side=side, psi=psi)

    def _replan_from_pre(self, plan, new_target_w):
        p = self._to_cart(new_target_w)
        q_pick, ok = self._ik(p, plan["R"], plan["q_pre"])
        if not ok:
            return False
        path, q = [], plan["q_pre"]
        for a in (0.33, 0.66):
            q, _ = self._ik(plan["p"] - PRE_DIST * plan["app"] + (p - plan["p"] + PRE_DIST * plan["app"]) * a, plan["R"], q)
            path.append(q)
        plan.update(p=p, q_pick=q_pick, path=path)
        return True

    def _crate_pose(self, crate, slot):
        """Joint pose holding the fruit just above crate slot `slot` (cached; never None)."""
        key = (crate, slot)
        if key in self.crate_pose_cache:
            return self.crate_pose_cache[key]
        cy = robot.CRATE_Y if crate == 0 else -robot.CRATE_Y
        sp = CRATE_SLOTS[slot]
        pos = np.array([robot.CRATE_X + sp[0], cy + sp[1], robot.CHASSIS_Z1 + sp[2] + 0.055])
        seeds = [self.templ["crate_L" if crate == 0 else "crate_R"]]
        seeds += [self.crate_pose_cache[(crate, k)] for k in (slot - 1, slot - 8, slot - 48)
                  if (crate, k) in self.crate_pose_cache]
        q_ok, self._crate_reach_ok = None, True
        for yaxis in ([1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [-1.0, 0, 0]):     # fingers' heading over the crate
            R = tool_mat([0, 0, -1.0], yaxis)
            for sd in seeds:
                q, ok = self._ik(pos, R, sd)
                if ok:
                    q_ok = q
                    break
            if q_ok is not None:
                break
        if q_ok is None:                       # fall back to the nearest slot the arm can reach
            self._crate_reach_ok = False
            done = [(np.linalg.norm(CRATE_SLOTS[k] - sp), q) for (c, k), q in self.crate_pose_cache.items() if c == crate]
            q_ok = min(done, key=lambda t: t[0])[1] if done else self.templ["crate_L" if crate == 0 else "crate_R"]
        self.crate_pose_cache[key] = q_ok
        return q_ok

    # ------------------------------------------------------------------ physical checks
    def _knife_cuts(self):
        """Fruit whose stem crosses the knife's cut window right now."""
        tcp, R = self._tcp()
        origin = tcp - R[:, 2] * robot.TCP_Z
        x0, x1, z0, z1 = robot.CUT_WINDOW
        B = self.berries
        near = np.where((B.state == "hanging") & (np.linalg.norm(B.center - tcp, axis=1) < 0.15))[0]
        cut = []
        for i in near:
            P = (B.pedicel(i) - origin) @ R
            for a, b in zip(P[:-1], P[1:]):
                if (a[1] - robot.CUT_Y) * (b[1] - robot.CUT_Y) <= 0 and a[1] != b[1]:
                    t = (robot.CUT_Y - a[1]) / (b[1] - a[1])
                    c = a + (b - a) * t
                    if x0 <= c[0] <= x1 and z0 <= c[2] <= z1:
                        cut.append(int(i))
                    break
        return cut

    def _drop_to_floor(self, i):
        B, d = self.berries, self.d
        p0 = d.mocap_pos[B.mocap[i]].copy()
        p1 = p0.copy()
        p1[2] = 0.025
        B.state[i] = "falling"
        B.falling.append(dict(i=i, t=0.0, T=math.sqrt(2 * (p0[2] - p1[2]) / 9.81), p0=p0, p1=p1,
                              q0=d.mocap_quat[B.mocap[i]].copy(), q1=lying_quat(B.rng), crate=None))

    # ------------------------------------------------------------------ one pick
    def _pick(self, key, target_w, others_w, where, avoid_psi=None):
        """One fruit. On a cut miss (stem outside the knife window) it re-grips once from another angle."""
        B, d, st = self.berries, self.d, self.stats
        t_start = self.t
        retry = avoid_psi is not None
        if not retry:
            st["attempts"] += 1
        plan = self._plan_pick(target_w, others_w, avoid_psi)
        rec = dict(t=round(self.t, 2), where=where, target=[round(v, 4) for v in target_w], outcome="", berry=-1,
                   peak_N=0.0, cycle_s=0.0, refine_mm=None)
        if plan is None and retry:
            st["cut_miss"] += 1
            return
        if plan is None:
            st["unreachable"] += 1
            self._mark(key, "unreachable")
            rec["outcome"] = "unreachable"
            self.log.append(rec)
            return
        self.status = f"{where}: approaching fruit"
        self.fingers = FINGER_PREOPEN
        self.blade = robot.BLADE_OPEN
        yield from self._move_joints(plan["q_pre"])
        if self.perception is not None:
            self.status = f"{where}: wrist camera check"
            mujoco.mj_forward(self.m, d)
            new = self.perception.refine(d, target_w)
            if new is None:
                # the wrist camera does not see the fruit where the map says it is: skip it
                st["not_confirmed"] += 1
                self._mark(key, "not_confirmed")
                rec["outcome"] = "not_confirmed"
                rec["cycle_s"] = round(self.t - t_start, 2)
                self.log.append(rec)
                return
            else:
                shift = float(np.linalg.norm(new - target_w))
                if self._replan_from_pre(plan, new):
                    rec["refine_mm"] = round(1000 * shift, 1)
                    st["refine_shift_mm"].append(shift * 1000)
                    target_w = new
        self.status = f"{where}: approaching fruit"
        yield from self._move_joints(plan["q_pick"], min_time=0.6, path=plan["path"])

        # ---- soft, force-limited grip
        self.status = f"{where}: soft grip ({GRIP_FORCE:.1f} N limit)"
        self._touch = {}
        self._set_grip_limit(GRIP_FORCE)
        self.fingers = 0.0
        yield from self._wait(0.45)
        tcp, _ = self._tcp()
        touched = [b for b, pads in self._touch.items() if len(pads) == 2 and B.state[b] == "hanging"]
        grasped = min(touched, key=lambda b: np.linalg.norm(B.pos(d, b) - tcp)) if touched else None
        if grasped is None:
            st["grasp_miss"] += 1
            rec["outcome"] = "grasp_miss"
        else:
            rec["berry"] = int(grasped)
            # ---- cut the stem
            self.status = f"{where}: cutting stem"
            self.blade = robot.BLADE_CLOSED
            yield from self._wait(0.18)
            cut = self._knife_cuts()
            yield from self._wait(0.12)
            self.blade = robot.BLADE_OPEN
            for i in cut:
                if i != grasped:
                    self._drop_to_floor(i)
                    st["collateral"] += 1
                    st["collateral_ripe"] += int(B.ripe[i])
                    self.events.append(f"{where}: neighbour stem cut, fruit {i} dropped")
            if grasped in cut:
                p, R = self._tcp()
                bp = d.mocap_pos[B.mocap[grasped]]
                B.held = (grasped, R.T @ (bp - p), qmul_batch(mat2quat(R.T), d.mocap_quat[B.mocap[grasped]]))
                B.state[grasped] = "held"
            elif retry:
                st["cut_miss"] += 1
                rec["outcome"] = "cut_miss"
            else:
                st["regrips"] += 1
                rec["outcome"] = "regrip"
        # ---- retract
        self.status = f"{where}: retract"
        if B.held is None:
            self._set_grip_limit(4.0)
            self.fingers = robot.FINGER_OPEN
            yield from self._wait(0.2)
        yield from self._move_joints(plan["q_pre"], min_time=0.4, path=plan["path"][::-1])
        if B.held is not None:
            i = B.held[0]
            crate = 0 if plan["side"] > 0 else 1
            fill = lambda c: len(B.crate_items[c]) + sum(f["crate"] == c for f in B.falling)
            if fill(crate) >= len(self.slots[crate]):
                crate = 1 - crate
            n = fill(crate)
            if n >= len(self.slots[crate]):
                st["crate_full"] += 1
                n = len(self.slots[crate]) - 1
            slot = self.slots[crate][n]
            q_crate = self._crate_pose(crate, slot)
            self.status = f"{where}: lowering into crate {'AB'[crate]}"
            yield from self._move_joints(q_crate)
            self.status = f"{where}: release"
            self._set_grip_limit(4.0)
            self.fingers = robot.FINGER_OPEN
            yield from self._wait(0.12)
            c = self.crate_bodies[crate]
            Rc = d.xmat[c].reshape(3, 3)
            p0 = Rc.T @ (d.mocap_pos[B.mocap[i]] - d.xpos[c])
            q0 = qmul_batch(mat2quat(Rc.T), d.mocap_quat[B.mocap[i]])
            sp = CRATE_SLOTS[min(slot, len(CRATE_SLOTS) - 1)] + np.r_[B.rng.uniform(-0.004, 0.004, 2), 0.0]
            p1 = sp
            B.falling.append(dict(i=i, t=0.0, T=max(0.08, math.sqrt(2 * max(p0[2] - p1[2], 0.005) / 9.81)),
                                  p0=p0, p1=p1, q0=q0, q1=lying_quat(B.rng), crate=crate))
            B.state[i] = "falling"
            B.held = None
            dmg = B.peak[i] > B.tolerance[i]
            st["picked"] += 1
            st["damaged"] += int(dmg)
            st["unripe_picked"] += int(not B.ripe[i])
            rec["outcome"] = "picked_damaged" if dmg else "picked"
            rec["peak_N"] = round(float(B.peak[i]), 2)
            st["cycle_times"].append(self.t - t_start)
        self._set_grip_limit(4.0)
        self._mark(key, "done")
        rec["cycle_s"] = round(self.t - t_start, 2)
        self.log.append(rec)
        if rec["outcome"] == "regrip":
            self.status = f"{where}: stem not in knife window, re-gripping from another angle"
            yield from self._pick(key, target_w, others_w, where, avoid_psi=plan["psi"])

    # ------------------------------------------------------------------ second look (wrist camera)
    def _second_look(self, x_arm, aisle, where):
        """Point the wrist camera at each half of the pick window to find fruit the mast missed."""
        ya = aisle_y(aisle)
        found = 0
        for wy in (ya - 0.555, ya + 0.555):
            for dx in (-0.22, 0.22):
                look_at = np.array([x_arm + dx, wy, 0.95])
                p = self._to_cart(look_at)
                side = 1.0 if p[1] > 0 else -1.0
                eye = p - np.array([0, side * 0.36, -0.05])
                R = tool_mat(p - eye)
                q, ok = self._ik(eye - R @ np.array([0, 0.0, 0.0]), R, self.templ["pre_L" if side > 0 else "pre_R"])
                if not ok:
                    continue
                self.status = f"{where}: wrist camera second look"
                yield from self._move_joints(q)
                mujoco.mj_forward(self.m, self.d)
                rgb, depth = self.perception.wrist.grab(self.d, self.perception.opt)
                self.perception.n_images += 1
                import perception as P
                dets = P.detect(self.perception.wrist, rgb, depth, max_range=0.7)
                found += self.perception.map.add(dets)
        return found

    # ------------------------------------------------------------------ mission
    def _mission(self):
        route, B = self.route, self.berries
        stops = list(route.stops)
        for kind, aisle, s0, s1 in route.legs:
            if kind == "in":
                if aisle not in self.aisles:
                    yield from self._drive_to(s1, AISLE_SPEED * 1.6, f"Aisle {aisle}: passing")
                    continue
                mine = [s for s, a in stops if a == aisle]
                for k, s_stop in enumerate(mine, 1):
                    where = f"Aisle {aisle} stop {k}/{len(mine)}"
                    yield from self._drive_to(s_stop, AISLE_SPEED, f"Aisle {aisle}: driving to stop {k}/{len(mine)}",
                                              scans=(0.34, 0.67))
                    x_arm = self.d.xpos[self.m.body("lift").id][0]
                    self._scan()
                    for rnd in range(2):
                        targets = self._targets(x_arm, aisle)
                        # left row first, then right; along the row in driving direction
                        targets.sort(key=lambda t: (t[1][1] < aisle_y(aisle), t[1][0], -t[1][2]))
                        for key, tw in targets:
                            if key[0] == "gt" and B.state[key[1]] != "hanging":
                                continue
                            others = [B.pos(self.d, i) for i in np.where(B.state == "hanging")[0]
                                      if np.linalg.norm(B.center[i] - tw) < 0.08]
                            others = [o for o in others if np.linalg.norm(o - tw) > 0.015]
                            yield from self._pick(key, tw, others, where)
                        if rnd == 0 and self.second_look:
                            if not (yield from self._second_look(x_arm, aisle, where)):
                                break
                        else:
                            break
                yield from self._move_joints(robot.GEN3_STOW)
                yield from self._drive_to(s1, AISLE_SPEED, f"Aisle {aisle}: end of row")
            elif kind == "out":
                yield from self._drive_to(s1, AISLE_SPEED * 1.6, f"Aisle {aisle}: returning to headland")
                if aisle in self.aisles:
                    yield from self._crate_swap(aisle)
            else:
                yield from self._drive_to(s1, TRANSFER_SPEED, "Switching track via headland cross-rail")

    def _crate_swap(self, aisle):
        B, d = self.berries, self.d
        self.status = f"Aisle {aisle} done: handing off full crates"
        yield from self._wait(1.0)
        spots = full_crate_spots()
        for k in (0, 1):
            if not B.crate_items[k]:
                continue
            for i, _, _ in B.crate_items[k]:
                B.state[i] = "shipped"
                d.mocap_pos[B.mocap[i]] = [0, 0, -5]
            B.crate_items[k] = []
            if self.crates_shipped < len(spots):
                d.mocap_pos[self.full_crates[self.crates_shipped]] = spots[self.crates_shipped]
            self.crates_shipped += 1
        yield from self._wait(1.5)

    # ------------------------------------------------------------------ reporting
    def summary(self):
        B, st = self.berries, self.stats
        in_scope = np.array([any(abs(B.center[i, 1] - aisle_y(a)) < 0.75 for a in self.aisles) for i in range(B.n)])
        ripe = B.ripe & in_scope
        left_ripe = int(np.sum(ripe & (B.state == "hanging")))
        bruised_on_plant = int(np.sum(in_scope & (B.state == "hanging") & (B.peak > B.tolerance)))
        ct = st["cycle_times"]
        det = None
        if self.perception is not None:
            items = self.perception.map.items
            tp = fp_u = fp_n = 0
            for it in items:
                dist = np.linalg.norm(B.center - it["pos"], axis=1)
                j = int(np.argmin(dist))
                if dist[j] < 0.025:
                    tp += int(B.ripe[j])
                    fp_u += int(not B.ripe[j])
                else:
                    fp_n += 1
            det = dict(map_entries=len(items), true_ripe=tp, unripe=fp_u, no_fruit=fp_n, images=self.perception.n_images)
        return dict(
            mode=self.mode, aisles=sorted(self.aisles), sim_time_min=round(self.t / 60, 1),
            ripe_in_scope=int(ripe.sum()), unripe_in_scope=int((~B.ripe & in_scope).sum()),
            picked=st["picked"], picked_pct=round(100 * st["picked"] / max(1, ripe.sum()), 1),
            damaged=st["damaged"], damaged_pct=round(100 * st["damaged"] / max(1, st["picked"]), 1),
            attempts=st["attempts"], grasp_miss=st["grasp_miss"], cut_miss=st["cut_miss"],
            unreachable=st["unreachable"], collateral_drops=st["collateral"], collateral_ripe=st["collateral_ripe"],
            unripe_picked=st["unripe_picked"], ripe_left_on_plants=left_ripe, bruised_on_plant=bruised_on_plant,
            crates_shipped=self.crates_shipped,
            cycle_time_mean_s=round(float(np.mean(ct)), 2) if ct else None,
            cycle_time_p90_s=round(float(np.percentile(ct, 90)), 2) if ct else None,
            fruit_per_hour=round(st["picked"] / max(self.t, 1e-9) * 3600, 0),
            wrist_refine_mean_mm=round(float(np.mean(st["refine_shift_mm"])), 1) if st["refine_shift_mm"] else None,
            not_confirmed=st["not_confirmed"], regrips=st["regrips"],
            detection=det,
        )

    def hud(self):
        st = self.stats
        ct = st["cycle_times"]
        crate = [len(self.berries.crate_items[k]) for k in (0, 1)]
        lines = [
            f"picked {st['picked']}   damaged {st['damaged']}   missed grasp/cut {st['grasp_miss']}/{st['cut_miss']}",
            f"out of reach {st['unreachable']}   dropped {st['collateral']}   crates A/B {crate[0]}/{crate[1]}  shipped {self.crates_shipped}",
            f"cycle {np.mean(ct):.1f} s/fruit" if ct else "cycle -",
        ]
        if self.perception is not None:
            lines.append(f"vision: {len(self.perception.map.items)} ripe fruit mapped, {self.perception.n_images} images")
        return "\n".join(lines)

    def _finish(self):
        if self.log_path is None:
            return
        out = Path(self.log_path)
        out.mkdir(parents=True, exist_ok=True)
        s = self.summary()
        (out / f"summary_{self.mode}.json").write_text(json.dumps(s, indent=2))
        with open(out / f"picks_{self.mode}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.log[0].keys()) if self.log else ["t"])
            w.writeheader()
            for r in self.log:
                w.writerow(r)
