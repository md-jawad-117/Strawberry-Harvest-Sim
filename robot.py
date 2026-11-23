"""
Harvesting robot model: rail cart (two swivelling bogies) + telescopic lift + Kinova Gen3 7-DoF
(with vision module) + "cradle and cut" soft end-effector (fin-ray fingers + stem blade).

build_scene.py calls robot_assets(), robot_body(), robot_actuators().
Runtime helpers (used by the viewer / previews): RobotIO, solve_ik(), harvest_poses().

Cart frame: +x forward, +y left, z up, origin on the rail centre line at floor level.
Tool frame (site "tcp", between the fingers): +z = approach direction, +y = "up" side where the
blade sits (the stem is cut above the calyx), +x = finger opening direction.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
GEN3_DIR = ROOT / "third_party" / "kinova_gen3"

# ---- main dimensions (metres) ---------------------------------------------------------
CART_L, CART_W = 1.10, 0.66
CHASSIS_Z0, CHASSIS_Z1 = 0.24, 0.44          # chassis bottom / deck height
BOGIE_X = 0.38                               # bogie pivot distance from cart centre
WHEEL_R, GAUGE = 0.05, 0.55
LIFT_X = 0.02                                # lift column position along the cart
PLATE_Z0 = 0.57                              # arm mounting plate height at lift = 0
LIFT_RANGE = (0.0, 0.38)                     # -> arm base 0.57 .. 0.95 m
LIFT_HOME = 0.10                             # -> arm base at 0.67 m (shoulder ~ fruit height)
MAST_X = 0.46
CRATE_X, CRATE_Y = -0.32, 0.16               # crate centres (two crates side by side)
CRATE_L, CRATE_W, CRATE_H = 0.40, 0.30, 0.12

ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
CONTINUOUS = {"joint_1", "joint_3", "joint_5", "joint_7"}
FINGER_OPEN, FINGER_HOLD = 0.030, 0.011      # finger slide position (per finger)
BLADE_OPEN, BLADE_CLOSED = 0.0, 0.064          # sliding knife travel (closes across the cut window)
CUT_Y = 0.025                                 # cut plane height above the fruit centre (tool +y)
CUT_WINDOW = (-0.022, 0.024, 0.062, 0.150)    # tool-frame x0, x1, z0, z1: knife cuts between the closed pads
TCP_Z = 0.10                                  # tcp (fruit centre between the pads) on the tool axis


def f(*v) -> str:
    out = []
    for x in v:
        s = f"{x:.5f}".rstrip("0").rstrip(".")
        out.append("0" if s in ("-0", "") else s)
    return " ".join(out)


def _quat_y(angle):
    return (math.cos(angle / 2), 0, math.sin(angle / 2), 0)


# ======================================================================================
# MJCF generation
# ======================================================================================
def robot_assets(meshdir_rel_to_project="third_party/kinova_gen3/assets"):
    """Mesh + material assets. Scene meshdir is scene/assets, so paths go up two levels."""
    tree = ET.parse(GEN3_DIR / "gen3.xml")
    meshes = []
    for m in tree.getroot().find("asset"):
        meshes.append(f'<mesh name="gen3_{m.get("name")}" file="../../{meshdir_rel_to_project}/{m.get("file")}"/>')
    # fin-ray finger outline (left finger; inner contact face at x = 0, outer face toward +x)
    return "\n".join("    " + s for s in meshes) + """
    <material name="gen3"         rgba="0.80 0.81 0.83 1" specular="0.6" shininess="0.6"/>
    <material name="gen3_dark"    rgba="0.12 0.12 0.13 1" specular="0.4"/>
    <material name="cart_body"    rgba="0.94 0.94 0.95 1" specular="0.4" shininess="0.4"/>
    <material name="cart_dark"    rgba="0.20 0.21 0.23 1" specular="0.3"/>
    <material name="cart_accent"  rgba="0.95 0.45 0.08 1" specular="0.4"/>
    <material name="rubber"       rgba="0.06 0.06 0.06 1"/>
    <material name="alu"          rgba="0.72 0.74 0.77 1" specular="0.8" shininess="0.7"/>
    <material name="wheel_steel"  rgba="0.45 0.47 0.5 1" specular="0.9" shininess="0.8"/>
    <material name="estop_red"    rgba="0.85 0.05 0.05 1" specular="0.5"/>
    <material name="estop_yellow" rgba="0.95 0.80 0.05 1"/>
    <material name="led_green"    rgba="0.15 1 0.35 1" emission="1"/>
    <material name="beacon"       rgba="1 0.6 0.1 0.9" emission="0.8"/>
    <material name="lens"         rgba="0.05 0.08 0.12 1" specular="1" shininess="1"/>
    <material name="crate_a"      rgba="0.12 0.42 0.75 1" specular="0.3"/>
    <material name="crate_b"      rgba="0.15 0.55 0.30 1" specular="0.3"/>
    <material name="foam"         rgba="0.85 0.86 0.80 1"/>
    <material name="tpu"          rgba="0.10 0.55 0.85 1" specular="0.3"/>
    <material name="gripper_body" rgba="0.25 0.26 0.28 1" specular="0.5"/>
    <material name="blade"        rgba="0.85 0.87 0.9 1" specular="1" shininess="1"/>
"""


def _finray_finger(name, sign):
    """Fin-ray finger: inner contact beam, slanted outer beam and cross ribs (TPU).
    sign=+1 left finger (opens toward +x), -1 right finger."""
    L, W, H = 0.075, 0.018, 0.011          # length, root width, half height
    g = []
    # inner (contact) beam along z at x = 0..0.0025
    g.append(f'<geom name="{name}_pad" type="box" pos="{f(sign*0.00125, 0, L/2)}" size="{f(0.00125, H, L/2)}" material="tpu" '
             f'conaffinity="2" condim="3" friction="1.2 0.01 0.001" solref="0.008 1" solimp="0.9 0.95 0.002"/>')
    # outer beam from (W, 0) at the root to (0.002, L) at the tip
    ang = math.atan2(W - 0.002, L)
    ln = math.hypot(W - 0.002, L) / 2
    g.append(f'<geom type="box" pos="{f(sign*(W+0.002)/2, 0, L/2)}" size="{f(0.00125, H, ln)}" '
             f'quat="{f(*_quat_y(-sign*ang))}" material="tpu"/>')
    # ribs
    for z in (0.014, 0.030, 0.046, 0.060):
        xo = W - (W - 0.002) * z / L
        g.append(f'<geom type="box" pos="{f(sign*xo/2, 0, z)}" size="{f(xo/2, H*0.9, 0.0009)}" material="tpu"/>')
    # root clamp
    g.append(f'<geom type="box" pos="{f(sign*W/2, 0, -0.004)}" size="{f(W/2+0.002, H+0.002, 0.005)}" material="gripper_body"/>')
    return "\n".join(g)


def end_effector_xml():
    """Cradle-and-cut end-effector, mounted on the Gen3 bracelet flange (tool +z outward)."""
    return f"""
<body name="ee" pos="0 0 -0.061525" quat="0 1 0 0">
  <geom type="cylinder" pos="0 0 0.004" size="0.037 0.004" material="gripper_body"/>
  <geom type="box" pos="0 0 0.031" size="0.046 0.030 0.023" material="gripper_body"/>
  <geom type="box" pos="0 -0.0302 0.031" size="0.040 0.0006 0.012" material="cart_accent"/>
  <geom type="box" pos="0 0 0.057" size="0.044 0.008 0.003" material="alu"/>
  <!-- knife carrier (+x) and anvil (-x) rails along the fingers, in the cut plane above the fruit -->
  <geom type="box" pos="0.037 0.022 0.058" size="0.004 0.010 0.006" material="gripper_body"/>
  <geom type="box" pos="-0.037 0.022 0.058" size="0.004 0.010 0.006" material="gripper_body"/>
  <geom type="box" pos="0.037 {CUT_Y + 0.004:.4f} 0.104" size="0.003 0.003 0.044" material="gripper_body"/>
  <geom type="box" pos="-0.037 {CUT_Y + 0.004:.4f} 0.104" size="0.003 0.003 0.044" material="gripper_body"/>
  <geom type="box" pos="-0.0335 {CUT_Y:.4f} 0.106" size="0.0015 0.0012 0.040" material="blade"/>
  <site name="tcp" pos="0 0 {TCP_Z}" size="0.005" rgba="1 0.1 0.1 0.5" group="4"/>
  <!-- recording camera (demo videos only): behind/above the fingers, looking at the grasp point -->
  <camera name="gripper_view" pos="0.05 0.1 0" xyaxes="-0.917 0.000 -0.399 -0.249 0.782 0.572" fovy="60"/>
  <site name="cut_point" pos="0 {CUT_Y:.4f} 0.106" size="0.003" rgba="1 1 0 0.6" group="4"/>
  <body name="finger_l" pos="0.008 0 0.062">
    <joint name="finger_l" type="slide" axis="1 0 0" range="0 0.032" damping="2" armature="0.01"/>
    {_finray_finger("finger_l", +1)}
  </body>
  <body name="finger_r" pos="-0.008 0 0.062">
    <joint name="finger_r" type="slide" axis="-1 0 0" range="0 0.032" damping="2" armature="0.01"/>
    {_finray_finger("finger_r", -1)}
  </body>
  <body name="blade" pos="0.034 {CUT_Y:.4f} 0.106">
    <joint name="blade" type="slide" axis="-1 0 0" range="0 0.066" damping="0.5" armature="0.002"/>
    <geom type="box" pos="-0.002 0 0" size="0.002 0.0008 0.040" material="blade"/>
    <geom type="box" pos="0.002 0.002 0" size="0.002 0.0015 0.040" material="gripper_body"/>
  </body>
</body>"""


def _gen3_body():
    """Gen3 kinematic tree from the Menagerie model: visual meshes only, gravity-compensated,
    with the end-effector attached to the bracelet."""
    root = ET.parse(GEN3_DIR / "gen3.xml").getroot()
    base = root.find("worldbody/body")

    def fix(body):
        body.set("gravcomp", "1")
        for el in list(body):
            if el.tag == "geom":
                if el.get("class") == "collision":
                    body.remove(el)
                else:
                    mesh = el.get("mesh")
                    for k in list(el.attrib):
                        del el.attrib[k]
                    el.set("type", "mesh"); el.set("mesh", f"gen3_{mesh}")
                    el.set("material", "gen3_dark" if mesh in ("base_link", "bracelet_with_vision_link") else "gen3")
            elif el.tag == "joint":
                el.set("armature", "0.1"); el.set("damping", "1")
            elif el.tag == "camera":
                el.set("name", "gen3_wrist_cam")
            elif el.tag == "site":
                body.remove(el)
            elif el.tag == "body":
                fix(el)

    fix(base)
    bracelet = base.find(".//body[@name='bracelet_link']")
    bracelet.append(ET.fromstring(end_effector_xml()))
    return ET.tostring(base, encoding="unicode")


def _bogie(name, x):
    axles = []
    for k, ax in enumerate((-0.10, 0.10)):
        wheels = []
        for wy in (-GAUGE / 2, GAUGE / 2):
            inner = -np.sign(wy)          # flange on the inside of each rail
            wheels.append(f'<geom type="cylinder" fromto="{f(0, wy-0.022, 0, 0, wy+0.022, 0)}" size="{WHEEL_R}" material="wheel_steel"/>')
            wheels.append(f'<geom type="cylinder" fromto="{f(0, wy+inner*0.022, 0, 0, wy+inner*0.030, 0)}" size="{WHEEL_R+0.012}" material="wheel_steel"/>')
            wheels.append(f'<geom type="cylinder" fromto="{f(0, wy-0.026, 0, 0, wy+0.026, 0)}" size="0.016" material="cart_dark"/>')
        axles.append(f"""
        <body name="{name}_axle{k}" pos="{f(ax, 0, 0)}">
          <joint name="{name}_axle{k}" type="hinge" axis="0 1 0"/>
          <geom type="cylinder" fromto="{f(0, -GAUGE/2+0.03, 0, 0, GAUGE/2-0.03, 0)}" size="0.012" material="cart_dark"/>
          {"".join(wheels)}
          <geom type="box" pos="0.03 0 0" size="0.03 0.004 0.004" material="cart_accent"/>
        </body>""")
    return f"""
      <body name="{name}" pos="{f(x, 0, WHEEL_R + 0.125)}">
        <joint name="{name}_yaw" type="hinge" axis="0 0 1"/>
        <geom type="box" pos="0 0 0.022" size="0.16 0.215 0.012" material="cart_dark"/>
        <geom type="cylinder" pos="0 0 0.045" size="0.06 0.012" material="alu"/>
        {"".join(axles)}
      </body>"""


def robot_body(chase_cam_xyaxes):
    zc = (CHASSIS_Z0 + CHASSIS_Z1) / 2
    hz = (CHASSIS_Z1 - CHASSIS_Z0) / 2
    hl, hw = CART_L / 2, CART_W / 2
    crates = []
    for k, (cy, mat) in enumerate(((CRATE_Y, "crate_a"), (-CRATE_Y, "crate_b"))):
        z0 = CHASSIS_Z1
        cx = CRATE_X
        crates.append(f"""
      <body name="crate_{'ab'[k]}" pos="{f(cx, cy, z0)}">
        <geom type="box" pos="0 0 0.005" size="{f(CRATE_L/2, CRATE_W/2, 0.005)}" material="{mat}"/>
        <geom type="box" pos="0 0 0.013" size="{f(CRATE_L/2-0.01, CRATE_W/2-0.01, 0.003)}" material="foam"/>
        <geom type="box" pos="{f(CRATE_L/2, 0, CRATE_H/2)}" size="{f(0.006, CRATE_W/2, CRATE_H/2)}" material="{mat}"/>
        <geom type="box" pos="{f(-CRATE_L/2, 0, CRATE_H/2)}" size="{f(0.006, CRATE_W/2, CRATE_H/2)}" material="{mat}"/>
        <geom type="box" pos="{f(0, CRATE_W/2, CRATE_H/2)}" size="{f(CRATE_L/2, 0.006, CRATE_H/2)}" material="{mat}"/>
        <geom type="box" pos="{f(0, -CRATE_W/2, CRATE_H/2)}" size="{f(CRATE_L/2, 0.006, CRATE_H/2)}" material="{mat}"/>
        <site name="crate_{'ab'[k]}_drop" pos="{f(0, 0, 0.17)}" size="0.01" rgba="0 1 0 0.4" group="4"/>
      </body>""")
    vents = "".join(
        f'<geom type="box" pos="{f(x, s*(hw+0.001), zc+0.02)}" size="0.035 0.001 0.004" material="cart_dark"/>'
        for x in (-0.12, -0.04, 0.04) for s in (-1, 1))
    return f"""
    <!-- ===================== HARVESTING ROBOT ===================== -->
    <body name="cart" pos="0 0 0">
      <joint name="cart_x" type="slide" axis="1 0 0"/>
      <joint name="cart_y" type="slide" axis="0 1 0"/>
      <joint name="cart_yaw" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0.3" mass="140" diaginertia="8 14 16"/>
      <!-- chassis: white body, dark skirt, orange stripe, deck -->
      <geom name="chassis" type="box" pos="{f(0, 0, zc)}" size="{f(hl, hw, hz)}" material="cart_body"/>
      <geom type="box" pos="{f(0, 0, CHASSIS_Z0+0.025)}" size="{f(hl+0.003, hw+0.003, 0.025)}" material="cart_dark"/>
      <geom type="box" pos="{f(0, 0, zc+0.045)}" size="{f(hl+0.002, hw+0.002, 0.009)}" material="cart_accent"/>
      <geom type="box" pos="{f(0, 0, CHASSIS_Z1+0.002)}" size="{f(hl-0.01, hw-0.01, 0.002)}" material="cart_dark"/>
      {vents}
      <!-- battery bay door + charging contacts (rear) -->
      <geom type="box" pos="{f(0.25, hw+0.001, zc-0.01)}" size="0.16 0.001 0.06" material="cart_dark"/>
      <geom type="box" pos="{f(-hl-0.03, 0, CHASSIS_Z0+0.08)}" size="0.004 0.08 0.015" material="alu"/>
      <!-- bumpers -->
      <geom type="box" pos="{f(hl+0.02, 0, CHASSIS_Z0+0.06)}" size="{f(0.02, hw-0.02, 0.03)}" material="rubber"/>
      <geom type="box" pos="{f(-hl-0.02, 0, CHASSIS_Z0+0.06)}" size="{f(0.02, hw-0.02, 0.03)}" material="rubber"/>
      <!-- status light bars along the deck edges -->
      <geom type="box" pos="{f(0, hw+0.003, CHASSIS_Z1-0.012)}" size="{f(hl-0.05, 0.003, 0.006)}" material="led_green"/>
      <geom type="box" pos="{f(0, -hw-0.003, CHASSIS_Z1-0.012)}" size="{f(hl-0.05, 0.003, 0.006)}" material="led_green"/>
      <!-- E-stops (front corners) -->
      <geom type="box" pos="{f(hl-0.05, hw-0.06, CHASSIS_Z1+0.02)}" size="0.035 0.035 0.02" material="estop_yellow"/>
      <geom type="cylinder" pos="{f(hl-0.05, hw-0.06, CHASSIS_Z1+0.05)}" size="0.022 0.012" material="estop_red"/>
      <geom type="box" pos="{f(hl-0.05, -hw+0.06, CHASSIS_Z1+0.02)}" size="0.035 0.035 0.02" material="estop_yellow"/>
      <geom type="cylinder" pos="{f(hl-0.05, -hw+0.06, CHASSIS_Z1+0.05)}" size="0.022 0.012" material="estop_red"/>
      <!-- compute / electronics box under the mast -->
      <geom type="box" pos="{f(MAST_X-0.04, 0, CHASSIS_Z1+0.06)}" size="0.07 0.18 0.06" material="cart_dark"/>
      <geom type="box" pos="{f(MAST_X+0.031, 0, CHASSIS_Z1+0.06)}" size="0.001 0.12 0.03" material="lens"/>

      <!-- sensor mast: aluminium profile, side-looking RGB-D scan cameras, beacon -->
      <geom type="box" pos="{f(MAST_X, 0, (CHASSIS_Z1+1.36)/2)}" size="{f(0.02, 0.02, (1.36-CHASSIS_Z1)/2)}" material="alu"/>
      <geom type="box" pos="{f(MAST_X, 0, 1.30)}" size="0.025 0.20 0.015" material="alu"/>
      <geom type="cylinder" pos="{f(MAST_X, 0, 1.39)}" size="0.022 0.03" material="beacon"/>
      <body name="scan_cam_l" pos="{f(MAST_X, 0.20, 1.27)}" quat="{f(*_scan_quat(+1))}">
        <geom type="box" size="0.045 0.015 0.014" pos="0 0 0.012" material="cart_dark"/>
        <geom type="cylinder" pos="0.02 0 -0.003" size="0.008 0.002" material="lens"/>
        <geom type="cylinder" pos="-0.02 0 -0.003" size="0.008 0.002" material="lens"/>
        <camera name="scan_left" pos="0 0 -0.006" xyaxes="1 0 0 0 1 0" fovy="58"/>
      </body>
      <body name="scan_cam_r" pos="{f(MAST_X, -0.20, 1.27)}" quat="{f(*_scan_quat(-1))}">
        <geom type="box" size="0.045 0.015 0.014" pos="0 0 0.012" material="cart_dark"/>
        <geom type="cylinder" pos="0.02 0 -0.003" size="0.008 0.002" material="lens"/>
        <geom type="cylinder" pos="-0.02 0 -0.003" size="0.008 0.002" material="lens"/>
        <camera name="scan_right" pos="0 0 -0.006" xyaxes="1 0 0 0 1 0" fovy="58"/>
      </body>
      <camera name="chase_cam" pos="-1.6 -0.05 1.85" xyaxes="{chase_cam_xyaxes}" fovy="60"/>

      {_bogie("bogie_f", BOGIE_X)}
      {_bogie("bogie_r", -BOGIE_X)}
      {"".join(crates)}

      <!-- telescopic lift column -->
      <geom type="box" pos="{f(LIFT_X, 0, CHASSIS_Z1+0.06)}" size="0.075 0.075 0.06" material="cart_dark"/>
      <geom type="box" pos="{f(LIFT_X, 0, CHASSIS_Z1+0.121)}" size="0.077 0.077 0.004" material="cart_accent"/>
      <body name="lift" pos="{f(LIFT_X, 0, PLATE_Z0)}" gravcomp="1">
        <joint name="lift" type="slide" axis="0 0 1" range="{f(*LIFT_RANGE)}" damping="200"/>
        <geom type="box" pos="0 0 -0.20" size="0.058 0.058 0.20" material="alu"/>
        <geom type="cylinder" pos="0 0 -0.006" size="0.08 0.006" material="cart_dark"/>
        {_gen3_body()}
      </body>
    </body>
"""


def _scan_quat(side):
    """Scan camera body orientation: local -z (view) points sideways, 12 deg ahead, 38 deg down."""
    yaw = side * math.radians(90 - 12)
    pitch = math.radians(38)
    view = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    z = -view
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return _mat2quat(R)


def _mat2quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (w, x, y, z)


def robot_actuators():
    return """
    <position name="lift"     joint="lift"     kp="30000" kv="3000" ctrlrange="0 0.38"/>
    <position name="joint_1"  joint="joint_1"  kp="2000" kv="100" forcerange="-105 105"/>
    <position name="joint_2"  joint="joint_2"  kp="2000" kv="100" forcerange="-105 105" ctrlrange="-2.2497 2.2497"/>
    <position name="joint_3"  joint="joint_3"  kp="2000" kv="100" forcerange="-105 105"/>
    <position name="joint_4"  joint="joint_4"  kp="2000" kv="100" forcerange="-105 105" ctrlrange="-2.5796 2.5796"/>
    <position name="joint_5"  joint="joint_5"  kp="500"  kv="50"  forcerange="-52 52"/>
    <position name="joint_6"  joint="joint_6"  kp="500"  kv="50"  forcerange="-52 52" ctrlrange="-2.0996 2.0996"/>
    <position name="joint_7"  joint="joint_7"  kp="500"  kv="50"  forcerange="-52 52"/>
    <!-- soft fingers: force-limited drive (max 4 N per finger) -->
    <position name="finger_l" joint="finger_l" kp="300" kv="10" forcerange="-4 4" ctrlrange="0 0.032"/>
    <position name="finger_r" joint="finger_r" kp="300" kv="10" forcerange="-4 4" ctrlrange="0 0.032"/>
    <position name="blade"    joint="blade"    kp="400" kv="8" ctrlrange="0 0.066"/>
"""


def ik_model_xml():
    """Robot only (no greenhouse): a small model for fast IK planning in the cart frame."""
    return f"""<mujoco model="harvester_only">
  <compiler angle="radian" autolimits="true" meshdir="assets"/>
  <option timestep="0.002"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>
{robot_assets()}
  </asset>
  <worldbody>
{robot_body("1 0 0 0 1 0")}
  </worldbody>
  <actuator>{robot_actuators()}</actuator>
</mujoco>
"""


# ======================================================================================
# runtime helpers
# ======================================================================================
class RobotIO:
    """Convenience indices into a compiled model."""

    def __init__(self, model):
        import mujoco
        self.m = model
        jn = lambda n: model.joint(n)
        self.qadr = {n: jn(n).qposadr[0] for n in
                     ["cart_x", "cart_y", "cart_yaw", "bogie_f_yaw", "bogie_r_yaw", "lift", "finger_l", "finger_r", "blade"]
                     + ARM_JOINTS + [f"bogie_{b}_axle{k}" for b in "fr" for k in (0, 1)]}
        self.dadr = {n: jn(n).dofadr[0] for n in self.qadr}
        self.act = {model.actuator(i).name: i for i in range(model.nu)}
        self.tcp = model.site("tcp").id
        self.arm_q = np.array([self.qadr[j] for j in ARM_JOINTS])
        self.arm_d = np.array([self.dadr[j] for j in ARM_JOINTS])
        self.mj = mujoco

    def place_cart(self, data, x, y, yaw, yf, yr, travelled):
        """Kinematically place the rail-guided cart (and turn its bogies/wheels)."""
        for n, v in (("cart_x", x), ("cart_y", y), ("cart_yaw", yaw), ("bogie_f_yaw", yf), ("bogie_r_yaw", yr)):
            data.qpos[self.qadr[n]] = v
            data.qvel[self.dadr[n]] = 0.0
        for b in "fr":
            for k in (0, 1):
                n = f"bogie_{b}_axle{k}"
                data.qpos[self.qadr[n]] = travelled / WHEEL_R
                data.qvel[self.dadr[n]] = 0.0

    def set_arm(self, data, q, ctrl_too=True):
        for j, v in zip(ARM_JOINTS, q):
            data.qpos[self.qadr[j]] = v
            if ctrl_too:
                data.ctrl[self.act[j]] = v

    def set_tool(self, data, fingers, blade, qpos_too=False):
        data.ctrl[self.act["finger_l"]] = fingers
        data.ctrl[self.act["finger_r"]] = fingers
        data.ctrl[self.act["blade"]] = blade
        if qpos_too:
            data.qpos[self.qadr["finger_l"]] = fingers
            data.qpos[self.qadr["finger_r"]] = fingers
            data.qpos[self.qadr["blade"]] = blade

    def set_lift(self, data, h):
        data.qpos[self.qadr["lift"]] = h
        data.ctrl[self.act["lift"]] = h


def solve_ik(model, data, io, pos, mat, seed, iters=300, tol=1e-4):
    """Damped least-squares IK for the 7 arm joints so that site 'tcp' reaches pos/mat (world frame).
    Uses `data` as scratch (cart / lift must already be set). Returns (q, pos_err, rot_err)."""
    import mujoco
    q = np.array(seed, float)
    lo = np.array([model.jnt_range[model.joint(j).id][0] if model.jnt_limited[model.joint(j).id] else -1e9 for j in ARM_JOINTS])
    hi = np.array([model.jnt_range[model.joint(j).id][1] if model.jnt_limited[model.joint(j).id] else 1e9 for j in ARM_JOINTS])
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    qt = np.zeros(4); mujoco.mju_mat2Quat(qt, np.asarray(mat, float).reshape(9))
    err_p = err_r = 1.0
    for _ in range(iters):
        data.qpos[io.arm_q] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        cur = data.site_xpos[io.tcp]
        cm = data.site_xmat[io.tcp]
        qc = np.zeros(4); mujoco.mju_mat2Quat(qc, cm)
        dr = np.zeros(3); mujoco.mju_subQuat(dr, qt, qc)
        e = np.concatenate([pos - cur, cm.reshape(3, 3) @ dr])
        err_p, err_r = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if err_p < tol and err_r < 1e-3:
            break
        mujoco.mj_jacSite(model, data, jp, jr, io.tcp)
        J = np.vstack([jp[:, io.arm_d], jr[:, io.arm_d]])
        W = np.diag([1, 1, 1, 0.3, 0.3, 0.3])
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), W @ e)
        q = np.clip(q + dq, lo, hi)
    return q, err_p, err_r


def _tool_mat(z_axis, y_axis):
    z = np.asarray(z_axis, float); z /= np.linalg.norm(z)
    y = np.asarray(y_axis, float); y -= z * np.dot(y, z); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


GEN3_HOME = [0.0, 0.26, math.pi, -2.27, 0.0, 0.96, math.pi / 2]
GEN3_STOW = [0.0, -0.35, math.pi, -2.55, 0.0, -0.87, math.pi / 2]    # Kinova "retract", folded upright
APPROACH_DIR = np.array([0.5, 0.7, 0.3]) / np.linalg.norm([0.5, 0.7, 0.3])


def harvest_poses(model, fruit_lateral=0.555, fruit_z=0.955, approach=0.10, verbose=False):
    """Joint-space poses (cart frame, cart at origin) for the scripted harvest cycle.
    Keys: stow, pre_L/pick_L/crate_L (left side, +y), and _R mirrored. Continuous joints are
    unwrapped so consecutive poses in a cycle are close."""
    import mujoco
    data = mujoco.MjData(model)
    io = RobotIO(model)
    io.place_cart(data, 0, 0, 0, 0, 0, 0)
    io.set_lift(data, LIFT_HOME)
    io.set_tool(data, FINGER_OPEN, BLADE_OPEN, qpos_too=True)
    arm_base_z = PLATE_Z0 + LIFT_HOME
    targets = {}
    for side, sgn in (("L", 1), ("R", -1)):
        # approach hanging fruit slightly forward along the row and from a little below:
        # keeps the Gen3 elbow well away from its limits (fruit is only ~0.55 m from the arm axis)
        y_dir = APPROACH_DIR * np.array([1, sgn, 1])
        pick = np.array([LIFT_X, sgn * fruit_lateral, fruit_z])
        targets[f"pick_{side}"] = (pick, _tool_mat(y_dir, [0, 0, 1]))
        targets[f"pre_{side}"] = (pick - approach * y_dir, _tool_mat(y_dir, [0, 0, 1]))
        drop = np.array([CRATE_X, sgn * CRATE_Y, CHASSIS_Z1 + 0.20])
        targets[f"crate_{side}"] = (drop, _tool_mat([0, 0, -1], [1, 0, 0]))

    seeds = []
    for j1 in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        for j4 in (-2.2, -1.5):
            seeds.append([j1, 0.4, math.pi, j4, 0.0, -0.9, math.pi / 2])
            seeds.append([j1, -0.4, 0.0, j4, 0.0, 0.9, math.pi / 2])
    poses = {"stow": np.array(GEN3_STOW)}
    order = ["pre_L", "pick_L", "crate_L", "pre_R", "pick_R", "crate_R"]
    for key in order:
        pos, mat = targets[key]
        pos_w = pos.copy()
        best = None
        pref = poses.get("pre_L" if key.endswith("L") else "pre_R") if key.startswith(("pick", "crate")) else None
        cand_seeds = ([pref] if pref is not None else []) + seeds
        lo = np.array([model.jnt_range[model.joint(j).id][0] if model.jnt_limited[model.joint(j).id] else -9 for j in ARM_JOINTS])
        hi = np.array([model.jnt_range[model.joint(j).id][1] if model.jnt_limited[model.joint(j).id] else 9 for j in ARM_JOINTS])
        for s in cand_seeds:
            q, ep, er = solve_ik(model, data, io, pos_w, mat, s)
            margin = np.min(np.minimum(q - lo, hi - q))
            score = ep * 10 + er - 0.05 * margin + (0.02 * np.linalg.norm(q - pref) if pref is not None else 0.0)
            if best is None or score < best[0]:
                best = (score, q, ep, er)
        poses[key] = best[1]
        if verbose:
            print(f"  IK {key:8s} pos err {best[2]*1000:6.2f} mm  rot err {math.degrees(best[3]):5.2f} deg")
    return poses


def unwrap_like(q_new, q_ref):
    """Shift continuous joints of q_new by 2*pi*k to be closest to q_ref."""
    q = np.array(q_new, float)
    for i, j in enumerate(ARM_JOINTS):
        if j in CONTINUOUS:
            q[i] += 2 * math.pi * round((q_ref[i] - q[i]) / (2 * math.pi))
    return q
