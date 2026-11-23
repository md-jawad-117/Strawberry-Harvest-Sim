"""
Render still images of the greenhouse from several cameras (no window needed).

    python render_preview.py          -> images/*.png
"""
import time
from pathlib import Path

import mujoco
from PIL import Image

import robot
from layout import CONFIG, Route

ROOT = Path(__file__).parent
SCENE = ROOT / "scene" / "greenhouse.xml"
OUT = ROOT / "images"

# (camera, hide walls/roof?)
CAMERAS = [
    ("overview", False),
    ("headland_view", False),
    ("aisle_view", False),
    ("robot_side", False),
    ("berries_closeup", False),
    ("chase_cam", False),
    ("gen3_wrist_cam", False),
    ("scan_left", False),
    ("cutaway", True),
]


def pose_robot(model, data, arm_pose="pick_L", s=None):
    """Park the cart in aisle 0 at CONFIG['robot_start_x'] with the arm in a harvest pose."""
    route = Route()
    io = robot.RobotIO(model)
    poses = robot.harvest_poses(model)
    if s is None:
        s = CONFIG["robot_start_x"] - route.xy[0, 0]
    x, y, yaw, yf, yr = route.vehicle_pose(s, robot.BOGIE_X)
    io.place_cart(data, x, y, yaw, yf, yr, s)
    io.set_lift(data, robot.LIFT_HOME)
    io.set_arm(data, poses[arm_pose])
    io.set_tool(data, robot.FINGER_HOLD if arm_pose.startswith("pick") else robot.FINGER_OPEN,
                robot.BLADE_OPEN, qpos_too=True)
    mujoco.mj_forward(model, data)
    return poses


def main(width=1600, height=900):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    pose_robot(model, data)
    print(f"bodies={model.nbody} geoms={model.ngeom} joints={model.njnt} actuators={model.nu}")

    OUT.mkdir(exist_ok=True)
    with mujoco.Renderer(model, height=height, width=width, max_geom=40000) as r:
        for cam, hide_cover in CAMERAS:
            opt = mujoco.MjvOption()
            opt.geomgroup[2] = 0 if hide_cover else 1
            t = time.perf_counter()
            r.update_scene(data, camera=cam, scene_option=opt)
            Image.fromarray(r.render()).save(OUT / f"{cam}.png")
            print(f"saved {OUT / (cam + '.png')}  ({(time.perf_counter() - t) * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
