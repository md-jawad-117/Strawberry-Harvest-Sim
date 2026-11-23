"""
Watch the harvester work in the interactive MuJoCo viewer.

    python view_scene.py               # full mission: cameras find ripe fruit, arm picks them into crates
    python view_scene.py --speed 3     # faster (simulated time runs 3x real time, if the PC keeps up)
    python view_scene.py --aisles 1    # only harvest aisle 1
    python view_scene.py --oracle      # skip the cameras: use known fruit positions (step-2 baseline)
    python view_scene.py --shadows     # lamp shadows (nicer, slower on integrated graphics)
    python view_scene.py --static      # no mission; move the joints yourself with the Control sliders

Mouse: left-drag rotate, right-drag pan, scroll zoom. Press ESC to free the camera.
Key "2" hides/shows the walls and roof; key "3" shows the fruit contact spheres.
The overlay shows what the robot is doing and the running harvest statistics.
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from harvester import Harvester

SCENE = Path(__file__).parent / "scene" / "greenhouse.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="no mission")
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed multiplier")
    ap.add_argument("--oracle", action="store_true", help="known fruit positions instead of cameras")
    ap.add_argument("--aisles", type=int, nargs="*", default=None)
    ap.add_argument("--shadows", action="store_true", help="enable lamp shadows (slower)")
    ap.add_argument("--free-cam", action="store_true", help="don't steer the follow camera automatically")
    ap.add_argument("--seconds", type=float, default=0, help="close automatically after N seconds (testing)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    if not args.shadows:
        model.light_castshadow[:] = 0
    print("Setting up the harvester (IK model, cameras)...")
    h = Harvester(model, data, mode="oracle" if args.oracle else "vision", aisles=args.aisles,
                  log_path=Path(__file__).parent / "results")
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = model.body("cart").id
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.6, -28, 25

        start = wall0 = time.perf_counter()
        sim0, frames = data.time, 0
        while v.is_running():
            frames += 1
            if args.seconds and time.perf_counter() - start > args.seconds:
                print(f"viewer fps: {frames / (time.perf_counter() - start):.1f}, simulated {data.time:.1f} s")
                break
            target = (time.perf_counter() - wall0) * args.speed
            budget = time.perf_counter() + 0.08          # keep the window responsive
            while data.time - sim0 < target and time.perf_counter() < budget:
                if not args.static:
                    h.tick()
                mujoco.mj_step(model, data)
            if data.time - sim0 < target - 0.5:          # can't keep up: let simulated time slip
                wall0 = time.perf_counter() - (data.time - sim0) / args.speed
            if not args.static and not args.free_cam:
                cx = h.cart[0]
                a = min(max((1.8 - cx) / 2.0, 0.0), 1.0)
                v.cam.azimuth += 0.03 * ((25 + a * 135) - v.cam.azimuth)
            if not args.static:
                try:
                    v.set_texts([
                        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         "Strawberry harvester", h.status),
                        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                         "Harvest", h.hud()),
                    ])
                except Exception:
                    pass
            v.sync()
            time.sleep(0.001)
    if h.done:
        print(h.summary())


if __name__ == "__main__":
    main()
