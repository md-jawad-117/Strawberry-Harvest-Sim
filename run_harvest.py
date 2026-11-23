"""
Run the harvest mission headless (no window) and write a report.

    python run_harvest.py --quick              # short test: aisle 0, 5 simulated minutes, both modes -> results_quick/ (about 3 min)
    python run_harvest.py --full               # whole mission in both modes, from scratch -> results/ (about 70 min)
    python run_harvest.py                      # camera-based, all aisles
    python run_harvest.py --oracle             # known fruit positions (baseline)
    python run_harvest.py --aisles 0           # only aisle 0
    python run_harvest.py --gif 40             # also record 40 s of picking: robot view + gripper camera, side by side

The scene is built automatically if scene/ is missing (--full always rebuilds it).
Each run writes summary_<mode>.json and picks_<mode>.csv (one row per pick attempt).
"""
import argparse
import math
import json
import time
from pathlib import Path

import mujoco

from harvester import Harvester

ROOT = Path(__file__).parent
FPS = 8


def gif_frame(left, right, status):
    """Robot view and gripper view side by side, with labels."""
    from PIL import Image, ImageDraw
    w, hgt = left.shape[1], left.shape[0]
    im = Image.new("RGB", (2 * w + 4, hgt + 22), (20, 20, 22))
    im.paste(Image.fromarray(left), (0, 22))
    im.paste(Image.fromarray(right), (w + 4, 22))
    d = ImageDraw.Draw(im)
    d.text((8, 5), "Robot view", fill=(255, 255, 255))
    d.text((w + 12, 5), "Gripper camera", fill=(255, 255, 255))
    st = status.split(": ", 1)[-1]
    d.text((2 * w - 6 * len(st), 5), st, fill=(255, 210, 80))
    return im.quantize(colors=128)


def run_mission(mode, aisles, out, second_look=True, gif=0, gif_start=0, max_minutes=0):
    """One headless mission. mode is "vision" (cameras) or "oracle" (known fruit positions)."""
    print(f"\n=== {mode} mode | aisles {aisles if aisles is not None else 'all'} | "
          f"{f'{max_minutes:g} simulated min' if max_minutes else 'full mission'} | results -> {out}/", flush=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene" / "greenhouse.xml"))
    model.light_castshadow[:] = 0
    data = mujoco.MjData(model)
    h = Harvester(model, data, mode=mode, aisles=aisles, log_path=ROOT / out, second_look=second_look)

    frames, rec = [], None
    if gif:
        PW, PH = 448, 252                                   # each panel
        rec = mujoco.Renderer(model, PH, PW, max_geom=20000)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation = 1.3, -22
        lift_id = model.body("lift").id

    wall0, last = time.perf_counter(), 0
    while not h.done:
        h.tick()
        mujoco.mj_step(model, data)
        if rec is not None and h.stats["attempts"] > 0 and h.t >= gif_start and len(frames) < gif * FPS and h.t * FPS >= last + 1:
            last = int(h.t * FPS)
            tcp = data.site_xpos[h.io.tcp]
            cam.lookat[:] = 0.5 * tcp + 0.5 * data.xpos[lift_id] + [0, 0, -0.1]
            cam.azimuth = math.degrees(h.cart[2]) + 180 + 15      # from behind the cart, along the aisle
            rec.update_scene(data, cam, opt)
            left = rec.render().copy()
            rec.update_scene(data, camera="gripper_view", scene_option=opt)
            right = rec.render().copy()
            frames.append(gif_frame(left, right, h.status))
        if int(h.t / 60) != int((h.t - model.opt.timestep) / 60):
            st = h.stats
            print(f"  sim {h.t/60:5.1f} min | wall {(time.perf_counter()-wall0)/60:4.1f} min | {h.status:45s} | "
                  f"picked {st['picked']} attempts {st['attempts']}", flush=True)
        if max_minutes and h.t > max_minutes * 60:
            h.done = True
            h._finish()
    if frames:
        frames[0].save(ROOT / "images" / "harvest.gif", save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
        print(f"saved images/harvest.gif ({len(frames)} frames)")
    return h.summary()


def print_table(results, out):
    """Side-by-side summary of the modes that were run."""
    rows = [("Ripe fruit picked", lambda s: s["picked"]),
            ("Pick attempts", lambda s: s["attempts"]),
            ("Bruised", lambda s: s["damaged"]),
            ("Unripe fruit picked", lambda s: s["unripe_picked"]),
            ("Stem not cut after re-grip", lambda s: s["cut_miss"]),
            ("Neighbour stems cut, fruit dropped", lambda s: s["collateral_drops"]),
            ("Average cycle time (s)", lambda s: s["cycle_time_mean_s"]),
            ("Fruit per hour", lambda s: s["fruit_per_hour"])]
    names = {"vision": "Cameras", "oracle": "Known positions"}
    print("\n" + f"{'':38s}" + "".join(f"{names[m]:>18s}" for m in results))
    for label, f in rows:
        cells = []
        for s in results.values():
            try:
                cells.append(f"{str(f(s)):>18s}")
            except (KeyError, TypeError):
                cells.append(f"{'-':>18s}")
        print(f"{label:38s}" + "".join(cells))
    print(f"\nFull reports: {out}/summary_<mode>.json and {out}/picks_<mode>.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short test: aisle 0, 5 simulated minutes, both modes, results_quick/")
    ap.add_argument("--full", action="store_true", help="rebuild the scene, then the whole mission in both modes, results/")
    ap.add_argument("--oracle", action="store_true", help="use known fruit positions instead of the cameras")
    ap.add_argument("--aisles", type=int, nargs="*", default=None)
    ap.add_argument("--no-second-look", action="store_true")
    ap.add_argument("--out", default=None, help="output folder (default results/, or results_quick/ with --quick)")
    ap.add_argument("--gif", type=float, default=0, help="seconds of picking to record as a GIF")
    ap.add_argument("--gif-start", type=float, default=0, help="start recording after this many simulated seconds")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this much simulated time")
    args = ap.parse_args()
    if args.quick and args.full:
        ap.error("choose either --quick or --full")

    if args.full or not (ROOT / "scene" / "greenhouse.xml").exists():
        import build_scene
        print("building the scene (about 1 minute) ...", flush=True)
        build_scene.build()

    if args.quick:
        modes, aisles, max_min, out = ("vision", "oracle"), [0], args.max_minutes or 5, args.out or "results_quick"
    elif args.full:
        modes, aisles, max_min, out = ("vision", "oracle"), None, 0, args.out or "results"
    else:
        modes, aisles, max_min, out = ("oracle" if args.oracle else "vision",), args.aisles, args.max_minutes, args.out or "results"

    t0 = time.perf_counter()
    results = {}
    for m in modes:
        results[m] = run_mission(m, aisles, out, second_look=not args.no_second_look,
                                 gif=args.gif if m == modes[0] else 0, gif_start=args.gif_start, max_minutes=max_min)
    if len(results) == 1:
        print(json.dumps(next(iter(results.values())), indent=2))
    print_table(results, out)
    print(f"total wall time {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
