# Harvester design: cart, arm, end-effector

Status: **built: all 4 steps** (see "As built" at the end). Numbers come from the current greenhouse layout (`layout.py`) and the Kinova Gen3 model in `third_party/kinova_gen3`.

## Constraints measured from the scene

| Item | Value |
|---|---|
| Rail gauge / aisle width between gutter walls | 0.55 m / 1.20 m |
| Free width between hanging fruit (both sides) | ~1.02 m |
| Fruit distance from aisle centre line | 0.53–0.58 m |
| Fruit height (top of fruit / bottom of fruit) | 0.96–1.02 m / ~0.91 m |
| Track curve radius at the headland | 0.75 m |

## 1. Arm: Kinova Gen3, 7-DoF, with vision module

- Reach: about 0.9 m. Shoulder→wrist-centre is 0.73 m, plus 0.17 m to the flange. Runs on 24 V DC with the controller in the base, and weighs about 8 kg.
- The wrist vision module (color + depth camera) is the close-up camera for ripeness checks and for finding the stem.
- **Mounting:** on the lift carriage on the cart centre line, base at **0.67 m** by default. That puts the shoulder at about 0.95 m, level with the fruit, so the arm reaches out sideways with the elbow bent and keeps good dexterity.
- **Workspace per stop:** reach-checked to ±0.6 m along the row, using ±0.45 m in practice. With **stops every 0.9 m** it covers both rows from one stop, by turning joint 1.

## 2. Cart (rail-guided)

```
  front                                                     rear
  ┌──────────┬───────────────────────┬───────────────────────┐
  │ mast +   │  vertical lift         │  crate A │ crate B    │  ← two 40×30 cm crates
  │ scan cams│  + Kinova Gen3         │  (fill)  │ (spare)    │    side by side
  ├──────────┴───────────────────────┴───────────────────────┤
  │  battery (48 V LiFePO4) · 24 V DC-DC for the arm · compute (Jetson) │
  └── bogie ──────────────────────────────────── bogie ───────┘
            on 0.55 m pipe rails, flanged wheels
```

- **Footprint:** about 1.10 m long × 0.66 m wide, so it stays well inside the 1.02 m gap between the hanging fruit.
- **Running gear:** two **swivelling bogies** (front and rear) with flanged wheels. These let a 1.1 m cart follow the 0.75 m-radius headland curves. A rigid 4-wheel frame would come off the rails.
- **Lift:** a vertical linear axis with 0.55–0.95 m base-height travel, so the same robot fits other gutter heights.
- **Scan cameras:** two RGB-D cameras on a 1.3 m mast, one per side, angled slightly ahead. They find ripe fruit in the next section *before* the cart stops there.
- **Crates:** two crates side by side at the rear. The arm fills A, then B. The headland dock is where full crates get swapped (later feature).
- **Power and compute:** a 48 V battery, a 24 V supply for the Gen3, and an onboard computer. A bumper, an E-stop and a status light bar are on the cart.

## 3. End-effector: hybrid "cradle and cut" (soft-fruit safe)

| Part | Purpose |
|---|---|
| Two **fin-ray soft fingers** (TPU, ~70 mm) | wrap around the fruit and cradle it without pinching |
| **Force-limited** drive (motor current + fingertip force sensor) | stops closing once a low grip force is reached |
| **Small blade / scissor** above the fingers | cuts the stem ~1 cm above the green cap, so the fruit is never pulled |
| Wrist camera (Gen3 vision module) | final alignment on the fruit and the stem |

**Pick sequence:**
1. Approach from the aisle side at fruit height.
2. Open the fingers and centre them on the fruit.
3. Close gently until the force limit is reached.
4. Cut the stem.
5. Retract.
6. Turn to the crate and *lower* the fruit to just above the crate bed before releasing it (never drop it).

## 4. How the simulation represents it

- **Strawberries:** each one becomes a body held to the plant by a weld constraint. "Cut" switches the weld off. The cut only succeeds when the blade is within ~1 cm of the stem point.
- **Fingers:** soft contact settings. The finger contact force is read on every pick.
- **Bruise model:** if peak grip force goes over the threshold (a tunable parameter), the fruit counts as *damaged*.
- **Simulation statistics:** picked / damaged / missed / unripe skipped, cycle time per fruit, crate fill level.

## Build order

1. Cart model (body, bogies, lift, mast cameras, crates) carrying the Gen3 + end-effector, still driven along the existing route
2. Inverse kinematics for the Gen3 (sideways approach poses) and the pick sequence on known fruit positions
3. Pickable strawberries (weld + cut), the force/bruise model and live statistics
4. Camera-based ripeness detection replaces the known positions

## As built (steps 1–4)

| Step | Where | What it does |
|---|---|---|
| 1 Cart + arm + gripper | `robot.py` | Rail cart with swivelling bogies, lift, mast cameras, two crates, Kinova Gen3, fin-ray fingers and sliding stem knife |
| 2 IK picking | `harvester.py` | Per fruit: chooses the approach angle (5 yaw options, avoiding neighbours), damped-least-squares IK on a small robot-only model, straight-line approach, crate drop poses per slot |
| 3 Pickable fruit + bruising | `harvester.py`, `build_scene.py` | Every fruit is a mocap body with an invisible contact sphere. Force-limited grip (1 N per finger); the knife cuts the fruit's own stem only if that stem crosses the knife window; neighbouring stems in the window are cut too and those fruit drop. Fruit pushed by a finger swing aside softly. Bruise if peak squeeze force exceeds the fruit's firmness (ripe ~3.0 N, fully ripe ~2.3 N) |
| 4 Camera detection | `perception.py` | Mast cameras: red-pixel segmentation, depth-edge and k-means blob splitting, back-projection to 3D, multi-view fruit map. Wrist camera re-measures each fruit before the grasp and does a "second look" for fruit the mast missed |

Changes from the original plan:
- **Fruit are mocap bodies, not welded free bodies.** 1,300 free bodies would be too slow for a laptop. A mocap fruit costs almost nothing until it is touched, and the harvester moves it when it is cut, carried, dropped and stored.
- **The knife slides instead of swinging.** It cuts in a 4.6 × 8.8 cm window between the closed finger pads, 2.5 cm above the fruit centre. A wider sweep (6.2 cm) picked the same number of fruit but cut 40 % more neighbouring stems. The planner does not see stems: when a cut misses, the robot re-grips once from another angle.
- **Fruit spacing:** the scene generator keeps at least 4 mm between neighbouring fruit (real fruit cannot overlap).

- **Future work, stem-aware cutting:** the planner does not use stems yet. The wrist camera could locate the stems above the target so the planner picks an approach angle with only the fruit's own stem in the knife window.

Run `python run_harvest.py` (cameras) or `python run_harvest.py --oracle` (known positions) for a full headless mission. Results go to `results/`.
