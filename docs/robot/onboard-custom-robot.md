# Onboard a custom robot: CAD to a trained policy

This tutorial follows Jonathan's planned robot: Raspberry Pi 5, a USB U2D2, four
Dynamixel XL430 actuators, and an apparent Waveshare Sense HAT (B) with an ICM-20948.
It also applies to another actuator count. The joint roles and mechanical layout are
still to be specified; four actuators do not necessarily mean four independent joints.

The deliverables are a **MuJoCo MJCF model (`.xml`) with meshes**, a **training task**,
a **robot profile (`.toml`)**, and an **exported policy (`.onnx`)**. Keep the original
CAD assembly as the editable mechanical source. A mesh alone is not a training model.

## 1. Know what you can do today

The current implementation boundary is documented in
[the robot-model design](../design/robotd-design.md#library-support-for-other-morphologies).
Read it before attempting hardware deployment.

You can prepare CAD, build and inspect an MJCF asset, validate a joint profile, and
implement a custom training environment now. **The stock daemon cannot yet run this
four-motor robot.** The hardware drivers, profile-driven daemon, client metadata, and
Pi deployment integration remain unfinished. Steps below explicitly mark that boundary;
there is no working `robotd --profile` command to run today.

Use three environments:

| Environment | Work |
| --- | --- |
| CAD workstation | Mechanical assembly, link exports, mass properties |
| Development/training machine | Model validation, training task, policy export and replay |
| Raspberry Pi 5 | Sensor/motor bring-up and measured deployment timing, once integrated |

The upstream [Microduck training repository](https://github.com/pollen-robotics/microduck_rl)
uses mjlab/MuJoCo Warp and PPO, with a CUDA GPU for its training path. Plan to train on
that machine and run inference on the Pi. A CPU MuJoCo viewer is useful for checking
assets before GPU training.

## 2. Prepare a mechanical handoff from CAD

Export the native assembly plus STEP files for preserving editable geometry. Export
STL or OBJ meshes **per rigid link** for simulation visuals. Group parts that cannot
move relative to each other; preserve separate bodies across moving joints.

Include a drawing or annotated screenshot showing the body axes and each joint's
positive rotation. Choose a consistent body convention, such as +X forward, +Y left,
+Z up, and use it in the simulator and the eventual IMU mount transform.

Prepare this table before choosing policy dimensions:

| Joint name | Parent → child | Axis in joint frame | DXL ID | Zero reference | Home (rad) | Min/max (rad) |
| --- | --- | --- | --- | --- | --- | --- |
| Replace with joint 1 role | … | … | … | … | … | … |
| Replace with joint 2 role | … | … | … | … | … | … |
| Replace with joint 3 role | … | … | … | … | … | … |
| Replace with joint 4 role | … | … | … | … | … | … |

For each rigid link, supply its mass, center of mass, and inertia tensor with the
coordinate frame and units identified. Include motors, battery, Pi, wiring and payload,
not just printed plastic. Record the IMU position and orientation relative to the body.

Use meters, kilograms, seconds and radians in the final simulation and runtime contract.
Check a known physical dimension after mesh import: a millimeter/meter error changes
both contact geometry and inferred inertia. Determine conservative joint travel from
mechanical clearance and wiring; the servo's full electrical travel is not the robot's
safe joint range.

If the CAD includes gears, coupled joints or a closed linkage, describe the coupling.
Do not silently turn a four-motor linkage into four unrelated hinges.

**Ready to proceed:** the link hierarchy, axes, units, mass properties and mapping from
actuators to joints are written down, even if some measurements are marked provisional.

## 3. Build the simulation asset in MJCF

Use **MJCF XML as the final MuJoCo asset**. URDF is a useful intermediate if your CAD
exporter produces one, but is optional. MuJoCo can import URDF; its native MJCF format
supports simulation details beyond that import. See the official
[modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html).

A suggested directory in your adaptation of the **training repository** is:

```text
assets/my_robot/
  robot.xml                 # articulated body, inertials, joints, actuators
  scene.xml                 # includes robot.xml; adds floor and lighting
  meshes/                   # per-link visual geometry
  calibration.csv           # measured zero references, directions and travel
  MODEL_NOTES.md            # units, coordinate frames, measurements and assumptions
```

These are proposed project files, not assets already supplied by this repository.
Keep the original CAD exports alongside them or in your mechanical repository.

You have two routes:

1. **CAD → URDF → MJCF:** inspect the CAD export for correct hierarchy, axes, limits
   and inertials, resolve its mesh paths, then import it with MuJoCo.
2. **CAD meshes → MJCF:** write the body hierarchy and joints directly, referencing
   the exported meshes and measured link properties.

For the URDF route, a Python environment with MuJoCo can compile and save the import:

```python
from pathlib import Path
import mujoco

source = Path("assets/my_robot/robot.urdf").resolve()
model = mujoco.MjModel.from_xml_path(str(source))
mujoco.mj_saveLastXML(str(source.with_suffix(".xml")), model)
```

Inspect the result and verify mesh paths from the saved location. Conversion is a
starting point: it does not create your training task, tune actuators or prove that
contacts and dynamics match the real mechanism.

Complete the asset with:

- A floating root for a freely moving robot, or a fixed root for a bolted-down mechanism.
- Explicit measured inertials rather than treating a decorative mesh as solid material.
- Simple collision geometry, with self-contact pairs chosen deliberately.
- Named hinge/slide joints and conservative travel limits.
- Actuators attached to the correct joints, with realistic effort and speed limitations.
- An IMU site/frame and named contact bodies relevant to the task.
- An initial pose that fits the mechanism and starts without unintended penetration.

Keep visual and collision geometry separate when a detailed visual mesh would give
poor contact behavior. Check the compiled inertias rather than hiding invalid CAD
inertias with automatic correction. MuJoCo's
[XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) describes the
body, inertial, joint, geom and actuator elements.

Do not copy Microduck's XL330 actuator parameters onto XL430s. Initially, a bounded
position actuator can test mapping and geometry; before transfer, identify the real
servo response, delay, friction/backlash and limits. An ideal position servo is not
an adequate final model of the hardware.

## 4. Inspect the asset before training

Create a separate Python environment on the development machine (Python 3.11+ is
convenient for reading TOML later):

```bash
python3 -m venv .venv-model
. .venv-model/bin/activate
python -m pip install mujoco
python -m mujoco.viewer --mjcf=assets/my_robot/scene.xml
```

The [official Python documentation](https://mujoco.readthedocs.io/en/stable/python.html)
covers the loader and viewer. These commands are for the model environment, not the
Pi's motor bus. Record the tested dependency versions when the model works.

In the viewer, check scale, floor height, joint axes, mass distribution and contact
surfaces. With the body supported or fixed temporarily, move **one simulated joint at
a time** through a small displacement and then through its permitted travel. Confirm
that the named joint, direction and child link agree with the CAD table.

Then test the intended free/fixed root, gravity and passive settling. A free robot
falling without control is not itself a model bug; exploding contacts, impossible
motion, or an immovable joint are reasons to fix the asset before training.

## 5. Write the robot profile and check the mapping

In this daemon repository, copy the provided example:

```bash
mkdir -p local/my_robot
cp deploy/profiles/four-joint.example.toml local/my_robot/robot.toml
```

Replace every placeholder name, ID, home, limit and action scale with your robot's
values. Keep `local/my_robot` as your working location or choose another; these commands
do not install anything. `action_scale` describes policy output scaling, not a substitute
for a calibrated encoder zero or a driver-level coordinate transform.

The authoritative profile and observation rules are in
[the robot-model design](../design/robotd-design.md#library-support-for-other-morphologies).
Validate with the actual Rust parser:

```bash
cargo run -p duck-control --example check-profile -- local/my_robot/robot.toml
```

The unedited four-joint example reports `4 joints, 21 observations, 4 actions`.
That is the proposed contract only when all four independent joints are policy-controlled
and the three listed commands are appropriate. A robot arm, wheeled platform or coupled
mechanism may need different commands or a different action representation.

Check the MuJoCo mapping by name instead of assuming that the first four positions
are the four joints. A floating base adds position and velocity coordinates. For a
model using direct joint transmissions, adapt these two paths and run:

```python
from pathlib import Path
import tomllib
import mujoco

profile = tomllib.loads(Path("local/my_robot/robot.toml").read_text())
model = mujoco.MjModel.from_xml_path("/absolute/path/to/assets/my_robot/scene.xml")
policy_names = profile["policy"]["joints"]
assert model.nu == len(policy_names), "actuator count differs from policy"

for slot, name in enumerate(policy_names):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint >= 0, f"missing joint: {name}"
    assert model.jnt_type[joint] in (
        mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
    ), f"expected a scalar joint: {name}"
    assert model.actuator_trntype[slot] == mujoco.mjtTrn.mjTRN_JOINT
    assert model.actuator_trnid[slot, 0] == joint, f"action order differs: {name}"
    print(slot, name, "qpos", model.jnt_qposadr[joint],
          "qvel", model.jnt_dofadr[joint])
```

This snippet assumes MJCF actuator order equals policy order; an environment with a
separate explicit action map should validate that map instead. Tendon/coupled transmissions
need their own mapping check. Also compare simulated homes, limits and directions with
the profile; a count check cannot detect a reversed knee.

## 6. Create the custom training task

Fork or branch [microduck_rl](https://github.com/pollen-robotics/microduck_rl), preserving
its existing training and export infrastructure. Follow its current installation
instructions and record the exact commit used. Inspect the registered tasks with:

```bash
uv run list-envs
```

Register a **new task for your morphology**. Changing a mesh path in a Microduck task
is insufficient: its body names, joint selections, rewards, reset poses and commands
also refer to the original robot. There is no registered task for your four-joint robot
yet, so a complete custom training command cannot be supplied at this stage.

Implement these pieces together:

| Task component | What must match your robot |
| --- | --- |
| Entity/model | MJCF file, joint and actuator names, contact bodies, initial pose |
| Actor observations | Runtime ordering, frames, units, history and normalization |
| Actions | Controlled joints, order, home offsets, scales, clipping and filtering |
| Commands | Behaviors the hardware can actually perform |
| Rewards | Your objective, with realistic motion and energy constraints |
| Resets/termination | Achievable starting states, falls/collisions, task completion |
| Actuator model | Measured response and limits of the installed XL430s |
| Randomization | Plausible mass, friction, delay, sensor noise and calibration variation |

Retain a 50 Hz policy rate unless you deliberately change both training and deployment.
Physics can run faster: for example, a proposed 0.002 s physics step with ten substeps
per action gives a 0.020 s policy interval. Validate numerical stability and servo dynamics
before treating those example settings as final.

Start with a modest task the mechanism can perform, such as maintaining a supported pose
or tracking small commands. Add locomotion and recovery only if the joint arrangement
makes them feasible. Extra simulator state may be available to a training critic, but
the deployed actor must use only signals the real robot can provide.

**Ready to train:** one environment can reset and step repeatedly with finite observations,
correct actions and sensible contacts; the observation vector matches the Rust builder
on a recorded sample. Then increase parallel environments and training duration.

## 7. Export and validate the policy

Save the model asset revision, profile, training configuration, dependency versions, seed
and checkpoint alongside the export. The desired deployment artifact is a self-contained
**ONNX actor**; CAD and MJCF remain development/training artifacts.

Use the upstream exporter as the starting point and adapt its task/model assumptions.
Include the actor's observation normalization in the export. Begin with a feed-forward
actor to make replay easier; use recurrence only with the supported explicit-state
contract described in [recurrent policies](../recurrent-policies.md).

For your proposed four-joint contract, test the export through `ProfilePolicy` using the
validated profile; the existing `policy-rehearsal` executable is still alpha-specific.
This custom replay tool has not been wired into a CLI yet. The profile-aware loader and
its test fixtures are in `duck-control`, not a replacement deployment command.

Before hardware use, record observation sequences from your environment and replay them
through the training actor, Python ONNX Runtime, and the Rust loader. Compare actions
within a stated numerical tolerance. Include resets, zero commands, large valid commands
and failure cases. For recurrent actors, compare sequences and reset boundaries too.

Do not reuse the tiny ONNX files in `duck-control/tests/fixtures` as robot policies:
they exercise shape validation and inference, not control behavior.

The [policy manifest](../policy-manifest.md) owns publishing metadata; the current
publisher/daemon path still targets Microduck and needs profile integration before it can
install this custom policy. Renaming a four-joint policy to an alpha filename does not
make it compatible.

## 8. Prepare and calibrate the physical robot

These are hardware preparation steps. Integrated control still depends on the unfinished
runtime work identified in step 1.

Connect the Pi to the U2D2 over USB and the XL430 TTL chain to the appropriate U2D2
connector. The U2D2 does **not** supply motor power: use a suitable separate motor supply
and the manufacturer's wiring arrangement. See the
[ROBOTIS U2D2 manual](https://emanual.robotis.com/docs/en/parts/interface/u2d2/) and the
[XL430 manual](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/) for your exact model.

With the mechanism supported and torque initially disabled:

1. Use a compatible Dynamixel configuration tool to identify each motor separately.
2. Assign unique IDs and a common baud rate, recording model and firmware versions.
3. Verify the selected operating mode and hardware travel/effort settings.
4. Establish encoder zero, positive direction and home pose for each physical joint.
5. Check small individual movements before attempting a coordinated pose.

Record the U2D2's stable serial-device path on the Pi where available; do not assume that
`/dev/ttyUSB0` will always identify it. Keep a practical way to cut motor power while
calibrating. Copy neither Microduck's knee offsets nor its XL330 EEPROM corrections.

For the Sense HAT, confirm the actual board revision. Waveshare documents the
[Sense HAT (B)](https://www.waveshare.com/wiki/Sense_HAT_%28B%29) as an I²C device with
an ICM-20948 at address 0x68. Enable I²C using the supported configuration for your Pi OS.
The new backend must read that sensor independently of the U2D2 and produce calibrated
body-frame orientation and gyro measurements. Record stationary gyro bias and verify
orientation under known tilts; a responding I²C address alone does not establish correct
sensor scaling, fusion or mounting.

## 9. Integrate, rehearse and deploy — pending implementation

The next software milestone is the **existing daemon running this profile**, with the
same control lifecycle and explicit enable/disable behavior. Before attempting it on the
Pi, complete the runtime work tracked in the
[robot-model design](../design/robotd-design.md#library-support-for-other-morphologies).
Select applicable body behaviors explicitly; do not run alpha sitting, mouth motion or
leg kinematics against an arbitrary joint array.

Once that integration exists:

1. Run the actual daemon against your custom MuJoCo body and exported policy. Exercise
   enable, disable, stale commands, sensor failures, restart and shutdown.
2. On the Pi, verify the full sensing/inference/write cycle meets the 20 ms budget.
   Record tail latency and missed deadlines, not just average ONNX inference time.
3. Start hardware checks without actuation, then try supported single-joint motion,
   small coordinated poses and a restrained low-amplitude policy trial.
4. Compare real and simulated motion; update actuator dynamics and retrain if needed.
5. Make deployment reproducible and verify restart behavior before unattended operation.

The existing [simulation tutorial](simulation.md) is for Microduck, not a ready-made
custom-body launcher. The [development deployment guide](dev-push.md) owns installation
instructions, and the [updater design](../design/updater-design.md) owns release and
rollback behavior; Pi-specific integration must be completed before applying those paths
to this robot.

## What to provide when the CAD is ready

Provide the native assembly or STEP export, separate link meshes, the joint table from
step 2, link mass properties, motor gearing/couplings, and the IMU mounting transform.
Also state the first behavior you want to train. With those inputs, the next concrete
artifacts are a robot-specific `robot.xml`/`scene.xml`, a measured profile, and a registered
training task. A trained policy follows after those artifacts pass simulation checks.
