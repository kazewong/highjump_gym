# High jump simulation

This is a high jump simulation environment built using the MuJoCo physics engine. There are a couple of intended functionalities:

1. Forward simulating a number of high jump to investigate the trade-offs between different jump models, such that we can develop a better understanding of the event
2. Given some data points from mocap, embed the person into the simulating environment to extract data such as center of mass velocity as well as stress in tendons.
3. Use reinforcement learning to find potential optimization for executing the events.

## Models

The project is anchored on the [MS-Human-700](./menagerie/ms_human_700) musculoskeletal model (full-body muscles and tendons), which is what makes tendon-stress analysis physically meaningful. A torque-actuated humanoid (e.g. [unitree_h1](./menagerie/unitree_h1)) is kept as a cheaper fallback for RL prototyping. Models are vendored via the `mujoco_menagerie` submodule and simulated with MuJoCo MJX.

## Roadmap

### Phase 0 — Foundation
- **Arena scene**: a reusable include adding high-jump geometry — two standards, a knockable crossbar (a separate free body resting on supports so bar displacement is detectable), a soft landing mat, and a runway plane, with parameterized bar height.
- **Loader + smoke test**: `loader.py` returning `(mjx.Model, mjx.Data)` for both the MS-Human-700 and torque-humanoid models composed with the arena, plus an MJX-compatibility check (the musculoskeletal model's many tendons/equality constraints may need tweaks vs CPU MuJoCo). Verified by a load → step → render smoke test.

### Phase 1 — Forward simulation & analysis (goal 1)
- A `JumpModel` abstraction: a policy (`state -> ctrl`) plus technique parameters (approach speed, takeoff angle). Implement 2–3 scripted variants (e.g. Fosbury flop vs straddle takeoff) so trade-offs are measurable.
- `analysis.py` extracting COM trajectory/velocity, bar clearance, peak tendon forces, and ground reaction. Shared with Phase 2.

### Phase 2 — Mocap embedding (goal 2)
- A `mocap/` package: keypoint loader, retargeting map from mocap markers to model bodies/sites, per-frame IK (batched over a clip in MJX) to fit joint angles, then inverse dynamics to recover joint torques and tendon stress.

### Phase 3 — Reinforcement learning (goal 3)
- A Brax/Gymnasium-style env wrapping the MJX model (reward: clear the bar without knocking it; penalize energy and tendon overload), trained batched in MJX. Prototype on the torque humanoid, then transfer to the musculoskeletal model.
