"""Exercise the MJX rollout path: single, batched, and CPU<->MJX parity.

    uv run python -m highjump_gym.mjx_demo

Runs three things on the MS-Human-700 + torque-motor scene:

1. A single closed-loop MJX rollout of the squat-jump controller (JIT + scan),
   reporting peak COM height and takeoff -- same metrics as the CPU demo.
2. A batched rollout that ``vmap``s a sweep of the push duration ``t_push`` in
   parallel, reporting each member's peak COM height (the Phase-3 search/RL
   primitive: many controllers stepped at once).
3. A parity check: the CPU recorder's exact ``ctrl`` sequence replayed through
   MJX open-loop, reporting how far the MJX trajectory drifts from the CPU one.

No GPU is required -- MJX runs on CPU too (just slower). The jump itself is still
untuned; this demo is about the *rollout machinery*, not jump quality.
"""

from __future__ import annotations

import mujoco
import numpy as np

from highjump_gym.analysis import peak_com_height, takeoff_velocity
from highjump_gym.controllers import SquatJump, load_torque_scene
from highjump_gym.jump_model import rollout as cpu_rollout
from highjump_gym.mjx_rollout import (
    batched_rollout,
    mjx_rollout,
    parity_check,
    squat_jump_control_fn,
)

DURATION = 1.0  # short horizon keeps the parity check meaningful


def main() -> None:
    import jax.numpy as jp

    model, actuated = load_torque_scene()
    print(f"scene: {len(actuated)} torque motors, model.nu={model.nu}, "
          f"dt={model.opt.timestep}")

    # Grounded crouch initial state (reuse the controller's init hook).
    controller = SquatJump()
    data = mujoco.MjData(model)
    controller.init(model, data)
    qpos0 = data.qpos.copy()
    print(f"crouch COM height at t=0: {float(data.subtree_com[model.body('pelvis').id][2]):.3f} m")

    # 1) Single closed-loop MJX rollout.
    print("\n[1] single MJX rollout (jit+scan)...")
    cfn = squat_jump_control_fn(model, controller)
    r = mjx_rollout(model, cfn, duration=DURATION, qpos0=qpos0, name="squat_jump")
    speed, angle = takeoff_velocity(r)
    print(f"    peak COM {peak_com_height(r):.3f} m, takeoff {speed:.2f} m/s @ {angle:.0f} deg")

    # 2) Batched sweep over t_push (parallel controllers).
    print("\n[2] batched MJX rollout: t_push sweep...")
    t_push_grid = jp.linspace(0.10, 0.26, 5)
    traj = batched_rollout(
        model, lambda p: squat_jump_control_fn(model, controller, t_push=p),
        t_push_grid, duration=DURATION, qpos0=qpos0,
    )
    peak_per_member = traj["com"][:, :, 2].max(axis=1)  # (B,)
    for tp, pk in zip(np.asarray(t_push_grid), peak_per_member):
        print(f"    t_push={float(tp):.3f}s -> peak COM {float(pk):.3f} m")

    # 3) CPU<->MJX parity (open-loop ctrl replay).
    print("\n[3] CPU<->MJX parity check (open-loop ctrl replay)...")
    r_cpu = cpu_rollout(
        model, controller, duration=DURATION,
        human_root="pelvis", bar_body="arena-crossbar", init=controller.init,
    )
    diffs = parity_check(model, r_cpu, qpos0=qpos0)
    print(f"    over {diffs['n_steps']} steps: COM max|Δ|={diffs['com_max_abs']:.2e} m "
          f"(mean {diffs['com_mean_abs']:.2e}), qpos max|Δ|={diffs['qpos_max_abs']:.2e}")
    ok = diffs["com_max_abs"] < 0.02
    print(f"    parity {'PASS' if ok else 'CHECK'} (COM agrees within "
          f"{'2 cm' if ok else 'tolerance exceeded'})")


if __name__ == "__main__":
    main()
