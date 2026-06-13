"""JumpModel interface and rollout recording (Phase 1).

A *jump model* is just a named controller -- a callable that maps the current
sim state to actuator commands. Different techniques and different model
fidelities are all different controllers behind the same interface, so the
analysis pipeline (``analysis.py``) can treat them uniformly.

This module ships only the interface, a rollout recorder, and a trivial
controller to exercise them. Real controllers (torque-space tracking first,
then trajectory optimisation) come in later rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import mujoco
import numpy as np

from highjump_gym.loader import load_scene

VERTICAL_AXIS = 2  # world z


@runtime_checkable
class JumpModel(Protocol):
    """A named policy over the scene's actuators.

    ``control`` returns an array of length ``model.nu`` (here: the 700 muscle
    activations; the crossbar free joint is unactuated).
    """

    name: str

    def control(
        self, model: mujoco.MjModel, data: mujoco.MjData, t: float
    ) -> np.ndarray: ...


@dataclass
class ConstantActivation:
    """Hold every actuator at a constant command.

    The trivial model used to exercise the pipeline: ``level=0`` is a passive
    drop (no muscle drive), small positive values apply uniform co-contraction.
    """

    level: float = 0.0
    name: str = "constant"

    def control(self, model, data, t):
        return np.full(model.nu, self.level, dtype=float)


@dataclass
class Rollout:
    """Recorded trajectory from stepping a jump model in the scene.

    Arrays are stacked over time (leading axis ``T``). ``com``/``com_vel`` track
    the human (the subtree rooted at ``human_root``); ``bar_pos`` tracks the
    crossbar so clearance/knock-off can be detected.
    """

    time: np.ndarray            # (T,)
    qpos: np.ndarray            # (T, nq)
    qvel: np.ndarray            # (T, nv)
    ctrl: np.ndarray            # (T, nu)
    actuator_force: np.ndarray  # (T, nu)   muscle/tendon forces
    tendon_length: np.ndarray   # (T, ntendon)
    com: np.ndarray             # (T, 3)    human COM position
    com_vel: np.ndarray         # (T, 3)    human COM linear velocity
    bar_pos: np.ndarray         # (T, 3)    crossbar position
    meta: dict = field(default_factory=dict)


def rollout(
    model: mujoco.MjModel,
    jump_model: JumpModel,
    duration: float = 1.5,
    *,
    human_root: str = "pelvis",
    bar_body: str = "arena-crossbar",
    keyframe: str | None = None,
) -> Rollout:
    """Step ``jump_model`` in ``model`` for ``duration`` seconds, recording state.

    Runs on CPU (``mj_step``); MJX batched rollouts are a Phase 3 concern. State
    is logged after each step, with ``mj_subtreeVel`` called so COM velocity is
    available.
    """
    data = mujoco.MjData(model)
    if keyframe is not None:
        mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    mujoco.mj_forward(model, data)

    root_id = model.body(human_root).id
    bar_id = model.body(bar_body).id
    n_steps = int(round(duration / model.opt.timestep))

    log: dict[str, list] = {
        k: [] for k in (
            "time", "qpos", "qvel", "ctrl", "actuator_force",
            "tendon_length", "com", "com_vel", "bar_pos",
        )
    }

    for _ in range(n_steps):
        data.ctrl[:] = np.asarray(
            jump_model.control(model, data, data.time), dtype=float
        )
        mujoco.mj_step(model, data)
        mujoco.mj_subtreeVel(model, data)  # fills subtree_linvel

        log["time"].append(data.time)
        log["qpos"].append(data.qpos.copy())
        log["qvel"].append(data.qvel.copy())
        log["ctrl"].append(data.ctrl.copy())
        log["actuator_force"].append(data.actuator_force.copy())
        log["tendon_length"].append(data.ten_length.copy())
        log["com"].append(data.subtree_com[root_id].copy())
        log["com_vel"].append(data.subtree_linvel[root_id].copy())
        log["bar_pos"].append(data.xpos[bar_id].copy())

    return Rollout(
        **{k: np.asarray(v) for k, v in log.items()},
        meta={"model": jump_model.name, "duration": duration},
    )
