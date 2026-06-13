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
from typing import Callable, Protocol, runtime_checkable

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
    athlete_top: np.ndarray | None = None  # (T,) highest athlete geom z, if tracked
    meta: dict = field(default_factory=dict)


def rollout(
    model: mujoco.MjModel,
    jump_model: JumpModel,
    duration: float = 1.5,
    *,
    human_root: str = "pelvis",
    bar_body: str = "arena-crossbar",
    keyframe: str | None = None,
    init: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    top_body: str | None = None,
) -> Rollout:
    """Step ``jump_model`` in ``model`` for ``duration`` seconds, recording state.

    Runs on CPU (``mj_step``); MJX batched rollouts are a Phase 3 concern. State
    is logged after each step, with ``mj_subtreeVel`` called so COM velocity is
    available.

    ``human_root`` names the body whose subtree COM is tracked (the human, or a
    fidelity-ladder athlete). ``init`` is an optional hook called after reset to
    set initial conditions, e.g. a takeoff velocity. ``top_body``, if given,
    records the highest z of that body's geoms each step (``athlete_top``), used
    to measure bar clearance for the fidelity ladder.
    """
    data = mujoco.MjData(model)
    if keyframe is not None:
        mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    if init is not None:
        init(model, data)
    mujoco.mj_forward(model, data)

    root_id = model.body(human_root).id
    bar_id = model.body(bar_body).id
    top_geoms = _subtree_geom_ids(model, top_body) if top_body is not None else None
    n_steps = int(round(duration / model.opt.timestep))

    keys = (
        "time", "qpos", "qvel", "ctrl", "actuator_force",
        "tendon_length", "com", "com_vel", "bar_pos", "athlete_top",
    )
    log: dict[str, list] = {k: [] for k in keys}

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
        if top_geoms is not None:
            log["athlete_top"].append(float(data.geom_xpos[top_geoms, 2].max()))

    return Rollout(
        time=np.asarray(log["time"]),
        qpos=np.asarray(log["qpos"]),
        qvel=np.asarray(log["qvel"]),
        ctrl=np.asarray(log["ctrl"]),
        actuator_force=np.asarray(log["actuator_force"]),
        tendon_length=np.asarray(log["tendon_length"]),
        com=np.asarray(log["com"]),
        com_vel=np.asarray(log["com_vel"]),
        bar_pos=np.asarray(log["bar_pos"]),
        athlete_top=np.asarray(log["athlete_top"]) if top_geoms is not None else None,
        meta={"model": jump_model.name, "duration": duration},
    )


def _subtree_geom_ids(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    """Geom indices for the whole subtree rooted at ``body_name``.

    Spans every segment of a multi-body athlete (e.g. an articulated chain), not
    just the named body, so its highest point is tracked correctly.
    """
    root = model.body(body_name).id
    in_subtree = np.zeros(model.nbody, dtype=bool)
    for body in range(model.nbody):
        ancestor = body
        while True:
            if ancestor == root:
                in_subtree[body] = True
                break
            if ancestor == 0:  # reached world without hitting root
                break
            ancestor = model.body_parentid[ancestor]
    return np.nonzero(in_subtree[model.geom_bodyid])[0]
