"""Batched, JIT-compiled rollouts on MJX (JAX backend).

The CPU recorder in :mod:`highjump_gym.jump_model` is fine for inspecting a
single jump, but Phase 3 (RL / trajectory optimisation) needs many rollouts in
parallel and gradients/vectorisation through the physics. This module provides
that on **MJX** (plain JAX -- no warp backend):

* :func:`mjx_rollout` -- a single closed-loop rollout via ``jax.lax.scan`` over
  ``mjx.step``, returning a :class:`~highjump_gym.jump_model.Rollout`.
* :func:`batched_rollout` -- ``jax.vmap`` over a batch of controller parameters,
  so a whole sweep/population is stepped at once (the RL/search primitive).
* :func:`squat_jump_control_fn` -- a pure-JAX reimplementation of the
  :class:`~highjump_gym.controllers.SquatJump` PD law, usable inside the scan.
* :func:`parity_check` -- replays a CPU rollout's exact ``ctrl`` sequence through
  MJX open-loop and compares trajectories, validating that the MJX integrator
  reproduces the CPU one (independent of any control-law reimplementation).

MJX runs in float32 by default, so MJX vs CPU trajectories diverge slowly in
contact-rich motion; the parity check uses a short horizon and a loose tolerance.

COM velocity is finite-differenced from ``subtree_com`` (MJX does not expose the
``mj_subtreeVel`` helper), matching the CPU recorder's ``com_vel`` to O(dt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from highjump_gym.jump_model import Rollout, _subtree_geom_ids

# A pure-JAX control law: (t, qpos, qvel) -> ctrl of length nu.
ControlFn = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True)
class _Refs:
    """Resolved indices/ids for a scene, shared by the rollout functions."""

    root_id: int
    bar_id: int
    top_geoms: np.ndarray | None
    dt: float


def _refs(
    model: mujoco.MjModel, root_body: str, bar_body: str, top_body: str | None
) -> _Refs:
    return _Refs(
        root_id=model.body(root_body).id,
        bar_id=model.body(bar_body).id,
        top_geoms=_subtree_geom_ids(model, top_body) if top_body else None,
        dt=float(model.opt.timestep),
    )


def _scan(mjx_model, d0, ctrl_seq_or_fn, n_steps, refs: _Refs, *, open_loop: bool):
    """Core ``lax.scan`` body shared by closed- and open-loop rollouts.

    ``ctrl_seq_or_fn`` is either an ``(n_steps, nu)`` array (open loop) or a
    :data:`ControlFn` (closed loop). Returns a dict of stacked trajectories.
    """
    top = refs.top_geoms
    top_idx = None if top is None else jp.asarray(top)

    def step(d, x):
        i, ext = x
        t = i * refs.dt
        ctrl = ext if open_loop else ctrl_seq_or_fn(t, d.qpos, d.qvel)
        d = d.replace(ctrl=ctrl)
        d = mjx.step(mjx_model, d)
        out = {
            "qpos": d.qpos,
            "qvel": d.qvel,
            "ctrl": d.ctrl,
            "actuator_force": d.actuator_force,
            "ten_length": d.ten_length,
            "com": d.subtree_com[refs.root_id],
            "bar_pos": d.xpos[refs.bar_id],
        }
        if top_idx is not None:
            out["athlete_top"] = jp.max(d.geom_xpos[top_idx, 2])
        return d, out

    idx = jp.arange(n_steps)
    ext = ctrl_seq_or_fn if open_loop else jp.zeros(n_steps)
    _, traj = jax.lax.scan(step, d0, (idx, ext))
    return traj


def _to_rollout(traj: dict, dt: float, name: str, has_top: bool) -> Rollout:
    """Convert stacked JAX trajectories to a CPU :class:`Rollout`."""
    arr = {k: np.asarray(v) for k, v in traj.items()}
    n = arr["qpos"].shape[0]
    time = (np.arange(1, n + 1) * dt).astype(float)  # state logged post-step
    com = arr["com"]
    com_vel = np.gradient(com, dt, axis=0)  # MJX has no mj_subtreeVel helper
    return Rollout(
        time=time,
        qpos=arr["qpos"],
        qvel=arr["qvel"],
        ctrl=arr["ctrl"],
        actuator_force=arr["actuator_force"],
        tendon_length=arr["ten_length"],
        com=com,
        com_vel=com_vel,
        bar_pos=arr["bar_pos"],
        athlete_top=arr["athlete_top"] if has_top else None,
        meta={"model": name, "backend": "mjx", "duration": n * dt},
    )


def mjx_rollout(
    model: mujoco.MjModel,
    control_fn: ControlFn,
    duration: float = 1.4,
    *,
    qpos0: np.ndarray | None = None,
    qvel0: np.ndarray | None = None,
    root_body: str = "pelvis",
    bar_body: str = "arena-crossbar",
    top_body: str | None = "pelvis",
    name: str = "mjx",
) -> Rollout:
    """One closed-loop MJX rollout, returned as a CPU :class:`Rollout`.

    ``control_fn(t, qpos, qvel) -> ctrl`` must be pure JAX (it runs inside the
    scan). ``qpos0``/``qvel0`` set the initial state -- e.g. the grounded crouch
    from :meth:`~highjump_gym.controllers.SquatJump.init`.
    """
    refs = _refs(model, root_body, bar_body, top_body)
    mjx_model = mjx.put_model(model)
    d0 = mjx.make_data(mjx_model)
    if qpos0 is not None:
        d0 = d0.replace(qpos=jp.asarray(qpos0, dtype=d0.qpos.dtype))
    if qvel0 is not None:
        d0 = d0.replace(qvel=jp.asarray(qvel0, dtype=d0.qvel.dtype))
    n_steps = int(round(duration / refs.dt))

    traj = jax.jit(
        lambda d: _scan(mjx_model, d, control_fn, n_steps, refs, open_loop=False)
    )(d0)
    return _to_rollout(traj, refs.dt, name, refs.top_geoms is not None)


def batched_rollout(
    model: mujoco.MjModel,
    make_control_fn: Callable[[jax.Array], ControlFn],
    params: jax.Array,
    duration: float = 1.4,
    *,
    qpos0: np.ndarray | None = None,
    qvel0: np.ndarray | None = None,
    root_body: str = "pelvis",
    bar_body: str = "arena-crossbar",
    top_body: str | None = "pelvis",
) -> dict[str, np.ndarray]:
    """``vmap`` a closed-loop rollout over a batch of controller parameters.

    ``make_control_fn(p)`` returns a :data:`ControlFn` for one parameter vector
    ``p``; ``params`` is a leading-batched array of such vectors. All members
    share the initial state ``qpos0``/``qvel0`` and the model. Returns a dict of
    batched trajectories (leading axis ``B``) -- the Phase-3 search/RL primitive.
    """
    refs = _refs(model, root_body, bar_body, top_body)
    mjx_model = mjx.put_model(model)
    base = mjx.make_data(mjx_model)
    if qpos0 is not None:
        base = base.replace(qpos=jp.asarray(qpos0, dtype=base.qpos.dtype))
    if qvel0 is not None:
        base = base.replace(qvel=jp.asarray(qvel0, dtype=base.qvel.dtype))
    n_steps = int(round(duration / refs.dt))

    def single(p):
        return _scan(
            mjx_model, base, make_control_fn(p), n_steps, refs, open_loop=False
        )

    traj = jax.jit(jax.vmap(single))(params)
    return {k: np.asarray(v) for k, v in traj.items()}


# --- JAX squat-jump control law (mirrors controllers.SquatJump) ---------------
def squat_jump_control_fn(
    model: mujoco.MjModel, controller=None, *, t_settle=None, t_push=None
) -> ControlFn:
    """Build a pure-JAX PD control law equivalent to :class:`SquatJump`.

    Resolves the motor actuators and their joint addresses/gains from ``model``
    and returns ``control_fn(t, qpos, qvel)`` implementing the same
    crouch->extend smoothstep + per-joint PD as the CPU controller, so MJX
    closed-loop rollouts track the same reference.

    ``t_settle``/``t_push`` override the controller's timing and may be *traced*
    JAX scalars -- so a batched sweep can ``vmap`` over them (see
    :func:`batched_rollout`).
    """
    from highjump_gym.controllers import SquatJump

    ctrl = controller or SquatJump()
    if ctrl._index is None:
        ctrl._resolve(model)

    a_ids, qadr, vadr, kp, kd, taumax, crouch, extend = [], [], [], [], [], [], [], []
    for joint, (aid, qa, va, g) in ctrl._index.items():
        a_ids.append(aid); qadr.append(qa); vadr.append(va)
        kp.append(g.kp); kd.append(g.kd); taumax.append(g.tau_max)
        crouch.append(ctrl.crouch.get(joint, 0.0))
        extend.append(ctrl.extend.get(joint, 0.0))

    a_ids = jp.asarray(a_ids); qadr = jp.asarray(qadr); vadr = jp.asarray(vadr)
    kp = jp.asarray(kp); kd = jp.asarray(kd); taumax = jp.asarray(taumax)
    crouch = jp.asarray(crouch); extend = jp.asarray(extend)
    nu = model.nu
    _t_settle = ctrl.t_settle if t_settle is None else t_settle
    _t_push = ctrl.t_push if t_push is None else t_push

    def control_fn(t, qpos, qvel):
        frac = jp.clip((t - _t_settle) / _t_push, 0.0, 1.0)
        s = 0.5 - 0.5 * jp.cos(jp.pi * frac)
        qref = crouch + (extend - crouch) * s
        q = qpos[qadr]
        qd = qvel[vadr]
        tau = jp.clip(kp * (qref - q) - kd * qd, -taumax, taumax)
        return jp.zeros(nu).at[a_ids].set(tau)

    return control_fn


# --- Parity check -------------------------------------------------------------
def parity_check(
    model: mujoco.MjModel,
    cpu_rollout: Rollout,
    qpos0: np.ndarray,
    qvel0: np.ndarray | None = None,
    *,
    root_body: str = "pelvis",
    bar_body: str = "arena-crossbar",
) -> dict[str, float]:
    """Replay ``cpu_rollout``'s ``ctrl`` sequence through MJX open-loop.

    Steps MJX from the same initial state applying the recorded per-step ``ctrl``
    (no feedback), so any divergence is the integrator/contact model alone, not a
    reimplemented control law. Returns max/mean absolute COM and qpos differences.
    """
    refs = _refs(model, root_body, bar_body, None)
    mjx_model = mjx.put_model(model)
    d0 = mjx.make_data(mjx_model)
    d0 = d0.replace(qpos=jp.asarray(qpos0, dtype=d0.qpos.dtype))
    if qvel0 is not None:
        d0 = d0.replace(qvel=jp.asarray(qvel0, dtype=d0.qvel.dtype))
    ctrl_seq = jp.asarray(cpu_rollout.ctrl, dtype=d0.ctrl.dtype)
    n_steps = ctrl_seq.shape[0]

    traj = jax.jit(
        lambda d, c: _scan(mjx_model, d, c, n_steps, refs, open_loop=True)
    )(d0, ctrl_seq)

    com_mjx = np.asarray(traj["com"])
    qpos_mjx = np.asarray(traj["qpos"])
    com_diff = np.abs(com_mjx - cpu_rollout.com)
    qpos_diff = np.abs(qpos_mjx - cpu_rollout.qpos)
    return {
        "com_max_abs": float(com_diff.max()),
        "com_mean_abs": float(com_diff.mean()),
        "qpos_max_abs": float(qpos_diff.max()),
        "qpos_mean_abs": float(qpos_diff.mean()),
        "n_steps": int(n_steps),
    }
