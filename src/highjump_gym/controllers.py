"""Torque-space takeoff controller for the MS-Human-700 model (Phase 1 top rung).

The MS-Human-700 model ships with only 700 muscle actuators (``ctrl`` in 0..1)
on an underactuated 6-DOF pelvis root, so getting a *first* jump directly in
muscle space is hard. The standard musculoskeletal-control stepping stone --
adopted here -- is **torque space first**: temporarily add joint torque motors
to the primary leg/spine DOFs, get a jump working with PD trajectory tracking,
and only then map the resulting joint torques onto the 700 muscles (next round,
bridging into the Phase 2 mocap reference).

What this module provides:

* :func:`add_torque_actuators` -- inject ``<motor>`` (direct joint torque)
  actuators onto a curated set of independent primary joints of an ``MjSpec``.
  The coupled knee joints (``knee_angle_*_translation/rotation/beta``) are driven
  by the model's equality constraints and are deliberately left unactuated.
* :func:`load_torque_scene` -- the human model + torque motors + arena, compiled.
* :class:`SquatJump` -- a :class:`~highjump_gym.jump_model.JumpModel` that starts
  in a loaded crouch (feet auto-dropped onto the floor), holds it to settle, then
  tracks an explosive crouch -> extension reference with per-joint PD, producing a
  vertical squat jump. Muscle actuators are held at 0.

The reference poses, timing, gains and torque limits are module-level constants
so they can be tuned by hand (same workflow as the fidelity-ladder knobs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from highjump_gym.loader import DEFAULT_BAR_HEIGHT, merge_with_arena

# --- Which joints get a torque motor -----------------------------------------
# Per-leg primary (independent) DOFs. Suffixes ``_r``/``_l`` are added below.
_LEG_BASE = [
    "hip_flexion",
    "hip_adduction",
    "hip_rotation",
    "knee_angle",
    "ankle_angle",
    "subtalar_angle",
    "mtp_angle",
]
LEG_JOINTS = [f"{j}_{side}" for side in ("r", "l") for j in _LEG_BASE]

# The active (uncommented) spine DOFs in Body_Torso_Simple.xml.
SPINE_JOINTS = [
    "L5_S1_FE", "L5_S1_LB", "L5_S1_AR",
    "T12_L1_FE", "T12_L1_LB", "T12_L1_AR",
    "T1_head_neck_FE", "T1_head_neck_LB", "T1_head_neck_AR",
]

ACTUATED_JOINTS = LEG_JOINTS + SPINE_JOINTS

# --- Per-joint actuation parameters (torque limit, PD gains) ------------------
# Keyed by the joint *class* (matched as a substring of the joint name). Torque
# limits are rough adult-athlete peaks (N*m); gains are PD on the tracked angle.
@dataclass(frozen=True)
class JointGains:
    tau_max: float  # N*m, ctrl clamp on the motor
    kp: float       # proportional gain (N*m / rad)
    kd: float       # derivative gain   (N*m / (rad/s))


# Matched longest-substring-first against the joint name.
GAINS: dict[str, JointGains] = {
    "hip_flexion": JointGains(tau_max=300.0, kp=250.0, kd=20.0),
    "hip_adduction": JointGains(tau_max=200.0, kp=200.0, kd=15.0),
    "hip_rotation": JointGains(tau_max=120.0, kp=120.0, kd=10.0),
    "knee_angle": JointGains(tau_max=300.0, kp=250.0, kd=20.0),
    "ankle_angle": JointGains(tau_max=250.0, kp=200.0, kd=12.0),
    "subtalar_angle": JointGains(tau_max=60.0, kp=40.0, kd=3.0),
    "mtp_angle": JointGains(tau_max=40.0, kp=25.0, kd=2.0),
}
# Spine triples (FE/LB/AR) share one gain set.
for _j in SPINE_JOINTS:
    GAINS[_j] = JointGains(tau_max=200.0, kp=180.0, kd=15.0)

DEFAULT_GAINS = JointGains(tau_max=100.0, kp=100.0, kd=8.0)


def gains_for(joint: str) -> JointGains:
    """PD gains + torque limit for a joint, by longest matching class prefix."""
    best = None
    for key, g in GAINS.items():
        if key in joint and (best is None or len(key) > len(best[0])):
            best = (key, g)
    return best[1] if best else DEFAULT_GAINS


# --- Reference poses (radians) -----------------------------------------------
# Only sagittal-relevant joints are listed; any actuated joint not named here
# tracks 0. Signs follow the model's joint ranges (verified in the MJCF):
#   hip_flexion  [-0.52, 2.0]  +ve = thigh forward/up (flex)
#   knee_angle   [0, 2.4]      +ve = bend
#   ankle_angle  [-0.68, 0.52] +ve = dorsiflex; -ve = plantarflex (push-off)
def _both(**base: float) -> dict[str, float]:
    """Expand sagittal targets to both legs (``_r`` and ``_l``)."""
    out: dict[str, float] = {}
    for joint, val in base.items():
        out[f"{joint}_r"] = val
        out[f"{joint}_l"] = val
    return out


CROUCH_POSE: dict[str, float] = {
    **_both(hip_flexion=1.1, knee_angle=1.4, ankle_angle=0.35),
    "L5_S1_FE": 0.25,  # slight trunk lean into the crouch
}

EXTEND_POSE: dict[str, float] = {
    **_both(hip_flexion=-0.1, knee_angle=0.0, ankle_angle=-0.5),  # triple extension
    "L5_S1_FE": -0.1,
}

# --- Phase timing (seconds) --------------------------------------------------
# Squat jump: the body STARTS in the loaded crouch (feet planted), holds it while
# contacts settle, then extends explosively. Starting crouched -- rather than
# dipping into a countermovement from standing -- avoids the floating-base leg
# swing that makes a naive joint-PD biped topple instead of load its legs.
T_SETTLE = 0.25   # hold the crouch so the feet settle flat on the floor
T_PUSH = 0.18     # crouch -> full extension (explosive concentric phase)
# After T_SETTLE + T_PUSH the body is airborne; hold EXTEND_POSE.

# Foot skin geoms used to drop the crouch onto the floor (see SquatJump.init).
_FOOT_GEOM_KEYS = ("calcn_skin", "toes_skin")
FOOT_CLEARANCE = 0.002  # m, gap left under the lowest foot geom at t=0


def _smoothstep(a: float, b: float, t: float) -> float:
    """Cosine ease from ``a`` at ``t=0`` to ``b`` at ``t=1`` (clamped)."""
    t = min(1.0, max(0.0, t))
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return a + (b - a) * s


def add_torque_actuators(
    spec: mujoco.MjSpec, joints: list[str] = ACTUATED_JOINTS
) -> list[str]:
    """Add a direct-torque ``<motor>`` actuator to each named joint, in place.

    Each motor is named ``motor_<joint>`` with ``gear=1`` so ``ctrl`` is the
    commanded joint torque (N*m), clamped to the joint's ``tau_max``. Returns the
    list of joints that were actuated (skipping any not present in the spec).
    """
    present = {j.name for j in spec.joints}
    added: list[str] = []
    for joint in joints:
        if joint not in present:
            continue
        g = gains_for(joint)
        act = spec.add_actuator()
        act.name = f"motor_{joint}"
        act.target = joint
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = 1.0
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.gear[0] = 1.0
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange[0] = -g.tau_max
        act.ctrlrange[1] = g.tau_max
        added.append(joint)
    return added


def load_torque_scene(
    bar_height: float = DEFAULT_BAR_HEIGHT,
    joints: list[str] = ACTUATED_JOINTS,
) -> tuple[mujoco.MjModel, list[str]]:
    """Compile the human model with torque motors injected, merged with the arena.

    Returns ``(model, actuated_joints)``. The injected motors are appended after
    the 700 muscle actuators; resolve them by name (``motor_<joint>``).
    """
    from highjump_gym.loader import HUMAN_XML

    spec = mujoco.MjSpec.from_file(str(HUMAN_XML))
    actuated = add_torque_actuators(spec, joints)
    merge_with_arena(spec, bar_height)
    return spec.compile(), actuated


@dataclass
class SquatJump:
    """PD reference-tracking controller producing a vertical squat jump.

    The body is initialised in the loaded ``CROUCH`` pose with its feet dropped
    flat onto the floor (:meth:`init`), holds it while contacts settle, then
    tracks ``CROUCH -> EXTEND`` on the actuated joints with per-joint PD,
    commanding joint torques directly (muscle actuators held at 0). The explosive
    triple extension drives the feet into the ground and launches the body.

    Joint/actuator indices are resolved lazily from the compiled model on the
    first ``control``/``init`` call and cached, so the controller stays a plain
    :class:`~highjump_gym.jump_model.JumpModel` with no construction-time handle.
    """

    name: str = "squat_jump"
    t_settle: float = T_SETTLE
    t_push: float = T_PUSH
    crouch: dict[str, float] = field(default_factory=lambda: dict(CROUCH_POSE))
    extend: dict[str, float] = field(default_factory=lambda: dict(EXTEND_POSE))
    # Resolved on first use: joint -> (act_id, qpos_adr, dof_adr, gains)
    _index: dict | None = field(default=None, repr=False)

    def _resolve(self, model: mujoco.MjModel) -> None:
        index: dict[str, tuple] = {}
        for aid in range(model.nu):
            aname = model.actuator(aid).name
            if not aname.startswith("motor_"):
                continue
            joint = aname[len("motor_"):]
            jid = model.joint(joint).id
            index[joint] = (
                aid,
                int(model.jnt_qposadr[jid]),
                int(model.jnt_dofadr[jid]),
                gains_for(joint),
            )
        self._index = index

    def init(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initial-condition hook: set the crouch pose, then drop it onto the floor.

        Writes the crouch joint angles, forward-evaluates kinematics, finds the
        lowest foot-skin geom, and slides the pelvis vertically (``pelvis_tz``) so
        that geom rests ``FOOT_CLEARANCE`` above the floor. This removes any
        manual pelvis-height guess: the crouch is always planted, whatever its
        depth. Pass this as ``rollout(..., init=controller.init)``.
        """
        if self._index is None:
            self._resolve(model)
        for joint, (_aid, qadr, _vadr, _g) in self._index.items():
            data.qpos[qadr] = self.crouch.get(joint, 0.0)
        mujoco.mj_forward(model, data)

        foot_geoms = [
            gid for gid in range(model.ngeom)
            if any(k in (model.geom(gid).name or "") for k in _FOOT_GEOM_KEYS)
        ]
        lowest_z = min(float(data.geom_xpos[gid, 2]) for gid in foot_geoms)
        tz_adr = int(model.jnt_qposadr[model.joint("pelvis_tz").id])
        data.qpos[tz_adr] += FOOT_CLEARANCE - lowest_z
        mujoco.mj_forward(model, data)

    def target(self, joint: str, t: float) -> float:
        """Reference angle for ``joint`` at time ``t`` (rad)."""
        crouch = self.crouch.get(joint, 0.0)
        extend = self.extend.get(joint, 0.0)
        if t < self.t_settle:
            return crouch
        return _smoothstep(crouch, extend, (t - self.t_settle) / self.t_push)

    def control(self, model, data, t):
        if self._index is None:
            self._resolve(model)
        ctrl = np.zeros(model.nu, dtype=float)
        for joint, (aid, qadr, vadr, g) in self._index.items():
            q = data.qpos[qadr]
            qd = data.qvel[vadr]
            tau = g.kp * (self.target(joint, t) - q) - g.kd * qd
            ctrl[aid] = float(np.clip(tau, -g.tau_max, g.tau_max))
        return ctrl
