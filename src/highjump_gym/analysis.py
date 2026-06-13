"""Analysis of jump rollouts: COM kinematics, bar clearance, tendon load.

These read a :class:`~highjump_gym.jump_model.Rollout` and are shared by Phase 1
(forward-sim trade-offs) and Phase 2 (mocap embedding), which produces the same
recorded quantities from fitted motion.

The takeoff heuristics (peak vertical COM velocity) are only meaningful once a
controller actually produces a jump; for the trivial passive model they still
compute, but describe a collapse rather than a jump.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highjump_gym.jump_model import VERTICAL_AXIS, Rollout


def peak_com_height(r: Rollout) -> float:
    """Highest the human COM reaches (m)."""
    return float(r.com[:, VERTICAL_AXIS].max())


def takeoff_index(r: Rollout) -> int:
    """Frame of maximum upward COM velocity -- a proxy for the takeoff instant."""
    return int(np.argmax(r.com_vel[:, VERTICAL_AXIS]))


def takeoff_velocity(r: Rollout) -> tuple[float, float]:
    """COM speed (m/s) and launch angle (deg above horizontal) at takeoff."""
    v = r.com_vel[takeoff_index(r)]
    speed = float(np.linalg.norm(v))
    horizontal = float(np.hypot(v[0], v[1]))
    angle = float(np.degrees(np.arctan2(v[VERTICAL_AXIS], horizontal)))
    return speed, angle


def bar_displacement(r: Rollout) -> float:
    """Max distance the crossbar moves from its resting pose (m)."""
    return float(np.linalg.norm(r.bar_pos - r.bar_pos[0], axis=1).max())


def bar_knocked(r: Rollout, drop_threshold: float = 0.1) -> bool:
    """True if the crossbar ever falls more than ``drop_threshold`` below rest."""
    z0 = r.bar_pos[0, VERTICAL_AXIS]
    return bool((z0 - r.bar_pos[:, VERTICAL_AXIS]).max() > drop_threshold)


def peak_body_top(r: Rollout) -> float:
    """Highest point any athlete geom reaches (m). Requires ``top_body`` tracking."""
    _require_top(r)
    return float(r.athlete_top.max())


def body_reach_over_com(r: Rollout) -> float:
    """Max height the body's top extends above its own COM (m).

    This is the fidelity-ladder payload: for a point mass it is ~the geom radius,
    but for an extended/arched body it is how much higher than the COM the body
    can clear -- i.e. how far the COM may pass *below* the bar.
    """
    _require_top(r)
    return float((r.athlete_top - r.com[:, VERTICAL_AXIS]).max())


def _require_top(r: Rollout) -> None:
    if r.athlete_top is None:
        raise ValueError("rollout has no athlete_top; pass top_body= to rollout()")


def peak_tendon_force(r: Rollout) -> float:
    """Largest magnitude actuator/tendon force over the whole rollout (N)."""
    return float(np.abs(r.actuator_force).max())


def peak_force_per_actuator(r: Rollout) -> np.ndarray:
    """Per-actuator peak |force| over time, shape ``(nu,)`` (N)."""
    return np.abs(r.actuator_force).max(axis=0)


@dataclass
class JumpSummary:
    peak_com_height: float
    takeoff_speed: float
    takeoff_angle_deg: float
    bar_displacement: float
    bar_knocked: bool
    peak_tendon_force: float


def summarize(r: Rollout) -> JumpSummary:
    """Roll up the headline metrics for one rollout."""
    speed, angle = takeoff_velocity(r)
    return JumpSummary(
        peak_com_height=peak_com_height(r),
        takeoff_speed=speed,
        takeoff_angle_deg=angle,
        bar_displacement=bar_displacement(r),
        bar_knocked=bar_knocked(r),
        peak_tendon_force=peak_tendon_force(r),
    )
