"""The fidelity ladder: compare jump *models* of increasing detail.

In flight a body's COM is a pure ballistic arc -- it depends only on the takeoff
COM velocity, not on the body model. What differs between fidelity levels is the
body's *extent around the COM*: how much higher than its COM the body can reach
while draping over the bar. That gap is the whole point of the Fosbury flop -- an
arched body clears a bar its COM never rises to -- and it is exactly what a point
mass cannot represent.

This module builds two low-fidelity athletes that plug into the same
``Rollout``/``analysis`` pipeline as the full muscle model:

* ``build_point_mass`` -- a single sphere (COM only).
* ``build_rigid_arch`` -- a rigid, downward-draping arch spanning the bar.

Both are launched ballistically (no actuators, no collisions) from the same
takeoff COM state, so the comparison isolates body geometry. The full
MS-Human-700 model is the top rung, added once it has a real controller.

    uv run python -m highjump_gym.fidelity
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from highjump_gym.analysis import (
    body_reach_over_com,
    peak_body_top,
    peak_com_height,
)
from highjump_gym.jump_model import (
    ConstantActivation,
    JumpModel,
    Rollout,
    rollout,
)
from highjump_gym.loader import DEFAULT_BAR_HEIGHT, merge_with_arena

ATHLETE_BODY = "athlete"
ATHLETE_JOINT = "athlete_free"


@dataclass
class Takeoff:
    """Takeoff condition: COM launch speed/angle and COM height at takeoff."""

    speed: float = 4.3          # m/s
    angle_deg: float = 69.0     # degrees above horizontal
    com_height: float = 1.25    # m

    @property
    def vx(self) -> float:
        return self.speed * math.cos(math.radians(self.angle_deg))

    @property
    def vz(self) -> float:
        return self.speed * math.sin(math.radians(self.angle_deg))


def _athlete_model(body_xml: str, bar_height: float) -> mujoco.MjModel:
    """Wrap an athlete body (one free body, non-colliding geoms) in the arena."""
    spec = mujoco.MjSpec.from_string(
        f'<mujoco model="athlete">\n'
        f'  <worldbody>\n'
        f'    <body name="{ATHLETE_BODY}" pos="0 0 0">\n'
        f'      <freejoint name="{ATHLETE_JOINT}"/>\n'
        f"{body_xml}"
        f"    </body>\n"
        f"  </worldbody>\n"
        f"</mujoco>\n"
    )
    return merge_with_arena(spec, bar_height).compile()


def build_point_mass(
    radius: float = 0.05, bar_height: float = DEFAULT_BAR_HEIGHT
) -> mujoco.MjModel:
    """A single sphere: the COM with essentially no extent."""
    geom = (
        f'      <geom name="pm" type="sphere" size="{radius}" '
        f'contype="0" conaffinity="0" rgba="0.9 0.3 0.3 1"/>\n'
    )
    return _athlete_model(geom, bar_height)


def build_rigid_arch(
    span: float = 1.1,
    sag: float = 0.6,
    radius: float = 0.04,
    n_seg: int = 8,
    bar_height: float = DEFAULT_BAR_HEIGHT,
) -> mujoco.MjModel:
    """A rigid arch draping downward in the bar (y-z) plane.

    Apex at the body-frame origin (local z=0), legs sagging to ``-sag``, so the
    COM sits below the apex. Built along the local x axis (arching up in the
    local x-z plane); ``simulate`` then yaws the whole body to the approach
    heading. It can pass over the bar with its COM below the apex -- the
    geometric basis of the flop.
    """
    xs = np.linspace(-span / 2, span / 2, n_seg + 1)
    zs = -sag * (2 * xs / span) ** 2  # parabola, apex (0,0), legs at -sag
    segments = "".join(
        f'      <geom type="capsule" size="{radius}" '
        f'fromto="{xs[i]:.4f} 0 {zs[i]:.4f} {xs[i + 1]:.4f} 0 {zs[i + 1]:.4f}" '
        f'contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>\n'
        for i in range(n_seg)
    )
    return _athlete_model(segments, bar_height)


def build_articulated(
    n_seg: int = 8,
    seg_len: float = 0.22,
    radius: float = 0.05,
    joint_damping: float = 5.0,
    kp: float = 300.0,
    bar_height: float = DEFAULT_BAR_HEIGHT,
) -> mujoco.MjModel:
    """A serial chain of ``n_seg`` rigid segments along the bar (y) axis.

    Straight at rest; hinge joints (axis x) let it curve in the bar/z plane into
    a ∩. Each hinge has a position servo so a scripted open-loop trajectory
    (:class:`ScriptedArch`) can bend it into the arch during flight. Geoms are
    non-colliding, so the COM stays ballistic and the only thing that changes is
    the body's configuration.
    """
    inner = ""
    for i in reversed(range(n_seg)):
        name = ATHLETE_BODY if i == 0 else f"seg{i}"
        pos = "0 0 0" if i == 0 else f"{seg_len} 0 0"
        free = f'<freejoint name="{ATHLETE_JOINT}"/>' if i == 0 else ""
        # hinge axis +y so positive servo angles arch the chain upward (∩) in
        # the local x-z plane; simulate() yaws the whole body to the heading.
        joint = (
            ""
            if i == 0
            else f'<joint name="hinge{i}" type="hinge" axis="0 1 0" '
            f'damping="{joint_damping}"/>'
        )
        geom = (
            f'<geom type="capsule" fromto="0 0 0 {seg_len} 0 0" size="{radius}" '
            f'contype="0" conaffinity="0" rgba="0.3 0.7 0.4 1"/>'
        )
        inner = f'<body name="{name}" pos="{pos}">{free}{joint}{geom}{inner}</body>'

    actuators = "".join(
        f'<position name="act{i}" joint="hinge{i}" kp="{kp}"/>'
        for i in range(1, n_seg)
    )
    spec = mujoco.MjSpec.from_string(
        f'<mujoco model="articulated">'
        f"<worldbody>{inner}</worldbody>"
        f"<actuator>{actuators}</actuator>"
        f"</mujoco>"
    )
    return merge_with_arena(spec, bar_height).compile()


@dataclass
class ScriptedArch:
    """Open-loop servo targets that bend the chain into a ∩ by ``t_full``.

    Every hinge is driven to the same angle (uniform curvature), ramped via a
    smoothstep from straight at ``t_start`` to a total turning angle of
    ``total_arch`` spread across the hinges by ``t_full`` (set near the COM apex
    so the body is fully draped as it crosses the bar).
    """

    total_arch: float = math.pi
    t_start: float = 0.0
    t_full: float = 0.4
    name: str = "articulated"

    def control(self, model, data, t):
        s = _smoothstep((t - self.t_start) / max(self.t_full - self.t_start, 1e-6))
        per_hinge = self.total_arch / max(model.nu, 1)
        return np.full(model.nu, per_hinge * s, dtype=float)


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _scene_refs(model: mujoco.MjModel) -> tuple[np.ndarray, float]:
    """COM offset of the athlete from its free-joint origin, and the bar's x."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    athlete = model.body(ATHLETE_BODY).id
    com_offset = data.subtree_com[athlete] - data.xpos[athlete]
    bar_x = float(data.xpos[model.body("arena-crossbar").id, 0])
    return com_offset, bar_x


def simulate(
    model: mujoco.MjModel,
    takeoff: Takeoff,
    *,
    name: str = "athlete",
    controller: JumpModel | None = None,
    duration: float = 1.2,
    approach_angle_deg: float = 30.0,
    from_left: bool = True,
) -> Rollout:
    """Launch the athlete ballistically from ``takeoff`` and record the rollout.

    The body (built along local x) is yawed so its long axis -- and its line of
    travel -- runs at ``approach_angle_deg`` off the bar line (0 deg = parallel,
    the Fosbury flop; 90 deg = straight across). ``from_left`` flips which side
    of the bar the athlete enters from. The takeoff point is chosen so the COM
    apex passes over the centre of the bar, making "does it clear" well posed.

    ``controller`` defaults to a passive (no-drive) model; an articulated body
    is driven by a :class:`ScriptedArch`. The body starts straight with zero
    joint velocity, so its COM velocity at takeoff equals the launch velocity.
    """
    if controller is None:
        controller = ConstantActivation(level=0.0, name=name)
    local_offset, bar_x = _scene_refs(model)
    g = -float(model.opt.gravity[2])
    t_apex = takeoff.vz / g

    # Horizontal heading: angle off the bar line (y), tilted toward crossing the
    # bar (+x). from_left sets the side the athlete enters from.
    a = math.radians(approach_angle_deg)
    side = -1.0 if from_left else 1.0
    heading = np.array([math.sin(a), side * math.cos(a), 0.0])

    # Yaw the body about z so its local-x long axis points along the heading.
    yaw = math.atan2(heading[1], heading[0])
    quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    world_offset = np.zeros(3)
    mujoco.mju_rotVecQuat(world_offset, local_offset, quat)

    # Global launch velocity, and a takeoff point putting the COM apex over the
    # centre of the bar (x = bar_x, y = 0 at t_apex).
    vel = takeoff.vx * heading + np.array([0.0, 0.0, takeoff.vz])
    com0 = np.array([bar_x - vel[0] * t_apex, -vel[1] * t_apex, takeoff.com_height])
    root_pos = com0 - world_offset

    jid = model.joint(ATHLETE_JOINT).id
    q = int(model.jnt_qposadr[jid])
    d = int(model.jnt_dofadr[jid])

    def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.qpos[q : q + 3] = root_pos
        data.qpos[q + 3 : q + 7] = quat
        data.qvel[d : d + 3] = vel
        data.qvel[d + 3 : d + 6] = 0.0

    return rollout(
        model,
        controller,
        duration=duration,
        human_root=ATHLETE_BODY,
        top_body=ATHLETE_BODY,
        init=init,
    )


def compare(takeoff: Takeoff | None = None, clearance_margin: float = 0.12) -> None:
    """Run the ladder and print the COM-vs-clearance comparison."""
    takeoff = takeoff or Takeoff()
    t_apex = takeoff.vz / 9.81  # arch should complete near the COM apex
    rollouts = {
        "point-mass": simulate(build_point_mass(), takeoff, name="point-mass"),
        "rigid-arch": simulate(build_rigid_arch(), takeoff, name="rigid-arch"),
        "articulated": simulate(
            build_articulated(),
            takeoff,
            name="articulated",
            controller=ScriptedArch(t_full=t_apex),
        ),
    }

    com_peak = peak_com_height(rollouts["point-mass"])  # identical across models
    bar = com_peak + clearance_margin

    print(f"takeoff: speed={takeoff.speed} m/s  angle={takeoff.angle_deg} deg  "
          f"COM height={takeoff.com_height} m")
    print(f"COM apex (ballistic, model-independent): {com_peak:.3f} m")
    print(f"test bar set {clearance_margin:.2f} m above COM apex -> {bar:.3f} m\n")
    print(f"{'model':<12}{'COM apex':>10}{'body top':>10}{'reach':>8}"
          f"{'clears bar':>12}{'COM below bar':>15}")
    for name, r in rollouts.items():
        top = peak_body_top(r)
        reach = body_reach_over_com(r)
        clears = top >= bar
        below = bar - peak_com_height(r)
        print(f"{name:<12}{peak_com_height(r):>10.3f}{top:>10.3f}{reach:>8.3f}"
              f"{str(clears):>12}{below:>15.3f}")

    print("\nAll three share one ballistic COM apex. The point mass cannot clear a "
          "bar above it; the rigid arch can (static drape); the articulated body "
          "reaches the same advantage dynamically -- taking off straight and "
          "arching into the ∩ in flight, the precursor to real flop technique.")


if __name__ == "__main__":
    compare()
