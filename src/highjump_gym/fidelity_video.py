"""Render the fidelity-ladder jumps to video for visual inspection.

    uv run python -m highjump_gym.fidelity_video

Writes one mp4 per rung (``fidelity_point_mass.mp4``, ``fidelity_rigid_arch.mp4``,
``fidelity_articulated.mp4``). The bar is placed just above the shared COM apex,
so you should see the point mass pass *under* the bar while the arch and the
articulated body drape *over* it with their COM below -- the flop signature.

Rendering needs a GL backend; set ``MUJOCO_GL=egl`` (headless GPU) or
``MUJOCO_GL=osmesa`` (CPU) if the default fails.
"""

from __future__ import annotations

from highjump_gym.analysis import bar_knocked
from highjump_gym.fidelity import (
    ARTICULATED_APEX_OFFSET,
    ARTICULATED_TAKEOFF_ROTATION_DEG,
    ARTICULATED_TAKEOFF_SPIN_DEG,
    ScriptedArch,
    Takeoff,
    build_articulated,
    build_point_mass,
    build_rigid_arch,
    simulate,
)
from highjump_gym.render import orbit_camera, render_rollout

GRAVITY = 9.81
RENDER_SECONDS = 2  # up, over, and starting back down -- before it nears the floor


def _camera(bar_height: float):
    """View looking face-on to the arch plane (~perpendicular to the 30-deg,
    from-left approach heading), so the ∩ arch and the COM-under-bar clearance
    read clearly. A small elevation keeps a little depth."""
    return orbit_camera(lookat=[1.0, 0.0, bar_height - 0.2], distance=8.0,
                        azimuth=30.0, elevation=-10.0)


def main() -> None:
    takeoff = Takeoff()
    t_apex = takeoff.vz / GRAVITY
    com_apex = takeoff.com_height + takeoff.vz**2 / (2 * GRAVITY)
    bar_height = com_apex + 0.0  # match compare(): bar just above the COM apex

    # (name, model, controller, takeoff_rotation_deg, takeoff_spin_deg, apex_offset)
    jobs = [
        ("point_mass", build_point_mass(bar_height=bar_height), None, 0.0, 0.0, 0.0),
        ("rigid_arch", build_rigid_arch(bar_height=bar_height), None, 0.0, 0.0, 0.0),
        ("articulated", build_articulated(bar_height=bar_height),
         ScriptedArch(t_full=t_apex), ARTICULATED_TAKEOFF_ROTATION_DEG,
         ARTICULATED_TAKEOFF_SPIN_DEG, ARTICULATED_APEX_OFFSET),
    ]

    print(f"bar at {bar_height:.3f} m (COM apex {com_apex:.3f} m)")
    for name, model, controller, rotation, spin, offset in jobs:
        r = simulate(
            model, takeoff, name=name, controller=controller,
            duration=RENDER_SECONDS, takeoff_rotation_deg=rotation,
            takeoff_spin_deg=spin, apex_offset=offset,
        )
        render_rollout(model, r, f"fidelity_{name}.mp4", camera=_camera(bar_height))
        print(f"  {name}: bar knocked = {bar_knocked(r)}")


if __name__ == "__main__":
    main()
