"""Drive the MS-Human-700 model through a torque-space squat jump.

    uv run python -m highjump_gym.controller_demo            # metrics only
    uv run python -m highjump_gym.controller_demo --video    # also write mp4

This is the Phase 1 top rung: the full muscle model jumping under a real
controller. Torque motors are injected on the primary leg/spine joints and a
PD reference-tracking controller (:class:`~highjump_gym.controllers.SquatJump`)
produces the jump; the muscle actuators are held at 0. The next round maps the
recorded joint torques onto the 700 muscles.

The bar is left at the loader default and is *not* the point here -- this rung
is about producing a real, ground-reaction jump and reading off its takeoff
kinematics and tendon/joint loads through the shared analysis pipeline. Tune the
pose/timing/gain constants in ``controllers.py`` if the jump stalls or topples.
"""

from __future__ import annotations

import argparse

from highjump_gym.analysis import summarize, takeoff_velocity
from highjump_gym.controllers import SquatJump, load_torque_scene
from highjump_gym.jump_model import rollout


def run(duration: float = 1.4):
    model, actuated = load_torque_scene()
    print(f"injected {len(actuated)} torque motors; model.nu = {model.nu}")
    controller = SquatJump()
    r = rollout(
        model, controller, duration=duration,
        human_root="pelvis", bar_body="arena-crossbar", top_body="pelvis",
        init=controller.init,
    )
    return model, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="render an mp4")
    ap.add_argument("--duration", type=float, default=1.4)
    args = ap.parse_args()

    model, r = run(duration=args.duration)

    s = summarize(r)
    speed, angle = takeoff_velocity(r)
    standing_com = float(r.com[0, 2])
    print(f"  standing COM height : {standing_com:.3f} m")
    print(f"  peak COM height     : {s.peak_com_height:.3f} m "
          f"(rise {s.peak_com_height - standing_com:+.3f} m)")
    print(f"  takeoff speed/angle : {speed:.2f} m/s @ {angle:.1f} deg")
    print(f"  peak joint torque   : {s.peak_tendon_force:.0f} N*m")

    if args.video:
        from highjump_gym.render import orbit_camera, render_rollout

        cam = orbit_camera(lookat=[0.2, 0.0, 1.0], distance=4.5,
                           azimuth=90.0, elevation=-8.0)
        render_rollout(model, r, "controller_jump.mp4", camera=cam)


if __name__ == "__main__":
    main()
