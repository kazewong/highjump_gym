"""Exercise the Phase 1 pipeline end to end.

    uv run python -m highjump_gym.phase1_demo

Loads the scene, rolls out the trivial constant-activation model, and prints the
analysis summary. This is a wiring check for the JumpModel / rollout / analysis
interfaces -- with no real controller yet the human just collapses, so the
"takeoff" numbers describe that collapse, not a jump.
"""

from __future__ import annotations

from highjump_gym.analysis import summarize
from highjump_gym.jump_model import ConstantActivation, rollout
from highjump_gym.loader import load_scene


def main() -> None:
    model = load_scene()
    jump_model = ConstantActivation(level=0.0, name="passive")

    r = rollout(model, jump_model, duration=1.5)
    s = summarize(r)

    print(f"model={r.meta['model']!r}  frames={len(r.time)}  nu={r.ctrl.shape[1]}")
    print(f"  peak COM height   : {s.peak_com_height:6.3f} m")
    print(f"  takeoff speed     : {s.takeoff_speed:6.3f} m/s")
    print(f"  takeoff angle     : {s.takeoff_angle_deg:6.1f} deg")
    print(f"  bar displacement  : {s.bar_displacement:6.3f} m")
    print(f"  bar knocked off   : {s.bar_knocked}")
    print(f"  peak tendon force : {s.peak_tendon_force:6.1f} N")


if __name__ == "__main__":
    main()
