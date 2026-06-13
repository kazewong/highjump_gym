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

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from highjump_gym.fidelity import (
    ScriptedArch,
    Takeoff,
    build_articulated,
    build_point_mass,
    build_rigid_arch,
    simulate,
)
from highjump_gym.analysis import bar_knocked
from highjump_gym.jump_model import Rollout

GRAVITY = 9.81
RENDER_FPS = 30
RENDER_SECONDS = 0.9  # up, over, and starting back down -- before it nears the floor


def _camera(bar_height: float) -> mujoco.MjvCamera:
    """View looking face-on to the arch plane (~perpendicular to the 30-deg,
    from-left approach heading), so the ∩ arch and the COM-under-bar clearance
    read clearly. A small elevation keeps a little depth."""
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([1.0, 0.0, bar_height - 0.2])
    cam.distance = 8.0
    cam.azimuth = 30.0
    cam.elevation = -10.0
    return cam


def render_rollout(
    model: mujoco.MjModel, r: Rollout, path: str, bar_height: float
) -> None:
    """Replay a rollout's qpos through the renderer and write an mp4."""
    import imageio.v2 as imageio

    cam = _camera(bar_height)
    every = max(1, int(round(1.0 / (RENDER_FPS * model.opt.timestep))))
    data = mujoco.MjData(model)

    frames = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for i in range(0, len(r.time), every):
            data.qpos[:] = r.qpos[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    imageio.mimwrite(path, frames, fps=RENDER_FPS)
    print(f"  wrote {len(frames)} frames -> {path}")


def main() -> None:
    takeoff = Takeoff()
    t_apex = takeoff.vz / GRAVITY
    com_apex = takeoff.com_height + takeoff.vz**2 / (2 * GRAVITY)
    bar_height = com_apex + 0.12  # match compare(): bar just above the COM apex

    jobs = [
        ("point_mass", build_point_mass(bar_height=bar_height), None),
        ("rigid_arch", build_rigid_arch(bar_height=bar_height), None),
        ("articulated", build_articulated(bar_height=bar_height),
         ScriptedArch(t_full=t_apex)),
    ]

    print(f"bar at {bar_height:.3f} m (COM apex {com_apex:.3f} m)")
    for name, model, controller in jobs:
        r = simulate(
            model, takeoff, name=name, controller=controller,
            duration=RENDER_SECONDS,
        )
        render_rollout(model, r, f"fidelity_{name}.mp4", bar_height)
        print(f"  {name}: bar knocked = {bar_knocked(r)}")


if __name__ == "__main__":
    main()
