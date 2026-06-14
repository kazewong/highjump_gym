"""Offscreen rendering of rollouts to mp4, shared by the demo scripts.

Replays a recorded :class:`~highjump_gym.jump_model.Rollout` (its ``qpos``
frames) through MuJoCo's offscreen :class:`mujoco.Renderer` and writes an mp4.
Centralising this keeps the demos (``fidelity_video``, ``controller_demo``, ...)
from each re-implementing the same render loop.

Rendering needs a GL backend; ``MUJOCO_GL`` is defaulted to ``egl`` (headless
GPU) on import. Set it to ``osmesa`` (CPU) in the environment if EGL is missing.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # read lazily when the GL context opens

import mujoco
import numpy as np

from highjump_gym.jump_model import Rollout


def orbit_camera(
    lookat, distance: float, azimuth: float, elevation: float
) -> mujoco.MjvCamera:
    """A free orbit camera aimed at ``lookat`` (x, y, z)."""
    cam = mujoco.MjvCamera()
    cam.lookat = np.asarray(lookat, dtype=float)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_rollout(
    model: mujoco.MjModel,
    rollout: Rollout,
    path: str,
    *,
    camera: mujoco.MjvCamera | None = None,
    fps: int = 30,
    height: int = 480,
    width: int = 640,
) -> str:
    """Replay ``rollout``'s qpos through the renderer and write an mp4 to ``path``.

    Frames are sampled to ``fps`` from the rollout's timestep. ``camera`` is an
    :class:`mujoco.MjvCamera` (e.g. from :func:`orbit_camera`); ``None`` uses the
    model's default free camera. Returns ``path``.
    """
    import imageio.v2 as imageio

    every = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    cam_arg = camera if camera is not None else -1
    data = mujoco.MjData(model)

    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i in range(0, len(rollout.time), every):
            data.qpos[:] = rollout.qpos[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam_arg)
            frames.append(renderer.render())

    imageio.mimwrite(path, frames, fps=fps)
    print(f"  wrote {len(frames)} frames -> {path}")
    return path
