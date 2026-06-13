"""Phase 0 smoke test: load -> step -> render, and probe MJX compatibility.

Run from the repo root with::

    uv run python -m highjump_gym.smoke_test

It works through four stages and prints a PASS/FAIL line for each:

1. Compile the bare MS-Human-700 model.
2. Compile the merged human + arena scene and confirm the crossbar free joint
   added 7 qpos.
3. CPU passive drop: step the scene under gravity, assert no NaNs, and render an
   mp4 (``highjump_smoke.mp4``) so the settling can be eyeballed.
4. MJX probe: put the scene on MJX and take a few steps. The MS-Human-700 model
   uses muscles / spatial tendons / equality constraints that MJX may not fully
   support, so this stage is wrapped to report exactly what (if anything) fails.

Rendering needs a GL backend; set ``MUJOCO_GL=egl`` (headless GPU) or
``MUJOCO_GL=osmesa`` (CPU) if the default fails.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from highjump_gym.loader import load_human, load_scene

DROP_SECONDS = 1.5
RENDER_FPS = 30
VIDEO_PATH = "highjump_smoke.mp4"


def _stats(model: mujoco.MjModel) -> str:
    return (
        f"nq={model.nq} nv={model.nv} nu={model.nu} "
        f"nbody={model.nbody} ntendon={model.ntendon} neq={model.neq}"
    )


def stage1_human() -> mujoco.MjModel:
    model = load_human()
    print(f"[1] PASS  bare human compiled: {_stats(model)}")
    return model


def stage2_scene(human: mujoco.MjModel) -> mujoco.MjModel:
    scene = load_scene()
    added = scene.nq - human.nq
    assert added == 7, f"expected +7 qpos from crossbar freejoint, got {added}"
    # Arena elements are merged under the 'arena-' prefix (loader.make_scene_spec);
    # this raises if the merge dropped the bar.
    scene.body("arena-crossbar")
    print(f"[2] PASS  human+arena merged: {_stats(scene)} (+{added} qpos for bar)")
    return scene


def stage3_drop(scene: mujoco.MjModel) -> None:
    data = mujoco.MjData(scene)
    mujoco.mj_forward(scene, data)

    n_steps = int(DROP_SECONDS / scene.opt.timestep)
    every = max(1, int(1.0 / (RENDER_FPS * scene.opt.timestep)))

    frames = []
    with mujoco.Renderer(scene, height=480, width=640) as renderer:
        for i in range(n_steps):
            mujoco.mj_step(scene, data)
            if not np.all(np.isfinite(data.qpos)):
                raise FloatingPointError(f"non-finite qpos at step {i}")
            if i % every == 0:
                renderer.update_scene(data, camera=-1)
                frames.append(renderer.render())

    pelvis_z = float(data.xpos[scene.body("pelvis").id, 2])
    print(f"[3] PASS  {n_steps} CPU steps, no NaNs; final pelvis z={pelvis_z:.3f} m")
    _write_video(frames)


def _write_video(frames: list) -> None:
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(VIDEO_PATH, frames, fps=RENDER_FPS)
        print(f"      wrote {len(frames)} frames -> {VIDEO_PATH}")
    except Exception as exc:  # noqa: BLE001 - rendering is best-effort
        print(f"      (skipped video write: {exc!r})")


def stage4_mjx(scene: mujoco.MjModel) -> None:
    try:
        import jax
        from mujoco import mjx

        mx = mjx.put_model(scene)
        dx = mjx.make_data(mx)
        step = jax.jit(mjx.step)
        for _ in range(5):
            dx = step(mx, dx)
        qpos = np.asarray(dx.qpos)
        ok = np.all(np.isfinite(qpos))
        print(f"[4] {'PASS' if ok else 'FAIL'}  MJX stepped 5x; finite qpos={ok}")
    except Exception as exc:  # noqa: BLE001 - this is exactly what we're probing
        print(f"[4] FAIL  MJX incompatibility: {type(exc).__name__}: {exc}")
        print("      (expected if muscles/spatial tendons aren't MJX-supported; "
              "next step is to identify and substitute the unsupported features)")


def main() -> None:
    human = stage1_human()
    scene = stage2_scene(human)
    stage3_drop(scene)
    stage4_mjx(scene)


if __name__ == "__main__":
    main()
