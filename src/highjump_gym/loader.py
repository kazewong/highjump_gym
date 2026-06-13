"""Model loading for the high-jump gym.

We deliberately avoid a cross-directory MJCF ``<include>`` of the MS-Human-700
model: its meshes use relative paths (``../geometry/*.stl``) that MuJoCo
resolves against the *main* file's directory, so including it from elsewhere
breaks asset resolution. Instead each model is loaded as an ``mjSpec`` from its
own location (assets resolve correctly) and the two are merged in memory with
``MjSpec.attach``.

The MS-Human-700 model is the parent of the merge so its joint/tendon/muscle
names stay unprefixed (analysis code can reference e.g. ``hip_flexion_r``
directly); arena elements are merged under an ``arena-`` prefix.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]

HUMAN_XML = _REPO_ROOT / "menagerie" / "ms_human_700" / "MS-Human-700.xml"
ARENA_XML = _PKG_DIR / "assets" / "arena.xml"

DEFAULT_BAR_HEIGHT = 2.0  # metres


def _find(elements, name: str):
    """Look up a spec element by name (MjSpec accessor names vary by version)."""
    for element in elements:
        if element.name == name:
            return element
    raise KeyError(f"{name!r} not found in spec")


def _set_z(element, z: float) -> None:
    """Set an element's world/local z while keeping x, y (pos may be a copy)."""
    pos = list(element.pos)
    pos[2] = z
    element.pos = pos


def _set_bar_height(arena: mujoco.MjSpec, height: float) -> None:
    """Place the crossbar and the two support pegs at ``height`` metres.

    A peg's top sits at ``height + 0.008`` (half the peg box thickness), so the
    bar rests with its centre at ``height + 0.023`` (bar radius 0.015). It starts
    1 mm above that (``height + 0.024``) and settles gently onto the pegs.
    """
    _set_z(_find(arena.geoms, "peg_l"), height)
    _set_z(_find(arena.geoms, "peg_r"), height)
    _set_z(_find(arena.bodies, "crossbar"), height + 0.024)


def make_scene_spec(bar_height: float = DEFAULT_BAR_HEIGHT) -> mujoco.MjSpec:
    """Return an ``MjSpec`` of the MS-Human-700 model merged with the arena."""
    human = mujoco.MjSpec.from_file(str(HUMAN_XML))
    arena = mujoco.MjSpec.from_file(str(ARENA_XML))

    _set_bar_height(arena, bar_height)

    # The human model ships a keyframe whose qpos length equals its own nq.
    # Adding the crossbar free joint changes nq, so the stale key would fail to
    # compile -- drop it and start from the default standing pose instead.
    for key in list(human.keys):
        human.delete(key)

    frame = human.worldbody.add_frame()
    human.attach(arena, frame=frame, prefix="arena-")
    return human


def load_scene(bar_height: float = DEFAULT_BAR_HEIGHT) -> mujoco.MjModel:
    """Compile the merged human + arena scene to an ``MjModel``."""
    return make_scene_spec(bar_height).compile()


def load_human() -> mujoco.MjModel:
    """The bare MS-Human-700 model (no arena), loaded from its own directory."""
    return mujoco.MjModel.from_xml_path(str(HUMAN_XML))
