"""High-jump simulation gym built on MuJoCo / MJX."""

from highjump_gym.analysis import JumpSummary, summarize
from highjump_gym.jump_model import ConstantActivation, JumpModel, Rollout, rollout
from highjump_gym.loader import (
    DEFAULT_BAR_HEIGHT,
    load_human,
    load_scene,
    make_scene_spec,
    merge_with_arena,
)

# fidelity.py (the ladder demo built on top of these) is intentionally not
# imported here: it is a runnable module (python -m highjump_gym.fidelity), and
# importing it in __init__ triggers a runpy double-import warning. Import it
# directly: `from highjump_gym.fidelity import compare`.

__all__ = [
    "DEFAULT_BAR_HEIGHT",
    "load_human",
    "load_scene",
    "make_scene_spec",
    "merge_with_arena",
    "JumpModel",
    "ConstantActivation",
    "Rollout",
    "rollout",
    "JumpSummary",
    "summarize",
]
