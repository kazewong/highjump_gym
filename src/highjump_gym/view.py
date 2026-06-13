"""Interactive viewer for eyeballing the rig.

Examples::

    uv run python -m highjump_gym.view                 # arena only
    uv run python -m highjump_gym.view --scene         # human + arena merged
    uv run python -m highjump_gym.view --bar-height 2.3

Opens the MuJoCo passive viewer; close the window to exit. Use this to sanity
check the crossbar/peg/standard geometry without running the full smoke test.
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer

from highjump_gym.loader import ARENA_XML, DEFAULT_BAR_HEIGHT, load_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        action="store_true",
        help="view the merged human + arena scene (default: arena only)",
    )
    parser.add_argument(
        "--bar-height",
        type=float,
        default=DEFAULT_BAR_HEIGHT,
        help="crossbar height in metres (only affects --scene)",
    )
    args = parser.parse_args()

    if args.scene:
        model = load_scene(bar_height=args.bar_height)
    else:
        model = mujoco.MjModel.from_xml_path(str(ARENA_XML))

    mujoco.viewer.launch(model)


if __name__ == "__main__":
    main()
