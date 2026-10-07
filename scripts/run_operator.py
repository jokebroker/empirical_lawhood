# SPDX-License-Identifier: MPL-2.0
"""Launch an external operator recipe using this checkout's helpers and bootstrap.

Run with the selected checkout's locked environment. This supplies import and
numerical-runtime context only; the recipe must still supply every input and
independent human act required by the public API.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from empirical_lawhood.cli.entry import enforce_single_thread_environment

    enforce_single_thread_environment()
    sys.argv = [str(args.script), *args.arguments]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
