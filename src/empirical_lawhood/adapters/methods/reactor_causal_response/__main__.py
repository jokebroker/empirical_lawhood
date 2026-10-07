"""Synthetic operand entrypoint; never acquires, issues or finalizes science."""

import argparse
from pathlib import Path
from .serialization import produce_calibration, produce_discovery


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("discovery", "calibration"))
    p.add_argument("--synthetic-software-fixture", action="store_true", required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fit", type=Path)
    a = p.parse_args()
    if a.operation == "discovery":
        produce_discovery(a.raw, a.output)
    else:
        if a.fit is None:
            p.error("calibration requires --fit")
        produce_calibration(a.raw, a.fit, a.output)


if __name__ == "__main__":
    main()
