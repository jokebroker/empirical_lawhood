"""Generate offline configuration resources from their closed typed owners."""

from __future__ import annotations

import argparse
from pathlib import Path

from empirical_lawhood.api.configuration_schemas import generated_resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "src/empirical_lawhood/resources/config_schemas"
    resources = generated_resources()
    actual = {path.name for path in root.glob("*.json")}
    if arguments.check and actual != set(resources):
        raise SystemExit("configuration schema resource inventory drifted")
    for name, payload in resources.items():
        path = root / name
        if arguments.check:
            if path.read_bytes() != payload:
                raise SystemExit(f"configuration schema resource drifted: {name}")
        else:
            path.write_bytes(payload)
    print(f"configuration schemas: {len(resources) - 1} projections; closed inventory current")


if __name__ == "__main__":
    main()
