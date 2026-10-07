# SPDX-License-Identifier: MPL-2.0
"""Check coverage source paths and the explicitly claimed measured owners."""

from __future__ import annotations

import argparse
from configparser import ConfigParser
import json
from pathlib import Path


CRITICAL_MEASURED_MODULES = frozenset({
    "src/empirical_lawhood/api/codecs.py",
    "src/empirical_lawhood/api/current_result_custody.py",
    "src/empirical_lawhood/api/integration_proof.py",
    "src/empirical_lawhood/api/matrix_numerical_provenance.py",
    "src/empirical_lawhood/kernel/numerical_provenance.py",
    "src/empirical_lawhood/kernel/parsed_nodes.py",
    "src/empirical_lawhood/adapters/simulators/_native_admission.py",
    "src/empirical_lawhood/kernel/serialization.py",
    "src/empirical_lawhood/kernel/decoding.py",
    "src/empirical_lawhood/adapters/_bounded_files.py",
    "src/empirical_lawhood/infrastructure/bounded_io.py",
    "src/empirical_lawhood/api/facade.py",
    "src/empirical_lawhood/infrastructure/sql/database.py",
    "src/empirical_lawhood/infrastructure/sql/migrations/versions/initial_catalog.py",
})


def validate_configuration(checkout: Path) -> None:
    config = ConfigParser()
    if not config.read(checkout / ".coveragerc"):
        raise ValueError("coverage configuration is missing")
    patterns = config.get("run", "include").split()
    if not patterns:
        raise ValueError("coverage source inventory is empty")
    for pattern in patterns:
        if not any(path.is_file() for path in checkout.glob(pattern)):
            raise ValueError(f"coverage include has no source match: {pattern}")


def validate_measured_inventory(checkout: Path, coverage_json: Path) -> None:
    root = checkout.resolve()
    measured = set()
    unexecuted = set()
    report = json.loads(coverage_json.read_text())
    for filename, details in report["files"].items():
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        try:
            selected = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        measured.add(selected)
        if selected in CRITICAL_MEASURED_MODULES:
            lines = details.get("executed_lines") if isinstance(details, dict) else None
            summary = details.get("summary") if isinstance(details, dict) else None
            covered = summary.get("covered_lines") if isinstance(summary, dict) else None
            if (not isinstance(lines, list) or not lines
                    or any(type(line) is not int or line <= 0 for line in lines)
                    or type(covered) is not int or covered <= 0):
                unexecuted.add(selected)
    missing = sorted(CRITICAL_MEASURED_MODULES - measured)
    if missing:
        raise ValueError(f"required measured coverage owners missing: {missing}")
    if unexecuted:
        raise ValueError(f"required coverage owners have no executed statements: {sorted(unexecuted)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--coverage-json", type=Path)
    args = parser.parse_args()
    validate_configuration(args.checkout)
    if args.coverage_json is not None:
        validate_measured_inventory(args.checkout, args.coverage_json)
    print("coverage source inventory verified" + (
        f"; {len(CRITICAL_MEASURED_MODULES)} required owners with executed statements"
        if args.coverage_json is not None else ""))


if __name__ == "__main__":
    main()
