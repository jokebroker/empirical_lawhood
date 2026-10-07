"""Receipt-backed target reference-validation checks for preparation sources.

Historical inputs require a separately verified export/import mapping to these
target descriptors and target custody. Original source or receipt bytes do not
become current qualifications through this reader.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, cast

from .fixtures import run_fixtures
from .schemas import QuantumTrajectoryPreparationQualificationConfig
from .types import Stage, Validity, Verdict


SOURCE_BEARING_FILES = (
    "bounded_intervals.py",
    "checkpoint.py",
    "comparator.py",
    "deterministic_reference.py",
    "evolution.py",
    "fixtures.py",
    "jumps.py",
    "operators.py",
    "receivers.py",
)

PREDECESSOR_ARTIFACTS = {
    "plan_sha256": "controls/plan.md",
    "config_sha256": "controls/config.json",
    "deterministic_reference_result_sha256": "stages/deterministic-reference/result.json",
    "path_conformance_result_sha256": "stages/path-source/result.json",
    "ensemble_conformance_result_sha256": "stages/ensemble-source/result.json",
    "closeout_sha256": "closeout/result.json",
}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _verified_artifact(root: Path, relative_path: str) -> tuple[str, str]:
    path = root / relative_path
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    payload = path.read_bytes()
    receipt_payload = receipt_path.read_bytes()
    receipt = json.loads(receipt_payload)
    value = receipt.get("value")
    if (
        not isinstance(value, Mapping)
        or value.get("relative_path") != relative_path
        or value.get("size_bytes") != len(payload)
        or value.get("sha256") != sha256(payload).hexdigest()
        or value.get("read_back") != "VERIFIED"
    ):
        raise RuntimeError(f"predecessor receipt verification failed: {relative_path}")
    return sha256(payload).hexdigest(), sha256(receipt_payload).hexdigest()


def run_base_compatibility(
    config: QuantumTrajectoryPreparationQualificationConfig,
    *,
    repository_root: Path,
    scientific_root: Path,
) -> dict[str, object]:
    predecessor = config.predecessor
    predecessor_root = scientific_root / str(predecessor["external_root"])
    identity_rows: list[dict[str, object]] = []
    passed = True
    try:
        for identity, relative_path in PREDECESSOR_ARTIFACTS.items():
            observed, _ = _verified_artifact(predecessor_root, relative_path)
            expected = str(predecessor[identity])
            row_passed = observed == expected
            passed = passed and row_passed
            identity_rows.append(
                {
                    "identity": identity,
                    "relative_path": relative_path,
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "passed": row_passed,
                }
            )
        _, closeout_receipt = _verified_artifact(
            predecessor_root,
            "closeout/result.json",
        )
        closeout_receipt_passed = closeout_receipt == predecessor["closeout_receipt_sha256"]
        passed = passed and closeout_receipt_passed

        implementation_document = json.loads(
            (predecessor_root / "controls/implementation-manifest.json").read_bytes()
        )
        implementation_value = implementation_document.get("value")
        if not isinstance(implementation_value, Mapping):
            raise RuntimeError("predecessor implementation manifest has no value object")
        implementation_passed = (
            implementation_value.get("implementation_sha256")
            == predecessor["implementation_sha256"]
        )
        passed = passed and implementation_passed
        files = cast(list[Mapping[str, object]], implementation_value["implementation_files"])
        frozen_hashes = {
            Path(str(row["relative_path"])).name: str(row["sha256"])
            for row in files
            if "/quantum_trajectory_reference_validation/" in str(row["relative_path"])
        }
        reference_source_package = repository_root / (
            "src/empirical_lawhood/adapters/simulators/quantum_trajectory_reference_validation"
        )
        preparation_source_package = repository_root / (
            "src/empirical_lawhood/adapters/simulators/quantum_trajectory_preparation_qualification"
        )
        file_rows: list[dict[str, object]] = []
        for name in SOURCE_BEARING_FILES:
            reference_source_sha256 = _sha(reference_source_package / name)
            preparation_source_sha256 = _sha(preparation_source_package / name)
            expected_file = frozen_hashes.get(name)
            row_passed = bool(expected_file == reference_source_sha256 == preparation_source_sha256)
            passed = passed and row_passed
            file_rows.append(
                {
                    "relative_name": name,
                    "frozen_reference_source_sha256": expected_file,
                    "reference_source_sha256": reference_source_sha256,
                    "preparation_source_sha256": preparation_source_sha256,
                    "normalization": "NONE_EXACT_BYTES",
                    "passed": row_passed,
                }
            )
        forbidden_imports = []
        for path in preparation_source_package.glob("*.py"):
            text = path.read_text()
            if any(
                "quantum_trajectory_reference_validation" in line
                for line in text.splitlines()
                if line.lstrip().startswith(("from ", "import "))
            ):
                forbidden_imports.append(path.name)
        passed = passed and not forbidden_imports
        fixture_rows = run_fixtures(config)
        expected_zero = {"unoccupied-basis-site", "zero-mass-prng-guard"}
        fixture_passed = bool(
            len(fixture_rows) == 21
            and all(
                row.validity
                == (
                    Validity.ZERO_PROJECTED_MASS
                    if row.fixture_id in expected_zero
                    else Validity.VALID
                )
                for row in fixture_rows
            )
        )
        passed = passed and fixture_passed
        return {
            "stage": Stage.BASE_COMPATIBILITY.value,
            "verdict": (Verdict.BASE_COMPATIBLE.value if passed else Verdict.BASE_STOP.value),
            "passed": passed,
            "predecessor_root": str(predecessor_root),
            "identity_rows": identity_rows,
            "closeout_receipt_expected_sha256": predecessor["closeout_receipt_sha256"],
            "closeout_receipt_observed_sha256": closeout_receipt,
            "closeout_receipt_passed": closeout_receipt_passed,
            "implementation_identity_passed": implementation_passed,
            "source_file_rows": file_rows,
            "forbidden_historical_runtime_imports": forbidden_imports,
            "truth_known_fixture_count": len(fixture_rows),
            "truth_known_fixtures_passed": fixture_passed,
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        return {
            "stage": Stage.BASE_COMPATIBILITY.value,
            "verdict": Verdict.BASE_STOP.value,
            "passed": False,
            "predecessor_root": str(predecessor_root),
            "identity_rows": identity_rows,
            "reason_code": type(error).__name__,
            "detail": str(error),
        }


__all__ = ["SOURCE_BEARING_FILES", "run_base_compatibility"]
