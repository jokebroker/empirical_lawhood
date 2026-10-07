"""Read-only target predecessor and source/action compatibility.

Historical inputs require an external verified export/import mapping to the
current descriptors, exact implementation identity and target custody. This
reader transfers no original qualification or receipt grant.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence
from empirical_lawhood.kernel.serialization import validate_sha256

from .basis import build_basis, build_boundary_current_operator, build_hamiltonian, hermiticity_residual, sparse_matrix_fingerprint
from .contracts import Action, QuantumResponseControlConfig, Verdict


REFERENCE_REQUIRED = {
    "stages/conformance/result.json": "SEMANTIC_CONTRACT_VALID",
    "stages/deterministic-reference/result.json": "DETERMINISTIC_REFERENCE_QUALIFIED",
    "stages/path-source/result.json": "PATH_SOURCE_QUALIFIED",
    "stages/ensemble-source/result.json": "ENSEMBLE_SOURCE_QUALIFIED",
}
@dataclass(frozen=True, slots=True)
class CompatibilityExpectedIdentities:
    """Caller supplied identities of the two predecessor evaluations."""

    reference_validation_implementation_sha256: str
    preparation_qualification_plan_sha256: str
    preparation_qualification_config_sha256: str
    preparation_qualification_implementation_sha256: str
    preparation_qualification_freeze_sha256: str
    preparation_qualification_closeout_sha256: str
    preparation_qualification_closeout_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            validate_sha256(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class CompatibilityCheck:
    check_id: str
    passed: bool
    observed: str
    expected: str
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    verdict: Verdict
    checks: tuple[CompatibilityCheck, ...]
    basis_sha256: str | None
    hamiltonian_sha256_by_action: Mapping[Action, str]
    current_sha256_by_action: Mapping[Action, str]
    reason_codes: tuple[str, ...]


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _document(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError(f"predecessor document is not an object: {path}")
    nested = value.get("value", value)
    if not isinstance(nested, Mapping):
        raise TypeError(f"predecessor value is not an object: {path}")
    return nested


def verify_payload_receipt(root: Path, relative_path: str) -> CompatibilityCheck:
    payload_path = root / relative_path
    receipt_path = payload_path.with_suffix(payload_path.suffix + ".receipt.json")
    if not payload_path.is_file() or not receipt_path.is_file():
        return CompatibilityCheck(
            f"receipt:{relative_path}",
            False,
            "MISSING",
            "PAYLOAD_AND_RECEIPT",
            "predecessor-payload-or-receipt-missing",
        )
    payload_sha = _sha_file(payload_path)
    receipt = _document(receipt_path)
    passed = (
        receipt.get("relative_path") == relative_path
        and receipt.get("size_bytes") == payload_path.stat().st_size
        and receipt.get("sha256") == payload_sha
    )
    return CompatibilityCheck(
        f"receipt:{relative_path}",
        passed,
        payload_sha,
        str(receipt.get("sha256")),
        None if passed else "predecessor-receipt-mismatch",
    )


def _check(
    check_id: str,
    observed: object,
    expected: object,
    reason: str,
) -> CompatibilityCheck:
    return CompatibilityCheck(
        check_id,
        observed == expected,
        str(observed),
        str(expected),
        None if observed == expected else reason,
    )


def evaluate_compatibility(
    *,
    config: QuantumResponseControlConfig,
    reference_validation_root: Path,
    preparation_qualification_root: Path,
    expected: CompatibilityExpectedIdentities,
    forbidden_import_hits: Sequence[str] = (),
) -> CompatibilityResult:
    checks: list[CompatibilityCheck] = []
    required_exports = (
        *((reference_validation_root, path) for path in REFERENCE_REQUIRED),
        *((preparation_qualification_root, path) for path in (
            "controls/freeze.json",
            "stages/base-compatibility/result.json",
            "stages/statistical-conformance/result.json",
            "stages/development/result.json",
            "stages/evaluation/result.json",
            "closeout/result.json",
        )),
    )
    missing = tuple(
        f"{root}/{path}"
        for root, path in required_exports
        if not (root / path).is_file()
        or not (root / (path + ".receipt.json")).is_file()
    )
    if missing:
        return CompatibilityResult(
            verdict=Verdict.PREDECESSOR_RECEIPT_STOP,
            checks=(CompatibilityCheck(
                "target-predecessor-export",
                False,
                ";".join(missing),
                "CURRENT_DESCRIPTORS_AND_VERIFIED_TARGET_CUSTODY",
                "target-predecessor-export-required",
            ),),
            basis_sha256=None,
            hamiltonian_sha256_by_action={},
            current_sha256_by_action={},
            reason_codes=("target-predecessor-export-required",),
        )
    for relative_path, verdict in REFERENCE_REQUIRED.items():
        checks.append(verify_payload_receipt(reference_validation_root, relative_path))
        document = _document(reference_validation_root / relative_path)
        checks.extend(
            (
                _check(
                    f"declared contract-verdict:{relative_path}",
                    document.get("verdict"),
                    verdict,
                    "declared contract-component-verdict-mismatch",
                ),
                _check(
                    f"declared contract-implementation:{relative_path}",
                    document.get("implementation_sha256"),
                    expected.reference_validation_implementation_sha256,
                    "declared contract-implementation-identity-mismatch",
                ),
            )
        )
    for relative_path in (
        "controls/freeze.json",
        "stages/base-compatibility/result.json",
        "stages/statistical-conformance/result.json",
        "stages/development/result.json",
        "stages/evaluation/result.json",
        "closeout/result.json",
    ):
        checks.append(verify_payload_receipt(preparation_qualification_root, relative_path))
    freeze = _document(preparation_qualification_root / "controls/freeze.json")
    closeout = _document(preparation_qualification_root / "closeout/result.json")
    evaluation_result = _document(preparation_qualification_root / "stages/evaluation/result.json")
    checks.extend(
        (
            _check(
                "declared contract-plan",
                closeout.get("plan_sha256"),
                expected.preparation_qualification_plan_sha256,
                "declared contract-plan-identity-mismatch",
            ),
            _check(
                "declared contract-config",
                closeout.get("config_sha256"),
                expected.preparation_qualification_config_sha256,
                "declared contract-config-identity-mismatch",
            ),
            _check(
                "declared contract-implementation",
                closeout.get("implementation_sha256"),
                expected.preparation_qualification_implementation_sha256,
                "declared contract-implementation-identity-mismatch",
            ),
            _check(
                "declared contract-freeze",
                freeze.get("freeze_sha256"),
                expected.preparation_qualification_freeze_sha256,
                "declared contract-freeze-identity-mismatch",
            ),
            _check(
                "declared contract-closeout-payload",
                _sha_file(preparation_qualification_root / "closeout/result.json"),
                expected.preparation_qualification_closeout_sha256,
                "declared contract-closeout-identity-mismatch",
            ),
            _check(
                "declared contract-closeout-receipt",
                _sha_file(preparation_qualification_root / "closeout/result.json.receipt.json"),
                expected.preparation_qualification_closeout_receipt_sha256,
                "declared contract-closeout-receipt-identity-mismatch",
            ),
            _check(
                'declared compatibility-evaluation verdict',
                evaluation_result.get("verdict"),
                "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_STRONG_SOURCE_PREPARATION_QUALIFIED",
                "declared contract-preparation-verdict-mismatch",
            ),
            _check(
                "declared contract-handoff",
                closeout.get("handoff"),
                "RESPONSE_CONTROL_STRONG_ONLY_PLANNING_ELIGIBLE",
                "declared contract-handoff-mismatch",
            ),
            _check(
                "declared contract-selected-cutoff",
                evaluation_result.get("selected_burnin"),
                200.0,
                "declared contract-cutoff-mismatch",
            ),
            _check(
                "declared contract-config-cutoff",
                str(config.cutoff),
                "200",
                "declared contract-cutoff-mismatch",
            ),
            _check(
                "historical-runtime-imports",
                tuple(forbidden_import_hits),
                (),
                "forbidden-historical-runtime-import",
            ),
        )
    )
    basis = build_basis(config.l_sites, config.particles)
    hamiltonians = {
        action: build_hamiltonian(
            basis,
            action=action,
            epsilon=float(config.epsilon),
            j_xy=float(config.j_xy),
            j_z=float(config.j_z),
        )
        for action in Action
    }
    currents = {
        action: build_boundary_current_operator(
            basis,
            action=action,
            epsilon=float(config.epsilon),
            j_xy=float(config.j_xy),
        )
        for action in Action
    }
    for action, matrix in hamiltonians.items():
        checks.append(
            _check(
                f"hamiltonian-hermitian:{action.value}",
                hermiticity_residual(matrix) <= 1e-12,
                True,
                "action-hamiltonian-not-hermitian",
            )
        )
    for action, matrix in currents.items():
        checks.append(
            _check(
                f"current-hermitian:{action.value}",
                hermiticity_residual(matrix) <= 1e-12,
                True,
                "boundary-current-not-hermitian",
            )
        )
    reasons = tuple(
        sorted(
            {
                check.reason_code
                for check in checks
                if not check.passed and check.reason_code is not None
            }
        )
    )
    if not reasons:
        verdict = Verdict.COMPATIBLE_STRONG_SOURCE_READY
    elif any("receipt" in reason for reason in reasons):
        verdict = Verdict.PREDECESSOR_RECEIPT_STOP
    elif any("action" in reason or "current" in reason for reason in reasons):
        verdict = Verdict.ACTION_COMPATIBILITY_STOP
    elif "forbidden-historical-runtime-import" in reasons:
        verdict = Verdict.IMPLEMENTATION_IDENTITY_STOP
    else:
        verdict = Verdict.SOURCE_COMPATIBILITY_STOP
    return CompatibilityResult(
        verdict=verdict,
        checks=tuple(checks),
        basis_sha256=basis.fingerprint,
        hamiltonian_sha256_by_action={
            action: sparse_matrix_fingerprint(matrix) for action, matrix in hamiltonians.items()
        },
        current_sha256_by_action={
            action: sparse_matrix_fingerprint(matrix) for action, matrix in currents.items()
        },
        reason_codes=reasons,
    )


__all__ = [
    "CompatibilityCheck",
    "CompatibilityExpectedIdentities",
    "CompatibilityResult",
    "evaluate_compatibility",
    "verify_payload_receipt",
]
