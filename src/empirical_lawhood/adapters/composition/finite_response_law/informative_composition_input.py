# SPDX-License-Identifier: MPL-2.0

"""Receipt-bound, retained-only informative-composition operands; no native acquisition or promotion."""

import base64
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.finite_response_law.development import develop_fold, summarize
from empirical_lawhood.adapters.methods.finite_response_law.fitting import Features, targets
from empirical_lawhood.adapters.methods.finite_response_law.oracle import evaluate_oracle
from empirical_lawhood.adapters.methods.finite_response_law.retained import FiniteResponseLawObservedPanel
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec, development_requests
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

ROOTS = tuple(
    f"{c}.prepared.r{r:03d}" for c, n in (("prepared-response", 16), ("information-response-prediction", 32)) for r in range(n)
)
ARRAYS = {
    "hold_observed": ((48, 5, 2, 2), "|u1"),
    "observed": ((48, 5, 4, 8, 2, 2), "|u1"),
    "parent_work": ((48, 5, 2), "<f8"),
    "valid": ((48, 5, 4, 8, 2, 2), "|u1"),
    "x": ((48, 24, 2), "<f8"),
    "y": ((48, 5, 4, 8, 2, 2), "<f8"),
    "z": ((48, 5, 24, 2), "<f8"),
}


@dataclass(frozen=True, slots=True)
class FiniteResponseLawInformativeCompositionParent(CanonicalRecord):
    """A producer's authenticated feature/panel join under original identities.

    Aliases are the retained method's fixed row labels, not new independent units.
    Each aliases exactly one original physical root in the publication census.
    Unobserved cells retain their masks; they are never completed by imputation.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/finite-response-law-informative-composition-parent'
    science: FiniteResponseLawScienceSpec
    root_identity_map: tuple[tuple[str, str], ...]
    arrays_base64: tuple[tuple[str, str], ...]
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        if (
            tuple(alias for alias, _ in self.root_identity_map) != ROOTS
            or len({root for _, root in self.root_identity_map}) != 48
            or tuple(key for key, _ in self.arrays_base64) != tuple(ARRAYS)
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError("FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_PARENT_CENSUS_OR_ROLE_MISMATCH")
        for alias, original in self.root_identity_map:
            validate_stable_id(original)
            if not original.endswith(alias.split(".", 1)[1]):
                raise ValueError("FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_ORIGINAL_ROOT_ALIAS_MISMATCH")
        self.operands()

    def operands(self) -> tuple[FiniteResponseLawObservedPanel, Features]:
        arrays = {}
        for key, text in self.arrays_base64:
            shape, dtype = ARRAYS[key]
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if len(text) != 4 * ((size + 2) // 3):
                raise ValueError("FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_ARRAY_SIZE_INVALID")
            raw = base64.b64decode(text, validate=True)
            if len(raw) != size or base64.b64encode(raw).decode() != text:
                raise ValueError("FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_ARRAY_ENCODING_INVALID")
            a = np.frombuffer(raw, dtype=dtype).reshape(shape)
            if dtype == "|u1":
                if np.any(a > 1):
                    raise ValueError("FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_MASK_NOT_BOOLEAN")
                a = a.astype(bool)
            arrays[key] = a
        panel = FiniteResponseLawObservedPanel(
            arrays["y"],
            arrays["observed"],
            arrays["valid"],
            arrays["parent_work"],
            arrays["hold_observed"],
            ROOTS,
        )
        features = Features(arrays["x"], arrays["z"], ROOTS)
        if not panel.observed[:16].all() or not panel.valid[:16].all():
            raise ValueError("FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_SUPPLEMENTAL_DEVELOPMENT_COMPLETE_TWO_FUTURE_PANEL_REQUIRED")
        return panel, features


def check_informative_composition_parent(parent, *, expected_plan_sha256: str) -> dict[str, object]:
    if parent.grant.parent.source_schema != FiniteResponseLawInformativeCompositionParent.SCHEMA:
        raise ValueError("FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_TYPED_RETAINED_PARENT_REQUIRED")
    record = decode_canonical_bytes(
        parent.raw, FiniteResponseLawInformativeCompositionParent, maximum_bytes=8 * 1024**2
    )
    if (
        record.science.plan_sha256 != expected_plan_sha256
        or tuple(sorted(root for _, root in record.root_identity_map))
        != parent.custody.original_root_ids
    ):
        raise ValueError("FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_PARENT_PLAN_OR_ROOTS_MISMATCH")
    return {
        "status": "FINITE_RESPONSE_LAW_RETAINED_INFORMATIVE_COMPOSITION_INPUT_READY",
        "binding_selected": "finite_response_law.development.develop_fold+summarize",
        "consumer_selected": True,
        "parent_custody_authenticated": True,
        "target_reveal_authorized": True,
        "target_analysis_authorized": True,
        "parent_sha256": parent.grant.parent.physical_sha256,
        "independent_units": 48,
        "complete_decision_roots": 16,
        "outer_folds": 4,
        "inner_folds": 3,
        "missing_port_keys": [],
    }


def analyze_informative_composition_parent(parent: FiniteResponseLawInformativeCompositionParent):
    """Retained scientific owner; callers must authenticate custody/analysis first."""
    panel, features = parent.operands()
    target = targets(panel)
    requests = development_requests(parent.science)
    oracle = evaluate_oracle(
        panel,
        requests["direction"][:16],
        requests["lower"][:16],
        parent.science,
        futures=2,
    )
    indices = np.arange(48, dtype=np.int64)
    folds = [
        develop_fold(
            panel,
            features,
            target,
            indices[indices % 4 != f],
            indices[indices % 4 == f],
        )
        for f in range(4)
    ]
    return summarize(
        panel,
        features,
        target,
        folds,
        requests["direction"],
        requests["lower"],
        oracle.feasible,
    )
