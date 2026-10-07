"""Preserved constructor geometry and complete eight-root Q gate.

This is pure readout arithmetic. Current authoring and provider reconstruction
must authenticate its operands and enforce the result before E native contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from empirical_lawhood.adapters.methods.preparation_applicability.operands import (
    ConstructorMeasurement,
)
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import effective_seed_ids

if TYPE_CHECKING:
    from empirical_lawhood.adapters.methods.finite_response_law.original_f import (
        OriginalFiniteResponseLaw,
    )


@dataclass(frozen=True, slots=True)
class ConstructorQualification:
    """Internal reduction result; current canonical reports remain separate."""

    complete: bool
    qualified: bool
    crossings: tuple[bool, ...]

    @property
    def disposition(self) -> str:
        if not self.complete:
            return "UNEVALUABLE"
        return "QUALIFIED" if self.qualified else "CONSTRUCTOR_NOT_QUALIFIED"


def constructor_crossing(
    measured: ConstructorMeasurement, lower: OriginalFiniteResponseLaw
) -> bool:
    if not measured.complete:
        return False
    z = np.asarray(measured.handoff, dtype=float).reshape(3, 24, 2)
    face = float(lower.center[17] + 6 * lower.scale[17])
    # Every N coordinate is supported in both views; each face margin must
    # strictly exceed its own numerical-view disagreement.
    return bool(
        (z[0, 17] > face).all()
        and all((abs(lower.normalize(z[1, :, v][None])) <= 6).all() for v in (0, 1))
        and min(z[0, 17] - face) > abs(z[0, 17, 0] - z[0, 17, 1])
        and min(face - z[1, 17]) > abs(z[1, 17, 0] - z[1, 17, 1])
    )


def qualify_constructor(
    measured: tuple[ConstructorMeasurement, ...],
    root_ids: tuple[str, ...],
    lower: OriginalFiniteResponseLaw,
) -> ConstructorQualification:
    """Require all eight complete Q units, at least six crossings, and numerics.

    No other Q maxima become eligibility gates. Root ordering binds the current
    declared census; published success counts never determine qualification.
    """
    if (
        len(root_ids) != 8
        or len(set(root_ids)) != 8
        or tuple(row.root_id for row in measured) != root_ids
    ):
        raise ValueError("constructor readout changes the complete assigned Q census")
    if not all(row.complete for row in measured):
        return ConstructorQualification(False, False, ())
    maxima = np.asarray([row.maxima for row in measured], dtype=float).reshape(8, 3, 7)
    crossings = tuple(constructor_crossing(row, lower) for row in measured)
    qualified = bool(sum(crossings) >= 6 and (maxima[:, :, 1] <= 1).all())
    return ConstructorQualification(True, qualified, crossings)


def require_qualified_constructor(stage, report) -> None:
    """Check the authenticated Q result before any E simulator contact.

    Callers authenticate exact current report bytes and receipt custody before
    passing this record. Fresh labels never substitute for disjoint allocations.
    """
    if (
        stage.phase != "E"
        or report.phase != "Q"
        or not report.complete
        or not report.constructor_qualified
        or report.disposition != "QUALIFIED"
        or report.lower_sha256 != stage.upstream[0].artifact.sha256
        or report.design_sha256 != stage.design.fingerprint()
        or report.source_sha256 != stage.source.object_fingerprint
        or set(report.root_ids) & set(stage.root_ids)
        or set(report.effective_seed_ids) & set(effective_seed_ids(stage.allocation))
    ):
        raise ValueError("E requires an authenticated complete qualified disjoint current Q result")
