"""Source-bound validity guards for the FreeGSNKE linear evolutive view."""

from __future__ import annotations

from enum import StrEnum
import math
from typing import Protocol


FREEGSNKE_LINEARIZATION_DOMAIN_TOLERANCE_PIXELS = 3


class FreeGsnkeLinearizationDomainStatus(StrEnum):
    READY = "READY"
    POLICY_MISMATCH = "POLICY_MISMATCH"
    DEPARTED = "DEPARTED"


class FreeGsnkeLinearizationDomainInspector(Protocol):
    tolerance: int

    def check_Myy(self, normalized_current: object) -> object: ...


def freegsnke_linearization_domain_status(
    *,
    inspector: FreeGsnkeLinearizationDomainInspector,
    normalized_current: object,
) -> FreeGsnkeLinearizationDomainStatus:
    """Fail closed when the source linearization leaves its exact pixel domain."""

    if inspector.tolerance != FREEGSNKE_LINEARIZATION_DOMAIN_TOLERANCE_PIXELS:
        return FreeGsnkeLinearizationDomainStatus.POLICY_MISMATCH
    if bool(inspector.check_Myy(normalized_current)):
        return FreeGsnkeLinearizationDomainStatus.DEPARTED
    return FreeGsnkeLinearizationDomainStatus.READY


def freegsnke_dynamic_gs_converged(
    *,
    requested_tolerance: float,
    achieved_tolerance: float,
) -> bool:
    """Expose upstream suppressed GS nonconvergence as an adverse outcome."""

    return bool(
        math.isfinite(requested_tolerance)
        and requested_tolerance > 0
        and math.isfinite(achieved_tolerance)
        and achieved_tolerance <= requested_tolerance
    )


__all__ = [
    "FREEGSNKE_LINEARIZATION_DOMAIN_TOLERANCE_PIXELS",
    "FreeGsnkeLinearizationDomainInspector",
    "FreeGsnkeLinearizationDomainStatus",
    "freegsnke_dynamic_gs_converged",
    "freegsnke_linearization_domain_status",
]
