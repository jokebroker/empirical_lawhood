"""Exact C scientific authority required before a D native preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from empirical_lawhood.kernel.evidence import EvidenceRung
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ScientificStatus

from .nomination_records import RegimeNominationPackage
from .science import METHOD, regime_system

if TYPE_CHECKING:
    from .law_terminal import RegimeJointLawResult
    from .qualification_pipeline import RegimeQualificationPackage


def require_scientific_release(
    nomination: RegimeNominationPackage,
    qualification: RegimeQualificationPackage,
    report: RegimeJointLawResult,
) -> None:
    """The method reduction cannot authorize source effects by itself."""
    system = regime_system()
    law = report.qualification
    if (
        not qualification.release_D
        or nomination.selected_route is None
        or qualification.nomination
        != ObjectIdentity.from_record(nomination.package_id, nomination)
        or law.dataset_or_projection
        != ObjectIdentity.from_record(qualification.package_id, qualification)
        or report.payload.qualification != law.dataset_or_projection
        or report.payload.route != nomination.selected_route
        or law.system_id != system.system_id
        or law.world_id != system.world.world_id
        or law.relation != system.relation
        or law.method_key != METHOD
        or law.scientific_status is not ScientificStatus.SUPPORTED
        or law.highest_supported_rung is not EvidenceRung.LOCAL_LAW
        or law.response_law is None
    ):
        raise ValueError("D requires the exact supported sole-owner C law and preparation")
