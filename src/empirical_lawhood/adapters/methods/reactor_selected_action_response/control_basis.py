"""Carry the exact sole-owner selected finite-bound law into the installed atlas/model-set owners."""

from functools import lru_cache
from empirical_lawhood.adapters.control.local_law_basis import qualified_local_basis

from empirical_lawhood.adapters.methods.contracts import (
    LawQualificationBatch,
)
from empirical_lawhood.kernel.models import ModelSetSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.geometry import (
    BatchAtlasAssemblyResult,
)

from .config import PREFIX
from .law_terminal import ClassicalLaw
from .science import classical_system


@lru_cache(maxsize=4)
def classical_control_basis(
    report: ClassicalLaw,
) -> tuple[LawQualificationBatch, BatchAtlasAssemblyResult, ModelSetSpec]:
    system = classical_system()
    qualification = report.qualification
    law = qualification.response_law
    if (
        qualification.scientific_status is not ScientificStatus.SUPPORTED
        or law is None
        or report.family.system != ObjectIdentity.from_record(system.system_id, system)
        or report.payload.route is None
    ):
        raise ValueError("Admission basis requires the exact supported selected joint reactor law")
    axes = report.family.axis_map
    if len(axes.bindings) != 1 or len(axes.qualification_view_ids) != 2:
        raise ValueError("Admission basis changed the frozen member and two-view roster")
    axis = axes.bindings[0]
    stem = f"{PREFIX}.qualified-parent"
    return qualified_local_basis(
        system=system,
        qualification=qualification,
        axis=axis,
        stem=stem,
        scope_description="One unchanged selected-route selected finite-bound law with measured zero-feed reference.",
        nontransport_description="One selected scalar feed word under an unchanged absolute jacket denominator.",
    )
