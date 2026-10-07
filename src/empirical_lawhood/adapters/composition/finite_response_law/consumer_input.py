# SPDX-License-Identifier: MPL-2.0

"""Typed finite-stage operands and retained builders for development preflight.

The package supplies no authority. Every consumed artifact must occur in a
separately authenticated parent inventory. No native or analysis task is entered.
"""

import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.composition.response_parent_reader import AuthenticatedResponseParent
from empirical_lawhood.adapters.methods.finite_response_law.completion.contracts import FiniteResponseLawAssignedRetainedCompletionConfig
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawControlConfig
from empirical_lawhood.adapters.methods.finite_response_law.method_records import FiniteResponseLawAssignedCalibrationMethodConfig
from empirical_lawhood.adapters.methods.finite_response_law.preparation_screen_results import FiniteResponseLawPreparationPolicyScreenConfig
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedEvaluationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_retention import FiniteResponseLawEvaluationRetention
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
)

from .authoring import build_native_authoring
from .exposure import native_seed_ids
from .method_authoring import build_method_authoring
from .preparation_policy_authoring import build_preparation_policy_authoring


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAdditionalParent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/finite-response-law-additional-parent'
    source_root: str
    parent_manifest: str
    custody: str
    reveal_record: str
    analysis_record: str

    def __post_init__(self) -> None:
        if any(
            not Path(getattr(self, item.name)).is_absolute() for item in fields(self)
        ):
            raise ValueError("FINITE_RESPONSE_LAW_ADDITIONAL_PARENT_ABSOLUTE_LOCATORS_REQUIRED")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAuthoringInput(CanonicalRecord):
    "Explicit scientific operands; original and current exposure stay distinct.\n\n    Assigned calibration retains the prior census declared *before* its native calibration parent;\n    the CLI also checks the full current census, where that parent is exposed.\n    This never makes a calibration root fresh again or authorizes reacquisition.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/finite-response-law-authoring-input'
    stage: str
    source: (
        FiniteResponseLawNativeConfig
        | FiniteResponseLawAssignedCalibrationConfig
        | FiniteResponseLawAssignedEvaluationConfig
        | FiniteResponseLawPreparationPolicyNativeConfig
    )
    primary_parent_sha256: str
    original_prior_unit_ids: tuple[str, ...]
    original_prior_seed_ids: tuple[str, ...]
    method: FiniteResponseLawAssignedCalibrationMethodConfig | None = None
    control: FiniteResponseLawControlConfig | None = None
    retention: FiniteResponseLawEvaluationRetention | None = None
    screen: FiniteResponseLawPreparationPolicyScreenConfig | None = None
    prospective_eligibility: ObjectIdentity | None = None
    runtime_context: ObjectIdentity | None = None
    completion: FiniteResponseLawAssignedRetainedCompletionConfig | None = None
    additional_parents: tuple[FiniteResponseLawAdditionalParent, ...] = ()
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_sha256

        validate_sha256(self.primary_parent_sha256)
        for name in ("original_prior_unit_ids", "original_prior_seed_ids"):
            require_sorted_unique_strings(
                getattr(self, name), field_name=name, allow_empty=False
            )
        if (
            self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
            or len(self.additional_parents) > 8
        ):
            raise ValueError("FINITE_RESPONSE_LAW_CONSUMER_ROLE_OR_PARENT_LIMIT")
        expected = {
            "supplemental-development": (
                FiniteResponseLawNativeConfig,
                "supplemental-development",
                False,
                False,
                False,
                False,
            ),
            "calibration-method": (
                FiniteResponseLawAssignedCalibrationConfig,
                "calibration",
                True,
                False,
                False,
                False,
            ),
            "prospective-evaluation": (
                FiniteResponseLawAssignedEvaluationConfig,
                "prospective-evaluation",
                False,
                True,
                False,
                False,
            ),
            "prospective-continuation": (
                FiniteResponseLawAssignedEvaluationConfig,
                "prospective-evaluation",
                False,
                True,
                True,
                False,
            ),
            "preparation-screening": (
                FiniteResponseLawPreparationPolicyNativeConfig,
                "preparation-screening",
                False,
                False,
                False,
                True,
            ),
        }.get(self.stage)
        actual = (
            type(self.source),
            self.source.stage,
            self.method is not None,
            self.control is not None,
            self.retention is not None,
            self.screen is not None,
        )
        if actual != expected or (self.stage == "preparation-screening") != (
            self.prospective_eligibility is not None
        ):
            raise ValueError("FINITE_RESPONSE_LAW_CONSUMER_STAGE_OPERANDS_MISMATCH")
        if self.method is not None and self.method.native_source != self.source:
            raise ValueError("FINITE_RESPONSE_LAW_METHOD_NATIVE_PARENT_MISMATCH")
        if self.control is not None and self.control.source != self.source:
            raise ValueError("FINITE_RESPONSE_LAW_CONTROL_SOURCE_MISMATCH")
        if self.retention is not None and self.retention.source != self.source:
            raise ValueError("FINITE_RESPONSE_LAW_CONTINUATION_SOURCE_MISMATCH")
        if (
            self.screen is not None
            and self.screen.projection.native_spec != self.source
        ):
            raise ValueError("FINITE_RESPONSE_LAW_SCREEN_SOURCE_MISMATCH")
        if self.completion is not None and (
            self.stage != "prospective-continuation"
            or self.completion.native_source != self.source
            or self.completion.control != self.control
            or self.completion.retention != self.retention
        ):
            raise ValueError("FINITE_RESPONSE_LAW_NONACQUIRING_COMPLETION_LINEAGE_MISMATCH")


def artifact_operands(value: object) -> tuple[ArtifactIdentity, ...]:
    """Enumerate scientific artifact operands, never executable metadata hashes."""
    found: dict[str, ArtifactIdentity] = {}

    def visit(item: object) -> None:
        if isinstance(item, ArtifactIdentity):
            if item.artifact_id in found and found[item.artifact_id] != item:
                raise ValueError("FINITE_RESPONSE_LAW_CONFLICTING_ARTIFACT_IDENTITY")
            found[item.artifact_id] = item
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found[key] for key in sorted(found))


def build_consumer_authoring(
    packet: FiniteResponseLawAuthoringInput,
    *,
    implementation_sha256: str,
    exposure: ObjectIdentity,
    current_units: tuple[str, ...],
    current_seeds: tuple[str, ...],
):
    """Use the retained scientific owners, preserving pre-acquisition cutoffs."""
    if not set(packet.original_prior_unit_ids) <= set(current_units) or not set(
        packet.original_prior_seed_ids
    ) <= set(current_seeds):
        raise ValueError("FINITE_RESPONSE_LAW_CURRENT_CENSUS_OMITS_ORIGINAL_PRIOR")
    if packet.completion is not None:
        from .completion_authoring import build_completion_authoring

        return build_completion_authoring(
            config=packet.completion,
            implementation_sha256=implementation_sha256,
            exposure=exposure,
            exposed_unit_ids=current_units,
            exposed_seed_ids=current_seeds,
        )
    if packet.stage == "calibration-method":
        assert packet.method is not None
        units = {root.physical_unit_id for root in packet.source.roots}
        seeds = set(native_seed_ids(packet.source))
        if not units <= set(current_units) or not seeds <= set(current_seeds):
            raise ValueError("FINITE_RESPONSE_LAW_CALIBRATION_METHOD_NATIVE_PARENT_NOT_IN_CURRENT_CENSUS")
        return build_method_authoring(
            config=packet.method,
            implementation_sha256=implementation_sha256,
            exposure=exposure,
            exposed_unit_ids=packet.original_prior_unit_ids,
            exposed_seed_ids=packet.original_prior_seed_ids,
        )
    if packet.stage == "preparation-screening":
        assert packet.screen is not None and packet.prospective_eligibility is not None
        if not {root.physical_unit_id for root in packet.source.roots} <= set(
            current_units
        ):
            raise ValueError("FINITE_RESPONSE_LAW_PREPARATION_SCREENING_RETAINED_PREFIX_CENSUS_INCOMPLETE")
        return build_preparation_policy_authoring(
            source=packet.source,
            screen=packet.screen,
            implementation_sha256=implementation_sha256,
            prospective_eligibility=packet.prospective_eligibility,
        )
    return build_native_authoring(
        source=packet.source,
        implementation_sha256=implementation_sha256,
        exposure=exposure,
        exposed_unit_ids=current_units,
        exposed_seed_ids=current_seeds,
        new_seed_ids=()
        if packet.retention is not None
        else native_seed_ids(packet.source),
        control=packet.control,
        retention=packet.retention,
    )


def validate_parent_join(
    packet: FiniteResponseLawAuthoringInput, parents: tuple[AuthenticatedResponseParent, ...]
) -> None:
    primary = parents[0]
    if packet.primary_parent_sha256 != primary.grant.parent.physical_sha256:
        raise ValueError("FINITE_RESPONSE_LAW_PRIMARY_PARENT_MISMATCH")
    from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation
    from empirical_lawhood.adapters.methods.finite_response_law.method_records import FiniteResponseLawAssignedQualificationReport
    from empirical_lawhood.adapters.methods.finite_response_law.preparation_screen_results import FiniteResponseLawRetainedProspectiveCloseoutReference
    from empirical_lawhood.adapters.methods.prepared_response.qualification import PreparedResponseSourceQualificationEvaluation
    from empirical_lawhood.kernel.decoding import decode_canonical_bytes

    parent_type = {
        "supplemental-development": PreparedResponseSourceQualificationEvaluation,
        "calibration-method": FiniteResponseLawAssignedCalibrationNativeEvaluation,
        "prospective-evaluation": FiniteResponseLawAssignedQualificationReport,
        "preparation-screening": FiniteResponseLawRetainedProspectiveCloseoutReference,
    }.get(packet.stage)
    if parent_type is not None:
        if primary.grant.parent.source_schema != parent_type.SCHEMA:
            raise ValueError("FINITE_RESPONSE_LAW_TARGET_TYPED_STAGE_PARENT_REQUIRED")
        decoded = decode_canonical_bytes(
            primary.raw, parent_type, maximum_bytes=8 * 1024**2
        )
        if packet.stage == "prospective-evaluation" and not decoded.eligible_for_prospective_evaluation:
            raise ValueError("FINITE_RESPONSE_LAW_PROSPECTIVE_EVALUATION_QUALIFIED_PARENT_REQUIRED")
        if packet.stage == "preparation-screening" and (
            packet.prospective_eligibility is None
            or packet.prospective_eligibility.object_schema != decoded.SCHEMA
            or packet.prospective_eligibility.object_fingerprint != decoded.fingerprint()
        ):
            raise ValueError("FINITE_RESPONSE_LAW_PREPARATION_SCREENING_ELIGIBILITY_IDENTITY_MISMATCH")
    allowed = {
        value
        for parent in parents
        for value in (
            *parent.custody.artifact_sha256s,
            *parent.custody.receipt_sha256s,
        )
    }
    if any(artifact.sha256 not in allowed for artifact in artifact_operands(packet)):
        raise ValueError("FINITE_RESPONSE_LAW_CONSUMED_ARTIFACT_CUSTODY_INCOMPLETE")
    primary_operand = (
        packet.method.native_evaluation
        if packet.method is not None
        else packet.retention.donor_closeout
        if packet.retention is not None
        else packet.control.qualification
        if packet.control is not None
        else packet.screen.prospective_closeout
        if packet.screen is not None
        else None
    )
    if primary_operand is not None and (
        primary_operand.sha256 != packet.primary_parent_sha256
        or primary_operand.payload_schema != primary.grant.parent.source_schema
    ):
        raise ValueError("FINITE_RESPONSE_LAW_PARENT_STAGE_SCHEMA_OR_IDENTITY_MISMATCH")
    retained_roots = {
        root for parent in parents for root in parent.custody.original_root_ids
    }
    if packet.stage in ("supplemental-development", "calibration-method", "prospective-continuation", "preparation-screening"):
        for root in packet.source.roots:
            match = re.match(r"^(.+?\.r[0-9]{3})(?:\.|$)", root.root_id)
            if match is None or match.group(1) not in retained_roots:
                raise ValueError("FINITE_RESPONSE_LAW_RETAINED_ORIGINAL_ROOT_CUSTODY_MISMATCH")


__all__ = [
    'FiniteResponseLawAdditionalParent',
    'FiniteResponseLawAuthoringInput',
    "artifact_operands",
    "build_consumer_authoring",
    "validate_parent_join",
]
