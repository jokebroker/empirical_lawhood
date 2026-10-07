from __future__ import annotations

from decimal import Decimal
import hashlib


from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.metatheory import MetatheoryApplicability, MetatheoryCellDisposition, MetatheoryEvidenceCeiling, MetatheoryMethodSelection
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceGateEvidence, ScientificSourceGateKind, ScientificSourceGateSpec, ScientificSourceInvalidUnitPolicy, ScientificSourceQualificationSpec
from empirical_lawhood.planning.source_pipelines import SourcePipelineCoordinate, SourcePipelineEdgeAccounting, SourcePipelineProfile, SourcePipelineQualificationDisposition, SourcePipelineQualificationReceipt
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityRegistry,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(object_id: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=_digest(f"{object_id}:{schema}"),
    )


def _method() -> MetatheoryMethodSelection:
    return MetatheoryMethodSelection(
        selection_id="selection.source-gate.synthetic",
        capability_key="executable-source-free-property-transport.source-gate.synthetic",
        capability_version="1.0.0",
        config=_identity("config.source-gate.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-source-qualification-gate/config'),
        implementation_sha256=_digest('source-gate-synthetic-implementation'),
    )


def _registry(*, implementation_sha256: str | None = None) -> CapabilityRegistry:
    method = _method()
    manifest = CapabilityManifest(
        capability_key=method.capability_key,
        capability_version=method.capability_version,
        kind=CapabilityKind.NUMERICAL_QUALIFIER,
        config_schema=method.config.object_schema,
        config_schema_sha256=_digest(method.config.object_schema),
        input_schema_ids=(ScientificSourceQualificationSpec.SCHEMA,),
        output_schema_ids=(ScientificSourceGateEvidence.SCHEMA,),
        permissions=(),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resource_ceiling=ResourceBudget(
            cpu_cores=1,
            memory_bytes=1024,
            gpu_devices=0,
            wall_time_seconds=1,
            source_scan_bytes=0,
            output_bytes=1024,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython",
        requires_clean_commit=False,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=("conformance.source-gate.synthetic",),
        implementation_sha256=implementation_sha256 or method.implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="registry.source-gate.synthetic", capabilities=(manifest,)
    )


def _spec(*, one_gate_not_applicable: bool = False) -> ScientificSourceQualificationSpec:
    operand = _identity(
        "operand.source.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/source-operand'
    )
    profile = _identity("source-profile.synthetic", SourcePipelineProfile.SCHEMA)
    pipeline_receipt = _pipeline_receipt_for_profile(profile)
    gates = []
    for kind in ScientificSourceGateKind:
        not_applicable = one_gate_not_applicable and kind is ScientificSourceGateKind.PREPARATION
        gates.append(
            ScientificSourceGateSpec(
                gate_id=f"gate.source.{kind.value.lower().replace('_', '-')}",
                gate_kind=kind,
                applicability=(
                    MetatheoryApplicability.NOT_APPLICABLE
                    if not_applicable
                    else MetatheoryApplicability.REQUIRED
                ),
                evaluator=None if not_applicable else _method(),
                required_operands=() if not_applicable else (operand,),
                independent_unit_definition_id="unit-definition.complete-source-unit",
                uncertainty_operation_id="uncertainty.exact-accounting",
                falsifier_ids=(f"falsifier.{kind.value.lower().replace('_', '-')}",),
                applicability_reason_codes=("TRUTH_KNOWN_PREPARATION",) if not_applicable else (),
            )
        )
    return ScientificSourceQualificationSpec(
        spec_id='source-qualification.synthetic',
        source_pipeline_profile=profile,
        source_pipeline_qualification=ObjectIdentity.from_record(
            pipeline_receipt.receipt_id,
            pipeline_receipt,
        ),
        evidence_profile_selection=_identity(
            "evidence-profile.synthetic",
            EvidenceProfileSelection.SCHEMA,
        ),
        source_identity=_identity("source.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/source'),
        source_materialization=_identity(
            "materialization.synthetic",
            'empirical-lawhood/methods/structural-transport/synthetic-input/materialization',
        ),
        gates=tuple(sorted(gates, key=lambda value: value.gate_id)),
        physical_unit_ids=("unit.synthetic.001", "unit.synthetic.002"),
        expected_physical_unit_count=2,
        invalid_unit_policy=ScientificSourceInvalidUnitPolicy.UNEVALUABLE,
        preparation=_identity(
            "preparation.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/preparation'
        ),
        observation_operator=_identity(
            "observation.synthetic",
            'empirical-lawhood/methods/structural-transport/synthetic-input/observation',
        ),
        numerical_view=_identity(
            "view.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/numerical-view'
        ),
        causal_cutoff=_identity(
            "cutoff.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/causal-cutoff'
        ),
        maximum_ordinary_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        maximum_structural_evidence_ceiling=(MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _pipeline_receipt_for_profile(
    profile: ObjectIdentity,
) -> SourcePipelineQualificationReceipt:
    coordinate = SourcePipelineCoordinate(
        coordinate_id="coordinate.source.synthetic",
        field_id="field.source.synthetic",
        native_unit="1",
        frame_id="frame.source.synthetic",
        clock_id="clock.source.synthetic",
    )
    source_artifact = _identity(
        "artifact.source.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/artifact'
    )
    output_artifact = _identity(
        "artifact.output.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/artifact'
    )
    accounting = SourcePipelineEdgeAccounting(
        edge_id="edge.source.synthetic",
        input_artifact=source_artifact,
        output_artifact=output_artifact,
        input_row_count=2,
        output_row_count=2,
        input_invalid_row_count=0,
        output_invalid_row_count=0,
        physical_unit_count=2,
        input_physical_unit_roster_sha256=_digest("units"),
        output_physical_unit_roster_sha256=_digest("units"),
        dropped_without_disposition_count=0,
        observed_input_coordinates=(coordinate,),
        observed_output_coordinates=(coordinate,),
    )
    receipt = SourcePipelineQualificationReceipt(
        receipt_id="pipeline-receipt.synthetic",
        profile=profile,
        disposition=SourcePipelineQualificationDisposition.QUALIFIED,
        source_result=_identity(
            "source-result.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/source-result'
        ),
        edge_accounting=(accounting,),
        output_artifact=output_artifact,
        output_content_sha256=_digest("output"),
        validator_receipts=(
            _identity(
                "validator-receipt.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/validator-receipt'
            ),
        ),
        publication=_identity(
            "publication.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/publication'
        ),
        recovery=_identity("recovery.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/recovery'),
        reason_codes=(),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return receipt


def _pipeline_receipt(
    spec: ScientificSourceQualificationSpec,
) -> SourcePipelineQualificationReceipt:
    receipt = _pipeline_receipt_for_profile(spec.source_pipeline_profile)
    assert ObjectIdentity.from_record(receipt.receipt_id, receipt) == (
        spec.source_pipeline_qualification
    )
    return receipt


def _evidence(
    spec: ScientificSourceQualificationSpec,
    *,
    opposed_gate: ScientificSourceGateKind | None = None,
    unevaluable_gate: ScientificSourceGateKind | None = None,
) -> tuple[ScientificSourceGateEvidence, ...]:
    values = []
    spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
    for gate in spec.gates:
        if gate.applicability is MetatheoryApplicability.NOT_APPLICABLE:
            continue
        invalid: tuple[str, ...]
        eligible: tuple[str, ...]
        reasons: tuple[str, ...]
        if gate.gate_kind is opposed_gate:
            disposition = MetatheoryCellDisposition.OPPOSED
            invalid = ("unit.synthetic.002",)
            eligible = ("unit.synthetic.001",)
            reasons = ("DECISIVE_FALSIFIER",)
        elif gate.gate_kind is unevaluable_gate:
            disposition = MetatheoryCellDisposition.UNEVALUABLE
            invalid = ("unit.synthetic.002",)
            eligible = ("unit.synthetic.001",)
            reasons = ("UNCERTAINTY_UNRESOLVED",)
        else:
            disposition = MetatheoryCellDisposition.SUPPORTED
            invalid = ()
            eligible = spec.physical_unit_ids
            reasons = ()
        values.append(
            ScientificSourceGateEvidence(
                evidence_id=f"evidence.{gate.gate_id}",
                qualification_spec=spec_identity,
                gate_spec=ObjectIdentity.from_record(gate.gate_id, gate),
                input_operands=gate.required_operands,
                method=_method(),
                observed_physical_unit_ids=spec.physical_unit_ids,
                eligible_physical_unit_ids=eligible,
                invalid_physical_unit_ids=invalid,
                metrics=(
                    NamedDecimal(
                        value_id=f"metric.{gate.gate_id}",
                        value=Decimal("0"),
                        unit="1",
                    ),
                ),
                publication=_identity(
                    f"publication.{gate.gate_id}",
                    'empirical-lawhood/methods/structural-transport/synthetic-input/publication',
                ),
                recovery=_identity(
                    f"recovery.{gate.gate_id}",
                    'empirical-lawhood/methods/structural-transport/synthetic-input/recovery',
                ),
                evidence_links=(
                    _identity(
                        f"link.{gate.gate_id}", 'empirical-lawhood/methods/structural-transport/synthetic-input/evidence-link'
                    ),
                ),
                disposition=disposition,
                reason_codes=reasons,
            )
        )
    return tuple(values)
