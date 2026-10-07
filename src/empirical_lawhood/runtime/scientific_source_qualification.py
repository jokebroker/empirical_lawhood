"""Identity-exact noncompensating scientific source qualification service."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryApplicability, MetatheoryCellDisposition
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceGateEvidence, ScientificSourceGateResult, ScientificSourceQualificationResult, ScientificSourceQualificationSpec, ScientificSourceQualificationStopKind, ScientificSourceQualificationStop
from empirical_lawhood.planning.source_pipelines import SourcePipelineQualificationDisposition, SourcePipelineQualificationReceipt
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


class ScientificSourceQualificationError(ValueError):
    """Exact scientific qualification inputs are incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class ScientificSourceQualificationService:
    capability_registry: CapabilityRegistry

    def stop(
        self,
        *,
        spec: ScientificSourceQualificationSpec,
        stop_kind: ScientificSourceQualificationStopKind,
        reason_codes: tuple[str, ...],
    ) -> ScientificSourceQualificationStop:
        return ScientificSourceQualificationStop(
            stop_id=f"stop.{spec.spec_id}.{stop_kind.value.lower().replace('_', '-')}",
            qualification_spec=ObjectIdentity.from_record(spec.spec_id, spec),
            stop_kind=stop_kind,
            reason_codes=reason_codes,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )

    def finalize(
        self,
        *,
        spec: ScientificSourceQualificationSpec,
        pipeline_receipt: SourcePipelineQualificationReceipt,
        gate_evidence: tuple[ScientificSourceGateEvidence, ...],
    ) -> ScientificSourceQualificationResult:
        spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        receipt_identity = ObjectIdentity.from_record(pipeline_receipt.receipt_id, pipeline_receipt)
        if receipt_identity != spec.source_pipeline_qualification:
            raise ScientificSourceQualificationError("SOURCE_PIPELINE_RECEIPT_IDENTITY_MISMATCH")
        if (
            pipeline_receipt.profile != spec.source_pipeline_profile
            or pipeline_receipt.disposition
            is not SourcePipelineQualificationDisposition.QUALIFIED
            or pipeline_receipt.publication is None
            or pipeline_receipt.recovery is None
        ):
            raise ScientificSourceQualificationError("SOURCE_PIPELINE_NOT_QUALIFIED_AND_RECOVERED")

        required_gates = tuple(
            gate for gate in spec.gates if gate.applicability is MetatheoryApplicability.REQUIRED
        )
        if len(gate_evidence) != len(required_gates):
            raise ScientificSourceQualificationError("GATE_EVIDENCE_ROSTER_INCOMPLETE")

        results: list[ScientificSourceGateResult] = []
        evidence_index = 0
        invalid_units: set[str] = set()
        eligible_units = set(spec.physical_unit_ids)
        for gate in spec.gates:
            gate_identity = ObjectIdentity.from_record(gate.gate_id, gate)
            if gate.applicability is MetatheoryApplicability.NOT_APPLICABLE:
                results.append(
                    ScientificSourceGateResult(
                        result_id=f"result.{gate.gate_id}",
                        qualification_spec=spec_identity,
                        gate_spec=gate_identity,
                        evidence=None,
                        disposition=MetatheoryCellDisposition.NOT_APPLICABLE,
                        decisive_witness_ids=(),
                        reason_codes=gate.applicability_reason_codes,
                    )
                )
                continue

            evidence = gate_evidence[evidence_index]
            evidence_index += 1
            if (
                evidence.qualification_spec != spec_identity
                or evidence.gate_spec != gate_identity
                or evidence.method != gate.evaluator
                or evidence.input_operands != gate.required_operands
                or evidence.observed_physical_unit_ids != spec.physical_unit_ids
            ):
                raise ScientificSourceQualificationError("GATE_EVIDENCE_IDENTITY_MISMATCH")
            assert gate.evaluator is not None
            try:
                manifest = self.capability_registry.resolve(
                    gate.evaluator.capability_key,
                    gate.evaluator.capability_version,
                )
            except KeyError as error:
                raise ScientificSourceQualificationError("GATE_EVALUATOR_UNREGISTERED") from error
            if (
                manifest.implementation_sha256 != gate.evaluator.implementation_sha256
                or manifest.config_schema != gate.evaluator.config.object_schema
                or ScientificSourceGateEvidence.SCHEMA not in manifest.output_schema_ids
            ):
                raise ScientificSourceQualificationError("GATE_EVALUATOR_BINDING_DRIFT")
            invalid_units.update(evidence.invalid_physical_unit_ids)
            eligible_units.intersection_update(evidence.eligible_physical_unit_ids)
            results.append(
                ScientificSourceGateResult(
                    result_id=f"result.{gate.gate_id}",
                    qualification_spec=spec_identity,
                    gate_spec=gate_identity,
                    evidence=ObjectIdentity.from_record(evidence.evidence_id, evidence),
                    disposition=evidence.disposition,
                    decisive_witness_ids=tuple(
                        sorted(value.object_id for value in evidence.evidence_links)
                    ),
                    reason_codes=evidence.reason_codes,
                )
            )

        applicable = tuple(
            value.disposition
            for value in results
            if value.disposition is not MetatheoryCellDisposition.NOT_APPLICABLE
        )
        if MetatheoryCellDisposition.OPPOSED in applicable:
            aggregate = MetatheoryAggregateDisposition.OPPOSED
        elif MetatheoryCellDisposition.UNEVALUABLE in applicable:
            aggregate = MetatheoryAggregateDisposition.UNEVALUABLE
        elif applicable and all(
            value is MetatheoryCellDisposition.SUPPORTED for value in applicable
        ):
            aggregate = MetatheoryAggregateDisposition.SUPPORTED
        else:
            raise ScientificSourceQualificationError("NO_APPLICABLE_SOURCE_GATES")

        link_by_id = {
            value.object_id: value
            for value in (
                pipeline_receipt.publication,
                pipeline_receipt.recovery,
                *(link for evidence in gate_evidence for link in evidence.evidence_links),
            )
            if value is not None
        }
        return ScientificSourceQualificationResult(
            result_id=f"result.{spec.spec_id}",
            qualification_spec=spec_identity,
            source_pipeline_qualification=receipt_identity,
            gate_evidence=gate_evidence,
            gate_results=tuple(sorted(results, key=lambda value: value.result_id)),
            complete_physical_unit_count=len(spec.physical_unit_ids),
            invalid_physical_unit_count=len(invalid_units),
            eligible_physical_unit_count=len(eligible_units - invalid_units),
            disposition=aggregate,
            maximum_ordinary_evidence_ceiling=spec.maximum_ordinary_evidence_ceiling,
            maximum_structural_evidence_ceiling=spec.maximum_structural_evidence_ceiling,
            evidence_links=tuple(link_by_id[key] for key in sorted(link_by_id)),
            outcome_access=spec.outcome_access,
            visibility_ceiling=spec.visibility_ceiling,
        )


__all__ = [
    "ScientificSourceQualificationError",
    "ScientificSourceQualificationService",
]
