"""Statically registered anomaly detectors over typed diagnostic measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.planning.discovery import (
    DiagnosticMeasureKind,
    ExplorationDiagnosticProjection,
    ExplorationEvidenceView,
)
from empirical_lawhood.planning.exploration import AnomalyKind, AnomalySignal


@dataclass(frozen=True, slots=True)
class DetectorDefinition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/exploration/detector-definition'

    detector_key: str
    detector_version: str
    measure_kind: DiagnosticMeasureKind
    anomaly_kind: AnomalyKind
    rationale: str

    def __post_init__(self) -> None:
        validate_stable_id(self.detector_key, field_name="detector_key")
        validate_semantic_version(self.detector_version)
        validate_nonempty(self.rationale, field_name="rationale")


@dataclass(frozen=True, slots=True)
class DetectorRegistry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/exploration/detector-registry'

    registry_id: str
    detectors: tuple[DetectorDefinition, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        require_sorted_unique_ids(
            self.detectors,
            attribute="detector_key",
            field_name="detectors",
        )
        if {detector.measure_kind for detector in self.detectors} != set(DiagnosticMeasureKind):
            raise ValueError("anomaly registry must cover the complete diagnostic measure family")

    def detect(
        self,
        snapshot: EvidenceSnapshot,
        projection: ExplorationDiagnosticProjection,
        view: ExplorationEvidenceView,
    ) -> tuple[AnomalySignal, ...]:
        expected_snapshot = ObjectIdentity.from_record(snapshot.snapshot_id, snapshot)
        if view.snapshot != expected_snapshot:
            raise ValueError("anomaly detector view binds another EvidenceSnapshot")
        if view.projection != ObjectIdentity.from_record(projection.projection_id, projection):
            raise ValueError("anomaly detector view binds another diagnostic projection")
        if (
            view.system_id != projection.system_id
            or view.relation != projection.relation
            or view.independent_unit_id != projection.independent_unit_id
            or view.grouping_clock_ids != projection.grouping_clock_ids
            or view.available_role_ids != projection.available_role_ids
            or view.available_upstream_schema_ids != projection.available_upstream_schema_ids
            or view.upstream_objects != projection.upstream_objects
            or view.measures != projection.measures
            or view.outcome_access is not projection.outcome_access
            or view.visibility_ceiling is not projection.visibility_ceiling
        ):
            raise ValueError("bound evidence view differs from its immutable projection")
        if not set(view.available_projection_ids).issubset(snapshot.projection_ids):
            raise ValueError("anomaly detector view requests an unavailable projection")
        if view.parent_visibility_ceiling is not snapshot.visibility_ceiling:
            raise ValueError("anomaly detector view lowers snapshot visibility")
        definitions = {definition.measure_kind: definition for definition in self.detectors}
        signals = []
        for measure in view.measures:
            if not measure.triggered:
                continue
            definition = definitions[measure.kind]
            signals.append(
                AnomalySignal(
                    signal_id=(
                        f"signal.{view.view_id}.{definition.detector_key}.{measure.measure_id}"
                    ),
                    snapshot_id=snapshot.snapshot_id,
                    detector_key=definition.detector_key,
                    detector_version=definition.detector_version,
                    kind=definition.anomaly_kind,
                    relation_id=measure.relation_id,
                    affected_quantity_ids=measure.affected_quantity_ids,
                    severity=measure.normalized_excess,
                    rationale=definition.rationale,
                    outcome_access=view.outcome_access,
                    parent_visibility_ceiling=view.visibility_ceiling,
                    visibility_ceiling=view.visibility_ceiling,
                    evidence_link_ids=measure.evidence_link_ids,
                )
            )
        return tuple(sorted(signals, key=lambda signal: signal.signal_id))


def bind_evidence_view(
    snapshot: EvidenceSnapshot,
    projection: ExplorationDiagnosticProjection,
) -> ExplorationEvidenceView:
    """Bind a verified materializable projection into one read-only snapshot view."""

    artifact_matches = tuple(
        artifact
        for artifact in snapshot.artifacts
        if artifact.payload_schema == projection.SCHEMA
        and artifact.sha256 == projection.fingerprint()
    )
    if len(artifact_matches) != 1:
        raise ValueError("snapshot must contain exactly one matching diagnostic projection")
    if projection.visibility_ceiling is not snapshot.visibility_ceiling:
        raise ValueError("diagnostic projection and snapshot visibility differ")
    if projection.outcome_access is not snapshot.outcome_access:
        raise ValueError("diagnostic projection and snapshot outcome access differ")
    if not set(projection.upstream_objects).issubset(snapshot.parent_objects):
        raise ValueError("diagnostic projection names upstream objects outside its snapshot")
    return ExplorationEvidenceView(
        view_id=f"view.{snapshot.snapshot_id}.{projection.projection_id}",
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        projection=ObjectIdentity.from_record(projection.projection_id, projection),
        system_id=projection.system_id,
        relation=projection.relation,
        independent_unit_id=projection.independent_unit_id,
        grouping_clock_ids=projection.grouping_clock_ids,
        available_projection_ids=snapshot.projection_ids,
        available_role_ids=projection.available_role_ids,
        available_upstream_schema_ids=projection.available_upstream_schema_ids,
        upstream_objects=projection.upstream_objects,
        measures=projection.measures,
        outcome_access=projection.outcome_access,
        parent_visibility_ceiling=snapshot.visibility_ceiling,
        visibility_ceiling=projection.visibility_ceiling,
    )


def default_detector_registry() -> DetectorRegistry:
    """Return the fixed exploration detector family; definitions never inspect positivity."""

    rows = (
        (
            "detector.coordinate-instability",
            DiagnosticMeasureKind.COORDINATE_INSTABILITY,
            AnomalyKind.COORDINATE_FAILURE,
            "Declared coordinates rotate or change beyond their frozen recurrence tolerance.",
        ),
        (
            "detector.information-limit",
            DiagnosticMeasureKind.EFFECTIVE_INDEPENDENT_UNITS,
            AnomalyKind.INFORMATION_LIMIT,
            "The physical independent-unit count cannot resolve the declared question.",
        ),
        (
            "detector.fragile-gate",
            DiagnosticMeasureKind.GATE_MARGIN,
            AnomalyKind.ADMISSION_FRAGILITY,
            "A required gate lies inside the predeclared fragility margin.",
        ),
        (
            "detector.unexpected-invariance",
            DiagnosticMeasureKind.INVARIANCE_DEVIATION,
            AnomalyKind.UNEXPECTED_NULL,
            "A source-backed action channel is unexpectedly invariant under the declared contrast.",
        ),
        (
            "detector.model-disagreement",
            DiagnosticMeasureKind.MODEL_DISAGREEMENT,
            AnomalyKind.CONTRADICTION,
            "Frozen plausible models disagree beyond the declared structural tolerance.",
        ),
        (
            "detector.numerical-disagreement",
            DiagnosticMeasureKind.NUMERICAL_DISAGREEMENT,
            AnomalyKind.COUNTERFEIT_RANK,
            "Nested numerical views disagree beyond the declared refinement tolerance.",
        ),
        (
            "detector.observer-plant-loss",
            DiagnosticMeasureKind.OBSERVER_PLANT_LOSS,
            AnomalyKind.TRANSPORT_FAILURE,
            (
                "The plant-to-observer or observer-to-selector interface loses declared "
                "response structure."
            ),
        ),
        (
            "detector.rank-instability",
            DiagnosticMeasureKind.RANK_INSTABILITY,
            AnomalyKind.COUNTERFEIT_RANK,
            "Local response rank is unstable across declared physical or numerical views.",
        ),
        (
            "detector.residual-memory",
            DiagnosticMeasureKind.RESIDUAL_MEMORY,
            AnomalyKind.RESIDUAL_MEMORY,
            "Held-out residual dependence exceeds the frozen closure tolerance.",
        ),
        (
            "detector.support-hole",
            DiagnosticMeasureKind.SUPPORT_COVERAGE,
            AnomalyKind.SUPPORT_GAP,
            "Declared receiver/action cells lack the frozen minimum support coverage.",
        ),
        (
            "detector.wrong-action-retention",
            DiagnosticMeasureKind.WRONG_ACTION_RETENTION,
            AnomalyKind.CONTRADICTION,
            "The matched wrong-action falsifier retains too much response advantage.",
        ),
    )
    definitions = tuple(
        sorted(
            (
                DetectorDefinition(
                    detector_key=key,
                    detector_version="1.0.0",
                    measure_kind=measure,
                    anomaly_kind=anomaly,
                    rationale=rationale,
                )
                for key, measure, anomaly, rationale in rows
            ),
            key=lambda definition: definition.detector_key,
        )
    )
    return DetectorRegistry(registry_id="exploration-anomaly-detectors", detectors=definitions)
