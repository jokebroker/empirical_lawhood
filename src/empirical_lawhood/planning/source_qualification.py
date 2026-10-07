"Action-bearing source/response qualification before local-law construction.\n\nThis additive carrier binds independent units, native continuation segments\nand scientific views. It neither constructs local laws nor authorizes effects.\n"

from dataclasses import dataclass, fields
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .observation_order import ObservationPhysicalUnit


@dataclass(frozen=True, slots=True)
class SourceQualificationSegment(CanonicalRecord):
    """One native continuation, with an optional authenticated predecessor."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-qualification-segment'

    segment_id: str
    acquisition_group_id: str
    physical_independent_unit_id: str
    predecessor_segment_id: str | None
    native_clock_id: str
    start: Decimal
    end: Decimal
    view_ids: tuple[str, ...]
    action_occurrence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "segment_id",
            "acquisition_group_id",
            "physical_independent_unit_id",
            "native_clock_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.predecessor_segment_id is not None:
            validate_stable_id(self.predecessor_segment_id, field_name="predecessor_segment_id")
        for name in ("start", "end"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.end <= self.start:
            raise ValueError("qualification segment must advance its native clock")
        require_sorted_unique_strings(self.view_ids, field_name="view_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.action_occurrence_ids, field_name="action_occurrence_ids"
        )
        if self.predecessor_segment_id is None and self.start != 0:
            raise ValueError("qualification root segment must begin at its native origin")


@dataclass(frozen=True, slots=True)
class SourceQualificationView(CanonicalRecord):
    """A pure scientific projection of one unit and numerical view."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-qualification-view'

    view_id: str
    physical_independent_unit_id: str
    numerical_view_id: str
    segment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("view_id", "physical_independent_unit_id", "numerical_view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.segment_ids, field_name="segment_ids", allow_empty=False)


@dataclass(frozen=True, slots=True)
class SourceQualificationRetainedPredecessor(CanonicalRecord):
    """An exposed, externally custodied terminal; never a new source task.

    Artifact identities and the producing receipt are authenticated through the
    normal external-input ports. The source adapter must decode and verify their
    native unit/view/clock correspondence before any continuation effect.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-qualification-retained-predecessor'
    ACCESS: ClassVar[OutcomeAccess] = OutcomeAccess.DEVELOPMENT_VISIBLE
    VISIBILITY: ClassVar[VisibilityCeiling] = VisibilityCeiling.OUTCOME_VISIBLE
    segment_id: str
    physical_independent_unit_id: str
    native_clock_id: str
    end: Decimal
    view_ids: tuple[str, ...]
    artifacts: tuple[ArtifactIdentity, ...]
    task_receipt: ArtifactIdentity
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in ("segment_id", "physical_independent_unit_id", "native_clock_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_decimal(self.end, field_name="end", minimum=Decimal(0))
        if self.end <= 0:
            raise ValueError("retained predecessor must have advanced its native clock")
        require_sorted_unique_strings(self.view_ids, field_name="view_ids", allow_empty=False)
        require_sorted_unique_ids(self.artifacts, attribute="artifact_id", field_name="artifacts")
        if not 1 <= len(self.artifacts) <= 8 or any(
            type(a.size_bytes) is not int or a.size_bytes <= 0
            for a in (*self.artifacts, self.task_receipt)
        ):
            raise ValueError("retained predecessor requires bounded nonempty artifacts and receipt")
        if (
            self.task_receipt.payload_schema != 'empirical-lawhood/runtime/canonical-task-receipt'
            or self.task_receipt.media_type != "application/json"
            or self.task_receipt.artifact_id in {a.artifact_id for a in self.artifacts}
            or self.outcome_access is not self.ACCESS
            or self.visibility_ceiling is not self.VISIBILITY
        ):
            raise ValueError(
                "retained predecessor requires its distinct exposed native task receipt"
            )


@dataclass(frozen=True, slots=True)
class ProspectiveRetainedPredecessor(SourceQualificationRetainedPredecessor):
    """A receipted pre-action origin whose outcomes remain prospective."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-retained-predecessor'
    ACCESS: ClassVar[OutcomeAccess] = OutcomeAccess.OUTCOME_BLIND
    VISIBILITY: ClassVar[VisibilityCeiling] = VisibilityCeiling.PROSPECTIVE


@dataclass(frozen=True, slots=True)
class FreshSourceQualificationExperiment(CanonicalRecord):
    "Finite measurement/response qualification, with no law/admission/controller overlay."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/fresh-source-qualification-experiment'

    extension_set_id: str
    evidence_profile: ObjectIdentity
    physical_units: tuple[ObservationPhysicalUnit, ...]
    segments: tuple[SourceQualificationSegment, ...]
    views: tuple[SourceQualificationView, ...]
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    source_capability: ObjectIdentity
    projection_capability: ObjectIdentity
    evaluator_capability: ObjectIdentity
    source_config: ObjectIdentity
    projection_config: ObjectIdentity
    evaluator_config: ObjectIdentity
    evaluator_task_id: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.extension_set_id, field_name="extension_set_id")
        if self.evidence_profile.object_schema != 'empirical-lawhood/planning/evidence-profile-selection':
            raise ValueError("qualification requires its strict evidence profile identity")
        validate_stable_id(self.evaluator_task_id, field_name="evaluator_task_id")
        for name, attribute in (
            ("physical_units", "physical_independent_unit_id"),
            ("segments", "segment_id"),
            ("views", "view_id"),
        ):
            values = getattr(self, name)
            require_sorted_unique_ids(values, attribute=attribute, field_name=name)
            if not values or len(values) > 8192:
                raise ValueError(f"qualification {name} is empty or exceeds its finite bound")
        for name in ("receiver_ids", "clock_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.evidence_ceiling not in {
            EvidenceCeiling.MEASUREMENT,
            EvidenceCeiling.RESPONSE,
        }:
            raise ValueError("source qualification cannot claim local law or later evidence")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("source qualification must separate acquisition from evaluator reveal")
        for owner in (
            self.source_capability,
            self.projection_capability,
            self.evaluator_capability,
        ):
            if owner.object_schema != 'empirical-lawhood/runtime/capability-manifest':
                raise ValueError("qualification owner must be an exact capability identity")
        if (
            len({self.source_capability, self.projection_capability, self.evaluator_capability})
            != 3
        ):
            raise ValueError(
                "qualification source, projection and evaluator must be separate owners"
            )
        units = {unit.physical_independent_unit_id for unit in self.physical_units}
        segments = {segment.segment_id: segment for segment in self.segments}
        views = {view.view_id: view for view in self.views}
        retained = (
            self.retained_predecessors if isinstance(self, PredecessorBoundSourceQualificationExperiment) else ()
        )
        require_sorted_unique_ids(
            retained, attribute="segment_id", field_name="retained_predecessors"
        )
        if len(retained) > 8192:
            raise ValueError("retained predecessor roster exceeds its finite bound")
        retained_by_id = {r.segment_id: r for r in retained}
        task_ids = set(segments) | set(views) | {self.evaluator_task_id}
        if len(task_ids) != len(segments) + len(views) + 1:
            raise ValueError("qualification task identities overlap")
        if len({segment.acquisition_group_id for segment in self.segments}) != len(segments):
            raise ValueError("qualification native segments repeat an acquisition group")
        if set(retained_by_id) & task_ids:
            raise ValueError("retained predecessor cannot also be a new execution task")
        used_retained = {
            s.predecessor_segment_id
            for s in self.segments
            if s.predecessor_segment_id in retained_by_id
        }
        if used_retained != set(retained_by_id):
            raise ValueError("qualification retained predecessor is unused")
        if len({r.task_receipt.artifact_id for r in retained}) != len(retained):
            raise ValueError("qualification retained predecessor repeats a native receipt")
        for prior in retained:
            if (
                prior.physical_independent_unit_id not in units
                or prior.native_clock_id not in self.clock_ids
                or not set(prior.view_ids) <= views.keys()
                or any(
                    views[v].physical_independent_unit_id != prior.physical_independent_unit_id
                    for v in prior.view_ids
                )
            ):
                raise ValueError("retained predecessor changes unit, views or native clock")
        roots: list[str] = []
        for segment in self.segments:
            if segment.physical_independent_unit_id not in units:
                raise ValueError("qualification segment has an unknown independent unit")
            if segment.native_clock_id not in self.clock_ids:
                raise ValueError("qualification segment has an unbound native clock")
            if not set(segment.view_ids) <= views.keys():
                raise ValueError("qualification segment has an unknown view")
            if segment.predecessor_segment_id is None:
                roots.append(segment.physical_independent_unit_id)
            else:
                parent = segments.get(segment.predecessor_segment_id) or retained_by_id.get(
                    segment.predecessor_segment_id
                )
                if parent is None or (
                    parent.physical_independent_unit_id != segment.physical_independent_unit_id
                    or parent.native_clock_id != segment.native_clock_id
                    or parent.end != segment.start
                    or parent.view_ids != segment.view_ids
                ):
                    raise ValueError(
                        "qualification continuation changes unit, views or native clock"
                    )
        # Strictly increasing clocks rule out cycles and must reach one root per unit.
        retained_units = {r.physical_independent_unit_id for r in retained}
        if (
            len(roots) != len(set(roots))
            or set(roots) & retained_units
            or set(roots) | retained_units != units
        ):
            raise ValueError(
                "qualification requires exactly one source origin per independent unit"
            )
        for view in self.views:
            expected = tuple(
                segment.segment_id for segment in self.segments if view.view_id in segment.view_ids
            )
            if view.segment_ids != expected or not expected:
                raise ValueError("qualification projection drops or adds native segments")
            if any(
                segments[key].physical_independent_unit_id != view.physical_independent_unit_id
                for key in expected
            ):
                raise ValueError("qualification projection pools independent units")


@dataclass(frozen=True, slots=True)
class PredecessorBoundSourceQualificationExperiment(FreshSourceQualificationExperiment):
    "Source qualification bound to explicit exposed retained origins.\n\n    The inherited segment's predecessor ID resolves to exactly one new segment or\n    retained terminal. Retained terminals are omitted from acquisition/projection\n    task counts.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/predecessor-bound-source-qualification-experiment'
    retained_predecessors: tuple[SourceQualificationRetainedPredecessor, ...]


@dataclass(frozen=True, slots=True)
class ProspectiveRetainedSourceQualification(PredecessorBoundSourceQualificationExperiment):
    "Prospective retained-source qualification with strictly outcome-blind retained origins."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-retained-source-qualification'
    retained_predecessors: tuple[ProspectiveRetainedPredecessor, ...]


@dataclass(frozen=True, slots=True)
class QualifiedSourceUseStage(CanonicalRecord):
    """Exact outcome-blind binding in a source/preparation/use composition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/qualified-source-use-stage'
    task_id: str
    stage: str
    dependency_task_ids: tuple[str, ...]
    owner: ObjectIdentity
    config: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        require_sorted_unique_strings(self.dependency_task_ids, field_name="dependency_task_ids")
        if self.stage not in {"PREPARE", "TRANSFORM", "CONTROLLER", "FREEZE", "EVALUATE"}:
            raise ValueError("source-use binding adds an undeclared scientific stage")
        if (
            self.task_id in self.dependency_task_ids
            or self.owner.object_schema != 'empirical-lawhood/runtime/capability-manifest'
        ):
            raise ValueError("source-use binding has a self dependency or non-capability owner")


@dataclass(frozen=True, slots=True)
class QualifiedSourceUseExperiment(PredecessorBoundSourceQualificationExperiment):
    "Compose the unchanged source census with separately owned control/use.\n\n    The inherited evidence ceiling bounds the SOURCE component only. This is\n    an additive carrier, not permission to promote fresh or predecessor-bound source-qualification records.\n    The outer experiment, qualified law, controller and controller-use owners govern use.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/qualified-source-use-experiment'
    stages: tuple[QualifiedSourceUseStage, ...]

    def source_qualification(self) -> PredecessorBoundSourceQualificationExperiment:
        """Explicit compatibility map to the unchanged measurement component."""
        return PredecessorBoundSourceQualificationExperiment(
            **{
                field.name: getattr(self, field.name)
                for field in fields(PredecessorBoundSourceQualificationExperiment)
            }
        )

    def __post_init__(self) -> None:
        PredecessorBoundSourceQualificationExperiment.__post_init__(self)
        require_sorted_unique_ids(self.stages, attribute="task_id", field_name="stages")
        if not 3 <= len(self.stages) <= 8193:
            raise ValueError("source-use topology exceeds its finite bound")
        stages = {stage.task_id: stage for stage in self.stages}
        source_ids = {s.segment_id for s in self.segments}
        view_ids = {v.view_id for v in self.views}
        if not source_ids | view_ids | {self.evaluator_task_id} <= stages.keys():
            raise ValueError("source-use topology drops its measurement census")
        extra = set(stages) - source_ids - view_ids - {self.evaluator_task_id}
        imports = (
            {r.segment_id for r in self.retained_predecessors}
            if isinstance(self, ProspectiveRetainedSourceUse)
            else set()
        )
        if imports and (
            not imports <= extra
            or any(stages[k].stage != "FREEZE" or stages[k].dependency_task_ids for k in imports)
        ):
            raise ValueError("prospective retained imports must be independent freeze tasks")
        if (
            not extra
            or not any(stages[k].stage == "CONTROLLER" for k in extra)
            or not any(stages[k].stage == "EVALUATE" for k in extra)
        ):
            raise ValueError("source-use requires separately declared control and evaluation")
        for key in extra:
            if stages[key].stage not in {"CONTROLLER", "FREEZE", "EVALUATE"}:
                raise ValueError("source-use overlay cannot invent native acquisitions or views")
        for segment in self.segments:
            stage = stages[segment.segment_id]
            predecessor = {segment.predecessor_segment_id} & (source_ids | imports)
            additional = set(stage.dependency_task_ids) - predecessor
            if (
                stage.stage != "PREPARE"
                or stage.owner != self.source_capability
                or stage.config != self.source_config
                or not predecessor <= set(stage.dependency_task_ids)
                or any(k not in extra or stages[k].stage != "CONTROLLER" for k in additional)
            ):
                raise ValueError("source-use changes a native continuation or its control guard")
        for view in self.views:
            stage = stages[view.view_id]
            if (
                stage.stage != "TRANSFORM"
                or stage.owner != self.projection_capability
                or stage.config != self.projection_config
                or stage.dependency_task_ids
                != tuple(
                    sorted(
                        (
                            *view.segment_ids,
                            *(
                                r.segment_id
                                for r in self.retained_predecessors
                                if r.segment_id in imports and view.view_id in r.view_ids
                            ),
                        )
                    )
                )
            ):
                raise ValueError("source-use changes an acquisition-sharing projection")
        terminal = stages[self.evaluator_task_id]
        if (
            terminal.stage != "FREEZE"
            or terminal.owner != self.evaluator_capability
            or terminal.config != self.evaluator_config
            or not view_ids <= set(terminal.dependency_task_ids)
            or any(
                k not in extra or stages[k].stage != "FREEZE"
                for k in set(terminal.dependency_task_ids) - view_ids
            )
        ):
            raise ValueError("source-use changes its sealed measurement census")
        pending = set(stages)
        while pending:
            ready = {
                k for k in pending if set(stages[k].dependency_task_ids) <= set(stages) - pending
            }
            if not ready:
                raise ValueError("source-use topology has a cycle or missing dependency")
            pending -= ready


@dataclass(frozen=True, slots=True)
class ProspectiveRetainedSourceUse(QualifiedSourceUseExperiment):
    """Explicit receipt imports; imports never enter the new acquisition census."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-retained-source-use'
    retained_predecessors: tuple[ProspectiveRetainedPredecessor, ...]

    def source_qualification(self) -> ProspectiveRetainedSourceQualification:
        return ProspectiveRetainedSourceQualification(
            **{f.name: getattr(self, f.name) for f in fields(PredecessorBoundSourceQualificationExperiment)}
        )
