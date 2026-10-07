"""Typed relational atlas and bounded candidate-formalism machinery.

This module deliberately does not pool substrate-native coefficients.  It
operates on structural propositions, typed context maps and independent-unit
summaries.  Numeric estimands are exposed only for one declared receiver gauge
and denominator family at a time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


FloatArray = npt.NDArray[np.float64]


class StructuralDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    MIXED = "MIXED"
    OPPOSED = "OPPOSED"
    NULL = "NULL"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"


@dataclass(frozen=True, slots=True)
class ParentDocument(CanonicalRecord):
    """A byte-verified, bounded parent document identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/parent-document'

    parent_id: str
    relative_locator: str
    payload_schema: str
    sha256: str
    size_bytes: int
    evidence_ceiling: str

    def __post_init__(self) -> None:
        validate_stable_id(self.parent_id, field_name="parent_id")
        validate_nonempty(self.relative_locator, field_name="relative_locator")
        validate_nonempty(self.payload_schema, field_name="payload_schema")
        validate_sha256(self.sha256)
        if self.size_bytes <= 0:
            raise ValueError("parent document must be nonempty")
        validate_nonempty(self.evidence_ceiling, field_name="evidence_ceiling")


@dataclass(frozen=True, slots=True)
class ContextMapRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/context-map-record'

    map_id: str
    source_context_id: str
    target_context_id: str
    map_kind: str
    compatibility_contract: str
    coefficient_transport_permitted: bool
    status: StructuralDisposition

    def __post_init__(self) -> None:
        for name, value in (
            ("map_id", self.map_id),
            ("source_context_id", self.source_context_id),
            ("target_context_id", self.target_context_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.map_kind, field_name="map_kind")
        validate_nonempty(self.compatibility_contract, field_name="compatibility_contract")
        if self.coefficient_transport_permitted and self.compatibility_contract == "NONE":
            raise ValueError("coefficient transport requires an explicit compatibility contract")


@dataclass(frozen=True, slots=True)
class MissingArrow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/missing-arrow'

    arrow_id: str
    context_id: str
    source_role: str
    target_role: str
    reason_code: str
    disposition: StructuralDisposition

    def __post_init__(self) -> None:
        validate_stable_id(self.arrow_id, field_name="arrow_id")
        validate_stable_id(self.context_id, field_name="context_id")
        for name, value in (
            ("source_role", self.source_role),
            ("target_role", self.target_role),
            ("reason_code", self.reason_code),
        ):
            validate_nonempty(value, field_name=name)
        if self.disposition not in {
            StructuralDisposition.UNEVALUABLE,
            StructuralDisposition.NOT_ATTEMPTED,
            StructuralDisposition.AUTHORITY_REQUIRED,
        }:
            raise ValueError("a missing arrow must retain an epistemic/authority disposition")


@dataclass(frozen=True, slots=True)
class NonEntailmentWitness(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/non-entailment-witness'

    witness_id: str
    antecedent: str
    consequent: str
    context_id: str
    witness_class: str

    def __post_init__(self) -> None:
        validate_stable_id(self.witness_id, field_name="witness_id")
        validate_stable_id(self.context_id, field_name="context_id")
        for name, value in (
            ("antecedent", self.antecedent),
            ("consequent", self.consequent),
            ("witness_class", self.witness_class),
        ):
            validate_nonempty(value, field_name=name)
        if self.antecedent == self.consequent:
            raise ValueError("non-entailment witness requires distinct propositions")


@dataclass(frozen=True, slots=True)
class RelationalAtlasCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/relational-atlas-cell'

    context_id: str
    denominator_id: str
    history_role: str
    action_chart_id: str
    receiver_id: str
    horizon_id: str
    independent_unit: str
    evidence_world: str
    native_action_units: tuple[str, ...]
    delivery_stages: tuple[str, ...]
    propositions: tuple[tuple[str, StructuralDisposition], ...]
    relations: tuple[str, ...]
    obstructions: tuple[str, ...]
    evidence_ceiling: str
    coefficient_namespace: str

    def __post_init__(self) -> None:
        for name, value in (
            ("context_id", self.context_id),
            ("denominator_id", self.denominator_id),
            ("action_chart_id", self.action_chart_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
            ("coefficient_namespace", self.coefficient_namespace),
        ):
            validate_stable_id(value, field_name=name)
        for name, value in (
            ("history_role", self.history_role),
            ("independent_unit", self.independent_unit),
            ("evidence_world", self.evidence_world),
            ("evidence_ceiling", self.evidence_ceiling),
        ):
            validate_nonempty(value, field_name=name)
        require_sorted_unique_strings(self.native_action_units, field_name="native_action_units")
        require_sorted_unique_strings(self.delivery_stages, field_name="delivery_stages")
        require_sorted_unique_strings(self.relations, field_name="relations")
        require_sorted_unique_strings(self.obstructions, field_name="obstructions")
        proposition_ids = tuple(value[0] for value in self.propositions)
        require_sorted_unique_strings(proposition_ids, field_name="proposition_ids")


@dataclass(frozen=True, slots=True)
class RelationalInvariantAtlas(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/relational-invariant-atlas'

    atlas_id: str
    information_cutoff_utc: str
    parent_documents: tuple[ParentDocument, ...]
    cells: tuple[RelationalAtlasCell, ...]
    context_maps: tuple[ContextMapRecord, ...]
    missing_arrows: tuple[MissingArrow, ...]
    non_entailment_witnesses: tuple[NonEntailmentWitness, ...]
    cross_substrate_numeric_pooling: bool
    promotion_permitted: bool
    evidence_ceiling: str

    def __post_init__(self) -> None:
        validate_stable_id(self.atlas_id, field_name="atlas_id")
        validate_nonempty(self.information_cutoff_utc, field_name="information_cutoff_utc")
        require_sorted_unique_ids(
            self.parent_documents,
            attribute="parent_id",
            field_name="parent_documents",
        )
        require_sorted_unique_ids(self.cells, attribute="context_id", field_name="cells")
        require_sorted_unique_ids(
            self.context_maps,
            attribute="map_id",
            field_name="context_maps",
        )
        require_sorted_unique_ids(
            self.missing_arrows,
            attribute="arrow_id",
            field_name="missing_arrows",
        )
        require_sorted_unique_ids(
            self.non_entailment_witnesses,
            attribute="witness_id",
            field_name="non_entailment_witnesses",
        )
        if self.cross_substrate_numeric_pooling or self.promotion_permitted:
            raise ValueError("outcome-visible atlas cannot pool or promote across substrates")
        namespaces = {cell.coefficient_namespace for cell in self.cells}
        if len(namespaces) != len(self.cells):
            raise ValueError("atlas coefficient namespaces must be denominator-local")


@dataclass(frozen=True, slots=True)
class CandidateFormalism(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/candidate-formalism'

    candidate_id: str
    level: int
    object_kind: str
    required_propositions: tuple[str, ...]
    required_relations: tuple[str, ...]
    predicts: tuple[str, ...]
    falsifiers: tuple[str, ...]
    separating_experiment: str

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.level < 0 or self.level > 5:
            raise ValueError("candidate formalism level must be in [0, 5]")
        validate_nonempty(self.object_kind, field_name="object_kind")
        for name, values in (
            ("required_propositions", self.required_propositions),
            ("required_relations", self.required_relations),
            ("predicts", self.predicts),
            ("falsifiers", self.falsifiers),
        ):
            require_sorted_unique_strings(values, field_name=name)
        validate_nonempty(self.separating_experiment, field_name="separating_experiment")


@dataclass(frozen=True, slots=True)
class CandidateAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/candidate-assessment'

    assessment_id: str
    candidate_id: str
    disposition: StructuralDisposition
    supporting_context_ids: tuple[str, ...]
    opposing_context_ids: tuple[str, ...]
    unevaluable_requirement_ids: tuple[str, ...]
    universal_claim_made: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        for name, values in (
            ("supporting_context_ids", self.supporting_context_ids),
            ("opposing_context_ids", self.opposing_context_ids),
            ("unevaluable_requirement_ids", self.unevaluable_requirement_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.universal_claim_made:
            raise ValueError("candidate assessment cannot universalize a local atlas")


@dataclass(frozen=True, slots=True)
class FormalVersionSpace(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/formal-version-space'

    version_space_id: str
    candidate_vocabulary_sha256: str
    assessments: tuple[CandidateAssessment, ...]
    surviving_candidate_ids: tuple[str, ...]
    observational_equivalence_classes: tuple[tuple[str, ...], ...]
    ranked_separating_experiments: tuple[str, ...]
    forced_winner: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.version_space_id, field_name="version_space_id")
        validate_sha256(self.candidate_vocabulary_sha256)
        require_sorted_unique_ids(
            self.assessments,
            attribute="assessment_id",
            field_name="assessments",
        )
        require_sorted_unique_strings(
            self.surviving_candidate_ids,
            field_name="surviving_candidate_ids",
        )
        require_sorted_unique_strings(
            self.ranked_separating_experiments,
            field_name="ranked_separating_experiments",
        )
        if self.forced_winner and len(self.surviving_candidate_ids) != 1:
            raise ValueError("forced winner and surviving candidate count disagree")


def decode_verified_parent(
    payload: bytes,
    *,
    expected_sha256: str,
    maximum_bytes: int = 16 * 1024**2,
) -> Mapping[str, Any]:
    """Fail closed on bytes, envelope and unbounded historical semantics."""

    validate_sha256(expected_sha256)
    if not payload or len(payload) > maximum_bytes:
        raise ValueError("parent payload violates its byte bound")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("parent payload digest differs")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict) or set(decoded) != {"schema", "value", "version"}:
        raise ValueError("parent payload is not an exact canonical envelope")
    if not isinstance(decoded["schema"], str) or not decoded["schema"].startswith("empirical-lawhood/"):
        raise ValueError("parent schema is invalid")
    if not isinstance(decoded["value"], dict):
        raise ValueError("parent value is not a mapping")
    return cast(Mapping[str, Any], decoded)


def _empirical_binding(cell_id: str) -> tuple[str, str, str, str, tuple[str, ...]]:
    if "gym-torax" in cell_id:
        return (
            "denominator.gym-torax-torax-simulation",
            "action-chart.gym-torax-current-ecrh",
            "receiver.gym-torax-six-shell-24",
            "horizon.gym-torax-126s",
            ("A", "W"),
        )
    if "pybamm" in cell_id:
        return (
            "denominator.pybamm-spme-thermal-chen2020",
            "action-chart.pybamm-current-ambient",
            "receiver.pybamm-electrothermal-full",
            "horizon.pybamm-900s",
            ("A", "K"),
        )
    if "boptest" in cell_id:
        return (
            "denominator.boptest-source-interface",
            "action-chart.unbound-historical",
            "receiver.boptest-observation",
            "horizon.uninstantiated",
            (),
        )
    if "mastu" in cell_id:
        return (
            "denominator.mastu-physical-archive",
            "action-chart.mastu-requested-flow",
            "receiver.mastu-archive",
            "horizon.mastu-archive",
            ("V",),
        )
    if "battery" in cell_id:
        return (
            "denominator.battery-physical-archive",
            "action-chart.battery-cycle-request",
            "receiver.battery-pack-capacity",
            "horizon.battery-cycle",
            ("A",),
        )
    return (
        f"denominator.{cell_id}",
        f"action-chart.{cell_id}",
        f"receiver.{cell_id}",
        f"horizon.{cell_id}",
        ("1",),
    )


def build_relational_atlas(
    *,
    parent_documents: tuple[ParentDocument, ...],
    non_entailment_document: Mapping[str, Any],
    information_cutoff_utc: str,
) -> RelationalInvariantAtlas:
    value = non_entailment_document.get("value")
    if not isinstance(value, Mapping):
        raise ValueError("non-entailment document value is absent")
    raw_cells = value.get("cells")
    raw_witnesses = value.get("minimal_witnesses")
    if not isinstance(raw_cells, list) or not isinstance(raw_witnesses, list):
        raise ValueError("non-entailment document operands are absent")
    cells: list[RelationalAtlasCell] = []
    missing: list[MissingArrow] = []
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise ValueError("non-entailment cell is malformed")
        cell_id = str(raw["cell_id"])
        denominator, action, receiver, horizon, units = _empirical_binding(cell_id)
        raw_propositions = raw["propositions"]
        if not isinstance(raw_propositions, Mapping):
            raise ValueError("cell propositions are malformed")
        propositions = tuple(
            sorted(
                (
                    str(key),
                    StructuralDisposition(str(disposition)),
                )
                for key, disposition in raw_propositions.items()
            )
        )
        supported = tuple(
            sorted(key for key, disposition in propositions if disposition is StructuralDisposition.SUPPORTED)
        )
        obstructions = tuple(
            sorted(key for key, disposition in propositions if disposition is StructuralDisposition.OPPOSED)
        )
        delivery_stages = (
            ("accepted", "applied", "realized", "requested")
            if dict(propositions).get("delivery_observability")
            is StructuralDisposition.SUPPORTED
            else ("requested",)
        )
        cells.append(
            RelationalAtlasCell(
                context_id=f"context.{cell_id}",
                denominator_id=denominator,
                history_role="retained-causal-history-or-explicitly-unbound",
                action_chart_id=action,
                receiver_id=receiver,
                horizon_id=horizon,
                independent_unit="complete-physical-or-simulated-preparation",
                evidence_world=str(raw["evidence_world"]),
                native_action_units=tuple(sorted(units)),
                delivery_stages=delivery_stages,
                propositions=propositions,
                relations=supported,
                obstructions=obstructions,
                evidence_ceiling="NON_PROMOTABLE_OUTCOME_VISIBLE",
                coefficient_namespace=f"namespace.{cell_id}",
            )
        )
        for proposition, disposition in propositions:
            if disposition is StructuralDisposition.UNEVALUABLE:
                missing.append(
                    MissingArrow(
                        arrow_id=f"missing.{cell_id}.{proposition}",
                        context_id=f"context.{cell_id}",
                        source_role="prepared-context",
                        target_role=proposition,
                        reason_code="OPERAND_ABSENT_OR_UNBOUND_HISTORICAL",
                        disposition=StructuralDisposition.UNEVALUABLE,
                    )
                )
    witnesses = []
    for index, raw in enumerate(raw_witnesses, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError("non-entailment witness is malformed")
        witnesses.append(
            NonEntailmentWitness(
                witness_id=f"witness.non-entailment-{index:03d}",
                antecedent=str(raw["antecedent"]),
                consequent=str(raw["consequent"]),
                context_id=f"context.{raw['witness']}",
                witness_class=str(raw["witness_class"]),
            )
        )
    maps = (
        ContextMapRecord(
            map_id="map.gym-full-to-operational-receiver",
            source_context_id="context.empirical-gym-torax-held-out",
            target_context_id="context.empirical-gym-torax-held-out",
            map_kind="receiver-pushforward",
            compatibility_contract="same-denominator-coordinate-projection",
            coefficient_transport_permitted=False,
            status=StructuralDisposition.MIXED,
        ),
        ContextMapRecord(
            map_id="map.pybamm-full-to-operational-receiver",
            source_context_id="context.empirical-pybamm-held-out",
            target_context_id="context.empirical-pybamm-held-out",
            map_kind="receiver-pushforward",
            compatibility_contract="same-denominator-coordinate-projection",
            coefficient_transport_permitted=False,
            status=StructuralDisposition.MIXED,
        ),
    )
    return RelationalInvariantAtlas(
        atlas_id="atlas.response-formalization-outcome-visible",
        information_cutoff_utc=information_cutoff_utc,
        parent_documents=tuple(sorted(parent_documents, key=lambda item: item.parent_id)),
        cells=tuple(sorted(cells, key=lambda item: item.context_id)),
        context_maps=tuple(sorted(maps, key=lambda item: item.map_id)),
        missing_arrows=tuple(sorted(missing, key=lambda item: item.arrow_id)),
        non_entailment_witnesses=tuple(
            sorted(witnesses, key=lambda item: item.witness_id)
        ),
        cross_substrate_numeric_pooling=False,
        promotion_permitted=False,
        evidence_ceiling="NON_PROMOTABLE_OUTCOME_VISIBLE_ATLAS",
    )


def candidate_formalisms() -> tuple[CandidateFormalism, ...]:
    values = (
        CandidateFormalism(
            'candidate.context-indexed-unrelated-maps',
            0,
            "context-indexed-unrelated-response-maps",
            (),
            (),
            ("local-response",),
            ("held-out-cross-context-relation",),
            "repeat one complete preparation under the same context",
        ),
        CandidateFormalism(
            'candidate.partial-word-action',
            1,
            "partial-representation-of-delivered-finite-words",
            ("action_image", "delivery_observability"),
            ("identity", "partial-composition"),
            ("held-out-word-composition",),
            ("delivery-mismatch", "identity-failure", "repeat-failure"),
            "prospective identity repeat and held-out word composition",
        ),
        CandidateFormalism(
            'candidate.temporal-cocycle',
            2,
            "time-indexed-cocycle",
            ("temporal_cocycle",),
            ("cocycle",),
            ("equal-lag-composition",),
            ("cocycle-defect", "nonrecurrence"),
            "fresh equal-lag and unequal-anchor temporal compositions",
        ),
        CandidateFormalism(
            'candidate.context-fibred-system',
            3,
            "fibred-partial-response-system",
            ("receiver_faithfulness",),
            ("context-map", "receiver-pushforward"),
            ("commuting-context-square",),
            ("map-path-dependence", "receiver-fold"),
            "held-out receiver and topology map commutation",
        ),
        CandidateFormalism(
            'candidate.overlap-gluing',
            4,
            "overlap-compatible-local-sections",
            ("transport_membership",),
            ("gluing", "overlap-compatibility"),
            ("whole-system-error-bound",),
            ("interface-obstruction",),
            "prospective cut/glue intervention with measured interface",
        ),
        CandidateFormalism(
            'candidate.topology-mixture',
            5,
            "topology-conditioned-concentrated-or-mixture-operator-family",
            (),
            ("topology-conditioning",),
            ("held-out-topology-law-class",),
            ("relabel-defect", "wrong-topology-map"),
            "matched relabel rewiring boundary and cut/glue factorial",
        ),
    )
    return tuple(sorted(values, key=lambda item: item.level))


def assess_version_space(
    atlas: RelationalInvariantAtlas,
    candidates: tuple[CandidateFormalism, ...],
) -> FormalVersionSpace:
    assessments = []
    for candidate in candidates:
        supporting: set[str] = set()
        opposing: set[str] = set()
        observed_requirements: set[str] = set()
        for cell in atlas.cells:
            propositions = dict(cell.propositions)
            dispositions = [propositions.get(item) for item in candidate.required_propositions]
            if dispositions and all(item is StructuralDisposition.SUPPORTED for item in dispositions):
                supporting.add(cell.context_id)
            if any(item is StructuralDisposition.OPPOSED for item in dispositions):
                opposing.add(cell.context_id)
            observed_requirements.update(
                item
                for item, disposition in zip(
                    candidate.required_propositions,
                    dispositions,
                    strict=True,
                )
                if disposition in {StructuralDisposition.SUPPORTED, StructuralDisposition.OPPOSED}
            )
        relation_inventory = {relation for cell in atlas.cells for relation in cell.relations}
        missing_requirements = set(candidate.required_propositions) - observed_requirements
        missing_requirements.update(set(candidate.required_relations) - relation_inventory)
        if candidate.level == 0:
            disposition = StructuralDisposition.SUPPORTED
        elif supporting and opposing:
            disposition = StructuralDisposition.MIXED
        elif supporting and not missing_requirements:
            disposition = StructuralDisposition.SUPPORTED
        elif opposing and not supporting:
            disposition = StructuralDisposition.OPPOSED
        else:
            disposition = StructuralDisposition.UNEVALUABLE
        assessments.append(
            CandidateAssessment(
                assessment_id=f"assessment.{candidate.candidate_id.removeprefix('candidate.')}",
                candidate_id=candidate.candidate_id,
                disposition=disposition,
                supporting_context_ids=tuple(sorted(supporting)),
                opposing_context_ids=tuple(sorted(opposing)),
                unevaluable_requirement_ids=tuple(sorted(missing_requirements)),
                universal_claim_made=False,
            )
        )
    surviving = tuple(
        sorted(
            item.candidate_id
            for item in assessments
            if item.disposition
            in {
                StructuralDisposition.SUPPORTED,
                StructuralDisposition.MIXED,
                StructuralDisposition.UNEVALUABLE,
            }
        )
    )
    vocabulary_sha = hashlib.sha256(
        b"".join(candidate.canonical_bytes() for candidate in candidates)
    ).hexdigest()
    separating = tuple(
        sorted(
            candidate.separating_experiment
            for candidate in candidates
            if candidate.candidate_id in surviving and candidate.level > 0
        )
    )
    return FormalVersionSpace(
        version_space_id="version-space.response-formalization",
        candidate_vocabulary_sha256=vocabulary_sha,
        assessments=tuple(sorted(assessments, key=lambda item: item.assessment_id)),
        surviving_candidate_ids=surviving,
        observational_equivalence_classes=(surviving,),
        ranked_separating_experiments=separating,
        forced_winner=False,
    )


def leave_one_evidence_family_sensitivity(
    atlas: RelationalInvariantAtlas,
) -> tuple[tuple[str, int, int], ...]:
    """Count retained witnesses after dropping each evidence-world family."""

    world_by_context = {cell.context_id: cell.evidence_world for cell in atlas.cells}
    worlds = sorted(set(world_by_context.values()))
    rows = []
    total = len(atlas.non_entailment_witnesses)
    for world in worlds:
        retained = sum(
            world_by_context.get(witness.context_id) != world
            for witness in atlas.non_entailment_witnesses
        )
        rows.append((world, retained, total))
    return tuple(rows)


def receiver_coarsening_sensitivity(
    receiver_frontier_document: Mapping[str, Any],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    value = receiver_frontier_document.get("value")
    if not isinstance(value, Mapping) or not isinstance(value.get("systems"), Mapping):
        raise ValueError("receiver frontier systems are absent")
    rows = []
    for system_id, raw_system in cast(Mapping[str, Any], value["systems"]).items():
        if not isinstance(raw_system, Mapping):
            raise ValueError("receiver frontier system is malformed")
        minimal = str(raw_system.get("minimal_faithful_group", "NONE"))
        groups = raw_system.get("receiver_groups", [])
        fully_faithful = tuple(
            sorted(
                str(group["group_id"])
                for group in groups
                if isinstance(group, Mapping) and bool(group.get("fully_faithful", False))
            )
        )
        rows.append((str(system_id), minimal, fully_faithful))
    return tuple(sorted(rows))


def floor_certified_rank(
    contrasts: FloatArray,
    *,
    absolute_floor: float,
    relative_tolerance: float,
) -> tuple[int, FloatArray]:
    if contrasts.ndim != 2 or contrasts.size == 0:
        raise ValueError("rank contrasts must be a nonempty matrix")
    singular = np.asarray(np.linalg.svd(contrasts, compute_uv=False), dtype=np.float64)
    threshold = max(absolute_floor, relative_tolerance * float(singular[0]))
    return int(np.sum(singular > threshold)), singular


def fit_affine_operator(states: FloatArray, targets: FloatArray, ridge: float = 1e-8) -> FloatArray:
    if states.ndim != 2 or targets.ndim != 2 or states.shape[0] != targets.shape[0]:
        raise ValueError("operator fitting arrays have incompatible shapes")
    design = np.column_stack((states, np.ones(states.shape[0], dtype=np.float64)))
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.asarray(
        np.linalg.solve(design.T @ design + penalty, design.T @ targets),
        dtype=np.float64,
    )


def affine_prediction(operator: FloatArray, states: FloatArray) -> FloatArray:
    if states.ndim != 2 or operator.ndim != 2 or operator.shape[0] != states.shape[1] + 1:
        raise ValueError("affine operator and state dimensions differ")
    return np.column_stack((states, np.ones(states.shape[0], dtype=np.float64))) @ operator


def rms(values: FloatArray) -> float:
    if values.size == 0:
        raise ValueError("RMS requires at least one value")
    return float(np.sqrt(np.mean(np.square(values))))


def operator_concentration(operators: Sequence[FloatArray], temperature: float) -> dict[str, float]:
    if not operators or temperature <= 0.0:
        raise ValueError("operator concentration requires operators and positive temperature")
    flattened = np.stack([operator.reshape(-1) for operator in operators])
    centroid = np.mean(flattened, axis=0)
    distances = np.linalg.norm(flattened - centroid, axis=1)
    weights = np.exp(-distances / temperature)
    weights /= np.sum(weights)
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, np.finfo(float).tiny))))
    return {
        "mean_distance": float(np.mean(distances)),
        "maximum_distance": float(np.max(distances)),
        "weight_entropy": entropy,
        "effective_operator_count": float(np.exp(entropy)),
    }


def bootstrap_mean_interval(
    independent_values: FloatArray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    if independent_values.ndim != 1 or independent_values.size < 2:
        raise ValueError("bootstrap requires at least two independent units")
    if resamples < 100 or not 0.5 < confidence_level < 1.0:
        raise ValueError("bootstrap controls are invalid")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        independent_values.size,
        size=(resamples, independent_values.size),
    )
    estimates = np.mean(independent_values[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.mean(independent_values)),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def functional_commutation_defect(
    mapped_source_output: FloatArray,
    target_output: FloatArray,
    scale: FloatArray,
) -> float:
    if mapped_source_output.shape != target_output.shape or scale.shape != target_output.shape[-1:]:
        raise ValueError("commutation arrays have incompatible shapes")
    if np.any(scale <= 0.0):
        raise ValueError("commutation scale must be positive")
    return rms((mapped_source_output - target_output) / scale)


def approximate_gluing_bound(
    *,
    contraction_factor: float,
    local_section_error: float,
    interface_error: float,
    map_error: float,
) -> float:
    if not 0.0 <= contraction_factor < 1.0:
        raise ValueError("gluing bound requires a contraction factor in [0, 1)")
    if min(local_section_error, interface_error, map_error) < 0.0:
        raise ValueError("gluing error components must be nonnegative")
    return (local_section_error + interface_error + map_error) / (1.0 - contraction_factor)
