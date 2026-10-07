"""Production finite-action identification over the exact projection companion.

The producer estimates member-local finite-word responses and emits raw method
evidence.  It never emits a terminal law verdict; the existing qualification
profile and sole response-law finalizer retain that authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
import hashlib
import random
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordSupportStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import LawRepresentationKind
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import HorizonSpec
from empirical_lawhood.planning.identification_evidence import IdentificationEvidenceProjection
from empirical_lawhood.planning.identification_evidence_extensions import ActionOccurrenceBinding, AuthorityAxis, ComputabilityEffectStatus, IdentificationEvidenceProjectionExtension, IdentificationMethodEvidenceKind, NumericalValidityAxis, ObservationValidityAxis, PhysicalSinkAxis, ScientificTerminalAxis
from .contracts import (
    CandidateClaimTemplate,
    CandidateEvaluatorImplementation,
    CandidateMethodEvidenceReceipt,
    FiniteActionCellDisposition,
    FiniteActionCompatibilitySetExtension,
    FiniteActionNativeValue,
    FiniteActionResponseEntry,
    LawCandidateAxisMap,
    LawCandidateEvidence,
    LawObligationTemplate,
    MethodEvidenceAvailability,
)
from .evidence_projection_extensions import require_method_projection_truth
from .law_assessment import CandidatePayloadPublisher


MAX_BOOTSTRAP_REPLICATES = 20_000
MAX_FINITE_ACTION_CELLS = 100_000


class FiniteActionProductionDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    UNEVALUABLE = "UNEVALUABLE"


class IndependentUnitResamplingMethod(StrEnum):
    COMPLETE_UNIT_BOOTSTRAP = "COMPLETE_UNIT_BOOTSTRAP"


@dataclass(frozen=True, slots=True)
class FiniteActionComparatorBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-comparator-binding'

    binding_id: str
    action_word_id: str
    comparator_action_word_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("action_word_id", self.action_word_id),
            ("comparator_action_word_id", self.comparator_action_word_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.action_word_id == self.comparator_action_word_id:
            raise ValueError("finite-action comparator must be another exact word")


@dataclass(frozen=True, slots=True)
class FiniteActionResponseQuantity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-response-quantity'

    binding_id: str
    projection_value_id: str
    quantity_id: str
    native_unit: str
    native_frame_id: str
    clock_id: str
    receiver_window_id: str
    maximum_heldout_absolute_error: Decimal
    maximum_refinement_absolute_delta: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("projection_value_id", self.projection_value_id),
            ("quantity_id", self.quantity_id),
            ("native_frame_id", self.native_frame_id),
            ("clock_id", self.clock_id),
            ("receiver_window_id", self.receiver_window_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(
            self.maximum_heldout_absolute_error,
            field_name="maximum_heldout_absolute_error",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.maximum_refinement_absolute_delta,
            field_name="maximum_refinement_absolute_delta",
            minimum=Decimal(0),
        )


@dataclass(frozen=True, slots=True)
class FiniteActionIdentificationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-identification-config'

    config_id: str
    method_key: str
    method_version: str
    implementation_sha256: str
    projection: ObjectIdentity
    projection_extension: ObjectIdentity
    horizon: ObjectIdentity
    prepared_denominator_id: str
    retained_history_ids: tuple[str, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    comparator_bindings: tuple[FiniteActionComparatorBinding, ...]
    axis_map: LawCandidateAxisMap
    support_cell_ids: tuple[str, ...]
    physical_independent_unit_ids: tuple[str, ...]
    calibration_split_ids: tuple[str, ...]
    heldout_split_ids: tuple[str, ...]
    estimation_view_id: str
    response_quantities: tuple[FiniteActionResponseQuantity, ...]
    minimum_complete_units: int
    resampling_method: IndependentUnitResamplingMethod
    bootstrap_replicates: int
    bootstrap_seed: int
    simultaneous_confidence_level: Decimal
    information_cutoff_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("config_id", self.config_id),
            ("method_key", self.method_key),
            ("prepared_denominator_id", self.prepared_denominator_id),
            ("information_cutoff_id", self.information_cutoff_id),
            ("estimation_view_id", self.estimation_view_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.method_key != "finite-action.compatibility-set":
            raise ValueError("finite-action config selects another method")
        validate_semantic_version(self.method_version)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.projection.object_schema != IdentificationEvidenceProjection.SCHEMA:
            raise ValueError("finite-action config requires an evidence projection")
        if (
            self.projection_extension.object_schema
            != IdentificationEvidenceProjectionExtension.SCHEMA
        ):
            raise ValueError("finite-action config requires the projection companion")
        if self.horizon.object_schema != HorizonSpec.SCHEMA:
            raise ValueError("finite-action config requires the exact response horizon")
        for name, values in (
            ("retained_history_ids", self.retained_history_ids),
            ("support_cell_ids", self.support_cell_ids),
            ("physical_independent_unit_ids", self.physical_independent_unit_ids),
            ("calibration_split_ids", self.calibration_split_ids),
            ("heldout_split_ids", self.heldout_split_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if set(self.calibration_split_ids) & set(self.heldout_split_ids):
            raise ValueError("finite-action calibration and held-out splits overlap")
        require_sorted_unique_ids(self.action_words, attribute="word_id", field_name="action_words")
        if not self.action_words:
            raise ValueError("finite-action config requires action words")
        word_ids = {value.word_id for value in self.action_words}
        if any(
            value.denominator_id != self.prepared_denominator_id
            or value.horizon_id != self.horizon.object_id
            or value.retained_history_id not in self.retained_history_ids
            for value in self.action_words
        ):
            raise ValueError("finite-action words change denominator/history/horizon")
        require_sorted_unique_ids(
            self.comparator_bindings, attribute="binding_id", field_name="comparator_bindings"
        )
        if {value.action_word_id for value in self.comparator_bindings} != word_ids:
            raise ValueError("finite-action comparators must cover every action word exactly once")
        if any(
            value.comparator_action_word_id not in word_ids for value in self.comparator_bindings
        ):
            raise ValueError("finite-action comparator names a foreign action word")
        require_sorted_unique_ids(
            self.response_quantities, attribute="binding_id", field_name="response_quantities"
        )
        if not self.response_quantities:
            raise ValueError("finite-action config requires response quantities")
        if any(
            self.estimation_view_id not in value.qualification_view_ids
            for value in self.axis_map.bindings
        ):
            raise ValueError("finite-action estimation view is absent from an axis binding")
        if self.minimum_complete_units < 2:
            raise ValueError("finite-action inference requires at least two physical units")
        if not 1 <= self.bootstrap_replicates <= MAX_BOOTSTRAP_REPLICATES:
            raise ValueError("finite-action bootstrap replicate count is outside bounds")
        if self.bootstrap_seed < 0:
            raise ValueError("finite-action bootstrap seed must be nonnegative")
        validate_decimal(
            self.simultaneous_confidence_level,
            field_name="simultaneous_confidence_level",
            minimum=Decimal(0),
        )
        if not Decimal(0) < self.simultaneous_confidence_level < Decimal(1):
            raise ValueError("finite-action confidence level must lie between zero and one")
        cell_count = (
            len(self.axis_map.bindings) * len(self.support_cell_ids) * len(self.action_words)
        )
        if cell_count > MAX_FINITE_ACTION_CELLS:
            raise ValueError("finite-action config exceeds its cell bound")


@dataclass(frozen=True, slots=True)
class FiniteActionCandidateScaffold(CanonicalRecord):
    """Status-free candidate metadata supplied by campaign authoring."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-candidate-scaffold'

    scaffold_id: str
    evidence_id: str
    candidate_id: str
    system: ObjectIdentity
    axis_map: LawCandidateAxisMap
    claim_unit_binding: ObjectIdentity
    claim_template: CandidateClaimTemplate
    obligation_template: LawObligationTemplate
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("scaffold_id", self.scaffold_id),
            ("evidence_id", self.evidence_id),
            ("candidate_id", self.candidate_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        if not self.evidence_links:
            raise ValueError("finite-action candidate scaffold requires evidence links")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("finite-action scaffold lowers evidence visibility")


@dataclass(frozen=True, slots=True)
class FiniteActionProductionCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-production-cell'

    cell_id: str
    denominator_member_id: str
    candidate_version_id: str
    support_cell_id: str
    action_word: ObjectIdentity
    comparator_action_word_id: str
    complete_physical_unit_ids: tuple[str, ...]
    excluded_physical_unit_ids: tuple[str, ...]
    response_estimates: tuple[FiniteActionNativeValue, ...]
    lower_bounds: tuple[FiniteActionNativeValue, ...]
    upper_bounds: tuple[FiniteActionNativeValue, ...]
    sink_operands: tuple[NamedDecimal, ...]
    disposition: FiniteActionProductionDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("support_cell_id", self.support_cell_id),
            ("comparator_action_word_id", self.comparator_action_word_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("finite-action cell requires the exact current action word")
        for name, string_values in (
            ("complete_physical_unit_ids", self.complete_physical_unit_ids),
            ("excluded_physical_unit_ids", self.excluded_physical_unit_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(string_values, field_name=name)
        if set(self.complete_physical_unit_ids) & set(self.excluded_physical_unit_ids):
            raise ValueError("finite-action complete and excluded units overlap")
        for name, record_values in (
            ("response_estimates", self.response_estimates),
            ("lower_bounds", self.lower_bounds),
            ("upper_bounds", self.upper_bounds),
            ("sink_operands", self.sink_operands),
        ):
            require_sorted_unique_ids(record_values, attribute="value_id", field_name=name)
        has_estimate = bool(self.response_estimates)
        if has_estimate != bool(self.lower_bounds) or has_estimate != bool(self.upper_bounds):
            raise ValueError("finite-action estimates require both interval bounds")
        if self.disposition is FiniteActionProductionDisposition.SUPPORTED:
            if not has_estimate or self.reason_codes:
                raise ValueError("supported finite-action cell requires estimates and no reasons")
        elif not self.reason_codes:
            raise ValueError("non-supported finite-action cell requires typed reasons")


@dataclass(frozen=True, slots=True)
class FiniteActionIdentificationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-identification-result'

    result_id: str
    config: ObjectIdentity
    projection: ObjectIdentity
    projection_extension: ObjectIdentity
    cells: tuple[FiniteActionProductionCell, ...]
    payload: FiniteActionCompatibilitySetExtension
    candidate_evidence: LawCandidateEvidence

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.config.object_schema != FiniteActionIdentificationConfig.SCHEMA:
            raise ValueError("finite-action result binds another config")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if not self.cells:
            raise ValueError("finite-action result requires complete cell accounting")
        expected = {
            (
                value.denominator_member_id,
                value.candidate_version_id,
                value.support_cell_id,
                value.action_word.object_id,
            )
            for value in self.cells
        }
        actual = {
            (
                value.denominator_member_id,
                value.candidate_version_id,
                value.support_cell_id,
                value.action_word.object_id,
            )
            for value in self.payload.entries
        }
        if expected != actual:
            raise ValueError("finite-action result and evaluator payload cell rosters differ")


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, Decimal(0)) / Decimal(len(values))


def _quantile(values: list[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    index = int(probability * Decimal(len(ordered) - 1))
    return ordered[index]


def _bootstrap_interval(
    values: tuple[Decimal, ...],
    *,
    replicates: int,
    confidence: Decimal,
    seed: int,
    simultaneous_count: int,
) -> tuple[Decimal, Decimal]:
    generator = random.Random(seed)
    samples = [
        _mean(tuple(values[generator.randrange(len(values))] for _ in values))
        for _ in range(replicates)
    ]
    alpha = (Decimal(1) - confidence) / Decimal(2 * simultaneous_count)
    return _quantile(samples, alpha), _quantile(samples, Decimal(1) - alpha)


def _one_factor_exchange(action: OccurrenceActionWord, comparator: OccurrenceActionWord) -> bool:
    if (
        action.denominator_id != comparator.denominator_id
        or action.retained_history_id != comparator.retained_history_id
        or action.horizon_id != comparator.horizon_id
        or len(action.occurrences) != len(comparator.occurrences)
    ):
        return False
    differences = sum(
        left.canonical_bytes() != right.canonical_bytes()
        for left, right in zip(action.occurrences, comparator.occurrences, strict=True)
    )
    return differences == 1


def _native_value(
    *,
    prefix: str,
    quantity: FiniteActionResponseQuantity,
    value: Decimal,
) -> FiniteActionNativeValue:
    return FiniteActionNativeValue(
        value_id=f"{prefix}.{quantity.quantity_id}",
        quantity_id=quantity.quantity_id,
        value=value,
        native_unit=quantity.native_unit,
        native_frame_id=quantity.native_frame_id,
        clock_id=quantity.clock_id,
    )


def _receipt(
    *,
    candidate_id: str,
    suffix: str,
    metric_id: str,
    observed: bool | None,
    evidence_link_ids: tuple[str, ...],
) -> CandidateMethodEvidenceReceipt:
    availability = (
        MethodEvidenceAvailability.NOT_APPLICABLE
        if observed is None
        else MethodEvidenceAvailability.OBSERVED
    )
    return CandidateMethodEvidenceReceipt(
        receipt_id=f"method-evidence.{candidate_id}.{suffix}",
        evidence_kind_id=f"finite-action.{suffix}",
        metrics=(NamedDecimal(metric_id, Decimal(int(observed)), "1"),)
        if observed is not None
        else (),
        artifact_ids=(),
        evidence_link_ids=evidence_link_ids,
        method_reason_codes=(),
        availability=availability,
    )


@dataclass(frozen=True, slots=True)
class FiniteActionIdentificationProducer:
    payload_publisher: CandidatePayloadPublisher

    def identify(
        self,
        *,
        projection: IdentificationEvidenceProjection,
        extension: IdentificationEvidenceProjectionExtension,
        config: FiniteActionIdentificationConfig,
        scaffold: FiniteActionCandidateScaffold,
    ) -> FiniteActionIdentificationResult:
        if config.projection != ObjectIdentity.from_record(projection.projection_id, projection):
            raise ValueError("finite-action config binds another projection")
        if config.projection_extension != ObjectIdentity.from_record(
            extension.extension_id, extension
        ):
            raise ValueError("finite-action config binds another projection extension")
        if extension.projection != config.projection or projection.information_cutoff_id != (
            config.information_cutoff_id
        ):
            raise ValueError("finite-action projection lineage or causal cutoff differs")
        require_method_projection_truth(
            extension=extension,
            method_kind=IdentificationMethodEvidenceKind.FINITE_ACTION,
        )
        if config.physical_independent_unit_ids != extension.independent_unit_ids:
            raise ValueError("finite-action config changes the independent-unit roster")
        if scaffold.axis_map != config.axis_map:
            raise ValueError("finite-action candidate axis map differs from config")
        if scaffold.obligation_template.physical_unit_count != len(
            config.physical_independent_unit_ids
        ):
            raise ValueError("finite-action obligation scope changes physical replication")
        if scaffold.claim_unit_binding != ObjectIdentity.from_record(
            projection.claim_unit_binding.binding_id,
            projection.claim_unit_binding,
        ):
            raise ValueError("finite-action scaffold changes the projection claim/unit binding")

        observations = {value.projected_observation_id: value for value in projection.observations}
        if len(observations) != len(projection.observations):
            raise ValueError("finite-action projection duplicates an observation identity")
        terminals = {value.episode_id: value for value in extension.terminal_dispositions}
        if {value.projected_observation_id for value in terminals.values()} != set(observations):
            raise ValueError("finite-action terminal roster differs from the projection")
        words = {value.word_id: value for value in config.action_words}
        word_identities = {
            value.word_id: ObjectIdentity.from_record(value.word_id, value)
            for value in config.action_words
        }
        occurrence_bindings: dict[str, list[ActionOccurrenceBinding]] = {}
        for value in extension.action_occurrences:
            if word_identities.get(value.action_word.object_id) != value.action_word:
                raise ValueError("finite-action episode names a foreign or altered action word")
            occurrence_bindings.setdefault(value.episode_id, []).append(value)
        actions_by_episode: dict[str, ObjectIdentity] = {}
        for episode_id, bindings in occurrence_bindings.items():
            action_identities = {value.action_word for value in bindings}
            if len(action_identities) != 1:
                raise ValueError("one episode contains occurrences from different action words")
            action_identity = next(iter(action_identities))
            word = words[action_identity.object_id]
            by_occurrence = {value.occurrence.occurrence_id: value.occurrence for value in bindings}
            if (
                len(by_occurrence) != len(bindings)
                or tuple(by_occurrence.get(value.occurrence_id) for value in word.occurrences)
                != word.occurrences
            ):
                raise ValueError("finite-action episode changes exact occurrence order or content")
            actions_by_episode[episode_id] = action_identity
        effects: dict[tuple[str, str], set[ComputabilityEffectStatus]] = {}
        for entry in extension.computability_entries:
            effects.setdefault(
                (
                    entry.coordinate.denominator_member_id,
                    entry.coordinate.numerical_view_id,
                ),
                set(),
            ).add(entry.status)
        comparators = {
            value.action_word_id: value.comparator_action_word_id
            for value in config.comparator_bindings
        }
        if any(
            not _one_factor_exchange(words[action], words[comparator])
            for action, comparator in comparators.items()
        ):
            raise ValueError("finite-action comparator is not a declared one-factor exchange")

        axes = {
            (value.denominator_member_id, value.candidate_version_member_id): value
            for value in config.axis_map.bindings
        }
        configured_splits = {
            *config.calibration_split_ids,
            *config.heldout_split_ids,
        }
        split_family_by_unit: dict[str, str] = {}
        for terminal in terminals.values():
            try:
                observation = observations[terminal.projected_observation_id]
            except KeyError as error:
                raise ValueError("finite-action terminal names an absent observation") from error
            if terminal.physical_independent_unit_id != observation.physical_unit_instance_id:
                raise ValueError("finite-action terminal changes the physical unit")
            if observation.physical_unit_instance_id not in config.physical_independent_unit_ids:
                raise ValueError("finite-action observation names a foreign physical unit")
            if observation.denominator_cell_id not in config.support_cell_ids:
                raise ValueError("finite-action observation names a foreign support cell")
            axis = axes.get((observation.member_id, observation.candidate_version_id))
            if axis is None or observation.qualification_view_id not in axis.qualification_view_ids:
                raise ValueError("finite-action observation lies outside the candidate axis map")
            if observation.split_id not in configured_splits:
                raise ValueError("finite-action observation names an undeclared split")
            family = (
                "calibration" if observation.split_id in config.calibration_split_ids else "heldout"
            )
            previous = split_family_by_unit.setdefault(
                observation.physical_unit_instance_id,
                family,
            )
            if previous != family:
                raise ValueError("one physical unit crosses calibration and held-out splits")
        if set(split_family_by_unit) != set(config.physical_independent_unit_ids):
            raise ValueError("finite-action split roster omits a physical unit")
        calibration_units = tuple(
            sorted(unit for unit, family in split_family_by_unit.items() if family == "calibration")
        )
        heldout_units = tuple(
            sorted(unit for unit, family in split_family_by_unit.items() if family == "heldout")
        )
        if not calibration_units or not heldout_units:
            raise ValueError("finite-action inference requires calibration and held-out units")

        values_by_coordinate: dict[tuple[str, str, str, str, str, str], dict[str, Decimal]] = {}
        invalid_units: dict[tuple[str, str, str, str], set[str]] = {}
        sink_by_cell: dict[tuple[str, str, str, str], dict[str, NamedDecimal]] = {}
        for episode_id, terminal in terminals.items():
            observation = observations[terminal.projected_observation_id]
            word_id = actions_by_episode[episode_id].object_id
            cell = (
                observation.member_id,
                observation.candidate_version_id,
                observation.denominator_cell_id,
                word_id,
            )
            sink_by_cell.setdefault(cell, {}).update(
                {value.value_id: value for value in terminal.sink_operands}
            )
            effect_statuses = effects.get(
                (observation.member_id, observation.qualification_view_id),
                set(),
            )
            valid = (
                terminal.physical_sink is PhysicalSinkAxis.NONE
                and terminal.observation_validity is ObservationValidityAxis.VALID
                and terminal.numerical_validity is NumericalValidityAxis.VALID
                and terminal.authority in {AuthorityAxis.NOT_REQUIRED, AuthorityAxis.AUTHORIZED}
                and terminal.scientific_status
                in {ScientificTerminalAxis.OBSERVED, ScientificTerminalAxis.NEGATIVE}
                and bool(effect_statuses)
                and effect_statuses
                <= {
                    ComputabilityEffectStatus.REPRESENTED,
                    ComputabilityEffectStatus.ASSUMPTION_CLOSED,
                }
            )
            if not valid:
                invalid_units.setdefault(cell, set()).add(terminal.physical_independent_unit_id)
                continue
            compact = {value.value_id: value for value in observation.compact_values}
            for quantity in config.response_quantities:
                try:
                    native = compact[quantity.projection_value_id]
                except KeyError as error:
                    raise ValueError("finite-action response value is absent") from error
                if native.unit != quantity.native_unit:
                    raise ValueError("finite-action response native unit drifted")
                coordinate = (
                    *cell,
                    terminal.physical_independent_unit_id,
                    observation.qualification_view_id,
                )
                bucket = values_by_coordinate.setdefault(coordinate, {})
                if quantity.quantity_id in bucket:
                    raise ValueError("finite-action projection duplicates a reduced response cell")
                bucket[quantity.quantity_id] = native.value

        cells: list[FiniteActionProductionCell] = []
        payload_entries: list[FiniteActionResponseEntry] = []
        evidence_link_ids = tuple(value.link_id for value in scaffold.evidence_links)
        calibration_unit_set = set(calibration_units)
        heldout_checks: list[bool] = []
        refinement_checks: list[bool] = []
        member_estimates: dict[tuple[str, str, str], list[Decimal]] = {}
        thresholds = {value.quantity_id: value for value in config.response_quantities}
        for axis in config.axis_map.bindings:
            member = axis.denominator_member_id
            candidate_version = axis.candidate_version_member_id
            for support in config.support_cell_ids:
                for word in config.action_words:
                    comparator_id = comparators[word.word_id]
                    key = (member, candidate_version, support, word.word_id)

                    def paired_for(
                        unit_ids: tuple[str, ...],
                        view_id: str,
                    ) -> dict[str, dict[str, Decimal]]:
                        paired_values: dict[str, dict[str, Decimal]] = {}
                        for unit in unit_ids:
                            action_values = values_by_coordinate.get((*key, unit, view_id))
                            comparator_values = values_by_coordinate.get(
                                (
                                    member,
                                    candidate_version,
                                    support,
                                    comparator_id,
                                    unit,
                                    view_id,
                                )
                            )
                            if action_values is None or comparator_values is None:
                                continue
                            if set(action_values) != set(thresholds) or set(
                                comparator_values
                            ) != set(thresholds):
                                continue
                            paired_values[unit] = {
                                quantity_id: action_values[quantity_id]
                                - comparator_values[quantity_id]
                                for quantity_id in thresholds
                            }
                        return paired_values

                    paired = paired_for(calibration_units, config.estimation_view_id)
                    complete_units = tuple(sorted(paired))
                    excluded_units = tuple(sorted(calibration_unit_set - set(complete_units)))
                    invalid_calibration_units = invalid_units.get(key, set()) & (
                        calibration_unit_set
                    )
                    estimates: list[FiniteActionNativeValue] = []
                    lowers: list[FiniteActionNativeValue] = []
                    uppers: list[FiniteActionNativeValue] = []
                    reasons: tuple[str, ...]
                    if word.support_status is not ActionWordSupportStatus.SUPPORTED:
                        disposition = FiniteActionProductionDisposition.OUTSIDE_SUPPORT
                        reasons = ("FINITE_ACTION_WORD_OUTSIDE_SUPPORT",)
                    elif not complete_units:
                        disposition = (
                            FiniteActionProductionDisposition.INVALID
                            if invalid_calibration_units
                            else FiniteActionProductionDisposition.UNEVALUABLE
                        )
                        reasons = (
                            "FINITE_ACTION_ALL_EPISODES_INVALID"
                            if invalid_calibration_units
                            else "FINITE_ACTION_NO_COMPLETE_UNIT_PAIRS",
                        )
                    elif len(complete_units) < config.minimum_complete_units:
                        disposition = FiniteActionProductionDisposition.PARTIAL
                        reasons = ("FINITE_ACTION_INSUFFICIENT_COMPLETE_UNITS",)
                    else:
                        interval_supported = True
                        coordinate_seed = int(
                            hashlib.sha256(".".join(key).encode()).hexdigest()[:16], 16
                        )
                        estimates_by_quantity: dict[str, Decimal] = {}
                        for quantity in config.response_quantities:
                            unit_values = tuple(
                                paired[unit][quantity.quantity_id] for unit in complete_units
                            )
                            estimate = _mean(unit_values)
                            lower, upper = _bootstrap_interval(
                                unit_values,
                                replicates=config.bootstrap_replicates,
                                confidence=config.simultaneous_confidence_level,
                                seed=config.bootstrap_seed + coordinate_seed,
                                simultaneous_count=len(config.response_quantities),
                            )
                            estimates.append(
                                _native_value(prefix="response", quantity=quantity, value=estimate)
                            )
                            lowers.append(
                                _native_value(prefix="lower", quantity=quantity, value=lower)
                            )
                            uppers.append(
                                _native_value(prefix="upper", quantity=quantity, value=upper)
                            )
                            estimates_by_quantity[quantity.quantity_id] = estimate
                            interval_supported = interval_supported and (lower > 0 or upper < 0)

                        heldout = paired_for(heldout_units, config.estimation_view_id)
                        heldout_complete = set(heldout) == set(heldout_units)
                        heldout_ok = heldout_complete and all(
                            abs(
                                _mean(
                                    tuple(
                                        heldout[unit][quantity.quantity_id]
                                        for unit in heldout_units
                                    )
                                )
                                - estimates_by_quantity[quantity.quantity_id]
                            )
                            <= quantity.maximum_heldout_absolute_error
                            for quantity in config.response_quantities
                        )
                        secondary_views = tuple(
                            value
                            for value in axis.qualification_view_ids
                            if value != config.estimation_view_id
                        )
                        refinement_complete = bool(secondary_views)
                        refinement_ok = bool(secondary_views)
                        for view_id in secondary_views:
                            refinement = paired_for(calibration_units, view_id)
                            refinement_complete = refinement_complete and set(refinement) == set(
                                calibration_units
                            )
                            if not refinement_complete:
                                refinement_ok = False
                                continue
                            refinement_ok = refinement_ok and all(
                                abs(
                                    _mean(
                                        tuple(
                                            refinement[unit][quantity.quantity_id]
                                            for unit in calibration_units
                                        )
                                    )
                                    - estimates_by_quantity[quantity.quantity_id]
                                )
                                <= quantity.maximum_refinement_absolute_delta
                                for quantity in config.response_quantities
                            )
                        heldout_checks.append(heldout_ok)
                        refinement_checks.append(refinement_ok)
                        for quantity_id, estimate in estimates_by_quantity.items():
                            member_estimates.setdefault(
                                (support, word.word_id, quantity_id), []
                            ).append(estimate)
                        failure_reasons: list[str] = []
                        if not interval_supported:
                            failure_reasons.append("FINITE_ACTION_INTERVAL_INCLUDES_ZERO")
                        if not heldout_complete:
                            failure_reasons.append("FINITE_ACTION_HELDOUT_INCOMPLETE")
                        elif not heldout_ok:
                            failure_reasons.append("FINITE_ACTION_HELDOUT_PREDICTION_FAILED")
                        if not refinement_complete:
                            failure_reasons.append("FINITE_ACTION_REFINEMENT_INCOMPLETE")
                        elif not refinement_ok:
                            failure_reasons.append("FINITE_ACTION_REFINEMENT_UNSTABLE")
                        if not heldout_complete or not refinement_complete:
                            disposition = FiniteActionProductionDisposition.PARTIAL
                        elif failure_reasons:
                            disposition = FiniteActionProductionDisposition.NOT_SUPPORTED
                        else:
                            disposition = FiniteActionProductionDisposition.SUPPORTED
                        reasons = tuple(sorted(failure_reasons))
                    suffix = f"{member}.{candidate_version}.{support}.{word.word_id}".replace(
                        "_", "-"
                    )
                    production_cell = FiniteActionProductionCell(
                        cell_id=f"finite-action-cell.{suffix}",
                        denominator_member_id=member,
                        candidate_version_id=candidate_version,
                        support_cell_id=support,
                        action_word=ObjectIdentity.from_record(word.word_id, word),
                        comparator_action_word_id=comparator_id,
                        complete_physical_unit_ids=complete_units,
                        excluded_physical_unit_ids=excluded_units,
                        response_estimates=tuple(
                            sorted(estimates, key=lambda value: value.value_id)
                        ),
                        lower_bounds=tuple(sorted(lowers, key=lambda value: value.value_id)),
                        upper_bounds=tuple(sorted(uppers, key=lambda value: value.value_id)),
                        sink_operands=tuple(
                            sorted(
                                sink_by_cell.get(key, {}).values(),
                                key=lambda value: value.value_id,
                            )
                        ),
                        disposition=disposition,
                        reason_codes=reasons,
                    )
                    cells.append(production_cell)
                    payload_disposition = {
                        FiniteActionProductionDisposition.SUPPORTED: (
                            FiniteActionCellDisposition.SUPPORTED
                        ),
                        FiniteActionProductionDisposition.NOT_SUPPORTED: (
                            FiniteActionCellDisposition.NOT_SUPPORTED
                        ),
                        FiniteActionProductionDisposition.OUTSIDE_SUPPORT: (
                            FiniteActionCellDisposition.OUTSIDE_SUPPORT
                        ),
                        FiniteActionProductionDisposition.PARTIAL: (
                            FiniteActionCellDisposition.UNEVALUABLE
                        ),
                        FiniteActionProductionDisposition.INVALID: (
                            FiniteActionCellDisposition.UNEVALUABLE
                        ),
                        FiniteActionProductionDisposition.UNEVALUABLE: (
                            FiniteActionCellDisposition.UNEVALUABLE
                        ),
                    }[disposition]
                    payload_entries.append(
                        FiniteActionResponseEntry(
                            entry_id=f"finite-action-entry.{suffix}",
                            denominator_member_id=member,
                            candidate_version_id=candidate_version,
                            qualification_view_ids=axis.qualification_view_ids,
                            support_cell_id=support,
                            action_word=ObjectIdentity.from_record(word.word_id, word),
                            comparator_action_word_ids=(comparator_id,),
                            physical_independent_unit_ids=config.physical_independent_unit_ids,
                            response_values=production_cell.response_estimates,
                            sink_values=(),
                            effort_values=(),
                            uncertainty_values=tuple(
                                sorted(
                                    (
                                        *production_cell.lower_bounds,
                                        *production_cell.upper_bounds,
                                    ),
                                    key=lambda value: value.value_id,
                                )
                            ),
                            disposition=payload_disposition,
                            evidence_link_ids=evidence_link_ids,
                            reason_codes=production_cell.reason_codes,
                        )
                    )

        payload = FiniteActionCompatibilitySetExtension(
            extension_id=f"finite-action-set.{config.config_id}",
            method_semantics=(
                "Member-local finite-word response lookup with complete-unit pairing and "
                "simultaneous intervals; no linearity inference or cross-member pooling."
            ),
            prepared_denominator_id=config.prepared_denominator_id,
            horizon=config.horizon,
            retained_history_ids=config.retained_history_ids,
            action_words=config.action_words,
            physical_independent_unit_ids=config.physical_independent_unit_ids,
            entries=tuple(sorted(payload_entries, key=lambda value: value.entry_id)),
            rejection_entry_ids=tuple(
                sorted(
                    value.entry_id
                    for value in payload_entries
                    if value.disposition is not FiniteActionCellDisposition.SUPPORTED
                )
            ),
            refusal_reason_codes=("finite-action-cell-unavailable",),
        )
        payload_bytes = payload.canonical_bytes()
        artifact = ArtifactIdentity(
            artifact_id=f"artifact.{payload.fingerprint()[:32]}",
            role="law-evaluator-payload",
            payload_schema=payload.SCHEMA,
            sha256=payload.fingerprint(),
            media_type="application/json",
            size_bytes=len(payload_bytes),
        )
        evaluator = ExecutableReference(
            reference_id=f"evaluator.{config.config_id}",
            capability_key="law-evaluator.finite-action",
            capability_version=config.method_version,
            evaluator_key="canonical-finite-action-law",
            payload=artifact,
            payload_format=SafePayloadFormat.CANONICAL_JSON,
            input_schema='empirical-lawhood/methods/law-evaluation-request',
            output_schema='empirical-lawhood/methods/law-evaluation-result',
            deterministic=True,
        )
        implementation = CandidateEvaluatorImplementation(
            implementation_id=f"implementation.{config.config_id}",
            capability_key="law-evaluator.finite-action",
            capability_version=config.method_version,
            evaluator_key=evaluator.evaluator_key,
            source_sha256=config.implementation_sha256,
        )
        publication = self.payload_publisher.publish_candidate_payload(
            payload=payload_bytes,
            evaluator=evaluator,
            implementation=ObjectIdentity.from_record(
                implementation.implementation_id, implementation
            ),
            decoder_schema='empirical-lawhood/methods/finite-action/compatibility-set-decoder',
            decoder_version="1.0.0",
            maximum_decode_bytes=4 * 1024 * 1024,
        )
        complete_cells = tuple(
            value
            for value in cells
            if value.disposition
            in {
                FiniteActionProductionDisposition.SUPPORTED,
                FiniteActionProductionDisposition.NOT_SUPPORTED,
            }
        )
        supported_word_count = sum(
            value.support_status is ActionWordSupportStatus.SUPPORTED
            for value in config.action_words
        )
        expected_complete_cells = (
            len(config.axis_map.bindings) * len(config.support_cell_ids) * supported_word_count
        )
        member_stability_checks = [
            len(values) == len(config.axis_map.bindings)
            and max(values) - min(values)
            <= thresholds[quantity_id].maximum_refinement_absolute_delta
            for (_support, _word, quantity_id), values in member_estimates.items()
        ]
        binary = {
            "recurrence": len(complete_cells) == expected_complete_cells,
            "one-factor-exchange": True,
            "causal-falsifiers": all(
                value.authority in {AuthorityAxis.NOT_REQUIRED, AuthorityAxis.AUTHORIZED}
                for value in extension.terminal_dispositions
            ),
            "history-order-closure": len(
                {value.retained_history_id for value in config.action_words}
            )
            == 1,
            "heldout-prediction": bool(heldout_checks) and all(heldout_checks),
            "member-refinement-stability": bool(refinement_checks)
            and all(refinement_checks)
            and bool(member_stability_checks)
            and all(member_stability_checks),
        }
        metric_ids = {
            "recurrence": "finite-action-recurrence-criterion",
            "one-factor-exchange": "finite-action-one-factor-exchange-criterion",
            "causal-falsifiers": "finite-action-causal-falsifiers-criterion",
            "history-order-closure": "finite-action-history-order-closure-criterion",
            "heldout-prediction": "finite-action-heldout-prediction-criterion",
            "member-refinement-stability": ("finite-action-member-refinement-stability-criterion"),
        }
        receipts = [
            _receipt(
                candidate_id=scaffold.candidate_id,
                suffix=suffix,
                metric_id=metric_ids[suffix],
                observed=value,
                evidence_link_ids=evidence_link_ids,
            )
            for suffix, value in binary.items()
        ]
        uncertainty_availability: dict[str, bool | None] = {
            "uncertainty.numerical": any(
                len(value.qualification_view_ids) > 1 for value in config.axis_map.bindings
            ),
            "uncertainty.aleatoric": len(calibration_units) > 1,
            "uncertainty.epistemic": len(config.axis_map.bindings) > 1
            and bool(member_stability_checks),
            "uncertainty.transport": None,
            "uncertainty.observation": all(
                value.observation_validity is ObservationValidityAxis.VALID
                for value in extension.terminal_dispositions
            ),
        }
        for suffix, available in uncertainty_availability.items():
            receipts.append(
                _receipt(
                    candidate_id=scaffold.candidate_id,
                    suffix=suffix,
                    metric_id=f"finite-action-{suffix.replace('.', '-')}-bound",
                    observed=available,
                    evidence_link_ids=evidence_link_ids,
                )
            )
        extension_binding = ExtensionBinding(
            namespace="finite-action-compatibility-set",
            schema=payload.SCHEMA,
            payload_sha256=publication.content_sha256,
        )
        candidate_evidence = LawCandidateEvidence(
            evidence_id=scaffold.evidence_id,
            candidate_id=scaffold.candidate_id,
            system=scaffold.system,
            dataset_or_projection=config.projection,
            config=ObjectIdentity.from_record(config.config_id, config),
            method_key=config.method_key,
            method_version=config.method_version,
            method_kind=LawMethodKind.NONLINEAR_LOCAL,
            representation_kind=LawRepresentationKind.FINITE_ACTION_OPERATOR,
            candidate_evaluator=evaluator,
            axis_map=scaffold.axis_map,
            claim_unit_binding=scaffold.claim_unit_binding,
            physical_independent_unit_ids=config.physical_independent_unit_ids,
            claim_template=scaffold.claim_template,
            obligation_template=scaffold.obligation_template,
            method_receipts=tuple(sorted(receipts, key=lambda value: value.receipt_id)),
            payload_publication=publication,
            evidence_links=scaffold.evidence_links,
            outcome_access=scaffold.outcome_access,
            parent_visibility_ceilings=scaffold.parent_visibility_ceilings,
            visibility_ceiling=scaffold.visibility_ceiling,
            candidate_extension=extension_binding,
        )
        return FiniteActionIdentificationResult(
            result_id=f"finite-action-result.{config.fingerprint()[:32]}",
            config=ObjectIdentity.from_record(config.config_id, config),
            projection=config.projection,
            projection_extension=config.projection_extension,
            cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
            payload=payload,
            candidate_evidence=candidate_evidence,
        )


__all__ = [
    'FiniteActionCandidateScaffold',
    'FiniteActionComparatorBinding',
    'FiniteActionIdentificationConfig',
    'FiniteActionIdentificationProducer',
    'FiniteActionIdentificationResult',
    'FiniteActionProductionCell',
    "FiniteActionProductionDisposition",
    'FiniteActionResponseQuantity',
    "IndependentUnitResamplingMethod",
]
