"""Deterministic construction and compatibility checks for projection truth."""

from __future__ import annotations

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.identification_evidence import (
    IdentificationEvidenceManifest,
    IdentificationEvidenceProjection,
)
from empirical_lawhood.planning.identification_evidence_extensions import ActionOccurrenceBinding, ComputabilityEffectCoordinate, ComputabilityEffectEntry, EpisodeTerminalDisposition, IdentificationEvidenceProjectionExtension, IdentificationMethodEvidenceKind, ProjectionCompatibilityStatus, ProjectionExtensionCompatibilityReceipt, ProjectionTruthRequirement

from .contracts import IdentificationDataset


_BASE_EXACT_FIELDS = frozenset(
    {
        "action.word.identity",
        "episode.roster",
        "independent-unit.roster",
        "manifest.identity",
        "projection.identity",
    }
)
_EXTENSION_FIELDS = frozenset(
    {
        "action.stage.clocks",
        "action.stage.receipts",
        "action.stage.values",
        "effect.computability",
        "terminal.authority",
        "terminal.numerical-validity",
        "terminal.observation-validity",
        "terminal.physical-sink",
        "terminal.scientific-status",
        "terminal.sink-operands",
    }
)
_ALL_FIELDS = _BASE_EXACT_FIELDS | _EXTENSION_FIELDS


def method_projection_truth_requirement(
    method_kind: IdentificationMethodEvidenceKind,
) -> ProjectionTruthRequirement:
    fields: tuple[str, ...]
    if method_kind is IdentificationMethodEvidenceKind.POINT_METHOD:
        fields = ("action.word.identity", "effect.computability")
    elif method_kind is IdentificationMethodEvidenceKind.FINITE_ACTION:
        fields = tuple(sorted(_ALL_FIELDS))
    else:
        fields = tuple(sorted(_ALL_FIELDS - {"terminal.sink-operands"}))
    return ProjectionTruthRequirement(
        requirement_id=f"projection-truth.{method_kind.value.lower().replace('_', '-')}",
        method_kind=method_kind,
        required_field_ids=fields,
    )


def build_identification_projection_extension(
    *,
    extension_id: str,
    manifest: IdentificationEvidenceManifest,
    projection: IdentificationEvidenceProjection,
    action_words: tuple[OccurrenceActionWord, ...],
    action_occurrences: tuple[ActionOccurrenceBinding, ...],
    terminal_dispositions: tuple[EpisodeTerminalDisposition, ...],
    declared_effect_coordinates: tuple[ComputabilityEffectCoordinate, ...],
    computability_entries: tuple[ComputabilityEffectEntry, ...],
) -> IdentificationEvidenceProjectionExtension:
    """Validate native classifications without authoring a proof verdict."""

    manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
    if projection.manifest != manifest_identity:
        raise ValueError("projection extension received a foreign manifest")
    projected = {value.projected_observation_id: value for value in projection.observations}
    if {value.projected_observation_id for value in terminal_dispositions} != set(projected):
        raise ValueError("terminal disposition roster differs from projected observations")
    words = {value.word_id: value for value in action_words}
    if len(words) != len(action_words):
        raise ValueError("projection extension received duplicate action words")
    for binding in action_occurrences:
        observation = projected.get(binding.projected_observation_id)
        if observation is None:
            raise ValueError("action occurrence binds a foreign projected observation")
        if observation.action_word != binding.action_word:
            raise ValueError("action occurrence substitutes the projected action word")
        try:
            word = words[binding.action_word.object_id]
        except KeyError as error:
            raise ValueError("action occurrence names an unavailable action word") from error
        if binding.action_word != ObjectIdentity.from_record(word.word_id, word):
            raise ValueError("action occurrence changes the exact action word identity")
        occurrences = {value.occurrence_id: value for value in word.occurrences}
        expected = occurrences.get(binding.occurrence.occurrence_id)
        if expected is None or expected != binding.occurrence:
            raise ValueError("action occurrence substitutes action content")
    return IdentificationEvidenceProjectionExtension(
        extension_id=extension_id,
        projection=ObjectIdentity.from_record(projection.projection_id, projection),
        manifest=manifest_identity,
        entered_episode_ids=tuple(sorted(value.episode_id for value in terminal_dispositions)),
        independent_unit_ids=projection.claim_unit_binding.independent_unit_instance_ids,
        action_occurrences=tuple(sorted(action_occurrences, key=lambda value: value.binding_id)),
        terminal_dispositions=tuple(
            sorted(terminal_dispositions, key=lambda value: value.episode_id)
        ),
        declared_effect_coordinates=tuple(
            sorted(declared_effect_coordinates, key=lambda value: value.coordinate_id)
        ),
        computability_entries=tuple(
            sorted(computability_entries, key=lambda value: value.entry_id)
        ),
    )


def assess_projection_extension_compatibility(
    *,
    receipt_id: str,
    extension: IdentificationEvidenceProjectionExtension,
    target_schema: str,
    consumer_id: str,
    required_field_ids: tuple[str, ...],
) -> ProjectionExtensionCompatibilityReceipt:
    """Describe exact lowering and refuse a consumer that needs lost truth."""

    exact: frozenset[str]
    lost: frozenset[str]
    if target_schema == IdentificationEvidenceProjectionExtension.SCHEMA:
        exact = _ALL_FIELDS
        lost = frozenset()
    elif target_schema in {
        IdentificationEvidenceProjection.SCHEMA,
        IdentificationDataset.SCHEMA,
    }:
        exact = _BASE_EXACT_FIELDS
        lost = _EXTENSION_FIELDS
    else:
        raise ValueError("projection extension target schema is not registered")
    unknown = set(required_field_ids) - _ALL_FIELDS
    if unknown:
        raise ValueError(f"consumer requests unknown projection truth fields: {sorted(unknown)!r}")
    incompatible = bool(set(required_field_ids) & lost)
    status = (
        ProjectionCompatibilityStatus.INCOMPATIBLE_REQUIRED_FIELD
        if incompatible
        else ProjectionCompatibilityStatus.COMPATIBLE_WITH_DECLARED_LOSS
        if lost
        else ProjectionCompatibilityStatus.EXACT
    )
    return ProjectionExtensionCompatibilityReceipt(
        receipt_id=receipt_id,
        extension=ObjectIdentity.from_record(extension.extension_id, extension),
        target_schema=target_schema,
        consumer_id=consumer_id,
        required_field_ids=tuple(sorted(required_field_ids)),
        exact_field_ids=tuple(sorted(exact)),
        lost_field_ids=tuple(sorted(lost)),
        status=status,
    )


def require_method_projection_truth(
    *,
    extension: IdentificationEvidenceProjectionExtension,
    method_kind: IdentificationMethodEvidenceKind,
) -> ProjectionExtensionCompatibilityReceipt:
    requirement = method_projection_truth_requirement(method_kind)
    receipt = assess_projection_extension_compatibility(
        receipt_id=f"compatibility.{requirement.requirement_id}",
        extension=extension,
        target_schema=IdentificationEvidenceProjectionExtension.SCHEMA,
        consumer_id=requirement.requirement_id,
        required_field_ids=requirement.required_field_ids,
    )
    if receipt.status is not ProjectionCompatibilityStatus.EXACT:
        raise ValueError("method projection truth is not exact")
    return receipt


__all__ = [
    "assess_projection_extension_compatibility",
    'build_identification_projection_extension',
    "method_projection_truth_requirement",
    "require_method_projection_truth",
]
