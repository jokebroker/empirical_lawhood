"""Finite held-source prediction, without invented action or physical-unit semantics.

Split groups constrain model access. They do not assert biological independence,
prospective exposure, response laws or controller efficacy. Model/feature science
belongs to the selected adapter; this carrier owns partitions and chronology.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)


class RetrospectiveExposure(StrEnum):
    DEVELOPMENT_EXPOSED = "DEVELOPMENT_EXPOSED"
    CURATOR_UNINSPECTED = "CURATOR_UNINSPECTED_NOT_NEW_BIOLOGY"
    SYNTHETIC = "SYNTHETIC_SOFTWARE_CONFORMANCE"


class RetrospectivePredictionRole(StrEnum):
    TRAINING_PROJECTION = "TRAINING_PROJECTION"
    FEATURE_PROJECTION = "FEATURE_PROJECTION"
    FIT = "FIT"
    CALIBRATE = "CALIBRATE"
    PREDICT = "PREDICT"
    COMMIT = "COMMIT"
    REVEAL = "REVEAL"
    EVALUATE = "EVALUATE"


class RetrospectivePredictionInputRole(StrEnum):
    TRAINING = "TRAINING"
    CALIBRATION = "CALIBRATION"
    TARGET_FEATURES = "TARGET_FEATURES"
    TARGET_LABELS = "TARGET_LABELS"


@dataclass(frozen=True, slots=True)
class RetrospectivePredictionSample(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-prediction-sample'
    sample_id: str
    source_materialization: ObjectIdentity
    source_record_locator: str
    split_group_ids: tuple[str, ...]
    shared_context_ids: tuple[str, ...]
    physical_independent_unit_id: str | None
    exposure: RetrospectiveExposure

    def __post_init__(self) -> None:
        validate_stable_id(self.sample_id, field_name="sample_id")
        validate_nonempty(self.source_record_locator, field_name="source_record_locator")
        if len(self.source_record_locator) > 1024:
            raise ValueError("native record locator exceeds its bound")
        for name in ("split_group_ids", "shared_context_ids"):
            values = getattr(self, name)
            require_sorted_unique_strings(
                values, field_name=name, allow_empty=name == "shared_context_ids"
            )
            if len(values) > 16:
                raise ValueError("historical sample has too many grouping coordinates")
            for value in values:
                validate_stable_id(value, field_name=name)
        if self.physical_independent_unit_id is not None:
            validate_stable_id(self.physical_independent_unit_id, field_name="physical_unit")
        if not isinstance(self.exposure, RetrospectiveExposure):
            raise ValueError("historical sample requires explicit exposure")


@dataclass(frozen=True, slots=True)
class RetrospectivePredictionFold(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-prediction-fold'
    fold_id: str
    training_sample_ids: tuple[str, ...]
    calibration_sample_ids: tuple[str, ...]
    target_sample_ids: tuple[str, ...]
    parent_anchor_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fold_id, field_name="fold_id")
        names = (
            "training_sample_ids",
            "calibration_sample_ids",
            "target_sample_ids",
            "parent_anchor_sample_ids",
        )
        for name in names:
            values = getattr(self, name)
            require_sorted_unique_strings(
                values,
                field_name=name,
                allow_empty=name in ("calibration_sample_ids", "parent_anchor_sample_ids"),
            )
            for value in values:
                validate_stable_id(value, field_name=name)
        training, calibration, target = (
            set(self.training_sample_ids),
            set(self.calibration_sample_ids),
            set(self.target_sample_ids),
        )
        if (
            training & calibration
            or training & target
            or calibration & target
            or not set(self.parent_anchor_sample_ids) <= training
        ):
            raise ValueError("historical split leaks samples or an unavailable parent anchor")


@dataclass(frozen=True, slots=True)
class RetrospectivePredictionInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-prediction-input'
    input_id: str
    fold_id: str
    role: RetrospectivePredictionInputRole
    artifact: ArtifactIdentity
    sample_ids: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(self.fold_id, field_name="fold_id")
        require_sorted_unique_strings(self.sample_ids, field_name="sample_ids", allow_empty=False)
        if (
            not isinstance(self.role, RetrospectivePredictionInputRole)
            or type(self.artifact.size_bytes) is not int
            or not 0 < self.artifact.size_bytes <= 16 * 1024**2
            or self.outcome_access
            not in (
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
            )
        ):
            raise ValueError("historical input exceeds its bounded held-source contract")
        if self.role is not RetrospectivePredictionInputRole.TARGET_FEATURES and (
            self.outcome_access is OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("label-bearing historical input cannot be outcome-blind")


@dataclass(frozen=True, slots=True)
class RetrospectivePredictionOwner(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-prediction-owner'
    role: RetrospectivePredictionRole
    owner: ObjectIdentity
    config: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            not isinstance(self.role, RetrospectivePredictionRole)
            or self.owner.object_schema != 'empirical-lawhood/runtime/capability-manifest'
        ):
            raise ValueError("historical stage requires a typed role and capability owner")


@dataclass(frozen=True, slots=True)
class RetrospectivePredictionExperiment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-prediction-experiment'
    extension_set_id: str
    experiment_id: str
    science_specification: ObjectIdentity
    evidence_profile_selection: ObjectIdentity
    source_pipeline_profile: ObjectIdentity
    samples: tuple[RetrospectivePredictionSample, ...]
    folds: tuple[RetrospectivePredictionFold, ...]
    inputs: tuple[RetrospectivePredictionInput, ...]
    owners: tuple[RetrospectivePredictionOwner, ...]
    maximum_evidence_ceiling: EvidenceCeiling = EvidenceCeiling.NON_PROMOTABLE
    grants_authority: bool = False
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND
    visibility_ceiling: VisibilityCeiling = VisibilityCeiling.PROSPECTIVE
    claim_scope: str = "HISTORICAL_PREDICTION_NOT_LAW_CONTROL_OR_NEW_BIOLOGY"

    def __post_init__(self) -> None:
        for name in ("extension_set_id", "experiment_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.evidence_profile_selection.object_schema
            != 'empirical-lawhood/planning/evidence-profile-selection'
            or self.source_pipeline_profile.object_schema
            != 'empirical-lawhood/planning/source-pipeline-profile'
        ):
            raise ValueError("historical prediction requires exact evidence/source roots")
        for name, attr in (
            ("samples", "sample_id"),
            ("folds", "fold_id"),
            ("inputs", "input_id"),
            ("owners", "role"),
        ):
            require_sorted_unique_ids(getattr(self, name), attribute=attr, field_name=name)
        if (
            not 2 <= len(self.samples) <= 4096
            or not 1 <= len(self.folds) <= 16
            or not 3 <= len(self.inputs) <= 64
            or {o.role for o in self.owners} != set(RetrospectivePredictionRole)
        ):
            raise ValueError("historical prediction omits roles or exceeds its finite roster")
        if (
            self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.claim_scope != "HISTORICAL_PREDICTION_NOT_LAW_CONTROL_OR_NEW_BIOLOGY"
        ):
            raise ValueError("historical carrier cannot promote evidence or grant authority")
        samples = {s.sample_id: s for s in self.samples}
        native_rows = {(s.source_materialization, s.source_record_locator) for s in self.samples}
        if len(native_rows) != len(samples):
            raise ValueError("historical sample IDs duplicate a native source record")
        owners = {o.role: o for o in self.owners}
        if owners[RetrospectivePredictionRole.EVALUATE].owner in {
            owners[RetrospectivePredictionRole.FIT].owner,
            owners[RetrospectivePredictionRole.CALIBRATE].owner,
            owners[RetrospectivePredictionRole.PREDICT].owner,
        }:
            raise ValueError("prediction and terminal evaluator require separate owners")
        folds = {fold.fold_id: fold for fold in self.folds}
        expected_roles = {
            RetrospectivePredictionInputRole.TRAINING,
            RetrospectivePredictionInputRole.TARGET_FEATURES,
            RetrospectivePredictionInputRole.TARGET_LABELS,
        }
        used: set[str] = set()
        for fold in self.folds:
            partitions = (
                fold.training_sample_ids,
                fold.calibration_sample_ids,
                fold.target_sample_ids,
            )
            if any(sample not in samples for part in partitions for sample in part):
                raise ValueError("fold selects a sample outside the declared roster")
            used.update(sample for part in partitions for sample in part)
            groups = [
                set(group for sample in part for group in samples[sample].split_group_ids)
                for part in partitions
            ]
            if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
                raise ValueError("historical partition splits an indivisible group")
            physical = [
                set(samples[s].physical_independent_unit_id for s in part) - {None}
                for part in partitions
            ]
            if physical[0] & physical[1] or physical[0] & physical[2] or physical[1] & physical[2]:
                raise ValueError("historical partition splits a known physical unit")
            selected = [item for item in self.inputs if item.fold_id == fold.fold_id]
            roles = expected_roles | (
                {RetrospectivePredictionInputRole.CALIBRATION}
                if fold.calibration_sample_ids
                else set()
            )
            if {item.role for item in selected} != roles or len(selected) != len(roles):
                raise ValueError("fold lacks its exact data-role partition")
            expected_samples = {
                RetrospectivePredictionInputRole.TRAINING: fold.training_sample_ids,
                RetrospectivePredictionInputRole.CALIBRATION: fold.calibration_sample_ids,
                RetrospectivePredictionInputRole.TARGET_FEATURES: fold.target_sample_ids,
                RetrospectivePredictionInputRole.TARGET_LABELS: fold.target_sample_ids,
            }
            for item in selected:
                if item.sample_ids != expected_samples[item.role]:
                    raise ValueError("historical input changes its assigned sample roster")
                if any(
                    samples[s].exposure is RetrospectiveExposure.DEVELOPMENT_EXPOSED
                    for s in item.sample_ids
                ) and not item.visibility_ceiling.is_at_least_as_restrictive_as(
                    VisibilityCeiling.OUTCOME_VISIBLE
                ):
                    raise ValueError("historical input conceals prior development exposure")
            labels = next(
                i.artifact
                for i in selected
                if i.role is RetrospectivePredictionInputRole.TARGET_LABELS
            )
            if any(
                i.role is not RetrospectivePredictionInputRole.TARGET_LABELS
                and (
                    i.artifact.artifact_id == labels.artifact_id
                    or i.artifact.sha256 == labels.sha256
                )
                for i in selected
            ):
                raise ValueError("target label artifact aliases a method input")
        if used != set(samples) or any(i.fold_id not in folds for i in self.inputs):
            raise ValueError("historical census contains unused samples or unknown folds")

    @property
    def config_identities(self) -> tuple[ObjectIdentity, ...]:
        return tuple(sorted({owner.config for owner in self.owners}, key=lambda x: x.object_id))
