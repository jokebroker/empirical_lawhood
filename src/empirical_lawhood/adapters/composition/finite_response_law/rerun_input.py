# SPDX-License-Identifier: MPL-2.0
"""Current forecast-reuse inputs, distinct from exposed development preflight.

Roles describe an allocation proposal, never scientific freshness or authority.
The existing numeric census, scientific builders and authenticated parent joins
remain authoritative. No source files, providers or simulator are opened here.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.composition.finite_response_law.assignment import (
    FiniteResponseLawCohortAssignment,
    validate_cohort_assignment,
)
from empirical_lawhood.adapters.composition.finite_response_law.consumer_input import FiniteResponseLawAdditionalParent
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawControlConfig
from empirical_lawhood.adapters.methods.finite_response_law.method_records import FiniteResponseLawAssignedCalibrationMethodConfig
from empirical_lawhood.adapters.methods.finite_response_law.nominated_package import CurrentNominationOperand
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import (
    FiniteResponseLawAssignedCalibrationConfig,
    FiniteResponseLawAssignedEvaluationConfig,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import ArtifactManifest

PUBLIC_CALIBRATION_MASTER_SEED = 706032
PUBLIC_EVALUATION_MASTER_SEED = 706064


@lru_cache(maxsize=1)
def original_development_exposure() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Preserve the original 16 Q2/32 CIR1 roots and actually consumed seed bits.

    All 48 units and 864 PCG64DXSM operands were checked against the original
    canary exclusion metadata (15817753 bytes, SHA256
    20a13c19e9e7b592dfd67cb27afcd7fc2c99259858c62f60bea3d0bc19bdf4d9).
    The recipe is the original prepared source's SHA256/root/purpose rule,
    including bridges and the further passive-probe digest. Historical strings
    are derivation operands, not current schema aliases or current authority.
    Original development requests add their separate PCG64 operands. No RNG
    is instantiated and no source, array or private location is opened here.
    """
    cohorts = (
        ('q', 'q2', 16, '3361fa40ba95753057681d978db9dc0d2c3de254826d96cc6238f5cfbc014a2e'),
        ('e', 'cir1', 32, '93d13c2f2f610a8bcb846ef19c8a7622e80fca03e5488f17e62b5945ffd79b3a'),
    )
    purposes = ('initial-ramp', 'initial-post-ramp', 'parent', 'audit0', 'audit1', 'audit2', 'audit3', 'task', 'passive-probes')
    units, seeds = set(), set()
    for stage, cohort, count, source_digest in cohorts:
        for index in range(count):
            unit = f'cc1-prepared-response-v1.{stage}.prepared.r{index:03d}.seed.{source_digest}'
            units.add(unit)
            for purpose in purposes:
                for bridge in (False,) if purpose == 'passive-probes' else (False, True):
                    purpose_id = f"{unit}.{purpose}{'.bridge' if bridge else ''}"
                    document = f'cc1-prepared-response-v1.sha256-stage-root-purpose.v1\0cc1-prepared-response-v1.seed.{source_digest}\0{purpose_id}\0{0}'
                    digest = sha256(document.encode()).hexdigest()
                    seeds.add(f'seed.pcg64dxsm.{digest[:32]}')
                    if purpose == 'passive-probes':
                        probe = sha256(f'cc1-prepared-response-v1.passive-probes\0{digest}\0q2-n4-fields12'.encode()).hexdigest()
                        seeds.add(f'seed.pcg64dxsm.{probe[:32]}')
            request = sha256(f'cc1-finite-lawhood-v1|20260914|development-request|{cohort}.prepared.r{index:03d}'.encode()).hexdigest()
            seeds.add(f'seed.pcg64.{request[:32]}')
    return tuple(sorted(units)), tuple(sorted(seeds))


@lru_cache(maxsize=1)
def public_rerun_sample_exposure() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reserve public sample numeric streams independently of caller labels."""
    from .assignment import assigned_native_seed_ids, proposed_scientific_seeds
    from empirical_lawhood.adapters.methods.finite_response_law.seed_commitments import FIXED_PASSIVE_PROBE_SEEDS, FIXED_SEEDS
    units, seeds = set(), set()
    for stage, count, master in (('calibration', 32, PUBLIC_CALIBRATION_MASTER_SEED), ('prospective-evaluation', 64, PUBLIC_EVALUATION_MASTER_SEED)):
        assignment = FiniteResponseLawCurrentAllocation(stage, f'empirical-lawhood.finite-response-law.{stage}.public-sample', FiniteResponseLawScienceSpec().plan_sha256, count, 'EXPOSED_DEVELOPMENT_NONPROMOTABLE', proposed_scientific_seeds(stage, master))
        units.update(assignment.physical_unit_ids)
        seeds.update(assigned_native_seed_ids(assignment))
    # Previously distributed fixed numerical operands remain exposed even if a
    # researcher supplies a new cohort namespace around the same integers.
    seeds.update(f'seed.pcg64.{seed:032x}' for row in FIXED_SEEDS.values() for seed in row.values())
    seeds.update(f'seed.pcg64dxsm.{seed >> 128:032x}' for seed in FIXED_PASSIVE_PROBE_SEEDS.values())
    return tuple(sorted(units)), tuple(sorted(seeds))


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCurrentExposure(CanonicalRecord):
    """Explicit known exposure; proposal labels never replace this census."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/current-exposure'
    census_id: str
    excluded_unit_ids: tuple[str, ...]
    excluded_seed_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.census_id)
        for name in ('excluded_unit_ids', 'excluded_seed_ids'):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
            for value in getattr(self, name):
                validate_stable_id(value)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCurrentPublication(CanonicalRecord):
    """One current issued run and its actual protected-outcome grants."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/current-publication'
    run_id: str
    issued_study: ObjectIdentity
    execution_authority: ObjectIdentity
    reveal_authority: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.run_id)
        _validate_current_grant_identities(self.issued_study, self.execution_authority, self.reveal_authority)


def _validate_current_grant_identities(issued, execution, reveal) -> None:
    from empirical_lawhood.planning.study_issue import StudyOperationAuthority
    from empirical_lawhood.runtime.study_issue import IssuedExecutableStudyManifest
    if issued.object_schema != IssuedExecutableStudyManifest.SCHEMA or any(value.object_schema != StudyOperationAuthority.SCHEMA for value in (execution, reveal)) or any(value.object_version != '1.0.0' for value in (issued, execution, reveal)):
        raise ValueError('FINITE_CURRENT_PARENT_ACTUAL_CURRENT_ISSUE_AND_AUTHORITY_IDENTITIES_REQUIRED')


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCurrentParent(CanonicalRecord):
    """Current published outputs and exact receipts, without imported old grants."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/current-parent'
    parent_id: str
    issued_study: ObjectIdentity
    execution_authority: ObjectIdentity
    reveal_authority: ObjectIdentity
    native_source: FiniteResponseLawAssignedCalibrationConfig
    manifests: tuple[ArtifactManifest, ...]
    # The pair identifies an exact successful stored receipt, not its first attempt.
    receipt_bindings: tuple[tuple[str, str, str], ...]
    additional_publications: tuple[FiniteResponseLawCurrentPublication, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.parent_id)
        if type(self.native_source) is not FiniteResponseLawAssignedCalibrationConfig:
            raise ValueError('FINITE_CURRENT_PARENT_ASSIGNED_CALIBRATION_REQUIRED')
        _validate_current_grant_identities(self.issued_study, self.execution_authority, self.reveal_authority)
        if not 1 <= len(self.manifests) <= 32 or not 1 <= len(self.receipt_bindings) <= 4:
            raise ValueError('FINITE_CURRENT_PARENT_INVENTORY_BOUND')
        keys = tuple(value.materialization.materialization_id for value in self.manifests)
        require_sorted_unique_strings(keys, field_name='materialization_ids', allow_empty=False)
        if tuple(sorted(set(self.receipt_bindings))) != self.receipt_bindings:
            raise ValueError('FINITE_CURRENT_PARENT_EXACT_SORTED_RECEIPTS_REQUIRED')
        for row in self.receipt_bindings:
            if not isinstance(row, tuple) or len(row) != 3:
                raise ValueError('FINITE_CURRENT_PARENT_RUN_TASK_RECEIPT_BINDING_REQUIRED')
            for value in row:
                validate_stable_id(value)
        publications = tuple(value.issued_study.object_id for value in self.additional_publications)
        if len(publications) > 3 or publications != tuple(sorted(set(publications))) or self.issued_study.object_id in publications:
            raise ValueError('FINITE_CURRENT_PARENT_DISTINCT_CURRENT_PUBLICATIONS_REQUIRED')


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCurrentResultInput(CanonicalRecord):
    """One selected current result, with its complete receipted sibling outputs."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/current-result-input'
    config_id: str
    publication: FiniteResponseLawCurrentPublication
    manifests: tuple[ArtifactManifest, ...]
    receipt_binding: tuple[str, str, str]
    result_artifact_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        validate_stable_id(self.result_artifact_id)
        if type(self.publication) is not FiniteResponseLawCurrentPublication or not isinstance(self.receipt_binding, tuple) or len(self.receipt_binding) != 3 or self.receipt_binding[0] != self.publication.run_id or not 1 <= len(self.manifests) <= 32:
            raise ValueError('FINITE_CURRENT_RESULT_EXACT_PUBLICATION_AND_RECEIPT_REQUIRED')
        for value in self.receipt_binding:
            validate_stable_id(value)
        keys = tuple(value.materialization.materialization_id for value in self.manifests)
        require_sorted_unique_strings(keys, field_name='materialization_ids', allow_empty=False)
        if sum(value.logical.logical_artifact_id == self.result_artifact_id for value in self.manifests) != 1:
            raise ValueError('FINITE_CURRENT_RESULT_EXACT_SELECTED_OUTPUT_REQUIRED')


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCurrentAllocation(FiniteResponseLawCohortAssignment):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/current-allocation'

    def __post_init__(self) -> None:
        validate_cohort_assignment(
            self, allowed_roles=('PROPOSED_UNRUN', 'EXPOSED_DEVELOPMENT_NONPROMOTABLE')
        )
        # The retained probe owner consumes only the leading 128 bits. Preserve
        # each full 256-bit commitment, but do not count low-bit changes as a
        # distinct independent stream.
        probes = tuple(seed >> 128 for purpose, _, seed in self.scientific_seeds if purpose == 'passive-probes')
        native = {seed for purpose, _, seed in self.scientific_seeds if purpose != 'passive-probes'}
        if len(set(probes)) != len(probes) or set(probes) & native:
            raise ValueError('FINITE_CURRENT_ALLOCATION_EFFECTIVE_PROBE_STREAM_COLLISION')


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRerunInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/rerun-input'

    config_id: str
    stage: str
    allocation: FiniteResponseLawCurrentAllocation
    nomination: CurrentNominationOperand
    original_prior_unit_ids: tuple[str, ...]
    original_prior_seed_ids: tuple[str, ...]
    primary_parent_sha256: str | None = None
    method: FiniteResponseLawAssignedCalibrationMethodConfig | None = None
    control: FiniteResponseLawControlConfig | None = None
    additional_parents: tuple[FiniteResponseLawAdditionalParent, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name='config_id')
        if type(self.allocation) is not FiniteResponseLawCurrentAllocation:
            raise ValueError('FINITE_RERUN_CURRENT_ALLOCATION_REQUIRED')
        if not isinstance(self.nomination, CurrentNominationOperand):
            raise ValueError('FINITE_RERUN_FIXED_NOMINATION_REQUIRED')
        if self.stage not in ('calibration', 'calibration-method', 'prospective-evaluation'):
            raise ValueError('FINITE_RERUN_UNSUPPORTED_STAGE')
        expected_stage = 'prospective-evaluation' if self.stage == 'prospective-evaluation' else 'calibration'
        if self.allocation.stage != expected_stage:
            raise ValueError('FINITE_RERUN_STAGE_ALLOCATION_MISMATCH')
        for name in ('original_prior_unit_ids', 'original_prior_seed_ids'):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        original_units, original_seeds = original_development_exposure()
        if not set(original_units) <= set(self.original_prior_unit_ids) or not set(original_seeds) <= set(self.original_prior_seed_ids):
            raise ValueError('FINITE_RERUN_ORIGINAL_48_ROOT_EFFECTIVE_EXPOSURE_REQUIRED')
        if len(self.additional_parents) > 8:
            raise ValueError('FINITE_RERUN_PARENT_LIMIT')
        if (self.stage == 'calibration-method') != (self.method is not None) or (self.stage == 'prospective-evaluation') != (self.control is not None):
            raise ValueError('FINITE_RERUN_STAGE_OPERANDS_MISMATCH')
        if self.stage == 'calibration':
            if self.primary_parent_sha256 is not None or self.additional_parents:
                raise ValueError('FINITE_RERUN_CALIBRATION_HAS_NO_NATIVE_PARENT')
        elif self.primary_parent_sha256 is None:
            raise ValueError('FINITE_RERUN_AUTHENTICATED_PARENT_REQUIRED')
        else:
            validate_sha256(self.primary_parent_sha256, field_name='primary_parent_sha256')
        source = self.source
        if (
            source.cohort_namespace != self.allocation.cohort_namespace
            or source.assignment_sha256 != self.allocation.fingerprint()
            or source.scientific_seeds != self.allocation.scientific_seeds
            or tuple(root.physical_unit_id for root in source.roots) != self.allocation.physical_unit_ids
        ):
            raise ValueError('FINITE_RERUN_ACTUAL_SOURCE_ALLOCATION_MISMATCH')
        if self.method is not None and (
            self.method.development_report != self.nomination.development_report
            or self.method.coefficients != self.nomination.coefficients
            or self.method.development_manifest != self.nomination.development_manifest
        ):
            raise ValueError('FINITE_RERUN_METHOD_CHANGED_FROZEN_NOMINATION')
        from .exposure import native_seed_collisions, native_seed_ids
        source_seeds = native_seed_ids(source)
        if set(self.allocation.physical_unit_ids) & set(original_units) or native_seed_collisions(source_seeds, original_seeds):
            raise ValueError('FINITE_RERUN_ORIGINAL_DEVELOPMENT_ALLOCATION_COLLISION')
        if self.evidence_role == 'PROPOSED_UNRUN':
            sample_units, sample_seeds = public_rerun_sample_exposure()
            if set(self.allocation.physical_unit_ids) & set(sample_units) or native_seed_collisions(source_seeds, sample_seeds):
                raise ValueError('FINITE_RERUN_PUBLIC_SAMPLE_CANNOT_BECOME_UNEXPOSED')

    @property
    def source(self) -> FiniteResponseLawAssignedCalibrationConfig | FiniteResponseLawAssignedEvaluationConfig:
        if self.method is not None:
            return self.method.native_source
        if self.control is not None:
            return self.control.source
        from empirical_lawhood.adapters.simulators.finite_response_law.calibration.discovery import SOURCE_CAPABILITY
        return FiniteResponseLawAssignedCalibrationConfig(
            'calibration', FiniteResponseLawScienceSpec(),
            ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY),
            None, (), self.allocation.cohort_namespace,
            self.allocation.fingerprint(), self.allocation.scientific_seeds,
        )

    @property
    def evidence_role(self) -> str:
        return self.allocation.evidence_role

    @property
    def completion(self) -> None:
        return None

    @property
    def retention(self) -> None:
        return None

    @property
    def screen(self) -> None:
        return None

    @property
    def prospective_eligibility(self) -> None:
        return None

    @property
    def runtime_context(self) -> None:
        # Current runtime authority is joined after issue, not a circular input
        # prerequisite for compiling an outcome-blind candidate.
        return None
