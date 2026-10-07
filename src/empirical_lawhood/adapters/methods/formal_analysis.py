"""Reusable standard algebra/calculus/geometry/dynamics analysis methods.

The four domain methods share bounded numerical helpers but retain distinct
registered capabilities and prerequisite-aware dispatch.  Missing mathematical
objects produce typed ``UNEVALUABLE`` rows; they are never replaced by a proxy.
"""

from __future__ import annotations

from dataclasses import fields, replace
from decimal import Decimal
import hashlib
from typing import Final

import numpy as np

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.formal_analysis import (
    FormalMethodBinding,
    FormalMethodCatalog,
    FormalMethodRole,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalDomain,
    FormalGapCoverage,
    FormalGapCoverageDisposition,
    FormalGapRegister,
)
from empirical_lawhood.planning.formal_results import (
    FormalAnalysisInputManifest,
    FormalAnalysisSample,
    FormalDomainAnalysisResult,
    FormalDomainMethodConfig,
    FormalGapAdjudication,
    FormalGapAdjudicationPanel,
    FormalGapResultStatus,
    FormalMultiplicityResult,
    FormalPanelEvaluatorConfig,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.providers import CapabilityOutputSemanticContract


FORMAL_METHOD_VERSION: Final = "1.0.0"
FORMAL_DOMAIN_CAPABILITY_KEYS: Final = {
    FormalDomain.ALGEBRA: "formal.algebra.standard",
    FormalDomain.CALCULUS: "formal.calculus.standard",
    FormalDomain.GEOMETRY: "formal.geometry.standard",
    FormalDomain.DYNAMICS: "formal.dynamics.standard",
}
FORMAL_PANEL_CAPABILITY_KEY: Final = "formal.panel.evaluator"
_CONFIG_SCHEMA_SHA256: Final = hashlib.sha256(
    FormalDomainMethodConfig.SCHEMA.encode("ascii")
).hexdigest()
_PANEL_CONFIG_SCHEMA_SHA256: Final = hashlib.sha256(
    FormalPanelEvaluatorConfig.SCHEMA.encode("ascii")
).hexdigest()


def _decimal(value: float | int) -> Decimal:
    return Decimal(f"{float(value):.15g}")


def _metric(value_id: str, value: float | int, unit: str = "1") -> NamedDecimal:
    return NamedDecimal(value_id=value_id, value=_decimal(value), unit=unit)


def _values(
    sample: FormalAnalysisSample,
    attribute: str,
    coordinate_ids: tuple[str, ...],
) -> np.ndarray:
    observed = {value.value_id: float(value.value) for value in getattr(sample, attribute)}
    if not set(coordinate_ids).issubset(observed):
        raise ValueError(f"formal sample omits {attribute} coordinates")
    return np.asarray([observed[value] for value in coordinate_ids], dtype=np.float64)


def _valid_samples(
    manifest: FormalAnalysisInputManifest,
    *,
    unit_id: str | None = None,
) -> tuple[FormalAnalysisSample, ...]:
    return tuple(
        value
        for value in manifest.samples
        if value.valid and (unit_id is None or value.independent_unit_id == unit_id)
    )


def _xy(
    manifest: FormalAnalysisInputManifest,
    samples: tuple[FormalAnalysisSample, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if len(samples) < 2:
        raise ValueError("formal method requires at least two valid observations")
    x = np.stack(
        [_values(value, "realized_actions", manifest.action_coordinate_ids) for value in samples]
    )
    y = np.stack(
        [_values(value, "receiver_values", manifest.receiver_coordinate_ids) for value in samples]
    )
    return _normalize_columns(x), _normalize_columns(y)


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    """Return offset- and scale-invariant dimensionless coordinates."""

    if values.ndim != 2 or not values.size:
        raise ValueError("formal coordinate normalization requires a nonempty matrix")
    center = np.mean(values, axis=0)
    centered = values - center
    scale = np.ptp(values, axis=0)
    fallback = np.max(np.abs(centered), axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, fallback)
    scale = np.where(
        scale > np.finfo(np.float64).eps,
        scale,
        np.ones_like(scale),
    )
    return np.asarray(centered / scale, dtype=np.float64)


def _normalized_receivers(
    manifest: FormalAnalysisInputManifest,
    samples: tuple[FormalAnalysisSample, ...],
) -> dict[str, np.ndarray]:
    raw = np.stack(
        [_values(value, "receiver_values", manifest.receiver_coordinate_ids) for value in samples]
    )
    normalized = _normalize_columns(raw)
    return {sample.sample_id: normalized[index] for index, sample in enumerate(samples)}


def _linear_metrics(
    manifest: FormalAnalysisInputManifest,
    samples: tuple[FormalAnalysisSample, ...],
) -> tuple[float, float]:
    x, y = _xy(manifest, samples)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    scale = max(float(np.linalg.norm(y)), np.finfo(np.float64).eps)
    relative_rmse = float(np.linalg.norm(residual) / scale)
    rank = float(np.linalg.matrix_rank(coefficients[1:, :]))
    return relative_rmse, rank


def _word_mean(
    manifest: FormalAnalysisInputManifest,
    samples: tuple[FormalAnalysisSample, ...],
    word_id: str,
    normalized_receivers: dict[str, np.ndarray],
) -> np.ndarray | None:
    selected = [
        normalized_receivers[value.sample_id]
        for value in samples
        if value.action_word_id == word_id
    ]
    return None if not selected else np.mean(np.stack(selected), axis=0)


def _word_residual(
    manifest: FormalAnalysisInputManifest,
    samples: tuple[FormalAnalysisSample, ...],
    words: tuple[str, ...],
) -> float:
    normalized = _normalized_receivers(manifest, samples)
    means = [_word_mean(manifest, samples, value, normalized) for value in words]
    if any(value is None for value in means):
        raise ValueError(f"formal action-word controls are absent: {words}")
    arrays = [value for value in means if value is not None]
    if len(arrays) == 2:
        return float(np.linalg.norm(arrays[0] - arrays[1]))
    return float(np.linalg.norm(arrays[0] + arrays[1] - arrays[2]))


def _view_convergence(
    manifest: FormalAnalysisInputManifest,
    samples: tuple[FormalAnalysisSample, ...],
) -> float:
    if len(manifest.numerical_view_ids) < 2:
        raise ValueError("formal convergence requires at least two numerical views")
    normalized = _normalized_receivers(manifest, samples)
    by_key: dict[tuple[str, str, str, Decimal], dict[str, np.ndarray]] = {}
    for value in samples:
        key = (
            value.independent_unit_id,
            value.action_word_id,
            value.clock_id,
            value.clock_coordinate,
        )
        by_key.setdefault(key, {})[value.numerical_view_id] = normalized[value.sample_id]
    coarse, refined = manifest.numerical_view_ids[:2]
    differences = [
        float(np.linalg.norm(values[coarse] - values[refined]))
        for values in by_key.values()
        if coarse in values and refined in values
    ]
    if not differences:
        raise ValueError("formal convergence views have no common observation keys")
    return float(np.max(differences))


def _transition_metrics(
    manifest: FormalAnalysisInputManifest,
    samples: tuple[FormalAnalysisSample, ...],
) -> tuple[float, float, float, float]:
    if not manifest.state_coordinate_ids:
        raise ValueError("formal dynamics requires declared state coordinates")
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for unit_id in manifest.preparation_unit_ids:
        unit = sorted(
            (value for value in samples if value.independent_unit_id == unit_id),
            key=lambda value: (value.clock_id, value.clock_coordinate, value.sample_id),
        )
        for left, right in zip(unit, unit[1:], strict=False):
            state = _values(left, "state_values", manifest.state_coordinate_ids)
            next_state = _values(right, "state_values", manifest.state_coordinate_ids)
            action = _values(left, "realized_actions", manifest.action_coordinate_ids)
            pairs.append((state, action, next_state))
    if len(pairs) < 3:
        raise ValueError("formal dynamics requires at least three time-aligned pairs")
    raw_states = np.stack([value[0] for value in pairs])
    raw_actions = np.stack([value[1] for value in pairs])
    raw_next_states = np.stack([value[2] for value in pairs])
    state_pair = _normalize_columns(np.vstack((raw_states, raw_next_states)))
    states = state_pair[: len(raw_states)]
    next_states = state_pair[len(raw_states) :]
    actions = _normalize_columns(raw_actions)
    design = np.column_stack((np.ones(len(states)), states, actions))
    coefficients = np.linalg.lstsq(design, next_states, rcond=None)[0]
    state_dimension = states.shape[1]
    a = coefficients[1 : state_dimension + 1, :].T
    b = coefficients[state_dimension + 1 :, :].T
    residual = next_states - design @ coefficients
    scale = max(float(np.linalg.norm(next_states)), np.finfo(np.float64).eps)
    relative_rmse = float(np.linalg.norm(residual) / scale)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    nonnormality = float(np.linalg.norm(a.T @ a - a @ a.T))
    controllability = b
    current = b
    for _ in range(1, state_dimension):
        current = a @ current
        controllability = np.column_stack((controllability, current))
    controllability_rank = float(np.linalg.matrix_rank(controllability))
    return relative_rmse, spectral_radius, nonnormality, controllability_rank


def _matrix_metric(
    manifest: FormalAnalysisInputManifest,
    role_ids: tuple[str, ...],
) -> float:
    selected = [value for value in manifest.declared_matrices if value.role_id in role_ids]
    if not selected:
        raise ValueError(f"formal declared structure is absent: {role_ids}")
    return float(
        max(
            np.linalg.norm(
                np.asarray([float(item) for item in value.values]).reshape(
                    len(value.row_ids),
                    len(value.column_ids),
                )
            )
            for value in selected
        )
    )


def _unit_metric(
    gap_id: str,
    manifest: FormalAnalysisInputManifest,
    samples: tuple[FormalAnalysisSample, ...],
) -> tuple[float, str, bool]:
    """Return metric, metric ID, and whether lower values support the question."""

    if gap_id == "gap.algebra.identity-inverse-repetition":
        return (
            _word_residual(manifest, samples, ("action.positive", "action.negative")),
            "formal.algebra.inverse-residual",
            True,
        )
    if gap_id == "gap.algebra.signed-opposition":
        return (
            _word_residual(
                manifest,
                samples,
                ("action.positive", "action.negative", "action.identity"),
            ),
            "formal.algebra.signed-opposition-residual",
            True,
        )
    word_pairs = {
        "gap.algebra.sequential-composition": (
            "action.sequential",
            "action.composed-reference",
        ),
        "gap.algebra.simultaneous-composition": (
            "action.simultaneous",
            "action.additive-reference",
        ),
        "gap.algebra.associativity-defect": (
            "action.parenthesization-left",
            "action.parenthesization-right",
        ),
        "gap.algebra.order-commutator": ("action.ab", "action.ba"),
        "gap.algebra.semigroup-embedding": (
            "action.repeat",
            "action.semigroup-reference",
        ),
        "gap.dynamics.hysteresis-return": (
            "action.return",
            "action.identity",
        ),
    }
    if gap_id in word_pairs:
        return (
            _word_residual(manifest, samples, word_pairs[gap_id]),
            f"formal.{gap_id.removeprefix('gap.').replace('.', '-')}-residual",
            True,
        )
    if gap_id in {
        "gap.algebra.restriction-gluing-transport",
        "gap.algebra.quotient-lumpability",
        "gap.algebra.falsifier-preservation",
        "gap.geometry.information-geometry",
        "gap.geometry.metric-structure",
        "gap.geometry.topology-restriction",
        "gap.geometry.cohomology",
        "gap.geometry.admission-margins",
        "gap.geometry.reachability-viability",
        "gap.dynamics.gramians",
        "gap.dynamics.preservation-barriers",
    }:
        return (
            _matrix_metric(
                manifest,
                (
                    gap_id,
                    f"structure.{gap_id.removeprefix('gap.')}",
                ),
            ),
            f"formal.{gap_id.removeprefix('gap.')}-matrix-norm",
            False,
        )
    if gap_id == "gap.calculus.numerical-view-convergence":
        return (
            _view_convergence(manifest, samples),
            "formal.calculus.maximum-view-difference",
            True,
        )
    if gap_id.startswith("gap.dynamics."):
        relative_rmse, spectral_radius, nonnormality, controllability_rank = _transition_metrics(
            manifest, samples
        )
        values = {
            "gap.dynamics.causal-cones-clock-transport": (
                relative_rmse,
                "formal.dynamics.clock-aligned-residual",
                True,
            ),
            "gap.dynamics.delay-relaxation": (
                relative_rmse,
                "formal.dynamics.delay-model-residual",
                True,
            ),
            "gap.dynamics.recurrence-stationarity": (
                relative_rmse,
                "formal.dynamics.recurrence-residual",
                True,
            ),
            "gap.dynamics.state-closure-memory": (
                relative_rmse,
                "formal.dynamics.state-closure-residual",
                True,
            ),
            "gap.dynamics.local-evolution-operator": (
                relative_rmse,
                "formal.dynamics.operator-residual",
                True,
            ),
            "gap.dynamics.koopman-operator": (
                relative_rmse,
                "formal.dynamics.koopman-residual",
                True,
            ),
            "gap.dynamics.sparse-evolution": (
                relative_rmse,
                "formal.dynamics.sparse-operator-residual",
                True,
            ),
            "gap.dynamics.stability-transient": (
                spectral_radius,
                "formal.dynamics.spectral-radius",
                True,
            ),
            "gap.dynamics.nonnormal-amplification": (
                nonnormality,
                "formal.dynamics.nonnormality",
                True,
            ),
            "gap.dynamics.controllability-observability": (
                controllability_rank,
                "formal.dynamics.controllability-rank",
                False,
            ),
        }
        if gap_id in values:
            return values[gap_id]
    relative_rmse, rank = _linear_metrics(manifest, samples)
    if gap_id in {
        "gap.calculus.local-jacobian-rank",
        "gap.geometry.tangent-rank",
        "gap.geometry.support-charts-atlas",
        "gap.geometry.receiver-fibers",
        "gap.geometry.decision-quotients",
        "gap.geometry.symmetry-gauge",
        "gap.geometry.boundary-strata",
    }:
        return rank, f"formal.{gap_id.removeprefix('gap.')}-rank", False
    return (
        relative_rmse,
        f"formal.{gap_id.removeprefix('gap.')}-relative-residual",
        True,
    )


def _classify(
    metrics: tuple[float, ...],
    *,
    lower_supports: bool,
    config: FormalDomainMethodConfig,
) -> FormalGapResultStatus:
    if lower_supports:
        statuses = [
            (
                FormalGapResultStatus.SUPPORTED
                if value <= float(config.convergence_tolerance)
                else FormalGapResultStatus.NOT_SUPPORTED
            )
            for value in metrics
        ]
    else:
        statuses = [
            (
                FormalGapResultStatus.SUPPORTED
                if value > float(config.zero_tolerance)
                else FormalGapResultStatus.NOT_SUPPORTED
            )
            for value in metrics
        ]
    return statuses[0] if len(set(statuses)) == 1 else FormalGapResultStatus.MIXED


def run_formal_domain_analysis(
    *,
    register: FormalGapRegister,
    manifest: FormalAnalysisInputManifest,
    config: FormalDomainMethodConfig,
    estimator_family_id: str,
) -> FormalDomainAnalysisResult:
    """Run one prerequisite-aware standard formal-domain method."""

    if manifest.domain is not config.domain:
        raise ValueError("formal input and method domains differ")
    if manifest.formal_gap_register != ObjectIdentity.from_record(
        register.register_id,
        register,
    ):
        raise ValueError("formal input binds another register")
    gaps = tuple(value for value in register.gaps if value.domain is config.domain)
    if config.gap_ids != tuple(value.gap_id for value in gaps):
        raise ValueError("formal domain config must bind its complete register family")
    rows: list[FormalGapAdjudication] = []
    for gap in gaps:
        unit_metrics: list[float] = []
        metric_id: str | None = None
        lower_supports = True
        supported_units: list[str] = []
        reasons: set[str] = set()
        for unit_id in manifest.preparation_unit_ids:
            samples = _valid_samples(manifest, unit_id=unit_id)
            observed_operands = {item for value in samples for item in value.operand_ids}
            observed_prerequisites = {item for value in samples for item in value.prerequisite_ids}
            observed_controls = {item for value in samples for item in value.control_ids}
            if not set(gap.required_operand_ids).issubset(observed_operands):
                reasons.add("FORMAL_REQUIRED_OPERAND_ABSENT")
                continue
            if not set(gap.support_prerequisite_ids).issubset(observed_prerequisites):
                reasons.add("FORMAL_SUPPORT_PREREQUISITE_ABSENT")
                continue
            if not set(gap.control_ids).issubset(observed_controls):
                reasons.add("FORMAL_PREDECLARED_CONTROL_ABSENT")
                continue
            try:
                value, current_metric_id, current_lower_supports = _unit_metric(
                    gap.gap_id,
                    manifest,
                    samples,
                )
            except (ValueError, np.linalg.LinAlgError):
                reasons.add("FORMAL_MATHEMATICAL_PREREQUISITE_UNAVAILABLE")
                continue
            if metric_id is not None and (
                current_metric_id != metric_id or current_lower_supports is not lower_supports
            ):
                raise ValueError("formal method changed estimand across units")
            metric_id = current_metric_id
            lower_supports = current_lower_supports
            unit_metrics.append(value)
            supported_units.append(unit_id)
        if not unit_metrics or len(unit_metrics) < gap.minimum_independent_units:
            rows.append(
                FormalGapAdjudication(
                    gap_id=gap.gap_id,
                    domain=gap.domain,
                    status=FormalGapResultStatus.UNEVALUABLE,
                    estimator_family_id=estimator_family_id,
                    metrics=(),
                    independent_unit_ids=tuple(sorted(supported_units)),
                    numerical_view_ids=manifest.numerical_view_ids,
                    decisive_falsifier_ids=gap.decisive_falsifier_ids,
                    reason_codes=tuple(
                        sorted(
                            {
                                *reasons,
                                "FORMAL_INDEPENDENT_UNIT_REQUIREMENT_NOT_MET",
                            }
                        )
                    ),
                    maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
                    outcome_access=manifest.outcome_access,
                )
            )
            continue
        assert metric_id is not None
        mean = float(np.mean(unit_metrics))
        uncertainty = (
            0.0
            if len(unit_metrics) == 1
            else float(np.std(unit_metrics, ddof=1) / np.sqrt(len(unit_metrics)))
        )
        status = _classify(
            tuple(unit_metrics),
            lower_supports=lower_supports,
            config=config,
        )
        rows.append(
            FormalGapAdjudication(
                gap_id=gap.gap_id,
                domain=gap.domain,
                status=status,
                estimator_family_id=estimator_family_id,
                metrics=tuple(
                    sorted(
                        (
                            _metric(metric_id, mean),
                            _metric("formal.independent-unit-standard-error", uncertainty),
                        ),
                        key=lambda value: value.value_id,
                    )
                ),
                independent_unit_ids=tuple(sorted(supported_units)),
                numerical_view_ids=manifest.numerical_view_ids,
                decisive_falsifier_ids=(
                    gap.decisive_falsifier_ids
                    if status is FormalGapResultStatus.NOT_SUPPORTED
                    else ()
                ),
                reason_codes=(),
                maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
                outcome_access=manifest.outcome_access,
            )
        )
    result_seed = (
        manifest.fingerprint(),
        config.fingerprint(),
        tuple(value.fingerprint() for value in rows),
    )
    result_digest = hashlib.sha256(repr(result_seed).encode("ascii")).hexdigest()
    return FormalDomainAnalysisResult(
        result_id=f"formal-domain-result.{result_digest[:32]}",
        domain=config.domain,
        input_manifest=ObjectIdentity.from_record(manifest.input_id, manifest),
        config=ObjectIdentity.from_record(config.config_id, config),
        rows=tuple(rows),
        outcome_access=manifest.outcome_access,
    )


def assemble_formal_gap_panel(
    *,
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
    domain_results: tuple[FormalDomainAnalysisResult, ...],
    configs: tuple[FormalDomainMethodConfig, ...],
) -> FormalGapAdjudicationPanel:
    """Reveal one exact 48-row panel from four sealed domain results."""

    if {value.domain for value in domain_results} != set(FormalDomain):
        raise ValueError("formal panel assembly requires four domain results")
    result_rows = {row.gap_id: row for value in domain_results for row in value.rows}
    assignments = {value.gap_id: value for value in coverage.assignments}
    gaps = {value.gap_id: value for value in register.gaps}
    rows: list[FormalGapAdjudication] = []
    for gap_id, gap in gaps.items():
        assignment = assignments[gap_id]
        if assignment.disposition is FormalGapCoverageDisposition.TEST_IN_THIS_ACT:
            row = result_rows.get(gap_id)
            if row is None:
                raise ValueError("tested formal gap lacks its domain result")
            rows.append(
                replace(
                    row,
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                )
            )
            continue
        status = (
            FormalGapResultStatus.NOT_APPLICABLE
            if assignment.disposition
            in {
                FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR,
                FormalGapCoverageDisposition.OUT_OF_SCOPE_WORLD,
            }
            else FormalGapResultStatus.UNEVALUABLE
        )
        rows.append(
            FormalGapAdjudication(
                gap_id=gap_id,
                domain=gap.domain,
                status=status,
                estimator_family_id=None,
                metrics=(),
                independent_unit_ids=(),
                numerical_view_ids=(),
                decisive_falsifier_ids=(),
                reason_codes=assignment.reason_codes,
                maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            )
        )
    config_by_domain = {value.domain: value for value in configs}
    multiplicity = tuple(
        FormalMultiplicityResult(
            family_id=next(
                value.multiplicity_family_id for value in register.gaps if value.domain is domain
            ),
            domain=domain,
            hypothesis_count=sum(value.domain is domain for value in register.gaps),
            familywise_alpha=config_by_domain[domain].familywise_alpha,
            per_hypothesis_alpha=(
                config_by_domain[domain].familywise_alpha
                / Decimal(sum(value.domain is domain for value in register.gaps))
            ),
            method_id="multiplicity.bonferroni-exact",
        )
        for domain in FormalDomain
    )
    panel_seed = (
        register.fingerprint(),
        coverage.fingerprint(),
        tuple(value.fingerprint() for value in rows),
    )
    panel_digest = hashlib.sha256(repr(panel_seed).encode("ascii")).hexdigest()
    return FormalGapAdjudicationPanel(
        panel_id=f"formal-gap-panel.{panel_digest[:32]}",
        formal_gap_register=ObjectIdentity.from_record(
            register.register_id,
            register,
        ),
        formal_gap_coverage=ObjectIdentity.from_record(
            coverage.coverage_id,
            coverage,
        ),
        domain_results=tuple(
            sorted(
                (ObjectIdentity.from_record(value.result_id, value) for value in domain_results),
                key=lambda value: value.object_id,
            )
        ),
        rows=tuple(sorted(rows, key=lambda value: value.gap_id)),
        multiplicity=tuple(sorted(multiplicity, key=lambda value: value.family_id)),
        maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


def formal_method_capability_manifests() -> tuple[CapabilityManifest, ...]:
    """Return four domain methods plus the sole formal-panel evaluator."""

    budget = ResourceBudget(
        cpu_cores=2,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=900,
        source_scan_bytes=2 * 1024**3,
        output_bytes=16 * 1024**2,
    )
    domain_manifests = [
        CapabilityManifest(
            capability_key=key,
            capability_version=FORMAL_METHOD_VERSION,
            kind=CapabilityKind.ANALYSIS,
            config_schema=FormalDomainMethodConfig.SCHEMA,
            config_schema_sha256=_CONFIG_SCHEMA_SHA256,
            input_schema_ids=(FormalAnalysisInputManifest.SCHEMA,),
            output_schema_ids=(FormalDomainAnalysisResult.SCHEMA,),
            permissions=tuple(
                sorted(
                    (
                        CapabilityPermission.READ_DEVELOPMENT,
                        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                    ),
                    key=lambda value: value.value,
                )
            ),
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            resource_ceiling=budget,
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                f"formal-{domain.value.lower()}-truth-known",
                "formal-independent-unit-multiplicity",
                "formal-missing-prerequisite-unevaluable",
            ),
            implementation_sha256=hashlib.sha256(
                f"formal-domain-method:{domain.value}:1.0.0".encode("ascii")
            ).hexdigest(),
        )
        for domain, key in FORMAL_DOMAIN_CAPABILITY_KEYS.items()
    ]
    evaluator = CapabilityManifest(
        capability_key=FORMAL_PANEL_CAPABILITY_KEY,
        capability_version=FORMAL_METHOD_VERSION,
        kind=CapabilityKind.EVALUATOR,
        config_schema=FormalPanelEvaluatorConfig.SCHEMA,
        config_schema_sha256=_PANEL_CONFIG_SCHEMA_SHA256,
        input_schema_ids=(
            FormalAnalysisInputManifest.SCHEMA,
            FormalDomainAnalysisResult.SCHEMA,
        ),
        output_schema_ids=(FormalGapAdjudicationPanel.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_SEALED_OUTCOMES,
                    CapabilityPermission.REVEAL_OUTCOMES,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                ),
                key=lambda value: value.value,
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        resource_ceiling=budget,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "formal-complete-48-row-panel",
            "formal-panel-evaluator-only-reveal",
        ),
        implementation_sha256=hashlib.sha256(b"formal-panel-evaluator:1.0.0").hexdigest(),
    )
    return tuple(sorted((*domain_manifests, evaluator), key=lambda value: value.registry_id))


def standard_formal_method_catalog(
    register: FormalGapRegister,
) -> FormalMethodCatalog:
    manifests = {value.capability_key: value for value in formal_method_capability_manifests()}
    bindings: list[FormalMethodBinding] = []
    for domain in FormalDomain:
        gaps = tuple(value for value in register.gaps if value.domain is domain)
        estimator_ids = {item for value in gaps for item in value.estimator_family_ids}
        multiplicity_ids = {value.multiplicity_family_id for value in gaps}
        capability = manifests[FORMAL_DOMAIN_CAPABILITY_KEYS[domain]]
        for role, family_ids in (
            (FormalMethodRole.ESTIMATOR, estimator_ids),
            (FormalMethodRole.MULTIPLICITY, multiplicity_ids),
        ):
            for family_id in sorted(family_ids):
                bindings.append(
                    FormalMethodBinding(
                        family_id=family_id,
                        role=role,
                        capability_key=capability.capability_key,
                        capability_version=capability.capability_version,
                        implementation_sha256=capability.implementation_sha256,
                        supported_gap_ids=tuple(value.gap_id for value in gaps),
                        input_schema_id=FormalAnalysisInputManifest.SCHEMA,
                        output_schema_id=FormalDomainAnalysisResult.SCHEMA,
                        maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
                        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                    )
                )
    return FormalMethodCatalog(
        catalog_id="formal-method-catalog.standard",
        bindings=tuple(sorted(bindings, key=lambda value: value.binding_id)),
    )


def standard_formal_method_configs(
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
) -> tuple[
    tuple[FormalDomainMethodConfig, ...],
    FormalPanelEvaluatorConfig,
]:
    """Return one closed configuration for every standard formal capability."""

    configs = tuple(
        FormalDomainMethodConfig(
            config_id=f"config.formal.{domain.value.lower()}.domain-method",
            domain=domain,
            gap_ids=tuple(value.gap_id for value in register.gaps if value.domain is domain),
            zero_tolerance=Decimal("1e-12"),
            materiality_threshold=Decimal("1e-6"),
            convergence_tolerance=Decimal("0.05"),
            familywise_alpha=Decimal("0.05"),
            method_version=FORMAL_METHOD_VERSION,
            maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        )
        for domain in FormalDomain
    )
    panel = FormalPanelEvaluatorConfig(
        config_id="config.formal.panel.standard",
        formal_gap_register=ObjectIdentity.from_record(
            register.register_id,
            register,
        ),
        formal_gap_coverage=ObjectIdentity.from_record(
            coverage.coverage_id,
            coverage,
        ),
        domain_config_ids=tuple(sorted(value.config_id for value in configs)),
        evaluator_version=FORMAL_METHOD_VERSION,
        maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
    )
    return configs, panel


class FormalCapabilityConfigDecoder:
    """Exact canonical decoder for one statically registered formal config."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        config: (
            FormalDomainMethodConfig
            | FormalPanelEvaluatorConfig
            | tuple[FormalDomainMethodConfig | FormalPanelEvaluatorConfig, ...]
        ),
    ) -> None:
        self.manifest = manifest
        self.configs = config if isinstance(config, tuple) else (config,)
        if not self.configs:
            raise ValueError("formal config decoder requires an accepted config")

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != self.manifest.config_schema:
            raise ValueError("formal config schema differs from its manifest")
        record_type = (
            FormalPanelEvaluatorConfig
            if self.manifest.capability_key == FORMAL_PANEL_CAPABILITY_KEY
            else FormalDomainMethodConfig
        )
        observed = decode_canonical_bytes(
            payload,
            record_type,
            maximum_bytes=1024 * 1024,
        )
        if observed not in self.configs:
            raise ValueError("formal config bytes differ from the static registration")


class FormalDomainTaskRunner:
    """Path-free runner for one registered development-domain method."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        register: FormalGapRegister,
        config: FormalDomainMethodConfig,
    ) -> None:
        self.manifest = manifest
        self.register = register
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        inputs = tuple(
            value
            for value in context.input_ports
            if value.payload_schema == FormalAnalysisInputManifest.SCHEMA
        )
        if len(inputs) != 1:
            raise ValueError("formal domain runner requires one input manifest")
        value = inputs[0]
        input_manifest = decode_canonical_bytes(
            value.read(),
            FormalAnalysisInputManifest,
            maximum_bytes=value.size_bytes,
        )
        if value.bytes_read != value.size_bytes:
            raise ValueError("formal domain runner did not consume its complete input")
        family_id = next(
            item
            for gap in self.register.gaps
            if gap.domain is self.config.domain
            for item in gap.estimator_family_ids
        )
        result = run_formal_domain_analysis(
            register=self.register,
            manifest=input_manifest,
            config=self.config,
            estimator_family_id=family_id,
        )
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("formal-domain-canonical-result", True, ()),),
        )


class FormalPanelTaskRunner:
    """Sole reveal runner; recompute all four methods on sealed evaluation data."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        register: FormalGapRegister,
        coverage: FormalGapCoverage,
        configs: tuple[FormalDomainMethodConfig, ...],
    ) -> None:
        self.manifest = manifest
        self.register = register
        self.coverage = coverage
        self.configs = configs

    def execute(self, context: TaskContext) -> RunnerResult:
        development_results: list[FormalDomainAnalysisResult] = []
        evaluation_inputs: list[FormalAnalysisInputManifest] = []
        for port in context.input_ports:
            payload = port.read()
            if port.bytes_read != port.size_bytes:
                raise ValueError("formal panel did not consume a complete input")
            if port.payload_schema == FormalDomainAnalysisResult.SCHEMA:
                development_results.append(
                    decode_canonical_bytes(
                        payload,
                        FormalDomainAnalysisResult,
                        maximum_bytes=port.size_bytes,
                    )
                )
            elif port.payload_schema == FormalAnalysisInputManifest.SCHEMA:
                evaluation_inputs.append(
                    decode_canonical_bytes(
                        payload,
                        FormalAnalysisInputManifest,
                        maximum_bytes=port.size_bytes,
                    )
                )
        if {value.domain for value in development_results} != set(FormalDomain):
            raise ValueError("formal panel lacks four frozen development results")
        if {value.domain for value in evaluation_inputs} != set(FormalDomain):
            raise ValueError("formal panel lacks four sealed evaluation manifests")
        configs = {value.domain: value for value in self.configs}
        evaluation_results = tuple(
            run_formal_domain_analysis(
                register=self.register,
                manifest=input_manifest,
                config=configs[input_manifest.domain],
                estimator_family_id=next(
                    item
                    for gap in self.register.gaps
                    if gap.domain is input_manifest.domain
                    for item in gap.estimator_family_ids
                ),
            )
            for input_manifest in sorted(
                evaluation_inputs,
                key=lambda value: value.domain.value,
            )
        )
        panel = assemble_formal_gap_panel(
            register=self.register,
            coverage=self.coverage,
            domain_results=evaluation_results,
            configs=self.configs,
        )
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=panel.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck("formal-panel-complete-48-row", len(panel.rows) == 48, ()),
                ReceiptCheck("formal-panel-sealed-evaluation-recomputed", True, ()),
            ),
        )


def formal_task_runners(
    *,
    registry: CapabilityRegistry,
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
    configs: tuple[FormalDomainMethodConfig, ...],
) -> tuple[TaskRunner, ...]:
    configs_by_domain = {value.domain: value for value in configs}
    runners: list[TaskRunner] = []
    for domain, key in FORMAL_DOMAIN_CAPABILITY_KEYS.items():
        runners.append(
            FormalDomainTaskRunner(
                registry.resolve(key, FORMAL_METHOD_VERSION),
                register,
                configs_by_domain[domain],
            )
        )
    runners.append(
        FormalPanelTaskRunner(
            registry.resolve(FORMAL_PANEL_CAPABILITY_KEY, FORMAL_METHOD_VERSION),
            register,
            coverage,
            configs,
        )
    )
    return tuple(sorted(runners, key=lambda value: value.manifest.registry_id))


def formal_output_semantic_contracts(
    manifests: tuple[CapabilityManifest, ...],
) -> tuple[CapabilityOutputSemanticContract, ...]:
    """Return exact canonical-envelope validators for formal outputs."""

    value_keys = {
        FormalDomainAnalysisResult.SCHEMA: tuple(
            sorted(field.name for field in fields(FormalDomainAnalysisResult))
        ),
        FormalGapAdjudicationPanel.SCHEMA: tuple(
            sorted(field.name for field in fields(FormalGapAdjudicationPanel))
        ),
    }
    return tuple(
        CapabilityOutputSemanticContract.from_manifest(
            manifest,
            payload_schema=payload_schema,
            profile=ArtifactProfile.CANONICAL_JSON,
            top_level_keys=("schema", "value", "version"),
            value_keys=value_keys[payload_schema],
        )
        for manifest in manifests
        for payload_schema in manifest.output_schema_ids
        if payload_schema in value_keys
    )


__all__ = [
    "FORMAL_DOMAIN_CAPABILITY_KEYS",
    "FORMAL_METHOD_VERSION",
    "FORMAL_PANEL_CAPABILITY_KEY",
    "assemble_formal_gap_panel",
    "formal_method_capability_manifests",
    "formal_output_semantic_contracts",
    "formal_task_runners",
    "run_formal_domain_analysis",
    "standard_formal_method_configs",
    "standard_formal_method_catalog",
    "FormalCapabilityConfigDecoder",
    "FormalDomainTaskRunner",
    "FormalPanelTaskRunner",
]
