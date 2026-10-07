"""Outcome-independent generation of the frozen physical scale morphism truth-case roster."""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import validate_stable_id

from .contracts import PHYSICAL_SCALE_MORPHISM_TRUTH_FIXTURE_IDS, PhysicalScaleMorphismTruthCase, PhysicalScaleMorphismTruthInputDatum, PhysicalScaleMorphismTruthSuiteConfig, PhysicalScaleMorphismTruthVariant, truth_input_sha256


_GENERATOR_ID = "generator.physical-scale-morphism-truth-input"


def _numeric(
    datum_id: str, value: float | int | str | Decimal, unit: str = "1"
) -> PhysicalScaleMorphismTruthInputDatum:
    return PhysicalScaleMorphismTruthInputDatum(
        datum_id=datum_id,
        numeric_value=Decimal(str(value)),
        categorical_value=None,
        unit=unit,
    )


def _category(datum_id: str, value: str) -> PhysicalScaleMorphismTruthInputDatum:
    return PhysicalScaleMorphismTruthInputDatum(
        datum_id=datum_id,
        numeric_value=None,
        categorical_value=value,
        unit="category",
    )


def _vector(prefix: str, values: np.ndarray, unit: str) -> tuple[PhysicalScaleMorphismTruthInputDatum, ...]:
    return tuple(
        _numeric(f"{prefix}-{index:03d}", float(value), unit) for index, value in enumerate(values)
    )


def _truth_input_data(
    fixture_id: str,
    variant: PhysicalScaleMorphismTruthVariant,
    seed: int,
) -> tuple[PhysicalScaleMorphismTruthInputDatum, ...]:
    rng = np.random.Generator(np.random.PCG64(seed))
    noisy = variant is PhysicalScaleMorphismTruthVariant.NOISY_REPEATED_BOARD
    values: list[PhysicalScaleMorphismTruthInputDatum] = []
    if fixture_id in {"exact-receiver-composition", "composition-discrimination"}:
        capacitances = np.linspace(0.8, 1.2, 16)
        if noisy:
            capacitances += rng.normal(0.0, 1e-8, size=16)
        values.extend(_vector("capacitance", capacitances, "F"))
    elif fixture_id == "scale-morphism":
        for n_cells in (16, 32, 64):
            voltages = np.full(n_cells, 0.5)
            if noisy:
                voltages += rng.normal(0.0, 1e-8, size=n_cells)
            values.extend(_vector(f"voltage-n{n_cells}", voltages, "V"))
    elif fixture_id == "saturation-coordinate":
        values.extend(
            (
                _category("saturation-left-label", "low"),
                _category("saturation-right-label", "high"),
            )
        )
        for coordinate in ("a", "d", "h", "r", "tau"):
            values.extend(
                (
                    _category(f"selective-{coordinate}-left-label", "low"),
                    _category(
                        f"selective-{coordinate}-right-label",
                        "low" if coordinate == "r" else "high",
                    ),
                )
            )
    elif fixture_id == "hidden-history":
        values.extend(
            _numeric(datum_id, value)
            for datum_id, value in (
                ("h0-present-defect", "0.001"),
                ("h0-future-defect", "0.4"),
                ("h1-present-defect", "0.001"),
                ("h1-future-defect", "0.001"),
            )
        )
    elif fixture_id == "future-closure":
        values.extend(
            _numeric(datum_id, value)
            for datum_id, value in (
                ("calibration-defect", "0.001"),
                ("observational-defect", "0.001"),
                ("held-future-defect", "0.4"),
                ("interventional-defect", "0.3"),
                ("decision-defect", "0.2"),
            )
        )
    elif fixture_id == "guard-false-safe":
        for datum_id, value in (
            ("fine-gate-0", "FAIL"),
            ("fine-gate-1", "PASS"),
            ("mapped-gate-0", "PASS"),
            ("mapped-gate-1", "PASS"),
        ):
            values.append(_category(datum_id, value))
    elif fixture_id == "hold-state-grammar":
        for datum_id, value in (
            ("safe-gate-0", "PASS"),
            ("safe-gate-1", "PASS"),
            ("unsafe-gate-0", "PASS"),
            ("unsafe-gate-1", "FAIL"),
            ("unqualified-gate-0", "UNEVALUABLE"),
            ("unsafe-mapped-disposition", "HOLD_MEASURED_SAFE"),
        ):
            values.append(_category(datum_id, value))
    elif fixture_id == "energy-mean-collision":
        left = np.tile(np.asarray([0.0, 2.0]), 8)
        right = np.ones(16)
        if noisy:
            left += rng.normal(0.0, 1e-8, size=16)
            right += rng.normal(0.0, 1e-8, size=16)
        values.extend(_vector("left-voltage", left, "V"))
        values.extend(_vector("right-voltage", right, "V"))
    elif fixture_id == "boundary-topology":
        for u_index in range(3):
            for duration_index in range(3):
                values.extend(
                    (
                        _category(
                            f"vertex-{u_index}-{duration_index}-target",
                            "FAIL" if u_index == 0 else "PASS",
                        ),
                        _category(
                            f"vertex-{u_index}-{duration_index}-sink",
                            "FAIL" if duration_index == 0 else "PASS",
                        ),
                    )
                )
    elif fixture_id == "numerical-qualification":
        perturbation = rng.normal(0, 0.01, size=4) if noisy else np.zeros(4)
        values.extend(_vector("capacitance", 0.001 * (1 + perturbation), "F"))
    elif fixture_id == "leakage-firewall":
        values.append(_numeric("protected-evaluation-outcome-count", 1, "count"))
    elif fixture_id == "unit-clock-grouping":
        values.extend(
            (
                _numeric("invalid-time-exponent", 1, "1"),
                _numeric("wrong-voltage-vector-length", 15, "count"),
                _category("invalid-voltage-value", "NONFINITE_NAN"),
            )
        )
    elif fixture_id == "power-and-precision":
        values.extend(
            (
                _numeric("adverse-count", 2, "count"),
                _numeric("adverse-board-count", 8, "count"),
                _numeric("adverse-rate-boundary", "0.05", "1"),
                _numeric("precision-location", 0, "1"),
                _numeric("precision-boundary", "0.01", "1"),
                _numeric("precision-standard-deviation", 1, "1"),
            )
        )
    elif fixture_id == "heterogeneous-local-support":
        for datum_id, value in (
            ("board-01-batch", "batch-a"),
            ("board-01-terminal", "supported"),
            ("board-01-boundary", "boundary-a"),
            ("board-01-gate", "target"),
            ("board-02-batch", "batch-b"),
            ("board-02-terminal", "supported"),
            ("board-02-boundary", "boundary-b"),
            ("board-02-gate", "sink"),
        ):
            values.append(_category(datum_id, value))
        values.extend(
            (
                _numeric("board-01-admitted-fraction", "0.8"),
                _numeric("board-02-admitted-fraction", "0.2"),
            )
        )
    else:  # pragma: no cover - the frozen fixture roster is exhaustive.
        raise ValueError("unknown physical scale morphism truth fixture")
    return tuple(sorted(values, key=lambda value: value.datum_id))


def default_truth_suite_config() -> PhysicalScaleMorphismTruthSuiteConfig:
    return PhysicalScaleMorphismTruthSuiteConfig(
        config_id="physical-scale-morphism-truth-suite",
        fixture_ids=PHYSICAL_SCALE_MORPHISM_TRUTH_FIXTURE_IDS,
        variants=tuple(PhysicalScaleMorphismTruthVariant),
        validation_seeds=(42017, 90173),
        fixtures_frozen_before_validation=True,
        truth_labels_hidden_from_method=True,
        protected_physical_outcome_count=0,
        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
    )


def generate_truth_cases(
    config: PhysicalScaleMorphismTruthSuiteConfig, *, case_id_prefix: str = "case"
) -> tuple[PhysicalScaleMorphismTruthCase, ...]:
    validate_stable_id(case_id_prefix, field_name="case_id_prefix")
    cases = []
    for fixture_id in config.fixture_ids:
        for index, variant in enumerate(config.variants):
            input_data = _truth_input_data(fixture_id, variant, config.validation_seeds[index])
            cases.append(
                PhysicalScaleMorphismTruthCase(
                    case_id=(
                        f"{case_id_prefix}.{fixture_id}."
                        f"{variant.value.lower().replace('_', '-')}"
                    ),
                    fixture_id=fixture_id,
                    variant=variant,
                    seed=config.validation_seeds[index],
                    generator_id=_GENERATOR_ID,
                    input_data=input_data,
                    input_data_sha256=truth_input_sha256(input_data),
                    physical_evidence_count=0,
                )
            )
    return tuple(sorted(cases, key=lambda value: value.case_id))


__all__ = ["default_truth_suite_config", "generate_truth_cases"]
