# SPDX-License-Identifier: MPL-2.0
"""Original typed operands are checked before any history-bundle publication."""

from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
import importlib

import pytest

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import CapabilityConfigRef
from empirical_lawhood.runtime.execution import TaskContext, WorkerIsolationProfile, WorkerOutputPort


FAMILIES = (
    ("split_cohort_history_budget", "SplitCohortHistoryBudget"),
    ("history_budget_phase_diagram", "HistoryBudgetPhaseDiagram"),
)
LONG = Decimal("1.23456789012345678901234567890123456789")
HASH = "1" * 64


@dataclass(frozen=True)
class DecimalOperandRecord(CanonicalRecord):
    SCHEMA = "empirical-lawhood/testing/fixtures/decimal-custody"
    value: Decimal


def history_fixture(family, residual):
    name, prefix = family
    contracts = importlib.import_module(f"empirical_lawhood.adapters.{name}.contracts")
    runtime = importlib.import_module(f"empirical_lawhood.adapters.{name}.runtime_contracts")
    provider = importlib.import_module(f"empirical_lawhood.adapters.{name}.runtime_provider")
    rank_type = getattr(contracts, prefix + "RankStep")
    forecast_type = getattr(contracts, prefix + "HistoryRankForecast")
    forecast_steps = tuple(rank_type(
        depth=depth, effective_rank=1, algebraic_comparator_rank=1, row_count_ceiling=1,
        largest_singular_value=Decimal(1), smallest_retained_relative_singular_value=Decimal(1),
        condition_number=Decimal(1), nonmonotone_effective_rank=False,
    ) for depth in range(32))
    forecasts = tuple(forecast_type(
        forecast_id=f"forecast-{index}", descriptor_sha256=HASH, rank_steps=forecast_steps,
        singular_spectra_sha256=HASH, k_full_effective=0, effective_rank_right_censored=False,
        probe_depths=(0,), algebraic_prediction_state=getattr(contracts, prefix + "ScientificState").SUPPORTED,
    ) for index in range(2))
    coordinate = getattr(contracts, prefix + "CoordinateLabel")(
        coordinate_id="coordinate-a", kind=getattr(contracts, prefix + "CoordinateKind").ABSOLUTE_DEPTH,
        scale_cells=16, depth=0, budget=None, resolution_epsilon=Decimal("0.1"), primary=True,
    )
    structural = getattr(contracts, prefix + "StructuralRankStep")(
        coordinate_id="coordinate-a", structural_rank=1, row_count=1, state_count=1,
        support_sha256=HASH, algorithm_id="synthetic-structural",
    )
    discrete = getattr(contracts, prefix + "DiscreteRankBracket")(
        coordinate_id="coordinate-a", lower_rank=1, upper_rank=1, state_count=1,
        precision_bits=384, residual_norm=residual, separation_ratio=None, matrix_sha256=HASH,
        compatibility=getattr(contracts, prefix + "RankCompatibility").COMPATIBLE,
        algorithm_id="synthetic-discrete",
    )
    conditioning = getattr(contracts, prefix + "ConditioningStep")(
        coordinate_id="coordinate-a", resolution_epsilon=Decimal("0.1"),
        largest_singular_value=Decimal(1), smallest_singular_value=Decimal(1),
        condition_number=Decimal(1), stable_rank=Decimal(1), effective_rank=1,
        right_censored=False, algorithm_id="synthetic-conditioning",
    )
    bundle = getattr(runtime, prefix + "HistoryBundle")(
        bundle_id="bundle-a", unit_id="unit-a", denominator_bundle_sha256=HASH,
        forecasts=forecasts, coordinates=(coordinate,), structural_rank_steps=(structural,),
        discrete_rank_brackets=(discrete,), conditioning_steps=(conditioning,),
        arrays_manifest_sha256=HASH, outcome_count=0,
    )
    runner = object.__new__(getattr(provider, prefix + "Runner"))
    runner.execution_count = 0
    runner._read = lambda context: None
    runner._dispatch = lambda context, inputs: {"bundle": bundle}
    context = TaskContext(
        run_id="synthetic-run", task_id="history", attempt_id="synthetic-run.history.attempt-001",
        config=CapabilityConfigRef("config-a", "empirical-lawhood/testing/fixtures/config", HASH, HASH, "config-artifact"),
        input_bindings=(), input_ports=(),
        output_ports=(WorkerOutputPort("history.bundle", bundle.SCHEMA, ArtifactProfile.CANONICAL_JSON, "application/json"),),
        permissions=(), outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resource_budget=ResourceBudget(1, 128 * 1024**2, 0, 10, 0, 1024**2),
        isolation_profile=WorkerIsolationProfile.TRUSTED_LOCAL,
    )
    return provider, runner, context, bundle


class CollectingEmitter:
    def __init__(self):
        self.writes = []

    def write(self, output_id, payload):
        self.writes.append((output_id, payload))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("streaming", [False, True])
def test_corrupted_bundle_refuses_before_any_output(family, streaming, monkeypatch):
    with localcontext() as context:
        context.prec = 28
        _provider, runner, task, bundle = history_fixture(family, LONG)
        # Inject a lossy writer independently of the corrected central codec.
        rounded = replace(bundle, discrete_rank_brackets=(
            replace(bundle.discrete_rank_brackets[0], residual_norm=Decimal("1.234567890123456789012345679")),
        )).canonical_bytes()
        monkeypatch.setattr(type(bundle), "canonical_bytes", lambda self: rounded)
        emitter = CollectingEmitter()
        with pytest.raises(ValueError, match=r"Decimal operand.*discrete_rank_brackets\[0\]\.residual_norm"):
            runner.execute_streaming(task, emitter) if streaming else runner.execute(task)
        assert emitter.writes == []
        assert bundle.discrete_rank_brackets[0].residual_norm == LONG


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("residual", [Decimal("0.1000"), Decimal("-0"), LONG])
@pytest.mark.parametrize("precision", [7, 28, 80])
def test_lossless_bundle_keeps_exact_payload_and_streaming_checks(family, residual, precision, monkeypatch):
    with localcontext() as context:
        context.prec = precision
        _provider, runner, task, bundle = history_fixture(family, residual)
        expected = bundle.canonical_bytes()
        calls = 0
        method = type(bundle).canonical_bytes

        def count_original(self):
            nonlocal calls
            if self is bundle:
                calls += 1
            return method(self)

        monkeypatch.setattr(type(bundle), "canonical_bytes", count_original)
        result = runner.execute(task)
        assert result.outputs[0].payload == expected
        assert calls == 1
        emitter = CollectingEmitter()
        checks = runner.execute_streaming(task, emitter)
        assert b"".join(payload for _output, payload in emitter.writes) == expected
        assert checks == result.checks
        assert calls == 2


@pytest.mark.parametrize("family", FAMILIES)
def test_aggregate_ceiling_precedes_guard_and_preserves_array_reference(family, monkeypatch):
    provider, runner, task, bundle = history_fixture(family, Decimal("0.1"))
    array = b"synthetic array bytes"
    runner._dispatch = lambda context, inputs: {"arrays": array, "bundle": bundle}
    task = replace(task, output_ports=(
        WorkerOutputPort("history.arrays", "empirical-lawhood/testing/fixtures/array", ArtifactProfile.RAW_SOURCE_BYTES, "application/octet-stream"),
        *task.output_ports,
    ))
    total = len(array) + len(bundle.canonical_bytes())
    exact = replace(task, resource_budget=replace(task.resource_budget, output_bytes=total))
    result = runner.execute(exact)
    assert result.outputs[0].payload is array
    monkeypatch.setattr(provider, "require_decimal_operands_preserved", lambda *args, **kwargs: pytest.fail("guard exceeded the aggregate ceiling"))
    too_small = replace(task, resource_budget=replace(task.resource_budget, output_bytes=total - 1))
    emitter = CollectingEmitter()
    with pytest.raises(ValueError, match="output byte budget"):
        runner.execute_streaming(too_small, emitter)
    assert emitter.writes == []


@pytest.mark.parametrize("family", FAMILIES)
def test_nested_rank_residual_round_trip_is_exact_at_default_precision(family):
    with localcontext() as context:
        context.prec = 28
        _provider, _runner, _task, bundle = history_fixture(family, LONG)
        payload = bundle.canonical_bytes()
        decoded = decode_canonical_bytes(payload, type(bundle), maximum_bytes=len(payload))
        assert decoded.discrete_rank_brackets[0].residual_norm == LONG


def test_long_operands_survive_archive_and_native_authoring_writers(tmp_path):
    from types import SimpleNamespace
    from empirical_lawhood.api.native_authoring import _write_records
    from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive

    record = DecimalOperandRecord(LONG)
    with localcontext() as context:
        context.prec = 7
        bundle = SimpleNamespace(authoring=record, payloads=(record,), decoder_registrations=(record,))
        author, payloads, decoders = _write_records(tmp_path, bundle, record)
        archive = CanonicalRecordArchive.pack("long-operand", record)
        for payload in (archive.unpack(), author.read_bytes(), payloads[0].read_bytes(),
                        decoders[0].read_bytes(), (tmp_path / "exposure-inspection.json").read_bytes()):
            decoded = decode_canonical_bytes(payload, DecimalOperandRecord, maximum_bytes=len(payload))
            assert decoded.value == LONG
            assert decoded.fingerprint() == record.fingerprint()
