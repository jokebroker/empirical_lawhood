"""Projection of the exact nine-schedule preparation-policy native panel."""

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.prepared_response.projection import NativeResponseArrays, reduce_native_response_arrays
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig, FiniteResponseLawPreparationPolicyRoot, PREPARATION_POLICY_SCHEDULE_IDS, preparation_policy_native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_instruments import preparation_policy_compact_interface
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_source import FiniteResponseLawPreparationPolicyNativeCheckpoint
from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawNativePhaseData
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_source_outputs import FiniteResponseLawPreparationPolicyNativeTaskResult, decode_preparation_policy_task_native
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_force_components
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, require_sorted_unique_ids


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-projection-config'
    native_spec: FiniteResponseLawPreparationPolicyNativeConfig

    @property
    def config_id(self) -> str:
        return f"{self.native_spec.spec_id}.projection"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyPanelCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-panel-cell'
    schedule_id: str
    pair_index: int
    future: str
    refinement: int
    values: tuple[Decimal | None, ...]
    observed: bool
    valid: bool
    selected_source: ObjectIdentity
    opposite_source: ObjectIdentity
    hold_source: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            self.schedule_id not in PREPARATION_POLICY_SCHEDULE_IDS
            or type(self.pair_index) is not int
            or self.pair_index not in range(4)
            or self.future not in ("future-1", "future-2")
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
            or len(self.values) != 8
            or type(self.observed) is not bool
            or type(self.valid) is not bool
            or any(
                value is not None
                and (not isinstance(value, Decimal) or not value.is_finite())
                for value in self.values
            )
            or self.valid
            and (not self.observed or any(value is None for value in self.values))
            or len({self.selected_source, self.opposite_source, self.hold_source}) != 3
            or any(
                source.object_schema != FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA
                for source in (self.selected_source, self.opposite_source, self.hold_source)
            )
        ):
            raise ValueError("preparation-policy panel cell changes chart, missingness or source roles")

    @property
    def cell_id(self) -> str:
        return (
            f"{self.schedule_id}.pair-{self.pair_index}.{self.future}.r{self.refinement}"
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyRootPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-root-panel'
    config: ObjectIdentity
    root: FiniteResponseLawPreparationPolicyRoot
    native_results: tuple[ObjectIdentity, ...]
    accounting: tuple[tuple[str, int, str], ...]
    prefix_features: tuple[tuple[Decimal, ...], ...]
    handoff_features: tuple[tuple[tuple[Decimal | None, ...], ...], ...]
    parent_work: tuple[tuple[Decimal | None, ...], ...]
    cells: tuple[FiniteResponseLawPreparationPolicyPanelCell, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_ids(
            self.native_results, attribute="object_id", field_name="native_results"
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if (
            self.config.object_schema != FiniteResponseLawPreparationPolicyProjectionConfig.SCHEMA
            or len(self.native_results) != 171
            or len(self.accounting) != 171
            or tuple(f"{row[0]}.result" for row in self.accounting)
            != tuple(result.object_id for result in self.native_results)
            or len(self.prefix_features) != 2
            or any(len(view) != 24 for view in self.prefix_features)
            or len(self.handoff_features) != 9
            or any(
                len(schedule) != 2 or any(len(view) != 24 for view in schedule)
                for schedule in self.handoff_features
            )
            or any(
                value is not None
                and (not isinstance(value, Decimal) or not value.is_finite())
                for schedule in self.handoff_features
                for view in schedule
                for value in view
            )
            or len(self.parent_work) != 9
            or any(len(row) != 2 for row in self.parent_work)
            or len(self.cells) != 144
        ):
            raise ValueError("preparation-policy root panel changes its complete 171-task tensor")

    @property
    def panel_id(self) -> str:
        return f"{self.root.stage_unit}.preparation-policy-root-panel"


def _arrays(value: FiniteResponseLawNativePhaseData) -> NativeResponseArrays:
    return NativeResponseArrays(
        value.ticks,
        value.positions,
        value.transfer,
        value.transfer_known,
        float(value.delivery.signed_force_work),
        float(value.delivery.absolute_force_work),
    )


def project_preparation_policy_root(
    config: FiniteResponseLawPreparationPolicyProjectionConfig,
    root: FiniteResponseLawPreparationPolicyRoot,
    inputs: tuple[tuple[FiniteResponseLawPreparationPolicyNativeTaskResult, bytes], ...],
) -> FiniteResponseLawPreparationPolicyRootPanel:
    if root not in config.native_spec.roots:
        raise ValueError("preparation-policy projection changes its assigned root")
    tasks = tuple(
        task for task in preparation_policy_native_invocations(config.native_spec) if task.root == root
    )
    if tuple(record.invocation for record, _ in inputs) != tasks:
        raise ValueError("preparation-policy projection requires its complete ordered source census")
    records = {record.invocation.task_id: record for record, _ in inputs}
    identities = {
        key: ObjectIdentity.from_record(record.result_id, record)
        for key, record in records.items()
    }
    pairs = {
        record.invocation.task_id: decode_preparation_policy_task_native(record, payload)
        for record, payload in inputs
    }
    accounting = []
    for task in tasks:
        record, pair = records[task.task_id], pairs[task.task_id]
        status = (
            record.unentered_reason
            if pair is None
            else "/".join(value.delivery.disposition for value in pair)
        )
        accounting.append(
            (task.task_id, record.completed_native_updates, str(status))
        )
        if task.phase == "future" and pair is not None:
            previous = pairs[task.dependency_task_ids[0]]
            if previous is None or record.predecessor != identities[task.dependency_task_ids[0]]:
                raise ValueError("preparation-policy projection changes future/preparation lineage")
            for value, parent in zip(pair, previous, strict=True):
                if (
                    parent.checkpoint is None
                    or value.delivery.incoming_checkpoint
                    != ObjectIdentity.from_record(parent.checkpoint.checkpoint_id, parent.checkpoint)
                    or not np.array_equal(value.positions[0], parent.positions[-1])
                    or not np.array_equal(value.momenta[0], parent.momenta[-1])
                ):
                    raise ValueError("preparation-policy future changes incoming state/checkpoint")
    prefix = root.retained_prefix
    assert prefix is not None
    handoff: list[tuple[tuple[Decimal | None, ...], ...]] = []
    work: list[tuple[Decimal | None, ...]] = []
    cells: list[FiniteResponseLawPreparationPolicyPanelCell] = []
    hold_word = next(word for word in config.native_spec.words if word.sign == 0)
    positive = tuple(word for word in config.native_spec.words if word.sign == 1)
    if len(positive) != 4:
        raise ValueError("preparation-policy projection changes its four positive-oriented pairs")
    for schedule in config.native_spec.schedules:
        prep = next(
            task
            for task in tasks
            if task.phase == "preparation" and task.schedule == schedule
        )
        prep_pair = pairs[prep.task_id]
        features: list[tuple[Decimal | None, ...]] = []
        works = []
        for refinement in (1, 2):
            phase = None if prep_pair is None else prep_pair[refinement - 1]
            works.append(None if phase is None else phase.delivery.parent_absolute_density_work)
            if phase is None or type(phase.checkpoint) is not FiniteResponseLawPreparationPolicyNativeCheckpoint:
                features.append((None,) * 24)
            else:
                checkpoint = phase.checkpoint
                interface = preparation_policy_compact_interface(
                    frame=records[prep.task_id].frame,
                    ticks=checkpoint.history_ticks,
                    positions=_decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4)),
                    momenta=_decode(checkpoint.history_momenta_base64, (31, 2, 3, 4, 4)),
                )
                features.append(
                    tuple(Decimal(str(float(interface_value))) for interface_value in interface.values)
                )
        handoff.append(tuple(features))
        work.append(tuple(works))
        for pair_index, word in enumerate(positive):
            opposite_word = replace(word, sign=-1)
            for future in ("future-1", "future-2"):
                selected_task = next(
                    task
                    for task in tasks
                    if task.schedule == schedule
                    and task.purpose == future
                    and task.word == word
                )
                opposite_task = replace(selected_task, word=opposite_word)
                hold_task = replace(selected_task, word=hold_word)
                for refinement in (1, 2):
                    selected_pair = pairs[selected_task.task_id]
                    opposite_pair = pairs[opposite_task.task_id]
                    hold_pair = pairs[hold_task.task_id]
                    phases = tuple(
                        None if value is None else value[refinement - 1]
                        for value in (selected_pair, opposite_pair, hold_pair)
                    )
                    observed = all(value is not None for value in phases)
                    valid = observed and all(
                        value.delivery.disposition == "COMPLETE" for value in phases if value
                    )
                    outputs: tuple[Decimal | None, ...] = (None,) * 8
                    if observed:
                        reduced = []
                        for task, phase_value in zip(
                            (selected_task, opposite_task), phases[:2], strict=True
                        ):
                            assert (
                                phase_value is not None
                                and task.word is not None
                                and phases[2] is not None
                            )
                            measurement, _ = reduce_native_response_arrays(
                                frame=records[task.task_id].frame,
                                word=task.word,
                                handoff_tick=4496,
                                readouts=(192,),
                                future=_arrays(phase_value),
                                paired_hold=_arrays(phases[2]),
                            )
                            reduced.append(measurement[0])
                            if phase_value.delivery.completed_intervals:
                                first = 4496 * refinement
                                components = prepared_force_components(
                                    task.word,
                                    native_step=first,
                                    invocation_tick=4496,
                                    refinement=refinement,
                                )
                                expected = (
                                    np.arange(phase_value.delivery.completed_intervals)
                                    < 64 * refinement
                                )[:, None] * components
                                if not np.array_equal(
                                    phase_value.realized_trace[:, 5:7], expected
                                ) or not np.array_equal(
                                    phase_value.realized_trace[:, 7:9], expected
                                ):
                                    valid = False
                        outputs_array = np.r_[
                            (reduced[0][:2] - reduced[1][:2]) / 2,
                            reduced[0][2:5],
                            reduced[1][2:5],
                        ]
                        outputs = tuple(
                            Decimal(str(float(value))) if np.isfinite(value) else None
                            for value in outputs_array
                        )
                        valid = valid and all(value is not None for value in outputs)
                    cells.append(
                        FiniteResponseLawPreparationPolicyPanelCell(
                            schedule.schedule_id,
                            pair_index,
                            future,
                            refinement,
                            outputs,
                            observed,
                            valid,
                            identities[selected_task.task_id],
                            identities[opposite_task.task_id],
                            identities[hold_task.task_id],
                        )
                    )
    if prefix.feature_instrument != REFERENCE_INSTRUMENT.identity:
        raise ValueError("preparation-policy prefix uses another feature instrument")
    return FiniteResponseLawPreparationPolicyRootPanel(
        ObjectIdentity.from_record(config.config_id, config),
        root,
        tuple(sorted(identities.values(), key=lambda value: value.object_id)),
        tuple(sorted(accounting)),
        prefix.prefix_features,
        tuple(handoff),
        tuple(work),
        tuple(sorted(cells, key=lambda value: value.cell_id)),
    )
