"Finite source qualification charter nomination with all assigned roots and both numerical views."

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.adapters.composition.prepared_response.task_charter import PreparedTaskCharterEntry, prepared_task_charter
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CONTEXTS, READOUTS
from .endpoint_check import check_prepared_endpoint
from .qualification_records import PreparedResponseSourceQualificationProjectionConfig, PreparedResponseSourceQualificationViewObservation


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-source-qualification-evaluation-config'
    config_id: str
    projection: PreparedResponseSourceQualificationProjectionConfig
    off_force_x_limit: Decimal = Decimal("0.125")
    relative_y_limit: Decimal = Decimal("0.05")
    transfer_difference_limit: Decimal = Decimal("0.05")
    parent_density_work_limit: Decimal = Decimal(32)
    absolute_resolution_floor: Decimal = Decimal("1e-10")
    force_component_tolerance: Decimal = Decimal("1e-12")
    numerical_rule: str = (
        "ALL_FIVE_READOUTS_PRIMARY_HALF_MAX_DISCREPANCY_AT_MOST_EPSILON_OVER_EIGHT"
    )
    preservation_rule: str = "INHERITED_MATCHED_HOLD_BOUNDS_WITH_TWO_VIEW_DECISION_AGREEMENT"
    parent_change_rule: str = (
        "ORACLE_CONTACT_CHANGE_OR_ABSOLUTE_ENDPOINT_CHANGE_FROM_HOLD_PARENT_IN_BOTH_VIEWS"
    )
    charter_order: tuple[PreparedTaskCharterEntry, ...] = prepared_task_charter()

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        exact = {
            "off_force_x_limit": Decimal("0.125"),
            "relative_y_limit": Decimal("0.05"),
            "transfer_difference_limit": Decimal("0.05"),
            "parent_density_work_limit": Decimal(32),
            "absolute_resolution_floor": Decimal("1e-10"),
            "force_component_tolerance": Decimal("1e-12"),
        }
        if any(
            not isinstance(getattr(self, key), Decimal) or getattr(self, key) != value
            for key, value in exact.items()
        ) or (
            self.charter_order != prepared_task_charter()
            or self.numerical_rule
            != "ALL_FIVE_READOUTS_PRIMARY_HALF_MAX_DISCREPANCY_AT_MOST_EPSILON_OVER_EIGHT"
            or self.preservation_rule
            != "INHERITED_MATCHED_HOLD_BOUNDS_WITH_TWO_VIEW_DECISION_AGREEMENT"
            or self.parent_change_rule
            != "ORACLE_CONTACT_CHANGE_OR_ABSOLUTE_ENDPOINT_CHANGE_FROM_HOLD_PARENT_IN_BOTH_VIEWS"
        ):
            raise ValueError("prepared source qualification evaluator changes the frozen finite qualification rule")


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationContextScreen(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-source-qualification-context-screen'
    context: str
    charter: PreparedTaskCharterEntry
    qualified_chart_roots: int
    two_direction_contact_roots: int
    distinct_command_roots: int
    parent_change_roots: int
    maximum_view_discrepancy: Decimal | None
    fixed_parent_oracle_counts: tuple[int, ...]
    fixed_parent_command_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS or any(
            type(value) is not int or not 0 <= value <= 16
            for value in (
                self.qualified_chart_roots,
                self.two_direction_contact_roots,
                self.distinct_command_roots,
                self.parent_change_roots,
            )
        ):
            raise ValueError("source qualification screens require the full sixteen assigned roots per context")
        if self.distinct_command_roots > self.two_direction_contact_roots:
            raise ValueError("source qualification distinct-command count exceeds its multi-direction contact count")
        if (
            type(self.fixed_parent_oracle_counts) is not tuple
            or len(self.fixed_parent_oracle_counts) != 5
            or type(self.fixed_parent_command_counts) is not tuple
            or len(self.fixed_parent_command_counts) != 5
            or any(
                type(row) is not tuple or len(row) != 9 for row in self.fixed_parent_command_counts
            )
            or any(
                type(value) is not int or not 0 <= value <= 144
                for value in (
                    *self.fixed_parent_oracle_counts,
                    *(x for row in self.fixed_parent_command_counts for x in row),
                )
            )
            or any(
                max(row) > oracle
                for row, oracle in zip(
                    self.fixed_parent_command_counts, self.fixed_parent_oracle_counts, strict=True
                )
            )
            or self.maximum_view_discrepancy is not None
            and (
                not isinstance(self.maximum_view_discrepancy, Decimal)
                or not self.maximum_view_discrepancy.is_finite()
                or self.maximum_view_discrepancy < 0
            )
        ):
            raise ValueError("source qualification contact counts or native resolution operand are inconsistent")

    @property
    def oracle_advantage(self) -> Decimal:
        return (
            Decimal(
                max(self.fixed_parent_oracle_counts)
                - max(x for row in self.fixed_parent_command_counts for x in row)
            )
            / 144
        )

    @property
    def passing(self) -> bool:
        return (
            self.qualified_chart_roots == 16
            and self.two_direction_contact_roots >= 12
            and self.distinct_command_roots >= 8
            and self.parent_change_roots >= 4
            and self.maximum_view_discrepancy is not None
            and self.maximum_view_discrepancy <= self.charter.epsilon / 8
            and self.oracle_advantage >= Decimal("0.20")
        )

    @property
    def reasons(self) -> tuple[str, ...]:
        conditions = {
            "CHART_DELIVERY_NUMERICAL_OR_PRESERVATION_UNQUALIFIED": self.qualified_chart_roots
            != 16,
            "INSUFFICIENT_TWO_DIRECTION_CONTACT": self.two_direction_contact_roots < 12,
            "INSUFFICIENT_ACTION_DISTINCT_CONTACT": self.distinct_command_roots < 8,
            "INSUFFICIENT_PARENT_CHANGE": self.parent_change_roots < 4,
            "ABSOLUTE_NUMERICAL_FLOOR_TOO_LARGE_OR_UNKNOWN": self.maximum_view_discrepancy is None
            or self.maximum_view_discrepancy > self.charter.epsilon / 8,
            "INSUFFICIENT_ORACLE_OVER_FIXED_COMMAND_CONTACT": self.oracle_advantage
            < Decimal("0.20"),
        }
        return tuple(sorted(key for key, failed in conditions.items() if failed))


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-source-qualification-evaluation'
    evaluation_id: str
    config: ObjectIdentity
    observations: tuple[ObjectIdentity, ...]
    screens: tuple[PreparedResponseSourceQualificationContextScreen, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        if (
            self.config.object_schema != PreparedResponseSourceQualificationEvaluationConfig.SCHEMA
            or type(self.observations) is not tuple
            or len(self.observations) != 64
            or tuple(v.object_id for v in self.observations)
            != tuple(sorted({v.object_id for v in self.observations}))
            or any(v.object_schema != PreparedResponseSourceQualificationViewObservation.SCHEMA for v in self.observations)
            or type(self.screens) is not tuple
            or tuple((v.charter, v.context) for v in self.screens)
            != tuple(
                (entry, context) for entry in prepared_task_charter() for context in CONTEXTS
            )
        ):
            raise ValueError(
                "source qualification terminal must retain the complete observation and ordered charter census"
            )

    @property
    def selected_charter(self) -> PreparedTaskCharterEntry | None:
        for first, second in zip(self.screens[::2], self.screens[1::2], strict=True):
            if first.passing and second.passing:
                return first.charter
        return None

    @property
    def disposition(self) -> str:
        return (
            "COMMON_CHARTER_NOMINATED"
            if self.selected_charter is not None
            else "SOURCE_QUALIFICATION_STOP_NO_COMMON_QUALIFIED_CHARTER"
        )

    @property
    def scientific_status(self) -> ScientificStatus:
        if self.selected_charter is not None:
            return ScientificStatus.SUPPORTED
        if any(
            first.qualified_chart_roots == second.qualified_chart_roots == 16
            for first, second in zip(self.screens[::2], self.screens[1::2], strict=True)
        ):
            return ScientificStatus.NOT_SUPPORTED
        return ScientificStatus.UNEVALUABLE

    @property
    def reasons(self) -> tuple[str, ...]:
        if self.selected_charter is not None:
            return ("COMMON_FINITE_CHARTER_NOMINATED",)
        return tuple(
            sorted(
                {
                    "NO_COMMON_QUALIFIED_CHARTER",
                    *(reason for screen in self.screens for reason in screen.reasons),
                }
            )
        )


def _context_screen(
    config: PreparedResponseSourceQualificationEvaluationConfig,
    charter: PreparedTaskCharterEntry,
    pairs: tuple[tuple[PreparedResponseSourceQualificationViewObservation, PreparedResponseSourceQualificationViewObservation], ...],
) -> PreparedResponseSourceQualificationContextScreen:
    selected_words = (0, *range(1, 9)) if charter.amplitude == 8 else (0, *range(9, 17))
    horizon = READOUTS.index(charter.horizon_ticks)
    contacts = np.zeros((16, 5, 9, 9), dtype=np.bool_)
    qualified = two_direction = distinct = changed = 0
    discrepancies = []
    for root_index, pair in enumerate(pairs):
        outputs = np.array(
            [
                [
                    [
                        [np.nan if v is None else float(v) for v in row]
                        for row in report.words[parent * 17 + word].outputs
                    ]
                    for word in selected_words
                ]
                for report in pair
                for parent in range(5)
            ],
            dtype=np.float64,
        ).reshape(2, 5, 9, 5, 7)
        displacements = np.array(
            [
                [
                    [np.nan, np.nan] if v is None else [float(x) for x in v]
                    for v in report.handoff_displacements
                ]
                for report in pair
            ]
        )
        absolute = outputs[..., :2] + displacements[:, :, None, None, :]
        discrepancy = (
            float(np.max(np.abs(absolute[0] - absolute[1])))
            if np.isfinite(absolute).all()
            else np.inf
        )
        discrepancies.append(max(float(config.absolute_resolution_floor), discrepancy))
        preservation = np.all(
            outputs[..., 2:5]
            <= np.array(
                [
                    float(config.off_force_x_limit),
                    float(config.relative_y_limit),
                    float(config.transfer_difference_limit),
                ]
            ),
            axis=-1,
        )
        parent_work = np.array(
            [
                [np.nan if v is None else float(v) for v in report.parent_density_work]
                for report in pair
            ]
        )
        parent_ok = parent_work <= float(config.parent_density_work_limit)
        delivered = np.array(
            [
                [
                    report.words[parent * 17 + word].delivery_complete
                    and report.words[parent * 17 + word].force_component_error is not None
                    and report.words[parent * 17 + word].force_component_error
                    <= config.force_component_tolerance
                    for parent in range(5)
                    for word in selected_words
                ]
                for report in pair
            ]
        ).reshape(2, 5, 9)
        root_qualified = bool(
            all(report.mode_disposition == "RESOLVED" for report in pair)
            and delivered.all()
            and np.isfinite(outputs).all()
            and np.isfinite(parent_work).all()
            and np.array_equal(preservation[0], preservation[1])
            and np.array_equal(parent_ok[0], parent_ok[1])
            and discrepancy <= float(charter.epsilon / 8)
        )
        qualified += int(root_qualified)
        velocity = pair[0].preparent_velocity
        if velocity is None:
            continue
        for parent in range(5):
            for word in range(9):
                if not (
                    delivered[:, parent, word].all()
                    and preservation[:, parent, word, horizon].all()
                    and parent_ok[:, parent].all()
                    and np.isfinite(absolute[:, parent, word, horizon]).all()
                ):
                    continue
                for draw in range(9):
                    contacts[root_index, parent, word, draw] = check_prepared_endpoint(
                        absolute_displacement=absolute[:, parent, word, horizon],
                        preparent_velocity=(float(velocity[0]), float(velocity[1])),
                        target_draw=draw,
                        horizon_ticks=charter.horizon_ticks,
                        epsilon=float(charter.epsilon),
                        distance_multiple=charter.distance_multiple,
                    )[0]
        root_contacts = contacts[root_index]
        has_directions = has_distinct = False
        for parent in range(5):
            draws = tuple(draw for draw in range(1, 9) if root_contacts[parent, :, draw].any())
            for first in draws:
                for second in draws:
                    if (first - 1) // 2 == (second - 1) // 2:
                        continue
                    has_directions = True
                    has_distinct |= any(
                        a != b
                        for a in np.flatnonzero(root_contacts[parent, :, first])
                        for b in np.flatnonzero(root_contacts[parent, :, second])
                    )
        two_direction += int(has_directions)
        distinct += int(has_distinct)
        oracle = root_contacts.any(axis=1)
        contact_change = np.any(oracle[1:] != oracle[0])
        endpoint_change = np.max(
            np.abs(absolute[:, 1:, :, horizon] - absolute[:, :1, :, horizon]), axis=-1
        )
        finite_change = np.any(
            np.all(
                np.isfinite(endpoint_change) & (endpoint_change >= float(charter.epsilon / 4)),
                axis=0,
            )
        )
        changed += int(root_qualified and (contact_change or finite_change))
    maximum = max(discrepancies)
    return PreparedResponseSourceQualificationContextScreen(
        pairs[0][0].root.context,
        charter,
        qualified,
        two_direction,
        distinct,
        changed,
        None if not np.isfinite(maximum) else Decimal(str(maximum)),
        tuple(int(v) for v in contacts.any(axis=2).sum(axis=(0, 2))),
        tuple(tuple(int(v) for v in row) for row in contacts.sum(axis=(0, 3))),
    )


def evaluate_prepared_response_source_qualification(
    config: PreparedResponseSourceQualificationEvaluationConfig, reports: tuple[PreparedResponseSourceQualificationViewObservation, ...]
) -> PreparedResponseSourceQualificationEvaluation:
    source = config.projection.native_spec
    by_view = {(v.root, v.refinement): v for v in reports}
    expected = {(root, refinement) for root in source.roots for refinement in (1, 2)}
    projection_id = ObjectIdentity.from_record(config.projection.config_id, config.projection)
    source_id = ObjectIdentity.from_record(source.spec_id, source)
    if (
        type(reports) is not tuple
        or len(reports) != len(expected)
        or set(by_view) != expected
        or any(v.projection_config != projection_id or v.source_spec != source_id for v in reports)
    ):
        raise ValueError(
            "source qualification evaluation requires every assigned root/view under its exact frozen configuration"
        )
    for root in source.roots:
        first, second = by_view[root, 1], by_view[root, 2]
        if (
            first.common_start,
            first.mode_disposition,
            first.preparent_velocity,
            first.source_results,
        ) != (
            second.common_start,
            second.mode_disposition,
            second.preparent_velocity,
            second.source_results,
        ):
            raise ValueError(
                "source qualification numerical views disagree on their shared source or pre-parent task frame"
            )
    screens = tuple(
        _context_screen(
            config,
            charter,
            tuple(
                (by_view[root, 1], by_view[root, 2])
                for root in source.roots
                if root.context == context
            ),
        )
        for charter in config.charter_order
        for context in CONTEXTS
    )
    return PreparedResponseSourceQualificationEvaluation(
        f"{source.spec_id}.q-evaluation",
        ObjectIdentity.from_record(config.config_id, config),
        tuple(
            sorted(
                (ObjectIdentity.from_record(v.report_id, v) for v in reports),
                key=lambda v: v.object_id,
            )
        ),
        screens,
    )
