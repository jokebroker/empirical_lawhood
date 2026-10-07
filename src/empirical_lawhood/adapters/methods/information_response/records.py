"information response prediction committed predictions, scalar projections and immutable comparison results."

from dataclasses import dataclass
from decimal import Decimal
import json
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_nonempty
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.adapters.simulators.information_response.contracts import InformationResponseNativeConfig
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import PreparedNativeInvocation
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedRoot
from .models import PROGRAMME


def validate_scalars(values: tuple[Decimal | None, ...], size: int) -> None:
    if (
        type(values) is not tuple
        or len(values) != size
        or any(
            v is not None and (not isinstance(v, Decimal) or not v.is_finite())
            for v in values
        )
    ):
        raise ValueError("information response prediction changes its finite scalar product shape/validity")


@dataclass(frozen=True, slots=True)
class InformationResponseCommittedPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/information-response/information-response-committed-prediction'
    )
    native_config: ObjectIdentity
    native_result: ObjectIdentity
    invocation: PreparedNativeInvocation
    model_bank: ObjectIdentity
    disposition: str
    gain: tuple[Decimal | None, ...]  # view,model,time,port,axis:280

    def __post_init__(self) -> None:
        if (
            self.native_config.object_schema != InformationResponseNativeConfig.SCHEMA
            or self.native_result.object_schema
            != 'empirical-lawhood/simulators/prepared-response/prepared-native-task-result'
            or self.model_bank.object_schema
            != 'empirical-lawhood/methods/information-response/information-response-model-bank'
        ):
            raise ValueError("information response prediction prediction detaches its source or bank")
        validate_scalars(self.gain, 280)
        if (
            self.disposition
            not in ("COMMITTED_AT_HANDOFF", "UNAVAILABLE", "NOT_A_HANDOFF")
            or (
                self.disposition == "COMMITTED_AT_HANDOFF"
                and (
                    self.invocation.phase != "parent"
                    or any(v is None for v in self.gain)
                )
            )
            or (self.disposition == "NOT_A_HANDOFF")
            != (self.invocation.phase != "parent")
        ):
            raise ValueError("information response prediction prediction changes its causal role")

    @property
    def prediction_id(self) -> str:
        return self.invocation.task_id + ".prediction"


@dataclass(frozen=True, slots=True)
class InformationResponseProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/information-response/information-response-projection-config'
    )
    config_id: str
    native_spec: InformationResponseNativeConfig

    def __post_init__(self) -> None:
        if not (
            self.config_id == f"{PROGRAMME}.projection-config"
            or (
                self.config_id.startswith("empirical-lawhood.information-response.")
                and self.config_id.endswith(".projection-config")
            )
        ):
            raise ValueError("information response prediction projection changes its config identity")


@dataclass(frozen=True, slots=True)
class InformationResponseEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/information-response/information-response-evaluation-config'
    )
    config_id: str
    projection: InformationResponseProjectionConfig

    def __post_init__(self) -> None:
        if (
            self.config_id
            != self.projection.config_id.removesuffix(".projection-config")
            + ".evaluation-config"
        ):
            raise ValueError("information response prediction evaluation changes its config identity")


@dataclass(frozen=True, slots=True)
class InformationResponseViewObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/information-response/information-response-view-observation'
    )
    projection_config: ObjectIdentity
    root: PreparedRoot
    refinement: int
    source_results: tuple[ObjectIdentity, ...]
    predictions: tuple[ObjectIdentity, ...]
    reasons: tuple[str, ...]
    observed_gain: tuple[Decimal | None, ...]  # parent,innovation,time,port,axis:200
    predicted_gain: tuple[Decimal | None, ...]  # parent,model,time,port,axis:700
    parent_work: tuple[Decimal | None, ...]
    force_work: tuple[Decimal | None, ...]  # parent,innovation,word,signed/absolute:80

    def __post_init__(self) -> None:
        if (
            self.projection_config.object_schema != InformationResponseProjectionConfig.SCHEMA
            or self.root.stage != 'prospective-evaluation'
            or not 0 <= self.root.index < 32
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
        ):
            raise ValueError("information response prediction view changes its assigned native carrier identity")
        for values, size in (
            (self.observed_gain, 200),
            (self.predicted_gain, 700),
            (self.parent_work, 5),
            (self.force_work, 80),
        ):
            validate_scalars(values, size)
        if (
            len(self.source_results) != 46
            or len(self.predictions) != 5
            or tuple(x.object_id for x in self.source_results)
            != tuple(sorted({x.object_id for x in self.source_results}))
        ):
            raise ValueError(
                "information response prediction view loses its complete native/prediction source census"
            )
        for reason in self.reasons:
            validate_nonempty(reason, field_name="reason")

    @property
    def report_id(self) -> str:
        return f"{self.root.root_id}.project.r{self.refinement}"


@dataclass(frozen=True, slots=True)
class InformationResponseEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/information-response/information-response-evaluation'
    evaluation_config: ObjectIdentity
    source_reports: tuple[ObjectIdentity, ...]
    context_results_json: str

    def __post_init__(self) -> None:
        if (
            self.evaluation_config.object_schema != InformationResponseEvaluationConfig.SCHEMA
            or len(self.source_reports) != 128
            or len(self.context_results_json) > 1_000_000
        ):
            raise ValueError("information response prediction evaluation changes its exact input census")
        results = json.loads(self.context_results_json)
        if tuple(sorted(results)) != ("assembling", "prepared"):
            raise ValueError("information response prediction evaluation changes its independent context results")
        for context in results.values():
            if set(context["experiments"]) != {"preparation-contrast", "signed-gain"}:
                raise ValueError("information response prediction loses one experiment")
            for experiment in context["experiments"].values():
                ScientificStatus(experiment["status"])

    @property
    def evaluation_id(self) -> str:
        return f"{PROGRAMME}.evaluation"

    @property
    def scientific_status(self) -> ScientificStatus:
        values = [
            e["status"]
            for c in json.loads(self.context_results_json).values()
            for e in c["experiments"].values()
        ]
        return combined_status(values)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                f"{context}.experiment-{number}.{experiment['status']}"
                for context, result in json.loads(self.context_results_json).items()
                for number, experiment in result["experiments"].items()
            )
        )


def combined_status(values: list[str]) -> ScientificStatus:
    if not values:
        raise ValueError("information response prediction cannot aggregate an empty result")
    if len(set(values)) == 1:
        return ScientificStatus(values[0])
    return ScientificStatus.MIXED
