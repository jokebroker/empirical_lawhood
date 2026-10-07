"Closed reactor bindings to existing controller, runtime and prepared controller-use owners.\n\nThe outer issued executable closure pins this adapter and every implementation.\nThe small static role payloads are canonical configuration, not executable code.\n"

from functools import lru_cache
from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import DeliveryControllerStudy, ImplementationBinding, ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import CoupledRealizationControllerEvaluationPlan
from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.adapters.simulators.reactor_regime_response.feed_word import ReactorScalarFeedDeliveryPort
from .config import ClassicalDesign, PREFIX

from empirical_lawhood.adapters.control.finite_bindings import (
    finite_owner_contracts,
    finite_implementation_payloads,
)

_OWNERS = finite_owner_contracts('ReactorScalarFeedDeliveryPort')


@dataclass(frozen=True, slots=True)
class ClassicalControllerBindingConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-controller-binding-config'
    role: ImplementationRole
    owner: str
    science_sha256: str

    def __post_init__(self) -> None:
        if (self.role, self.owner) not in {
            (r, o) for r, o, _, _ in _OWNERS
        } or self.science_sha256 != ClassicalDesign().fingerprint():
            raise ValueError(
                "reactor controller role configuration changes its closed owner or science"
            )

    @property
    def config_id(self) -> str:
        return f"{PREFIX}.admission.{self.role.value.lower().replace('_', '-')}.config"


def implementation_configurations() -> tuple[ClassicalControllerBindingConfig, ...]:
    return tuple(
        sorted(
            (
                ClassicalControllerBindingConfig(r, owner, ClassicalDesign().fingerprint())
                for r, owner, _, _ in _OWNERS
            ),
            key=lambda c: c.config_id,
        )
    )


@lru_cache(maxsize=1)
def implementation_payloads() -> tuple[tuple[ImplementationBinding, bytes], ...]:
    return finite_implementation_payloads(
        tuple((c.config_id, c.role, c) for c in implementation_configurations()),
        delivery_owner='ReactorScalarFeedDeliveryPort',
        implementation_namespace="reactor-selected-action-response-finite-native-binding",
    )


def control_services(
    study: DeliveryControllerStudy,
    evaluation_plan: CoupledRealizationControllerEvaluationPlan | None,
    delivery: ReactorScalarFeedDeliveryPort,
) -> ControllerStudyComposition:
    from empirical_lawhood.adapters.control.finite_campaign import finite_control_services

    expected = tuple(
        b
        for b, _ in implementation_payloads()
        if evaluation_plan is not None or b.role is not ImplementationRole.OUTCOME_EVALUATOR
    )
    if study.implementations != expected:
        raise ValueError("classical reactor changed its frozen implementation owners")
    return finite_control_services(study, evaluation_plan, delivery)
