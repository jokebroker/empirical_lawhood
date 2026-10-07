"Static staged-pulse bindings to unchanged generic finite controller owners."

from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from empirical_lawhood.adapters.control.finite_bindings import (
    finite_implementation_payloads,
    finite_owner_contracts,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole
from .config import ClassicalDesign, PREFIX

OWNERS = finite_owner_contracts('ClassicalPulseDeliveryPort')


@dataclass(frozen=True, slots=True)
class ClassicalControllerBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-controller-binding'
    role: ImplementationRole
    owner: str
    design_sha256: str

    def __post_init__(self) -> None:
        if (self.role, self.owner) not in {
            (r, o) for r, o, _, _ in OWNERS
        } or self.design_sha256 != ClassicalDesign().fingerprint():
            raise ValueError("Staged-pulse controller binding changed its scientific owner")

    @property
    def config_id(self) -> str:
        return f"{PREFIX}.admission.{self.role.value.lower().replace('_', '-')}.config"


@lru_cache(maxsize=1)
def implementation_configurations() -> tuple[ClassicalControllerBinding, ...]:
    return tuple(
        sorted(
            (
                ClassicalControllerBinding(r, o, ClassicalDesign().fingerprint())
                for r, o, _, _ in OWNERS
            ),
            key=lambda c: c.config_id,
        )
    )


@lru_cache(maxsize=1)
def implementation_payloads() -> tuple[tuple[ImplementationBinding, bytes], ...]:
    return finite_implementation_payloads(
        tuple((c.config_id, c.role, c) for c in implementation_configurations()),
        delivery_owner='ClassicalPulseDeliveryPort',
        implementation_namespace="finite-reactor-staged-pulse-response",
    )
