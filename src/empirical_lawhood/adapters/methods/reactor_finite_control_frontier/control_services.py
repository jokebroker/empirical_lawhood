"Static bindings to the existing admission/compiler/runtime/prepared controller-use owners."

from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from empirical_lawhood.adapters.control.finite_bindings import (
    finite_owner_contracts,
    finite_implementation_payloads,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole
from .config import PREFIX, FrontierDesign

OWNERS = finite_owner_contracts('FrontierPulseDeliveryPort')


@dataclass(frozen=True, slots=True)
class FrontierControllerBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-controller-binding'
    role: ImplementationRole
    owner: str
    design_sha256: str

    def __post_init__(self) -> None:
        if (self.role, self.owner) not in {
            (r, o) for r, o, _, _ in OWNERS
        } or self.design_sha256 != FrontierDesign().fingerprint():
            raise ValueError("controller binding changed its installed owner or science")

    @property
    def config_id(self) -> str:
        return f"{PREFIX}.admission.{self.role.value.lower().replace('_', '-')}.config"


@lru_cache(maxsize=1)
def implementation_configurations() -> tuple[FrontierControllerBinding, ...]:
    return tuple(
        sorted(
            (
                FrontierControllerBinding(r, o, FrontierDesign().fingerprint())
                for r, o, _, _ in OWNERS
            ),
            key=lambda c: c.config_id,
        )
    )


@lru_cache(maxsize=1)
def implementation_payloads() -> tuple[tuple[ImplementationBinding, bytes], ...]:
    return finite_implementation_payloads(
        tuple((c.config_id, c.role, c) for c in implementation_configurations()),
        delivery_owner='FrontierPulseDeliveryPort',
        implementation_namespace="finite-reactor-finite-control-frontier",
    )
