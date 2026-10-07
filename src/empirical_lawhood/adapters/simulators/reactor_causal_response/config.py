"""Closed native acquisition roster; fitted operands arrive through typed DAG edges."""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.methods.reactor_causal_response.config import EmpiricalRecipe
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BATCH_SOURCE_SHA256
from .design import ROLES


@dataclass(frozen=True, slots=True)
class EmpiricalNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-causal-response/empirical-native-config'
    config_id: str = "reactor-empirical-native"
    recipe_sha256: str = EmpiricalRecipe().fingerprint()
    source_sha256: str = BATCH_SOURCE_SHA256
    python_version: str = "3.11.14"
    numpy_version: str = "2.4.6"

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if name != "SCHEMA" and (
                getattr(self, name) != field.default
                or type(getattr(self, name)) is not type(field.default)
            ):
                raise ValueError(f"unamended native acquisition configuration: {name}")


NATIVE_TASKS = tuple(
    (f"empirical.reactor-empirical-{role}-{i:03d}", role, i)
    for role, n, _ in ROLES
    for i in range(n)
)
SOURCE_IMPLEMENTATION = sha256(
    b"reactor-empirical-native:declared-episode-rosters-causal-feedback-same-tape-refinement"
).hexdigest()
