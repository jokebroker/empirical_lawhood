"""Native source binding for the exact frozen local-law qualification design."""

from dataclasses import dataclass
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import LocalQualificationDesign
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BATCH_SOURCE_SHA256


@dataclass(frozen=True, slots=True)
class LocalNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-local-domain-qualification/local-native-config'
    design: LocalQualificationDesign
    config_id: str = "reactor-local-domain-native-source"
    source_sha256: str = BATCH_SOURCE_SHA256
    python_version: str = "3.11.14"
    numpy_version: str = "2.4.6"

    def __post_init__(self) -> None:
        if (
            self.config_id != "reactor-local-domain-native-source"
            or self.source_sha256 != BATCH_SOURCE_SHA256
            or self.python_version != "3.11.14"
            or self.numpy_version != "2.4.6"
        ):
            raise ValueError("local native source binding differs")
