"Retained source and preparation-domain identities for new staged-pulse bindings."

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import ClassicalDesign
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import OLD_PREPARED_DOMAIN_SHA256
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BATCH_SOURCE_SHA256
from empirical_lawhood.kernel.serialization import CanonicalRecord


@dataclass(frozen=True, slots=True)
class ClassicalNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-staged-pulse-response/classical-native-config'
    design: ClassicalDesign
    prepared_domain: FeedDomain
    config_id: str = "reactor-staged-pulse-response-native-config"
    source_sha256: str = BATCH_SOURCE_SHA256

    def __post_init__(self) -> None:
        if (
            self.design != ClassicalDesign()
            or self.config_id != "reactor-staged-pulse-response-native-config"
            or self.source_sha256 != BATCH_SOURCE_SHA256
            or self.prepared_domain.fingerprint() != OLD_PREPARED_DOMAIN_SHA256
        ):
            raise ValueError("Staged-pulse source or retained preparation domain changed")
