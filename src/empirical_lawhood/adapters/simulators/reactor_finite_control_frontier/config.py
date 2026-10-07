"""The retained source pin and empirical d11010 eligibility, without a state law."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import FrontierDesign
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import OLD_PREPARED_DOMAIN_SHA256
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BATCH_SOURCE_SHA256
from empirical_lawhood.kernel.serialization import CanonicalRecord


@dataclass(frozen=True, slots=True)
class FrontierNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-finite-control-frontier/frontier-native-config'
    design: FrontierDesign
    prepared_domain: FeedDomain
    config_id: str = "reactor-finite-control-frontier-native-config"
    source_sha256: str = BATCH_SOURCE_SHA256

    def __post_init__(self) -> None:
        if (
            self.design != FrontierDesign()
            or self.config_id != "reactor-finite-control-frontier-native-config"
            or self.source_sha256 != BATCH_SOURCE_SHA256
            or self.prepared_domain.fingerprint() != OLD_PREPARED_DOMAIN_SHA256
        ):
            raise ValueError("frontier source/denominator differs from its retained pin")
