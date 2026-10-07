"""Pinned native preparation and old interior-eligibility source binding."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.methods.reactor_regime_response.config import ReactorRegimeResponseDesign
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BATCH_SOURCE_SHA256
from empirical_lawhood.kernel.serialization import CanonicalRecord

# The retained preparation is authenticated by its original publication bytes.
# These fingerprint the retained preparation rebuilt under the current public
# schemas and record identifiers; archival source identities remain unchanged.
OLD_PREPARED_DESIGN_SHA256 = "da7823d5184389a4718d254d55a2980f888e51c787a04ce5ba79e5b0a93b2fcc"
OLD_PREPARED_DOMAIN_SHA256 = "7ae75d6a3431e53f0b43955e1a31a1f6b77c1d58550a71dc1537a8fac1d2c0dd"


@dataclass(frozen=True, slots=True)
class ReactorRegimeNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-regime-response/reactor-regime-native-config'

    design: ReactorRegimeResponseDesign
    prepared_domain: FeedDomain
    config_id: str = "reactor-regime-native"
    old_prepared_design_sha256: str = OLD_PREPARED_DESIGN_SHA256
    source_sha256: str = BATCH_SOURCE_SHA256
    python_version: str = "3.11.14"
    numpy_version: str = "2.4.6"

    def __post_init__(self) -> None:
        if (
            self.config_id != "reactor-regime-native"
            or self.old_prepared_design_sha256 != OLD_PREPARED_DESIGN_SHA256
            or self.prepared_domain.fingerprint() != OLD_PREPARED_DOMAIN_SHA256
            or self.prepared_domain.domain_id != "d11010"
            or self.prepared_domain.actions != (1, 4, 7)
            or self.source_sha256 != BATCH_SOURCE_SHA256
            or self.python_version != "3.11.14"
            or self.numpy_version != "2.4.6"
        ):
            raise ValueError("reactor regime-response study native source or preparation pin differs")
