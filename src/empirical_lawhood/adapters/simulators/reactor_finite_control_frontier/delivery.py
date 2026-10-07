"""One committed macro, twelve actual native deliveries, no mid-word reselection."""

from dataclasses import dataclass, field


from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import BatchDeliverySession
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_delivery import deliver_feed_word
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_callback import FiniteCommittedBranchOwner
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorDelivery
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.control import ScientificCommitmentKind
from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole
from empirical_lawhood.runtime.controller_runtime import DeliveryPortResult
from .words import PulseProjection


@dataclass
class FrontierPulseDeliveryPort:
    implementation_binding: ImplementationBinding
    session: BatchDeliverySession
    mapping: PulseProjection
    last_deliveries: list[ReactorDelivery] = field(default_factory=list)
    consumed: bool = False

    def deliver(
        self, *, action_word: OccurrenceActionWord, commitment_kind: ScientificCommitmentKind
    ) -> DeliveryPortResult:
        if (
            self.consumed
            or self.implementation_binding.role is not ImplementationRole.DELIVERY
            or commitment_kind is not ScientificCommitmentKind.ACTION
            or action_word != self.mapping.word
            or self.mapping.pulse.rate_kg_s <= 0
        ):
            raise ValueError("pulse delivery requires one unconsumed exact ACTION commitment")
        self.consumed = True
        return deliver_feed_word(
            self.session,
            action_word,
            self.mapping.mappings,
            self.last_deliveries,
        )


# The binding keeps its closed word contract; the recorder is shared with
# the bounded continuation.


@dataclass
class FrontierCommittedBranchOwner(FiniteCommittedBranchOwner):
    def __post_init__(self) -> None:
        if not isinstance(self.prepared.delivery.mapping, PulseProjection):
            raise ValueError("Finite-control frontier owner requires its exact closed pulse binding")
        super().__post_init__()
