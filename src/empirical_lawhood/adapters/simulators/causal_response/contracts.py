"Additive causal response prediction assignment with an explicit mechanical compatibility map."

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.methods.causal_response.models import CausalResponseModelBank, PROGRAMME
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedNativeSpec, PreparedRoot, PreparedForceWord
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import PreparedNativeInvocation


@dataclass(frozen=True, slots=True)
class CausalResponseNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/causal-response/causal-response-native-config'
    recipe: PreparedNativeSpec
    model_bank: CausalResponseModelBank
    root_indices: tuple[int, ...] = tuple(range(32))
    purposes: tuple[str, ...] = ('common-response', 'prospective-task')
    mechanical_assignment: str = "PREPARED_REFERENCE_MECHANICS_DISTINCT_CAUSAL_PREDICTION_ASSIGNMENT"

    def __post_init__(self) -> None:
        if (
            self.recipe.stage != 'prospective-evaluation'
            or self.recipe.selected_amplitude != Decimal(16)
            or (
                self.root_indices != tuple(range(32))
                or any(type(i) is not int for i in self.root_indices)
                or self.purposes != ('common-response', 'prospective-task')
                or self.mechanical_assignment
                != "PREPARED_REFERENCE_MECHANICS_DISTINCT_CAUSAL_PREDICTION_ASSIGNMENT"
                or self.recipe.implementation_plan_sha256 != self.model_bank.plan_sha256
                or self.recipe.seed_sha256
                in {r.seed_sha256 for c in self.model_bank.contexts for r in c.training_roots}
            )
        ):
            raise ValueError("causal response prediction source changes its mechanical compatibility or fresh assignment")

    @property
    def spec_id(self) -> str:
        return f"{PROGRAMME}.native-config"

    @property
    def roots(self) -> tuple[PreparedRoot, ...]:
        return tuple(r for r in self.recipe.roots if r.index in self.root_indices)

    @property
    def words(self) -> tuple[PreparedForceWord, ...]:
        return tuple(w for w in self.recipe.words if w.sign and w.direction_index < 2)


def native_invocations(config: CausalResponseNativeConfig) -> tuple[PreparedNativeInvocation, ...]:
    source = ObjectIdentity.from_record(config.recipe.spec_id, config.recipe)
    tasks = []
    for root in config.roots:
        tasks.append(PreparedNativeInvocation(source, root, "prefix", None, None, "initial-ramp"))
        for parent in PARENTS:
            tasks.append(PreparedNativeInvocation(source, root, "parent", parent, None, "parent"))
            for purpose in config.purposes:
                for word in config.words:
                    tasks.append(
                        PreparedNativeInvocation(source, root, "future", parent, word, purpose)
                    )
    result = tuple(sorted(tasks, key=lambda t: t.task_id))
    if len(result) != 2944 or sum(t.maximum_native_updates for t in result) != 3_210_240:
        raise ValueError("causal response prediction native roster changes its exact task/update census")
    return result
