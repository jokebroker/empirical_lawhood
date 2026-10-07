"""Current native software conformance metadata, separate from scientific Q."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.matrix_inputs import MatrixRootAllocation
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes, validate_stable_id, validate_sha256

NATIVE_SOURCE_PATHS = (
    "adapters/simulators/preparation_applicability/contracts.py",
    "adapters/simulators/preparation_applicability/records.py",
    "adapters/simulators/preparation_applicability/native.py",
    "adapters/simulators/constructed_preparation_applicability/records.py",
    "adapters/simulators/constructed_preparation_applicability/native.py",
    "adapters/simulators/prepared_response/source.py",
    "adapters/simulators/prepared_response/contracts.py",
    "adapters/simulators/prepared_response/instruments.py",
    "adapters/simulators/finite_response_law/instruments.py",
    "adapters/simulators/finite_response_law/preparation_policy_instruments.py",
    "adapters/simulators/finite_response_law/preparation_policy_contracts.py",
    "adapters/simulators/six_matrix_response/model.py",
    "adapters/simulators/six_matrix_response/passive_probe.py",
    "adapters/simulators/six_matrix_response/response_observer.py",
)


def current_native_implementation():
    package=files("empirical_lawhood")
    entries=[]
    for path in NATIVE_SOURCE_PATHS:
        with package.joinpath(path).open("rb") as handle:
            raw=handle.read(2*1024**2+1)
        if len(raw)>2*1024**2:
            raise ValueError("native source owner exceeds its fixed scan bound")
        entries.append((path,len(raw),sha256(raw).hexdigest()))
    return ObjectIdentity("preparation-applicability.current-native-implementation",
        "empirical-lawhood/preparation-applicability/native-implementation","1.0.0",
        sha256(canonical_json_bytes(tuple(entries))).hexdigest())


@dataclass(frozen=True,slots=True)
class PreparationApplicabilityNativeConformance(CanonicalRecord):
    SCHEMA: ClassVar[str]="empirical-lawhood/preparation-applicability/native-conformance"
    conformance_id: str
    implementation: ObjectIdentity
    python_version: str
    numpy_version: str
    cells: tuple[tuple[str,int,int,str],...]
    allocation_sha256s: tuple[str,...]
    allocations: tuple[MatrixRootAllocation,...]
    ceiling: str="SOFTWARE_CONFORMANCE_NOT_SCIENTIFIC_QUALIFICATION"

    def __post_init__(self):
        validate_stable_id(self.conformance_id)
        for digest in self.allocation_sha256s:
            validate_sha256(digest)
        expected=tuple((cohort,view) for cohort in ("q2","cir1","constructed") for view in (1,2))
        if (tuple((row[0],row[1]) for row in self.cells)!=expected
            or any(row[2]!=16*row[1] or row[3]!="COMPLETE" for row in self.cells)
            or len(set(self.allocation_sha256s))!=3
            or tuple(root.cohort for root in self.allocations)!=("q2","cir1","constructed")
            or self.allocation_sha256s!=tuple(root.fingerprint() for root in self.allocations)
            or self.implementation.object_schema!="empirical-lawhood/preparation-applicability/native-implementation"
            or self.ceiling!="SOFTWARE_CONFORMANCE_NOT_SCIENTIFIC_QUALIFICATION"):
            raise ValueError("native conformance omits its actual source/view cells")


def reject_conformance_reuse(stage,conformance):
    """Actual short native checks expose their own root/stream allocations."""
    exposed={(seed.generator,seed.effective_seed) for root in conformance.allocations for seed in root.scientific_seeds}
    current={(seed.generator,seed.effective_seed) for root in stage.allocation.roots for seed in root.scientific_seeds}
    if stage.allocation.bootstrap_seed is not None:
        current.add(("PCG64",stage.allocation.bootstrap_seed))
    if (current&exposed or {root.root_id for root in stage.allocation.roots}&{root.root_id for root in conformance.allocations}):
        raise ValueError("current preparation reuses actually exposed native-conformance roots or effective streams")
