'Narrow ambient pressure superconductor excluded solver control records and strict excluded solver control configuration decoding.\n\nThis module deliberately covers only the excluded Pb solver control.  Search,\nmaterial admission, transverse-response and manufacturing records are added\nonly after this control qualifies the numerical route.\n'

from __future__ import annotations

from empirical_lawhood._required_inputs import required_external_path

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import ClassVar, Final, Mapping

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


PLAN_ID: Final = 'ambient-pressure-superconductor-response-study'
CAMPAIGN_ID: Final = 'ambient-pressure-superconductor-excluded-solver-control-initial'
CORRECTIVE_CAMPAIGN_ID: Final = 'ambient-pressure-superconductor-excluded-solver-control-corrective'
CONFIG_SCHEMA: Final = 'empirical-lawhood/simulators/ambient-pressure-superconductor/excluded-solver-control/config'
CONFIG_VERSION: Final = "1.0.0"
CAPABILITY_VERSION: Final = "1.0.0"
FIXED_EXECUTOR_IMPLEMENTATION_ID: Final = 'executor.ambient-pressure-superconductor-qe-docker-loop'
EXTERNAL_ROOT: Final = 'runs/ambient-pressure-300k-superconductor/excluded-solver-control-pb-smoke'
CORRECTIVE_EXTERNAL_ROOT: Final = 'runs/ambient-pressure-300k-superconductor/excluded-solver-control-pb-smoke-corrective-solver-control'
MAXIMUM_CONFIG_BYTES: Final = 256 * 1024
MAXIMUM_OUTPUT_BYTES: Final = 8 * 1024 * 1024

PREPARATION_CAPABILITY_KEY: Final = 'open-sim.ambient-pressure-superconductor-material-preparation.excluded-solver-control'
SOLVER_CAPABILITY_KEY: Final = 'open-sim.ambient-pressure-superconductor-qe-solver.excluded-solver-control'
EVALUATOR_CAPABILITY_KEY: Final = 'evaluator.ambient-pressure-superconductor-solver-smoke.excluded-solver-control'
ADJUDICATION_SCHEMA: Final = 'empirical-lawhood/simulators/ambient-pressure-superconductor/excluded-solver-control/scientific-adjudication'
SOURCE_MANIFEST_SOURCE_ID: Final = 'source.ambient-pressure-superconductor-excluded-solver-control-solver-environment'


class ConfigLifecycle(StrEnum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"


class SolverSmokeDisposition(StrEnum):
    PASS = "INITIAL_SOLVER_CONTROL_PASS"
    FAIL = "INITIAL_SOLVER_CONTROL_FAIL"
    CORRECTIVE_PASS = "CORRECTIVE_SOLVER_CONTROL_PASS"
    CORRECTIVE_FAIL = "CORRECTIVE_SOLVER_CONTROL_FAIL"
    RESOURCE_INADEQUATE = "RESOURCE_PATH_INADEQUATE"


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f'{field_name} must be an object with string keys')
    return value


def _keys(value: Mapping[str, object], expected: tuple[str, ...], *, field_name: str) -> None:
    if tuple(sorted(value)) != tuple(sorted(expected)):
        raise ValueError(f'{field_name} keys differ from the closed excluded solver control schema')


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{field_name} must be a nonempty string')
    return value


def _integer(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{field_name} must be an integer >= {minimum}')
    return value


def _decimal(value: object, *, field_name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f'{field_name} must be a decimal string')
    try:
        result = Decimal(value)
    except Exception as error:
        raise ValueError(f'{field_name} must be a decimal string') from error
    validate_decimal(result, field_name=field_name)
    return result


def _sha(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    result = _text(value, field_name=field_name)
    validate_sha256(result, field_name=field_name)
    return result


def _reject_binary_floats(value: object, *, field_name: str = "config") -> None:
    if isinstance(value, float):
        raise ValueError(f'{field_name} contains a binary float')
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_binary_floats(item, field_name=f'{field_name}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_binary_floats(item, field_name=f'{field_name}[{index}]')


@dataclass(frozen=True, slots=True)
class ExcludedSolverControlConfig:
    'Closed excluded solver control control configuration; no paths, argv or imports are data.'

    payload_sha256: str
    lifecycle: ConfigLifecycle
    campaign_id: str
    revision: str
    freeze_id: str | None
    frozen_at_utc: str | None
    solver_profile_id: str
    source_release_id: str
    source_archive_sha256: str
    source_archive_bytes: int
    source_member_count: int
    source_expanded_bytes: int
    container_image_digest: str
    environment_image_sha256: str
    executor_implementation_id: str
    qe_binary_sha256: str
    input_sha256: str
    pseudopotential_sha256: str
    control_case_id: str
    composition: str
    functional: str
    pseudopotential_family: str
    wavefunction_cutoff_Ry: Decimal
    charge_density_cutoff_Ry: Decimal
    k_mesh: tuple[int, int, int]
    smearing_Ry: Decimal
    convergence_threshold_Ry: Decimal
    expected_atom_count: int
    expected_electron_count: Decimal
    storage_root: str
    external_root: str
    minimum_free_bytes: int
    resources: tuple[tuple[str, int | bool], ...]

    def resource(self, key: str) -> int | bool:
        try:
            return dict(self.resources)[key]
        except KeyError as error:
            raise ValueError(f'unknown excluded solver control resource key {key}') from error


@dataclass(frozen=True, slots=True)
class ExcludedSolverControlSourceManifest(CanonicalRecord):
    'Outcome-blind identity of the exact held excluded solver control solver/control closure.\n\n    The manifest is the small source-resolution operand.  The fixed executor\n    independently re-hashes the large environment image and its consumed\n    binary, input and pseudopotential immediately before execution.\n    '

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/excluded-solver-control-source-manifest'

    source_id: str
    release_id: str
    upstream_archive_sha256: str
    upstream_archive_bytes: int
    upstream_member_count: int
    upstream_expanded_bytes: int
    upstream_scf_input_sha256: str
    derived_scf_input_sha256: str
    derivation_id: str
    pseudopotential_sha256: str
    environment_profile_id: str
    environment_image_sha256: str
    environment_image_bytes: int
    container_image_digest: str
    container_image_bytes: int
    binary_sha256s: tuple[tuple[str, str], ...]
    build_manifest_sha256: str
    control_component_scope: str
    scientific_scope_limitations: tuple[str, ...]
    network_required: bool

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "release_id",
            "derivation_id",
            "environment_profile_id",
            "control_component_scope",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "upstream_archive_sha256",
            "upstream_scf_input_sha256",
            "derived_scf_input_sha256",
            "pseudopotential_sha256",
            "environment_image_sha256",
            "build_manifest_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        for name in (
            "upstream_archive_bytes",
            "upstream_member_count",
            "upstream_expanded_bytes",
            "environment_image_bytes",
            "container_image_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not self.container_image_digest.startswith("sha256:"):
            raise ValueError("container image identity must be a SHA-256 digest")
        validate_sha256(
            self.container_image_digest.removeprefix("sha256:"),
            field_name="container_image_digest",
        )
        if not self.binary_sha256s or tuple(sorted(self.binary_sha256s)) != self.binary_sha256s:
            raise ValueError("binary_sha256s must be sorted and nonempty")
        binary_names = tuple(name for name, _digest in self.binary_sha256s)
        if len(binary_names) != len(set(binary_names)):
            raise ValueError("binary_sha256s contains duplicate binary identities")
        for name, digest in self.binary_sha256s:
            validate_stable_id(name, field_name="binary_sha256s.name")
            validate_sha256(digest, field_name=f'binary_sha256s.{name}')
        require_sorted_unique_strings(
            self.scientific_scope_limitations,
            field_name="scientific_scope_limitations",
        )
        if self.network_required:
            raise ValueError('the excluded solver control held source closure must be executable without network')


def decode_config(document: Mapping[str, object], *, payload_sha256: str) -> ExcludedSolverControlConfig:
    _reject_binary_floats(document)
    _keys(
        document,
        (
            "authority",
            "campaign_id",
            "control",
            "lifecycle",
            "plan_id",
            "resources",
            "schema",
            "solver",
            "source",
            "storage",
            "version",
        ),
        field_name="config",
    )
    if document["schema"] != CONFIG_SCHEMA or document["version"] != CONFIG_VERSION:
        raise ValueError('excluded solver control config schema/version differs')
    if document["plan_id"] != PLAN_ID or document["campaign_id"] not in {
        CAMPAIGN_ID,
        CORRECTIVE_CAMPAIGN_ID,
    }:
        raise ValueError('excluded solver control plan/campaign identity differs')
    campaign_id = _text(document["campaign_id"], field_name="campaign_id")

    lifecycle = _mapping(document["lifecycle"], field_name="lifecycle")
    _keys(lifecycle, ("freeze_id", "frozen_at_utc", "revision", "status"), field_name="lifecycle")
    status = ConfigLifecycle(_text(lifecycle["status"], field_name="lifecycle.status"))
    revision = _text(lifecycle["revision"], field_name="lifecycle.revision")
    if revision != "A1":
        raise ValueError('excluded solver control config must bind plan amendment A1')
    freeze_id = lifecycle["freeze_id"]
    frozen_at = lifecycle["frozen_at_utc"]
    if status is ConfigLifecycle.DRAFT:
        if freeze_id is not None or frozen_at is not None:
            raise ValueError('draft excluded solver control config cannot claim a freeze')
        parsed_freeze_id = parsed_frozen_at = None
    else:
        parsed_freeze_id = _text(freeze_id, field_name="lifecycle.freeze_id")
        validate_stable_id(parsed_freeze_id, field_name="lifecycle.freeze_id")
        parsed_frozen_at = _text(frozen_at, field_name="lifecycle.frozen_at_utc")

    authority = _mapping(document["authority"], field_name="authority")
    _keys(
        authority,
        (
            "authorization_basis",
            "external_scientific_write",
            "network_during_execution",
            "physical_actuation",
            "public_source_acquisition",
        ),
        field_name="authority",
    )
    if (
        authority["authorization_basis"] != "OWNER_REQUEST_2026-08-02"
        or authority["external_scientific_write"] != "OWNER_AUTHORIZED"
        or authority["public_source_acquisition"] != "OWNER_AUTHORIZED"
        or authority["network_during_execution"] != "PROHIBITED"
        or authority["physical_actuation"] != "PROHIBITED"
    ):
        raise ValueError('excluded solver control authority scope differs')

    source = _mapping(document["source"], field_name="source")
    _keys(
        source,
        (
            "archive_bytes",
            "archive_sha256",
            "expanded_bytes",
            "license",
            "member_count",
            "release_id",
            "upstream_locator",
        ),
        field_name="source",
    )
    if source["release_id"] != "EPW-6.1-QE-7.6" or source["license"] != "GPL-2.0-or-later":
        raise ValueError('excluded solver control source release/license differs')
    if (
        source["upstream_locator"]
        != "https://gitlab.com/QEF/q-e/-/archive/EPW-6.1/q-e-EPW-6.1.tar.gz"
    ):
        raise ValueError('excluded solver control source locator differs')

    solver = _mapping(document["solver"], field_name="solver")
    _keys(
        solver,
        (
            "container_image_digest",
            "environment_image_sha256",
            "executor_implementation_id",
            "profile_id",
            "qe_binary_sha256",
        ),
        field_name="solver",
    )
    if solver["profile_id"] != 'ambient-pressure-superconductor-qe76-epw61-local-ext4':
        raise ValueError('excluded solver control solver profile is not statically registered')
    image_digest = _text(solver["container_image_digest"], field_name="container_image_digest")
    if not image_digest.startswith("sha256:"):
        raise ValueError('excluded solver control runtime container image must use its local content identity')
    _sha(image_digest.removeprefix("sha256:"), field_name="container_image_digest")
    environment_sha = _sha(
        solver["environment_image_sha256"], field_name="environment_image_sha256"
    )
    if solver["executor_implementation_id"] != FIXED_EXECUTOR_IMPLEMENTATION_ID:
        raise ValueError('excluded solver control executor implementation identity differs')
    binary_sha = _sha(solver["qe_binary_sha256"], field_name="qe_binary_sha256", allow_empty=True)
    if status is ConfigLifecycle.FROZEN and not binary_sha:
        raise ValueError('frozen excluded solver control config requires the qualified pw.x digest')

    control = _mapping(document["control"], field_name="control")
    _keys(
        control,
        (
            "charge_density_cutoff_Ry",
            "composition",
            "control_case_id",
            "convergence_threshold_Ry",
            "expected_atom_count",
            "expected_electron_count",
            "functional",
            "input_sha256",
            "k_mesh",
            "pseudopotential_family",
            "pseudopotential_sha256",
            "smearing_Ry",
            "wavefunction_cutoff_Ry",
        ),
        field_name="control",
    )
    if control["control_case_id"] != "epw-pb-wosoc-scf" or control["composition"] != "Pb":
        raise ValueError('excluded solver control control must remain the excluded Pb SCF case')
    if (
        control["functional"] != "PZ-LDA"
        or control["pseudopotential_family"] != "EPW-bundled-Pb-fully-relativistic-noSOC"
    ):
        raise ValueError('excluded solver control control closure differs')
    k_mesh_raw = control["k_mesh"]
    if not isinstance(k_mesh_raw, list) or len(k_mesh_raw) != 3:
        raise ValueError("control.k_mesh must be a three-integer list")
    k_mesh = tuple(_integer(value, field_name="control.k_mesh", minimum=1) for value in k_mesh_raw)
    if k_mesh != (14, 14, 14):
        raise ValueError('excluded solver control control k mesh differs from the frozen derived control')

    storage = _mapping(document["storage"], field_name="storage")
    _keys(storage, ("external_root", "minimum_free_bytes", "storage_root"), field_name="storage")
    expected_external_root = (
        CORRECTIVE_EXTERNAL_ROOT if campaign_id == CORRECTIVE_CAMPAIGN_ID else EXTERNAL_ROOT
    )
    if (
        storage["storage_root"] != str(required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT"))
        or storage["external_root"] != expected_external_root
    ):
        raise ValueError('excluded solver control storage route differs')

    resources = _mapping(document["resources"], field_name="resources")
    resource_keys = (
        "cpu_cores",
        "gpu_devices",
        "maximum_all_inputs_bytes",
        "memory_bytes",
        "network_required",
        "output_bytes",
        "scratch_bytes",
        "wall_time_seconds",
    )
    _keys(resources, resource_keys, field_name="resources")
    parsed_resources: list[tuple[str, int | bool]] = []
    for key in resource_keys:
        value = resources[key]
        if key == "network_required":
            if not isinstance(value, bool):
                raise ValueError("resources.network_required must be boolean")
            parsed: int | bool = value
        else:
            parsed = _integer(value, field_name=f'resources.{key}', minimum=0)
        parsed_resources.append((key, parsed))
    if resources["network_required"] is not False or resources["gpu_devices"] != 0:
        raise ValueError('excluded solver control execution is CPU-only and network-disabled')
    if resources["cpu_cores"] != 2:
        raise ValueError('excluded solver control is frozen serial with two MPI ranks')

    return ExcludedSolverControlConfig(
        payload_sha256=payload_sha256,
        lifecycle=status,
        campaign_id=campaign_id,
        revision=revision,
        freeze_id=parsed_freeze_id,
        frozen_at_utc=parsed_frozen_at,
        solver_profile_id=_text(solver["profile_id"], field_name="solver.profile_id"),
        source_release_id=_text(source["release_id"], field_name="source.release_id"),
        source_archive_sha256=_sha(source["archive_sha256"], field_name="source.archive_sha256"),
        source_archive_bytes=_integer(
            source["archive_bytes"], field_name="source.archive_bytes", minimum=1
        ),
        source_member_count=_integer(
            source["member_count"], field_name="source.member_count", minimum=1
        ),
        source_expanded_bytes=_integer(
            source["expanded_bytes"], field_name="source.expanded_bytes", minimum=1
        ),
        container_image_digest=image_digest,
        environment_image_sha256=environment_sha,
        executor_implementation_id=FIXED_EXECUTOR_IMPLEMENTATION_ID,
        qe_binary_sha256=binary_sha,
        input_sha256=_sha(control["input_sha256"], field_name="control.input_sha256"),
        pseudopotential_sha256=_sha(
            control["pseudopotential_sha256"], field_name="control.pseudopotential_sha256"
        ),
        control_case_id=_text(control["control_case_id"], field_name="control.control_case_id"),
        composition=_text(control["composition"], field_name="control.composition"),
        functional=_text(control["functional"], field_name="control.functional"),
        pseudopotential_family=_text(
            control["pseudopotential_family"], field_name="control.pseudopotential_family"
        ),
        wavefunction_cutoff_Ry=_decimal(
            control["wavefunction_cutoff_Ry"], field_name="control.wavefunction_cutoff_Ry"
        ),
        charge_density_cutoff_Ry=_decimal(
            control["charge_density_cutoff_Ry"], field_name="control.charge_density_cutoff_Ry"
        ),
        k_mesh=(k_mesh[0], k_mesh[1], k_mesh[2]),
        smearing_Ry=_decimal(control["smearing_Ry"], field_name="control.smearing_Ry"),
        convergence_threshold_Ry=_decimal(
            control["convergence_threshold_Ry"], field_name="control.convergence_threshold_Ry"
        ),
        expected_atom_count=_integer(
            control["expected_atom_count"], field_name="control.expected_atom_count", minimum=1
        ),
        expected_electron_count=_decimal(
            control["expected_electron_count"], field_name="control.expected_electron_count"
        ),
        storage_root=_text(storage["storage_root"], field_name="storage.storage_root"),
        external_root=EXTERNAL_ROOT,
        minimum_free_bytes=_integer(
            storage["minimum_free_bytes"], field_name="storage.minimum_free_bytes", minimum=1
        ),
        resources=tuple(sorted(parsed_resources)),
    )


def decode_config_bytes(payload: bytes) -> ExcludedSolverControlConfig:
    if not isinstance(payload, bytes) or not payload or len(payload) > MAXIMUM_CONFIG_BYTES:
        raise ValueError('excluded solver control config bytes are absent or oversized')
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('excluded solver control config is not strict JSON') from error
    if not isinstance(document, dict):
        raise ValueError('excluded solver control config root must be an object')
    return decode_config(document, payload_sha256=sha256(payload).hexdigest())


def load_config(path: Path) -> tuple[ExcludedSolverControlConfig, bytes]:
    payload = path.read_bytes()
    return decode_config_bytes(payload), payload


@dataclass(frozen=True, slots=True)
class MaterialPreparationRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-preparation-record'

    preparation_id: str
    config_sha256: str
    candidate_id: str
    source_release_id: str
    control_case_id: str
    prototype_family_id: str
    parent_preparation_id: str
    composition: tuple[tuple[str, Decimal], ...]
    requested_action: str
    accepted_action: str
    applied_action: str
    realized_action: str
    mechanism_lane: str
    split_partition: str
    outcome_visibility: str

    def __post_init__(self) -> None:
        for name in (
            "preparation_id",
            "candidate_id",
            "source_release_id",
            "control_case_id",
            "prototype_family_id",
            "parent_preparation_id",
            "mechanism_lane",
            "split_partition",
            "outcome_visibility",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.config_sha256, field_name="config_sha256")
        if tuple(sorted(self.composition)) != self.composition or not self.composition:
            raise ValueError("composition must be sorted and nonempty")
        for element, amount in self.composition:
            validate_stable_id(element, field_name="composition.element")
            validate_decimal(amount, field_name="composition.amount", minimum=Decimal("0"))
        if any(
            getattr(self, name) != "HOLD"
            for name in ("requested_action", "accepted_action", "applied_action", "realized_action")
        ):
            raise ValueError('the excluded solver control excluded control admits only HOLD/no-op action')


@dataclass(frozen=True, slots=True)
class NumericalViewRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/numerical-view-record'

    view_id: str
    config_sha256: str
    solver_profile_id: str
    source_release_id: str
    source_archive_sha256: str
    container_image_digest: str
    environment_image_sha256: str
    executor_implementation_id: str
    qe_binary_sha256: str
    functional: str
    pseudopotential_family: str
    pseudopotential_sha256: str
    input_sha256: str
    precision: str
    wavefunction_cutoff_Ry: Decimal
    charge_density_cutoff_Ry: Decimal
    k_mesh: tuple[int, int, int]
    smearing_Ry: Decimal
    convergence_threshold_Ry: Decimal
    device_class: str
    network_required: bool

    def __post_init__(self) -> None:
        for name in (
            "view_id",
            "solver_profile_id",
            "source_release_id",
            "executor_implementation_id",
            "functional",
            "pseudopotential_family",
            "precision",
            "device_class",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "config_sha256",
            "environment_image_sha256",
            "source_archive_sha256",
            "pseudopotential_sha256",
            "input_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.qe_binary_sha256:
            validate_sha256(self.qe_binary_sha256, field_name="qe_binary_sha256")
        for name in (
            "wavefunction_cutoff_Ry",
            "charge_density_cutoff_Ry",
            "smearing_Ry",
            "convergence_threshold_Ry",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal("0"))
        if len(self.k_mesh) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.k_mesh
        ):
            raise ValueError("k_mesh must contain three positive integers")
        if self.network_required:
            raise ValueError('excluded solver control solver execution must remain network-disabled')


@dataclass(frozen=True, slots=True)
class SolverControlInputRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/solver-control-input-record'

    input_id: str
    config_sha256: str
    preparation_fingerprint: str
    numerical_view_fingerprint: str
    control_case_id: str
    expected_atom_count: int
    expected_electron_count: Decimal
    required_output_operands: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(self.control_case_id, field_name="control_case_id")
        for name in ("config_sha256", "preparation_fingerprint", "numerical_view_fingerprint"):
            validate_sha256(getattr(self, name), field_name=name)
        if isinstance(self.expected_atom_count, bool) or self.expected_atom_count <= 0:
            raise ValueError("expected_atom_count must be positive")
        validate_decimal(
            self.expected_electron_count, field_name="expected_electron_count", minimum=Decimal("0")
        )
        require_sorted_unique_strings(
            self.required_output_operands, field_name="required_output_operands"
        )


@dataclass(frozen=True, slots=True)
class SolverRunObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/solver-run-observation'

    observation_id: str
    config_sha256: str
    solver_profile_id: str
    executor_implementation_id: str
    qe_binary_sha256: str
    control_case_id: str
    completed: bool
    converged: bool
    exit_code: int
    total_energy_Ry: Decimal | None
    estimated_accuracy_Ry: Decimal | None
    electron_count: Decimal | None
    atom_count: int | None
    scf_iterations: int | None
    wall_time_seconds: Decimal
    peak_scratch_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "solver_profile_id",
            "executor_implementation_id",
            "control_case_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("config_sha256", "stdout_sha256", "stderr_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.qe_binary_sha256:
            validate_sha256(self.qe_binary_sha256, field_name="qe_binary_sha256")
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise ValueError("exit_code must be an integer")
        for name in ("total_energy_Ry", "estimated_accuracy_Ry", "electron_count"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        if self.atom_count is not None and (
            isinstance(self.atom_count, bool) or self.atom_count <= 0
        ):
            raise ValueError("atom_count must be positive when present")
        if self.scf_iterations is not None and (
            isinstance(self.scf_iterations, bool) or self.scf_iterations <= 0
        ):
            raise ValueError("scf_iterations must be positive when present")
        validate_decimal(
            self.wall_time_seconds, field_name="wall_time_seconds", minimum=Decimal("0")
        )
        if isinstance(self.peak_scratch_bytes, bool) or self.peak_scratch_bytes < 0:
            raise ValueError("peak_scratch_bytes must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.converged and (not self.completed or self.exit_code != 0):
            raise ValueError("a converged solver run must complete with exit code zero")


@dataclass(frozen=True, slots=True)
class ExcludedSolverControlSolverSmokeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/excluded-solver-control-solver-smoke-result'

    result_id: str
    config_sha256: str
    observation_fingerprint: str
    disposition: SolverSmokeDisposition
    truth_known_control: bool
    target_contact_count: int
    extracted_operands: tuple[str, ...]
    missing_operands: tuple[str, ...]
    units_verified: bool
    exact_executor_binding_verified: bool
    restart_tested: bool
    restart_disposition: str
    resource_enforcement_limitations: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(self.observation_fingerprint, field_name="observation_fingerprint")
        for name in (
            "extracted_operands",
            "missing_operands",
            "resource_enforcement_limitations",
            "reason_codes",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if not self.truth_known_control or self.target_contact_count != 0:
            raise ValueError('excluded solver control must remain truth-known and target-disjoint')
        validate_stable_id(self.restart_disposition, field_name="restart_disposition")


__all__ = [
    "ADJUDICATION_SCHEMA",
    'ExcludedSolverControlConfig',
    'ExcludedSolverControlSourceManifest',
    'ExcludedSolverControlSolverSmokeResult',
    "CAMPAIGN_ID",
    "CAPABILITY_VERSION",
    "CORRECTIVE_CAMPAIGN_ID",
    "CORRECTIVE_EXTERNAL_ROOT",
    "CONFIG_SCHEMA",
    "CONFIG_VERSION",
    "ConfigLifecycle",
    "EVALUATOR_CAPABILITY_KEY",
    "EXTERNAL_ROOT",
    "FIXED_EXECUTOR_IMPLEMENTATION_ID",
    "MAXIMUM_CONFIG_BYTES",
    "MAXIMUM_OUTPUT_BYTES",
    "MaterialPreparationRecord",
    "NumericalViewRecord",
    "PLAN_ID",
    "PREPARATION_CAPABILITY_KEY",
    "SOLVER_CAPABILITY_KEY",
    "SOURCE_MANIFEST_SOURCE_ID",
    "SolverControlInputRecord",
    "SolverRunObservation",
    "SolverSmokeDisposition",
    "decode_config",
    "decode_config_bytes",
    "load_config",
]
