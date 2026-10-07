"""Target-owned prepared response source qualification/information response prediction/causal response prediction authoring over explicit held inputs.

This port composes the retained scientific builders and selected candidate
providers. Historical model banks are development inputs, never new evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters._bounded_files import read_bounded_contained
from empirical_lawhood.adapters._strict_json import loads_external_json
from empirical_lawhood.adapters.composition.causal_response.authoring import CausalResponseAuthoringBundle, CausalResponseCandidateContextProvider, build_causal_response_authoring
from empirical_lawhood.adapters.composition.causal_response.exposure import CausalResponseExposureInspection, causal_response_seed_ids
from empirical_lawhood.adapters.composition.information_response.authoring import InformationResponseAuthoringBundle, InformationResponseCandidateContextProvider, build_information_response_authoring
from empirical_lawhood.adapters.composition.information_response.exposure import InformationResponseExposureInspection, information_response_seed_ids
from empirical_lawhood.adapters.composition.prepared_response.exposure import PreparedExposureInspection, prepared_seed_ids
from empirical_lawhood.adapters.composition.prepared_response.qualification_design import PreparedResponseSourceQualificationAuthoringBundle, PreparedResponseSourceQualificationCandidateContextProvider, build_prepared_response_source_qualification_authoring
from empirical_lawhood.adapters.composition.response_geometry_prospective.exposure import ResponseGeometryAssayExposureSource
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.causal_response.models import CausalResponseModelBank
from empirical_lawhood.adapters.methods.information_response.models import InformationResponseModelBank
from empirical_lawhood.adapters.simulators.causal_response.contracts import CausalResponseNativeConfig
from empirical_lawhood.adapters.simulators.causal_response.executable_binding import SOURCE_BINDING as CAUSAL_RESPONSE_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.causal_response.extension_bundle import SOURCE_CAPABILITY as CAUSAL_RESPONSE_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.information_response.contracts import InformationResponseNativeConfig
from empirical_lawhood.adapters.simulators.information_response.executable_binding import SOURCE_BINDING as INFORMATION_RESPONSE_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.information_response.extension_bundle import SOURCE_CAPABILITY as INFORMATION_RESPONSE_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedNativeSpec, PreparedRootRandomness, prepared_native_member, prepared_numerical_view, validate_prepared_seed_census
from empirical_lawhood.adapters.simulators.prepared_response.executable_binding import SOURCE_QUALIFICATION_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.prepared_response.extension_bundle import SOURCE_QUALIFICATION_SOURCE_CAPABILITY
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding

ROUTES = ("source-qualification", "information-prediction", "causal-response-prediction")
_MAX_PLAN = 2 * 1024**2
_MAX_PRIOR = 64 * 1024**2
_MAX_BANK = 16 * 1024**2


@dataclass(frozen=True, slots=True)
class PreparedResponseNativeAuthoringInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/prepared-response/prepared-response-native-authoring-input'
    config_id: str
    route: str
    target_prefix: str
    seed_label: str
    source_seed_sha256: str
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    root_seed_census: tuple[PreparedRootRandomness, ...] = ()

    def __post_init__(self) -> None:
        family = {"source-qualification": "prepared-response", "information-prediction": "information-response", "causal-response-prediction": "causal-response"}.get(self.route, "invalid")
        for name in ("config_id", "target_prefix", "seed_label"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.route not in ROUTES
            or not self.config_id.startswith(f"empirical-lawhood-{family}-")
            or not self.target_prefix.startswith(f"empirical-lawhood.{family}.")
            or not self.seed_label.startswith(f"empirical-lawhood-{family}-")
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError(
                "prepared response authoring input requires a target development identity"
            )
        object.__setattr__(self, "root_seed_census", validate_prepared_seed_census(
            "qualification" if self.route == "source-qualification" else "prospective-evaluation",
            self.source_seed_sha256, self.root_seed_census,
        ))


def _read_external(root: Path, path: Path, maximum: int) -> tuple[bytes, str]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError(
            "SIX_MATRIX_RESPONSE_SOURCE_ROOT_REQUIRED: absolute real held-source directory"
        )
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("SIX_MATRIX_RESPONSE_SOURCE_MEMBER_REQUIRED: absolute real held file")
    if not path.is_relative_to(root):
        raise ValueError("prepared response source member is outside its declared held root")
    raw = read_bounded_contained(root, path, maximum_bytes=maximum)
    if not 0 < len(raw) <= maximum:
        raise ValueError("prepared response held source member changed its byte ceiling")
    return raw, path.relative_to(root).as_posix()


def _read_plan(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("SIX_MATRIX_RESPONSE_PLAN_REQUIRED: absolute real design plan")
    raw = read_bounded_contained(path.parent, path, maximum_bytes=_MAX_PLAN)
    if not 0 < len(raw) <= _MAX_PLAN:
        raise ValueError("prepared response design plan changed its byte ceiling")
    return raw


def _implementation_digest(repo_root: Path) -> str:
    package = repo_root / "src/empirical_lawhood"
    if not Path(__file__).resolve().is_relative_to(package.resolve()):
        raise ValueError(
            "SIX_MATRIX_RESPONSE_EXECUTING_SOURCE_MISMATCH: use the selected target checkout"
        )
    members = (repo_root / "pyproject.toml", repo_root / "uv.lock")
    selected = (
        *members,
        *(
            p
            for p in package.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        ),
    )
    inventory = []
    for path in sorted(selected):
        if path.is_symlink() or path.stat().st_size > 64 * 1024**2:
            raise ValueError("prepared response target source contains a linked or oversized member")
        inventory.append(
            (
                path.relative_to(repo_root).as_posix(),
                sha256(path.read_bytes()).hexdigest(),
            )
        )
    return sha256(canonical_json_bytes(tuple(inventory))).hexdigest()


def _prior_census(raw: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    document = loads_external_json(raw)
    if not isinstance(document, dict) or not isinstance(document.get("schema"), str):
        raise TypeError("prepared response prior inspection is not a typed document")
    if document["schema"] != PreparedExposureInspection.SCHEMA:
        raise ValueError(
            "PREPARED_RESPONSE_PRIOR_VERIFIED_TARGET_EXPORT_REQUIRED: "
            "original source receipts require a separately verified target inventory"
        )
    value = document.get("value")
    if not isinstance(value, dict):
        raise TypeError("prepared response prior inspection lacks its typed value")
    unions: list[tuple[str, ...]] = []
    for names in (
        ("excluded_unit_ids", "proposed_unit_ids"),
        ("excluded_seed_ids", "proposed_seed_ids"),
    ):
        combined: set[str] = set()
        for name in names:
            entries = value.get(name)
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"prepared response prior inspection lacks {name}")
            selected = tuple(entries)
            require_sorted_unique_strings(selected, field_name=name, allow_empty=False)
            for entry in selected:
                validate_stable_id(entry, field_name=name)
            combined.update(selected)
        unions.append(tuple(sorted(combined)))
    return unions[0], unions[1]


def _require_target_bank_document(document: object, expected_schema: str) -> bytes:
    """Accept current records; an original-source export never transfers authority.

    A separately verified export must bind original hashes and interpretation
    to the target identity and target custody before this authoring input exists.
    This reader cannot relabel foreign schemas, original roots or source grants.
    """
    def current(value: object) -> None:
        if isinstance(value, dict):
            if "schema" in value and (
                not isinstance(value["schema"], str)
                or not value["schema"].startswith("empirical-lawhood/")
            ):
                raise ValueError("SOURCE_MODEL_BANK_VERIFIED_EXPORT_REQUIRED")
            for item in value.values():
                current(item)
        elif isinstance(value, list):
            for item in value:
                current(item)

    if not isinstance(document, dict) or document.get("schema") != expected_schema:
        raise ValueError("SOURCE_MODEL_BANK_VERIFIED_EXPORT_REQUIRED")
    current(document)
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


@dataclass(frozen=True, slots=True)
class PreparedNativeAuthoring:
    """Scientific preparation handed to the application compiler without authority."""

    bundle: PreparedResponseSourceQualificationAuthoringBundle | InformationResponseAuthoringBundle | CausalResponseAuthoringBundle
    exposure: (
        PreparedExposureInspection | InformationResponseExposureInspection | CausalResponseExposureInspection
    )
    context_provider_type: (
        type[PreparedResponseSourceQualificationCandidateContextProvider]
        | type[InformationResponseCandidateContextProvider]
        | type[CausalResponseCandidateContextProvider]
    )
    selection_summary: dict[str, object]


def prepare_prepared_response_native_authoring(
    config: PreparedResponseNativeAuthoringInput,
    *,
    repo_root: Path,
    source_root: Path | None,
    plan: Path | None,
    design_packet: Path | None,
    prior_exposure: Path | None,
    model_bank: Path | None,
) -> PreparedNativeAuthoring:
    """Prepare source-owned scientific records and native providers without execution."""

    if source_root is None:
        raise ValueError("SIX_MATRIX_RESPONSE_SOURCE_ROOT_REQUIRED")
    if plan is None:
        raise ValueError("SIX_MATRIX_RESPONSE_PLAN_REQUIRED")
    if design_packet is None:
        raise ValueError("SIX_MATRIX_RESPONSE_DESIGN_PACKET_REQUIRED")
    if prior_exposure is None:
        raise ValueError("SIX_MATRIX_RESPONSE_PRIOR_EXPOSURE_REQUIRED")
    if config.route != "source-qualification" and model_bank is None:
        raise ValueError("SIX_MATRIX_RESPONSE_MODEL_BANK_REQUIRED")
    if config.route == "source-qualification" and model_bank is not None:
        raise ValueError("prepared source qualification does not consume an outcome-fitted model bank")
    plan_bytes = _read_plan(plan)
    design_packet_bytes = _read_plan(design_packet)
    prior_bytes, prior_locator = _read_external(source_root, prior_exposure, _MAX_PRIOR)
    excluded_units, excluded_seeds = _prior_census(prior_bytes)
    plan_sha = sha256(plan_bytes).hexdigest()
    design_packet_sha = sha256(design_packet_bytes).hexdigest()
    bank_sha: str | None = None
    bank: InformationResponseModelBank | CausalResponseModelBank | None = None
    if model_bank is not None:
        bank_bytes, _ = _read_external(source_root, model_bank, _MAX_BANK)
        bank_sha = sha256(bank_bytes).hexdigest()
        bank_type = InformationResponseModelBank if config.route == "information-prediction" else CausalResponseModelBank
        bank = decode_canonical_bytes(
            _require_target_bank_document(loads_external_json(bank_bytes), bank_type.SCHEMA),
            bank_type,
            maximum_bytes=_MAX_BANK,
        )
        if bank.plan_sha256 != plan_sha:
            raise ValueError(
                "SIX_MATRIX_RESPONSE_MODEL_PLAN_MISMATCH: fitted bank differs from supplied plan"
            )
        excluded_units = tuple(
            sorted(
                set(excluded_units)
                | {
                    root.physical_unit_id
                    for context in bank.contexts
                    for root in context.training_roots
                }
            )
        )
        excluded_seeds = tuple(
            sorted(
                set(excluded_seeds)
                | {
                    root.seed_sha256
                    for context in bank.contexts
                    for root in context.training_roots
                }
            )
        )
    seed = config.source_seed_sha256
    lock_sha = sha256((repo_root / "uv.lock").read_bytes()).hexdigest()
    implementation_sha = _implementation_digest(repo_root)
    source_capability = (
        SOURCE_QUALIFICATION_SOURCE_CAPABILITY
        if config.route == "source-qualification"
        else INFORMATION_RESPONSE_SOURCE_CAPABILITY
        if config.route == "information-prediction"
        else CAUSAL_RESPONSE_SOURCE_CAPABILITY
    )
    recipe = PreparedNativeSpec(
        'qualification' if config.route == "source-qualification" else 'prospective-evaluation',
        seed,
        plan_sha,
        lock_sha,
        ObjectIdentity.from_record(source_capability.capability_key, source_capability),
        prepared_native_member(),
        tuple(prepared_numerical_view(value) for value in (1, 2)),
        None if config.route == "source-qualification" else Decimal(16),
        root_seed_census=config.root_seed_census,
    )
    source: PreparedNativeSpec | InformationResponseNativeConfig | CausalResponseNativeConfig
    if config.route == "source-qualification":
        source = recipe
        proposed_seeds = prepared_seed_ids(recipe)
        exposure_type = PreparedExposureInspection
        builder = build_prepared_response_source_qualification_authoring
        provider_type = PreparedResponseSourceQualificationCandidateContextProvider
    elif config.route == "information-prediction":
        assert isinstance(bank, InformationResponseModelBank)
        source = InformationResponseNativeConfig(recipe, bank)
        proposed_seeds = information_response_seed_ids(source)
        exposure_type = InformationResponseExposureInspection
        builder = build_information_response_authoring
        provider_type = InformationResponseCandidateContextProvider
    else:
        assert isinstance(bank, CausalResponseModelBank)
        source = CausalResponseNativeConfig(recipe, bank)
        proposed_seeds = causal_response_seed_ids(source)
        exposure_type = CausalResponseExposureInspection
        builder = build_causal_response_authoring
        provider_type = CausalResponseCandidateContextProvider
    proposed_units = tuple(sorted(root.physical_unit_id for root in source.roots))
    collisions = tuple(sorted(set(excluded_units) & set(proposed_units)))
    seed_collisions = tuple(sorted(set(excluded_seeds) & set(proposed_seeds)))
    if collisions or seed_collisions:
        raise ValueError("SIX_MATRIX_RESPONSE_EXPOSED_ROOT_OR_STREAM_COLLISION")
    prior = ResponseGeometryAssayExposureSource(
        prior_locator, sha256(prior_bytes).hexdigest(), len(prior_bytes), None, None
    )
    sources = (prior,)
    exposure = exposure_type(
        f"{config.target_prefix}.exposure-inspection",
        datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sha256(Path(__file__).read_bytes()).hexdigest(),
        source,
        (prior_locator.rsplit("/", 1)[0],),
        sources,
        sha256(canonical_json_bytes(sources)).hexdigest(),
        excluded_units,
        excluded_seeds,
        proposed_units,
        proposed_seeds,
        collisions,
        seed_collisions,
        (),
    )
    bundle = builder(
        design_packet_sha256=design_packet_sha,
        implementation_sha256=implementation_sha,
        exposure=exposure,
        prefix=config.target_prefix,
    )
    source_binding = (
        SOURCE_QUALIFICATION_SOURCE_BINDING
        if config.route == "source-qualification"
        else INFORMATION_RESPONSE_SOURCE_BINDING
        if config.route == "information-prediction"
        else CAUSAL_RESPONSE_SOURCE_BINDING
    )
    source_records = tuple(
        record
        for record in bundle.payloads
        if type(record)
        in (
            type(source),
            FreshSourceQualificationExperiment,
            FreshSourceQualificationSubstrateBinding,
        )
    )
    if len(source_records) != 3 or {type(record) for record in source_records} != {
        type(source),
        FreshSourceQualificationExperiment,
        FreshSourceQualificationSubstrateBinding,
    }:
        raise ValueError("SIX_MATRIX_RESPONSE_SOURCE_RECORDS_INCOMPLETE")
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.provider_factory(
        source_binding.binding_id
    )
    registry = GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry
    native_provider = factory.build_provider(
        registry=registry, records=source_records, platform_ports=()
    )
    native_runners = native_provider.runners(registry)
    if (
        len(native_runners) != 1
        or native_runners[0].manifest.capability_key != source_capability.capability_key
    ):
        raise ValueError("SIX_MATRIX_RESPONSE_NATIVE_RUNNER_BINDING_MISMATCH")
    return PreparedNativeAuthoring(
        bundle=bundle,
        exposure=exposure,
        context_provider_type=provider_type,
        selection_summary={
            "route": config.route,
            "experiment_id": bundle.authoring.base.draft.experiment.experiment_id,
            "independent_units": len(proposed_units),
            "nested_views_per_unit": 2,
            "prior_excluded_units_examined": len(excluded_units),
            "prior_excluded_streams_examined": len(excluded_seeds),
            "prior_exposure_sha256": prior.content_sha256,
            "prior_custody_authenticated": False,
            "model_bank_sha256": bank_sha,
            "design_packet_sha256": design_packet_sha,
            "native_plan_sha256": plan_sha,
            "native_source_binding_selected": source_binding.binding_id,
            "native_source_provider_built": True,
            "native_source_runner_selected": type(native_runners[0]).__name__,
            "native_contact": False,
            "implementation_worktree_sha256": implementation_sha,
            "model_bank_role": "EXPOSED_OUTCOME_FITTED_DEVELOPMENT"
            if bank is not None
            else None,
            "evidence_role": config.evidence_role,
            "campaign_issued": False,
            "native_tasks_executed": 0,
            "prospective_issue_eligible": False,
        },
    )


__all__ = [
    'PreparedResponseNativeAuthoringInput',
    "PreparedNativeAuthoring",
    'prepare_prepared_response_native_authoring',
]
