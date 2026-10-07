"""Generate the explicit static executable-binding and codec aggregate module."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Sequence

from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityProviderFactoryRegistry, GeneratedExecutableBindingAggregate, compose_executable_binding_aggregate
from empirical_lawhood.runtime.static_codecs import CanonicalRecordCodecRegistration, CanonicalRecordCodecRegistry


try:
    from ._descriptor_discovery import (
        discover_descriptors as _discover_descriptors,
        module_name,
        require_safe_output,
    )
except ImportError:  # Explicit script invocation outside the checkout.
    from _descriptor_discovery import (
        discover_descriptors as _discover_descriptors,
        module_name,
        require_safe_output,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src/empirical_lawhood"
ALLOWLISTED_ROOTS = (
    PACKAGE_ROOT / "adapters/control",
    PACKAGE_ROOT / "adapters/methods",
    PACKAGE_ROOT / "adapters/physical",
    PACKAGE_ROOT / "adapters/simulators",
)
OUTPUT = PACKAGE_ROOT / "adapters/composition/generated_executable_bindings.py"
DESCRIPTOR_NAME = "executable_binding.py"
CONTRIBUTION_SYMBOL = "EXECUTABLE_BINDING_CONTRIBUTION"
FACTORIES_SYMBOL = "EXECUTABLE_BINDING_FACTORIES"
RECORD_TYPES_SYMBOL = "EXECUTABLE_RECORD_TYPES"


@dataclass(frozen=True, slots=True)
class LoadedExecutableDescriptor:
    path: Path
    module_name: str
    contribution: ExecutableBindingContribution
    factories: tuple[object, ...]
    record_types: tuple[type[CanonicalRecord], ...]


def _module_name(path: Path) -> str:
    return module_name(path, package_root=PACKAGE_ROOT)


def discover_descriptors() -> tuple[Path, ...]:
    return _discover_descriptors(
        package_root=PACKAGE_ROOT,
        roots=ALLOWLISTED_ROOTS,
        name=DESCRIPTOR_NAME,
        kind="executable",
    )


def load_descriptors(paths: tuple[Path, ...]) -> tuple[LoadedExecutableDescriptor, ...]:
    loaded: list[LoadedExecutableDescriptor] = []
    for path in paths:
        module_name = _module_name(path)
        module = importlib.import_module(module_name)
        contribution = getattr(module, CONTRIBUTION_SYMBOL, None)
        factories = getattr(module, FACTORIES_SYMBOL, None)
        record_types = getattr(module, RECORD_TYPES_SYMBOL, None)
        if not isinstance(contribution, ExecutableBindingContribution):
            raise ValueError(f"executable descriptor omits {CONTRIBUTION_SYMBOL}: {path}")
        if not isinstance(factories, tuple):
            raise ValueError(f"executable descriptor omits {FACTORIES_SYMBOL}: {path}")
        if not isinstance(record_types, tuple) or any(
            not isinstance(value, type) or not issubclass(value, CanonicalRecord)
            for value in record_types
        ):
            raise ValueError(f"executable descriptor omits {RECORD_TYPES_SYMBOL}: {path}")
        loaded.append(
            LoadedExecutableDescriptor(
                path=path,
                module_name=module_name,
                contribution=contribution,
                factories=factories,
                record_types=record_types,
            )
        )
    ordered = tuple(sorted(loaded, key=lambda value: value.contribution.contribution_id))
    contribution_ids = tuple(value.contribution.contribution_id for value in ordered)
    if tuple(sorted(set(contribution_ids))) != contribution_ids:
        raise ValueError("executable descriptors contain duplicate contribution IDs")
    return ordered


def _codec_registrations(
    descriptors: tuple[LoadedExecutableDescriptor, ...],
    aggregate: GeneratedExecutableBindingAggregate,
) -> tuple[
    tuple[CanonicalRecordCodecRegistration, ...],
    tuple[type[CanonicalRecord], ...],
]:
    record_types = tuple(
        sorted(
            (record_type for descriptor in descriptors for record_type in descriptor.record_types),
            key=lambda value: value.SCHEMA,
        )
    )
    schemas = tuple(value.SCHEMA for value in record_types)
    if tuple(sorted(set(schemas))) != schemas:
        raise ValueError("executable descriptors duplicate a canonical record schema")
    all_decoders = tuple(
        sorted(
            (
                registration
                for binding in aggregate.bindings
                for registration in binding.issued_decoder_registrations
            ),
            key=lambda value: value.payload_schema,
        )
    )
    by_decoder_schema: dict[str, StudyExtensionDecoderRegistration] = {}
    for decoder in all_decoders:
        previous = by_decoder_schema.setdefault(decoder.payload_schema, decoder)
        if previous != decoder:
            raise ValueError("executable bindings duplicate an issued decoder schema")
    decoders = tuple(by_decoder_schema[schema] for schema in sorted(by_decoder_schema))
    decoder_schemas = tuple(value.payload_schema for value in decoders)
    if schemas != decoder_schemas:
        raise ValueError("executable record types differ from issued decoder schemas")
    by_schema = {value.SCHEMA: value for value in record_types}
    registrations = tuple(
        CanonicalRecordCodecRegistration(
            codec_id=(
                "executable-canonical-json."
                + decoder.payload_schema.replace("/", ".").replace("_", "-")
            ),
            record_schema=decoder.payload_schema,
            record_version=decoder.payload_version,
            maximum_bytes=decoder.maximum_payload_bytes,
            decoder_key=decoder.decoder_key,
            decoder_version=decoder.decoder_version,
            decoder_implementation_sha256=decoder.implementation_sha256,
        )
        for decoder in decoders
    )
    for registration in registrations:
        record_type = by_schema[registration.record_schema]
        if record_type.VERSION != registration.record_version:
            raise ValueError("executable codec record version differs from its decoder")
    CanonicalRecordCodecRegistry(
        registry_id="canonical-codecs.generated-executable-bindings",
        registrations=registrations,
        record_types=by_schema,
    )
    return registrations, record_types


def compose_loaded_descriptors(
    descriptors: tuple[LoadedExecutableDescriptor, ...],
) -> tuple[
    GeneratedExecutableBindingAggregate,
    ExecutableCapabilityProviderFactoryRegistry,
    tuple[CanonicalRecordCodecRegistration, ...],
    tuple[type[CanonicalRecord], ...],
]:
    contributions = tuple(value.contribution for value in descriptors)
    aggregate = compose_executable_binding_aggregate(
        contributions,
        discovery_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
    )
    factories = tuple(
        sorted(
            (factory for descriptor in descriptors for factory in descriptor.factories),
            key=lambda value: value.binding.binding_id,  # type: ignore[attr-defined]
        )
    )
    factory_registry = ExecutableCapabilityProviderFactoryRegistry(
        aggregate=aggregate,
        factories=factories,  # type: ignore[arg-type]
    )
    codec_registrations, record_types = _codec_registrations(descriptors, aggregate)
    return aggregate, factory_registry, codec_registrations, record_types


def _render_codec_registration(value: CanonicalRecordCodecRegistration) -> str:
    return "\n".join(
        (
            "    CanonicalRecordCodecRegistration(",
            f"        codec_id={json.dumps(value.codec_id)},",
            f"        record_schema={json.dumps(value.record_schema)},",
            f"        record_version={json.dumps(value.record_version)},",
            f"        maximum_bytes={value.maximum_bytes},",
            f"        decoder_key={json.dumps(value.decoder_key)},",
            f"        decoder_version={json.dumps(value.decoder_version)},",
            "        decoder_implementation_sha256="
            f"{json.dumps(value.decoder_implementation_sha256)},",
            "    ),",
        )
    )


def render_generated_module(paths: tuple[Path, ...]) -> str:
    descriptors = load_descriptors(paths)
    aggregate, _factories, codec_registrations, _record_types = compose_loaded_descriptors(
        descriptors
    )
    aliases = {
        value.contribution.contribution_id: f"_descriptor_{index:03d}"
        for index, value in enumerate(descriptors)
    }
    imports: list[str] = []
    for descriptor in descriptors:
        alias = aliases[descriptor.contribution.contribution_id]
        imports.extend(
            (
                f"from {descriptor.module_name} import (",
                f"    {CONTRIBUTION_SYMBOL} as {alias}_contribution,",
                f"    {FACTORIES_SYMBOL} as {alias}_factories,",
                f"    {RECORD_TYPES_SYMBOL} as {alias}_record_types,",
                ")",
            )
        )
    contribution_values = ",\n".join(
        f"    {aliases[value.contribution.contribution_id]}_contribution" for value in descriptors
    )
    factory_values = ",\n".join(
        f"                *{aliases[value.contribution.contribution_id]}_factories"
        for value in descriptors
    )
    record_type_values = ",\n".join(
        f"            *{aliases[value.contribution.contribution_id]}_record_types"
        for value in descriptors
    )
    codec_values = "\n".join(_render_codec_registration(value) for value in codec_registrations)
    codec_fingerprint = hashlib.sha256(canonical_json_bytes(codec_registrations)).hexdigest()
    return "\n".join(
        (
            '"""Generated by scripts/generate_executable_binding_aggregate.py; do not edit."""',
            "",
            "import hashlib",
            "from typing import cast",
            "",
            "from empirical_lawhood.adapters.composition.generated_extension_bundles import (",
            "    GENERATED_EXTENSION_BUNDLE_AGGREGATE,",
            ")",
            "from empirical_lawhood.kernel.serialization import canonical_json_bytes",
            "from empirical_lawhood.runtime.executable_bindings import (",
            "    ExecutableCapabilityProviderFactoryRegistry,",
            "    ExecutableFactory,",
            "    compose_executable_binding_aggregate,",
            ")",
            "from empirical_lawhood.runtime.static_codecs import (",
            "    CanonicalRecordCodecRegistration,",
            "    CanonicalRecordCodecRegistry,",
            ")",
            *imports,
            "",
            "EXECUTABLE_BINDING_CONTRIBUTIONS = (",
            f"{contribution_values},",
            ")",
            "",
            "GENERATED_EXECUTABLE_BINDING_AGGREGATE = compose_executable_binding_aggregate(",
            "    EXECUTABLE_BINDING_CONTRIBUTIONS,",
            "    discovery_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,",
            ")",
            "_EXECUTABLE_FACTORIES: tuple[ExecutableFactory, ...] = tuple(",
            "    sorted(",
            "        cast(",
            "            tuple[ExecutableFactory, ...],",
            "            (",
            f"{factory_values},",
            "            ),",
            "        ),",
            "        key=lambda value: value.binding.binding_id,",
            "    )",
            ")",
            "EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY = ExecutableCapabilityProviderFactoryRegistry(",
            "    aggregate=GENERATED_EXECUTABLE_BINDING_AGGREGATE,",
            "    factories=_EXECUTABLE_FACTORIES,",
            ")",
            "",
            "EXECUTABLE_RECORD_TYPES = tuple(",
            "    sorted(",
            "        (",
            f"{record_type_values},",
            "        ),",
            "        key=lambda value: value.SCHEMA,",
            "    )",
            ")",
            "_EXECUTABLE_CODEC_REGISTRATIONS = (",
            f"{codec_values}",
            ")",
            "EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY = CanonicalRecordCodecRegistry(",
            '    registry_id="canonical-codecs.generated-executable-bindings",',
            "    registrations=_EXECUTABLE_CODEC_REGISTRATIONS,",
            "    record_types={value.SCHEMA: value for value in EXECUTABLE_RECORD_TYPES},",
            ")",
            "",
            f'EXPECTED_AGGREGATE_FINGERPRINT = "{aggregate.fingerprint()}"',
            "EXPECTED_CODEC_REGISTRY_FINGERPRINT = (",
            f'    "{codec_fingerprint}"',
            ")",
            "if GENERATED_EXECUTABLE_BINDING_AGGREGATE.fingerprint() != EXPECTED_AGGREGATE_FINGERPRINT:",
            '    raise RuntimeError("generated executable aggregate fingerprint drifted")',
            "if (",
            "    hashlib.sha256(canonical_json_bytes(_EXECUTABLE_CODEC_REGISTRATIONS)).hexdigest()",
            "    != EXPECTED_CODEC_REGISTRY_FINGERPRINT",
            "):",
            '    raise RuntimeError("generated executable codec registry fingerprint drifted")',
            "",
            "__all__ = [",
            '    "EXECUTABLE_BINDING_CONTRIBUTIONS",',
            '    "EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY",',
            '    "EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY",',
            '    "EXECUTABLE_RECORD_TYPES",',
            '    "EXPECTED_AGGREGATE_FINGERPRINT",',
            '    "EXPECTED_CODEC_REGISTRY_FINGERPRINT",',
            '    "GENERATED_EXECUTABLE_BINDING_AGGREGATE",',
            "]",
            "",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated output drifts")
    arguments = parser.parse_args(argv)
    expected = render_generated_module(discover_descriptors()).encode("utf-8")
    require_safe_output(OUTPUT, package_root=PACKAGE_ROOT)
    if arguments.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != expected:
            raise SystemExit("generated executable binding aggregate is missing or stale")
        return 0
    OUTPUT.write_bytes(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
