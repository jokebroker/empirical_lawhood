"""Canonical config and output bindings for adapter-authored protocols."""

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
)
from empirical_lawhood.runtime.plans import OutputTemplate

CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"


def capability_config_ref(
    config: CanonicalRecord, identity: ObjectIdentity, manifest: CapabilityManifest
) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        identity.object_id,
        config.SCHEMA,
        manifest.config_schema_sha256,
        config.fingerprint(),
        f"config-artifact.{identity.object_id}",
    )


def protocol_outputs(
    records: tuple[tuple[str, str], ...], hdf5: tuple[str, str] | None = None
) -> tuple[OutputTemplate, ...]:
    outputs = [
        OutputTemplate(
            name,
            schema,
            ArtifactProfile.CANONICAL_JSON,
            CANONICAL_MEDIA_TYPE,
            ".canonical.json",
        )
        for name, schema in records
    ]
    if hdf5 is not None:
        outputs.append(
            OutputTemplate(
                *hdf5, ArtifactProfile.AUDITED_HDF5, "application/x-hdf5", ".h5"
            )
        )
    return tuple(sorted(outputs, key=lambda value: value.output_id))


def record_stable_id(record: CanonicalRecord) -> str:
    for field in ("extension_set_id", "binding_id", "config_id", "profile_id"):
        value = getattr(record, field, None)
        if isinstance(value, str):
            return value
    raise ValueError("native authoring record lacks its declared stable identity")
