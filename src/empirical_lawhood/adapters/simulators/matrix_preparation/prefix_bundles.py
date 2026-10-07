"""Lossless four-root input materializations with original per-root custody."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeSegmentResult
from .contracts import CANONICAL_MEDIA_TYPE, CONTEXTS, DEVELOPMENT, PREFIX_BUNDLE_MAXIMUM_BYTES, PREFIX_BUNDLE_SCHEMA, PreparationPrefixBundleRef, PreparationRoot, RetainedPreparationPrefix


@dataclass(frozen=True, slots=True)
class PreparationPrefixBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = PREFIX_BUNDLE_SCHEMA
    context: str
    group_index: int
    prefixes: tuple[RetainedPreparationPrefix, ...]
    records: tuple[ResponseGeometryDevelopmentNativeSegmentResult, ...]

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or type(self.group_index) is not int
            or not 0 <= self.group_index < 16
            or len(self.records) != 4
            or tuple(p.root for p in self.prefixes)
            != tuple(PreparationRoot(self.context, 4 * self.group_index + i) for i in range(4))
        ):
            raise ValueError("retained input bundle must preserve four exact consecutive roots")
        for prefix, record in zip(self.prefixes, self.records, strict=True):
            if (
                type(record) is not ResponseGeometryDevelopmentNativeSegmentResult
                or record.source_config != prefix.source_config
                or record.source_config != self.records[0].source_config
                or record.segment.root.root_id != prefix.source_root_id
                or record.segment.phase != "prefix"
                or record.segment.end_tick != prefix.root.landmark
                or record.fingerprint() != prefix.artifact.sha256
                or len(record.canonical_bytes()) != prefix.artifact.size_bytes
            ):
                raise ValueError("retained input bundle changes an original root/checkpoint hash")

    @property
    def bundle_id(self) -> str:
        return f"{DEVELOPMENT}.prefix-bundle.{self.context}.g{self.group_index:02d}"

    @property
    def reference(self) -> PreparationPrefixBundleRef:
        payload = self.canonical_bytes()
        if len(payload) > PREFIX_BUNDLE_MAXIMUM_BYTES:
            raise ValueError("retained four-root bundle exceeds its declared bound")
        return PreparationPrefixBundleRef(
            self.context,
            self.group_index,
            ArtifactIdentity(
                f"artifact.{self.bundle_id}",
                "authenticated-retained-four-root-input",
                self.SCHEMA,
                self.fingerprint(),
                CANONICAL_MEDIA_TYPE,
                len(payload),
            ),
        )

    def for_root(self, prefix: RetainedPreparationPrefix) -> ResponseGeometryDevelopmentNativeSegmentResult:
        if prefix not in self.prefixes:
            raise ValueError("native task cannot select another bundle's root")
        return self.records[self.prefixes.index(prefix)]
