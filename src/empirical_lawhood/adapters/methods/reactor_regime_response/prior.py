"""Authenticated cross-issue package carrier for reactor C and D candidates.

The composition layer verifies the retained manifest on the external plane
before constructing this value.  A C worker checks the same B receipt and
content identity again after the public runtime imports the exact bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt


@dataclass(frozen=True, slots=True)
class RegimePriorArtifact:
    payload: bytes
    manifest: ArtifactManifest
    receipt: CanonicalTaskReceipt
    expected_task_id: str

    def __post_init__(self) -> None:
        if (
            self.expected_task_id not in (
                "regime.fit", "regime.nomination", "regime.calibration",
                "regime.qualification", "regime.law-qualification",
            )
            or self.receipt.task_id != self.expected_task_id
            or self.receipt.operational_status is not OperationalStatus.SUCCEEDED
            or self.manifest.publication is None
            or self.manifest.logical not in self.receipt.output_logical_artifacts
            or self.manifest.materialization not in self.receipt.output_materializations
            or self.manifest.materialization.size_bytes != len(self.payload)
            or self.manifest.logical.content_sha256 != sha256(self.payload).hexdigest()
        ):
            raise ValueError("reactor prior phase artifact lacks exact successful custody")
