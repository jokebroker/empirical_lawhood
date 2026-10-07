"""Injected held-source and external-scratch boundaries for Virtual Cell runners."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import BinaryIO, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess

from .contracts import VirtualCellSourceObject


class VirtualCellSourcePort(Protocol):
    """Open only a manifest-bound object under the supplied outcome authority."""

    def verify_available(self, source: VirtualCellSourceObject) -> None: ...

    def open_object(
        self,
        source: VirtualCellSourceObject,
        *,
        outcome_access: OutcomeAccess,
    ) -> AbstractContextManager[BinaryIO]: ...


class VirtualCellScratchWorkspace(Protocol):
    """One exact nonexisting workspace below the authoritative external root."""

    @property
    def workspace_id(self) -> str: ...

    def reserve_path(self, filename: str) -> Path: ...

    def open_existing(self, filename: str, mode: str) -> BinaryIO: ...

    def close(self) -> None: ...


class VirtualCellScratchPort(Protocol):
    """Allocate no-fallback scratch for one frozen execution attempt."""

    def allocate(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt_id: str,
    ) -> VirtualCellScratchWorkspace: ...


__all__ = [
    "VirtualCellScratchPort",
    "VirtualCellScratchWorkspace",
    "VirtualCellSourcePort",
]
