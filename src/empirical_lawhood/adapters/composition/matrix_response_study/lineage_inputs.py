"""Explicit parent identities for reusable Six-matrix response design constructors."""

# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from empirical_lawhood.kernel.serialization import (
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class MatrixResponseLineageInputs:
    """Caller supplied parent records, with no embedded earlier campaign bytes."""

    values: Mapping[str, object]

    def _get(self, name: str) -> object:
        if name not in self.values:
            raise ValueError(f"Six-matrix response parent input {name} is required")
        return self.values[name]

    def sha256(self, name: str) -> str:
        value = self._get(name)
        if not isinstance(value, str):
            raise ValueError(f"Six-matrix response parent input {name} must be a SHA-256")
        validate_sha256(value, field_name=name)
        return value

    def git_commit(self, name: str) -> str:
        value = self._get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"Six-matrix response parent input {name} must be a Git commit ID")
        return value

    def locator(self, name: str) -> str:
        value = self._get(name)
        if not isinstance(value, str):
            raise ValueError(f"Six-matrix response parent input {name} must be a relative locator")
        validate_relative_locator(value)
        return value

    def stable_id(self, name: str) -> str:
        value = self._get(name)
        if not isinstance(value, str):
            raise ValueError(f"Six-matrix response parent input {name} must be a stable ID")
        validate_stable_id(value, field_name=name)
        return value

    def positive_int(self, name: str) -> int:
        value = self._get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Six-matrix response parent input {name} must be a positive integer")
        return value

    def stable_ids(self, name: str) -> tuple[str, ...]:
        value = self._get(name)
        if not isinstance(value, tuple) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"Six-matrix response parent input {name} must be a tuple of IDs")
        require_sorted_unique_strings(value, field_name=name, allow_empty=False)
        for item in value:
            validate_stable_id(item, field_name=name)
        return value

    def locators(self, name: str) -> tuple[str, ...]:
        value = self._get(name)
        if not isinstance(value, tuple) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"Six-matrix response parent input {name} must be a tuple of locators")
        require_sorted_unique_strings(value, field_name=name, allow_empty=False)
        for item in value:
            validate_relative_locator(item)
        return value


__all__ = ['MatrixResponseLineageInputs']
