"""Explicit path inputs for retained, externally provisioned adapter routines."""

# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
from pathlib import Path
import re


class ExternalInputRequired(RuntimeError):
    """A caller must supply one named external input before native work."""


def required_external_path(variable: str) -> Path:
    """Reject an absent path instead of inventing an operator storage default."""
    raw = os.environ.get(variable)
    if not raw:
        raise ExternalInputRequired(f"{variable} must name the required external input path")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{variable} must be an absolute path")
    return path


def required_external_sha256(variable: str) -> str:
    """Require an exact caller-provided digest for an external input.

    A digest from an earlier private run cannot identify a new target record.
    The adapter must subsequently compare this expectation to observed bytes.
    """
    value = os.environ.get(variable)
    if value is None:
        raise ExternalInputRequired(f"{variable} must name the required external SHA-256")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{variable} must be a lowercase SHA-256")
    return value
