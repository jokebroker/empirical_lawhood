"""Diagnose incomplete development authoring without replacing failed inputs.

SPDX-License-Identifier: MPL-2.0
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(slots=True)
class AuthoringOutputProgress:
    directory: Path
    stage: str = "input export"


@contextmanager
def report_incomplete_output(directory: Path) -> Iterator[AuthoringOutputProgress]:
    """Enter only after a selected empty output directory is accepted."""

    progress = AuthoringOutputProgress(directory)
    try:
        yield progress
    except Exception as error:
        diagnostic = (
            f"Authoring output incomplete at {directory.absolute()} "
            f"during {progress.stage}. Retain this directory and use a fresh output path."
        )
        # Keep the original exception type, errno, filenames, cause and exit
        # handling. Notes alone would not appear in the CLI's str(error).
        if isinstance(error, OSError) and error.strerror is not None:
            error.strerror = f"{error.strerror}; {diagnostic}"
        else:
            error.args = (f"{error}; {diagnostic}", *error.args[1:])
        raise
