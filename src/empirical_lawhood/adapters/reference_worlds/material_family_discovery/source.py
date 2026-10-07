"""Bounded, custody-checked NIMS source reader for SDCB-SC."""

from __future__ import annotations

from contextlib import AbstractContextManager, closing
import csv
import hashlib
import io
from pathlib import Path
import stat
from typing import BinaryIO, Iterator

from .contracts import MaterialSourceManifest, MaterialSourceObject


_HASH_CHUNK_BYTES = 8 * 1024**2
_EXPECTED_COLUMNS = (
    "num",
    "refno",
    "commt",
    "name",
    "element",
    "ma1",
    "ma2",
    "mb1",
    "mb2",
    "mc1",
    "mc2",
    "md1",
    "md2",
    "me1",
    "me2",
    "mf1",
    "mf2",
    "mg1",
    "mg2",
    "mh1",
    "mh2",
    "mi1",
    "mi2",
    "mj1",
    "mj2",
    "mo1",
    "mo2",
)


class MaterialSourceError(ValueError):
    """Typed bounded-source or schema failure."""


class HeldMaterialFamilySource:
    """Open only exact, hash-qualified objects below one non-symlink root."""

    def __init__(self, *, source_root: Path, manifest: MaterialSourceManifest) -> None:
        self._source_root = source_root.absolute()
        self._manifest = manifest
        self._objects = {value.object_id: value for value in manifest.objects}
        self._verified: set[str] = set()
        observed = self._source_root.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise MaterialSourceError("held material source root is not a real directory")
        if self._source_root.resolve(strict=True) != self._source_root:
            raise MaterialSourceError("held material source root resolves through a symlink")

    def object(self, object_id: str) -> MaterialSourceObject:
        try:
            return self._objects[object_id]
        except KeyError as error:
            raise MaterialSourceError("source object is not bound by the manifest") from error

    def _path(self, source: MaterialSourceObject) -> Path:
        if self._objects.get(source.object_id) != source:
            raise MaterialSourceError("source object differs from the held manifest")
        path = (self._source_root / source.relative_locator).absolute()
        try:
            path.relative_to(self._source_root)
        except ValueError as error:
            raise MaterialSourceError("source object escapes the held root") from error
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise MaterialSourceError("source object is not a regular non-symlink file")
        return path

    def verify(self, source: MaterialSourceObject) -> None:
        identity = source.fingerprint()
        if identity in self._verified:
            return
        path = self._path(source)
        if path.stat().st_size != source.size_bytes:
            raise MaterialSourceError("source object byte count differs")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        if digest.hexdigest() != source.sha256:
            raise MaterialSourceError("source object SHA-256 differs")
        self._verified.add(identity)

    def open(self, object_id: str) -> AbstractContextManager[BinaryIO]:
        source = self.object(object_id)
        self.verify(source)
        return closing(self._path(source).open("rb"))


def parse_nims_supercon(
    stream: BinaryIO,
    *,
    maximum_rows: int = 40_000,
    maximum_line_bytes: int = 128 * 1024,
) -> tuple[tuple[str, ...], Iterator[dict[str, str]]]:
    """Parse the exact three-header-row, 191-column O&M table.

    The iterator shares the supplied stream and must be consumed before the
    caller closes it.  CP932 is required by the dated source bytes; replacement
    decoding is forbidden.
    """

    text = io.TextIOWrapper(stream, encoding="cp932", errors="strict", newline="")
    reader = csv.reader(text, delimiter="\t")
    try:
        numeric_header = tuple(next(reader))
        descriptive_header = tuple(next(reader))
        columns = tuple(next(reader))
    except StopIteration as error:
        raise MaterialSourceError("NIMS O&M table lacks its three header rows") from error
    if len(numeric_header) != 191 or len(descriptive_header) != 191 or len(columns) != 191:
        raise MaterialSourceError("NIMS O&M header width differs")
    missing = tuple(value for value in _EXPECTED_COLUMNS if value not in columns)
    if missing:
        raise MaterialSourceError(f"NIMS O&M required columns are missing: {missing}")

    def records() -> Iterator[dict[str, str]]:
        observed = 0
        for row in reader:
            observed += 1
            if observed > maximum_rows:
                raise MaterialSourceError("NIMS O&M row bound exceeded")
            if len(row) != len(columns):
                raise MaterialSourceError("NIMS O&M row width differs")
            if sum(len(value.encode("cp932")) for value in row) > maximum_line_bytes:
                raise MaterialSourceError("NIMS O&M line bound exceeded")
            yield dict(zip(columns, row, strict=True))

    return columns, records()


__all__ = ["HeldMaterialFamilySource", "MaterialSourceError", "parse_nims_supercon"]
