"""Bounded, lossless software operand transport; no pickle or scientific custody."""

from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
import numpy as np
from .numerical import FrozenFit, SupportCell
from .discovery import Rows, discover
from .calibration import RootOperands, calibrate


def fit_data(model: FrozenFit) -> dict[str, Any]:
    return asdict(model)


def read_fit(data: dict[str, Any]) -> FrozenFit:
    if set(data) != {
        "family",
        "penalty",
        "mean",
        "scale",
        "operator",
        "support",
        "training_digest",
        "rows",
        "columns",
    }:
        raise ValueError("unknown/missing fit field")
    support = []
    for c in data["support"]:
        if set(c) != {"clock_stratum", "action", "roots", "lower", "upper"}:
            raise ValueError("unknown/missing support field")
        support.append(
            SupportCell(
                c["clock_stratum"],
                c["action"],
                tuple(c["roots"]),
                tuple(c["lower"]),
                tuple(c["upper"]),
            )
        )
    return FrozenFit(
        data["family"],
        data["penalty"],
        tuple(data["mean"]),
        tuple(data["scale"]),
        tuple(map(tuple, data["operator"])),
        tuple(support),
        data["training_digest"],
        data["rows"],
        tuple(data["columns"]),
    )


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def produce_discovery(raw_path: Path, output: Path) -> None:
    with np.load(raw_path, allow_pickle=False) as raw:
        rows = Rows(
            tuple(raw["roots"].tolist()),
            tuple(raw["roles"].tolist()),
            raw["clocks"],
            raw["actions"],
            raw["features"],
            raw["labels"],
            raw["delivery_valid"],
        )
    result = discover(rows, require_full_census=False)
    output.write_bytes(json_bytes(asdict(result)))


def read_roots(raw_path: Path) -> tuple[RootOperands, ...]:
    with np.load(raw_path, allow_pickle=False) as raw:
        keys = set(RootOperands.__dataclass_fields__) - {"root"}
        if set(raw.files) != keys | {"roots"}:
            raise ValueError("raw root operand fields differ")
        # NPZ members are decompressed on every lookup. Slice one retained
        # array per field, rather than retaining 32 copies of the entire panel.
        arrays = {key: raw[key] for key in keys}
        return tuple(
            RootOperands(str(root), **{key: arrays[key][i] for key in keys})
            for i, root in enumerate(raw["roots"])
        )


def produce_calibration(raw_path: Path, fit_path: Path, output: Path) -> None:
    model = read_fit(json.loads(fit_path.read_bytes()))
    output.write_bytes(json_bytes(asdict(calibrate(read_roots(raw_path), model))))
