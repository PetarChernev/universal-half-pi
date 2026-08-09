"""Persistence helpers for raw acquisition datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

from xarray import Dataset


JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | Mapping[str, "JsonValue"]
)


def persist_dataset(
    dataset: Dataset,
    path: Path,
    metadata: Mapping[str, JsonValue],
) -> None:
    """Write an acquisition dataset and a JSON description of its circuits."""
    dataset.to_netcdf(path)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
