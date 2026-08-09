"""Persistence helpers for raw acquisition datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def persist_dataset(dataset: Any, path: Path, metadata: dict[str, Any]) -> None:
    """Write an acquisition dataset and a JSON description of its circuits."""
    dataset.to_netcdf(path)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
