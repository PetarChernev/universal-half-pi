"""Persistence helpers for raw acquisition datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def persist_dataset(dataset: Any, path: Path) -> None:
    """Write an acquisition dataset to a NetCDF file."""
    dataset.to_netcdf(path)
