"""Structured experiment results with notebook-friendly DataFrame views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .acquisition import Locus
from .points import ExperimentPoint


class ResultStatus(str, Enum):
    """Outcome of one technique at one point and locus."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


def _readonly_artifacts(artifacts: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    copied = {}
    for name, values in artifacts.items():
        array = np.asarray(values).copy()
        array.setflags(write=False)
        copied[name] = array
    return MappingProxyType(copied)


@dataclass(frozen=True, eq=False)
class TechniqueResult:
    """Metrics and diagnostic arrays produced by one technique for one locus."""

    point_id: str
    target_id: str
    technique_id: str
    locus: Locus
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    status: ResultStatus = ResultStatus.SUCCESS
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.point_id or not self.target_id or not self.technique_id:
            raise ValueError("Point, target, and technique IDs cannot be empty.")
        metric_values = {name: float(value) for name, value in self.metrics.items()}
        if not all(np.isfinite(value) for value in metric_values.values()):
            raise ValueError("Technique metrics must be finite.")
        object.__setattr__(self, "locus", tuple(self.locus))
        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
        object.__setattr__(self, "artifacts", _readonly_artifacts(self.artifacts))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if self.status is ResultStatus.SUCCESS and self.error is not None:
            raise ValueError("Successful results cannot contain an error message.")

    @classmethod
    def failed(
        cls,
        *,
        point: ExperimentPoint,
        technique_id: str,
        locus: Locus,
        error: Exception,
    ) -> TechniqueResult:
        """Construct a failed result suitable for record-and-continue runs."""

        return cls(
            point_id=point.point_id,
            target_id=point.target.operation_id,
            technique_id=technique_id,
            locus=locus,
            status=ResultStatus.FAILED,
            error=f"{type(error).__name__}: {error}",
        )


@dataclass(frozen=True)
class PointResult:
    """All technique outputs associated with one experimental point."""

    point: ExperimentPoint
    techniques: tuple[TechniqueResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "techniques", tuple(self.techniques))


@dataclass(frozen=True)
class CalibrationResult:
    """Scalar calibration information retained with an experiment run."""

    calibration_id: str
    metrics_by_locus: Mapping[Locus, Mapping[str, float]]

    def __post_init__(self) -> None:
        normalized = {
            tuple(locus): MappingProxyType({name: float(value) for name, value in metrics.items()})
            for locus, metrics in self.metrics_by_locus.items()
        }
        object.__setattr__(self, "metrics_by_locus", MappingProxyType(normalized))


@dataclass(frozen=True)
class ExperimentResult:
    """Canonical output of an experiment run.

    The structured representation supports optional techniques and array-valued
    artifacts.  ``metrics_dataframe`` and ``legacy_dataframe`` provide familiar
    tabular views for notebook exploration and CSV export.
    """

    points: tuple[PointResult, ...]
    calibrations: tuple[CalibrationResult, ...] = ()
    run_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "calibrations", tuple(self.calibrations))

    @staticmethod
    def _point_columns(point: ExperimentPoint) -> dict[str, Any]:
        columns: dict[str, Any] = {
            "point_id": point.point_id,
            "target_id": point.target.operation_id,
        }
        columns.update(point.target.metadata())
        columns.update(dict(point.resolved_parameters))
        columns.update(point.metadata)
        return columns

    def metrics_dataframe(self) -> pd.DataFrame:
        """Return one wide metric row per point, technique, and locus."""

        rows = []
        for point_result in self.points:
            for result in point_result.techniques:
                row = {
                    "run_id": self.run_id,
                    **self._point_columns(point_result.point),
                    "technique_id": result.technique_id,
                    "locus": ",".join(result.locus),
                    "status": result.status.value,
                    "error": result.error,
                }
                row.update(result.metrics)
                rows.append(row)
        return pd.DataFrame(rows)

    def long_dataframe(self) -> pd.DataFrame:
        """Return one row per scalar metric for tidy plotting and grouping."""

        rows = []
        for point_result in self.points:
            point_columns = self._point_columns(point_result.point)
            for result in point_result.techniques:
                for metric, value in result.metrics.items():
                    rows.append(
                        {
                            "run_id": self.run_id,
                            **point_columns,
                            "technique_id": result.technique_id,
                            "locus": ",".join(result.locus),
                            "metric": metric,
                            "value": value,
                            "status": result.status.value,
                        }
                    )
        return pd.DataFrame(rows)

    def legacy_dataframe(self) -> pd.DataFrame:
        """Return one row per point and locus with all technique metrics merged."""

        rows_by_key: dict[tuple[str, Locus], dict[str, Any]] = {}
        for point_result in self.points:
            for result in point_result.techniques:
                key = (point_result.point.point_id, result.locus)
                row = rows_by_key.setdefault(
                    key,
                    {
                        "sequence": point_result.point.target.operation_id,
                        **self._point_columns(point_result.point),
                        "qubit": result.locus[0] if len(result.locus) == 1 else ",".join(result.locus),
                    },
                )
                row.update(result.metrics)
                if result.status is not ResultStatus.SUCCESS:
                    row[f"{result.technique_id}_status"] = result.status.value
                    row[f"{result.technique_id}_error"] = result.error
        return pd.DataFrame(rows_by_key.values())
