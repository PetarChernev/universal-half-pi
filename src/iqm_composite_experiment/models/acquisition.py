"""Backend-neutral acquisition plans and normalized measurement data."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np

Locus: TypeAlias = tuple[str, ...]
"""Physical components on which one logical operation acts."""

CircuitRecipe: TypeAlias = Callable[[Any], object]
"""Deferred circuit construction using a backend's circuit factory.

IQM Pulla 14 creates its schedule builder during compilation, so circuits cannot be
built when a technique creates its plan.  A recipe keeps the scientific plan
backend-neutral and builds the concrete circuit only when the backend is ready.
"""


@dataclass(frozen=True)
class PlannedCircuit:
    """One deferred circuit and the coordinates that identify its meaning."""

    circuit_id: str
    build: CircuitRecipe
    coordinates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.circuit_id:
            raise ValueError("Circuit IDs cannot be empty.")
        object.__setattr__(self, "coordinates", MappingProxyType(dict(self.coordinates)))


@dataclass(frozen=True)
class AcquisitionPlan:
    """Circuits that can be compiled and executed as one hardware batch."""

    technique_id: str
    measurement_key: str
    shots: int
    circuits: tuple[PlannedCircuit, ...]

    def __post_init__(self) -> None:
        if not self.technique_id or not self.measurement_key:
            raise ValueError("Technique IDs and measurement keys cannot be empty.")
        if self.shots < 1:
            raise ValueError("An acquisition needs at least one shot.")
        object.__setattr__(self, "circuits", tuple(self.circuits))
        if not self.circuits:
            raise ValueError("An acquisition plan cannot be empty.")
        identifiers = [circuit.circuit_id for circuit in self.circuits]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Circuit IDs must be unique within an acquisition plan.")


def _readonly_vector(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(vector)):
        raise ValueError("Measurement probabilities must be finite.")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, eq=False)
class ProbabilityData:
    """Raw and optionally readout-corrected P(1) values for one locus."""

    raw_p1: np.ndarray
    corrected_p1: np.ndarray | None = None

    def __post_init__(self) -> None:
        raw = _readonly_vector(self.raw_p1)
        corrected = None if self.corrected_p1 is None else _readonly_vector(self.corrected_p1)
        if corrected is not None and len(corrected) != len(raw):
            raise ValueError("Raw and corrected probability arrays must have equal length.")
        object.__setattr__(self, "raw_p1", raw)
        object.__setattr__(self, "corrected_p1", corrected)

    @property
    def p1(self) -> np.ndarray:
        """Return corrected probabilities when available, otherwise raw values."""

        return self.raw_p1 if self.corrected_p1 is None else self.corrected_p1


@dataclass(frozen=True)
class MeasurementData:
    """Normalized probability observations returned by an experiment backend."""

    plan: AcquisitionPlan
    probabilities: Mapping[Locus, ProbabilityData]
    execution_id: str | None = None

    def __post_init__(self) -> None:
        normalized = {tuple(locus): data for locus, data in self.probabilities.items()}
        expected = len(self.plan.circuits)
        for locus, data in normalized.items():
            if len(data.raw_p1) != expected:
                raise ValueError(
                    f"Locus {locus!r} returned {len(data.raw_p1)} values for "
                    f"{expected} planned circuits."
                )
        object.__setattr__(self, "probabilities", MappingProxyType(normalized))

    def for_locus(self, locus: Locus) -> ProbabilityData:
        """Return observations for ``locus`` using tuple-normalized lookup."""

        return self.probabilities[tuple(locus)]

    def with_corrected(self, corrected: Mapping[Locus, np.ndarray]) -> MeasurementData:
        """Return a copy containing corrected probabilities for every locus."""

        missing = set(self.probabilities) - {tuple(locus) for locus in corrected}
        if missing:
            raise KeyError(f"No readout correction was provided for loci: {sorted(missing)}.")
        values = {
            locus: ProbabilityData(data.raw_p1, corrected[tuple(locus)])
            for locus, data in self.probabilities.items()
        }
        return MeasurementData(self.plan, values, self.execution_id)
