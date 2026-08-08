"""Strict conversion of IQM xarray datasets into framework measurements."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from ...models.acquisition import AcquisitionPlan, Locus, MeasurementData, ProbabilityData


class ProbabilitySource(str, Enum):
    """Dataset quantity interpreted as P(1) by the IQM adapter."""

    EXA_PROBABILITY = "exa_probability"
    THRESHOLDED_READOUT = "thresholded_readout"


class IQMDatasetDecoder:
    """Decode one IQM dataset without fuzzy variable-name matching.

    ``EXA_PROBABILITY`` uses IQM's post-processed excited-state probability,
    which may already include assignment correction. ``THRESHOLDED_READOUT``
    averages binary single-shot readout and is appropriate when this framework's
    independent readout correction is enabled.
    """

    def __init__(self, source: ProbabilitySource = ProbabilitySource.EXA_PROBABILITY) -> None:
        self.source = source

    def decode(
        self,
        dataset: Any,
        plan: AcquisitionPlan,
        loci: tuple[Locus, ...],
        *,
        execution_id: str | None = None,
    ) -> MeasurementData:
        """Decode all requested loci and preserve the planned circuit order."""

        probabilities = {}
        for locus in loci:
            if len(locus) != 1:
                raise ValueError("The IQM probability decoder currently supports one-qubit loci.")
            qubit = locus[0]
            suffix = (
                "excited_state_probability"
                if self.source is ProbabilitySource.EXA_PROBABILITY
                else "readout"
            )
            variable_name = f"{qubit}__{plan.measurement_key}_{suffix}"
            if variable_name not in dataset:
                available = list(getattr(dataset, "data_vars", ()))
                raise KeyError(
                    f"Dataset variable {variable_name!r} is missing. Available variables: {available}."
                )
            values = self._ordered_values(dataset[variable_name], len(plan.circuits))
            probabilities[locus] = ProbabilityData(values)
        return MeasurementData(plan, probabilities, execution_id)

    def _ordered_values(self, data_array: Any, circuit_count: int) -> np.ndarray:
        values = np.asarray(data_array.data)
        dimensions = tuple(getattr(data_array, "dims", ()))
        if "circuit_index" in dimensions:
            axis = dimensions.index("circuit_index")
            values = np.moveaxis(values, axis, 0)
        elif values.shape[:1] != (circuit_count,):
            values = values.reshape(-1)

        if values.shape[0] != circuit_count:
            raise ValueError(
                f"Dataset contains {values.shape[0]} circuit values; expected {circuit_count}."
            )
        if values.ndim > 1:
            values = values.mean(axis=tuple(range(1, values.ndim)))
        if np.iscomplexobj(values):
            raise ValueError(
                "The selected dataset variable contains complex values and cannot "
                "be interpreted directly as P(1)."
            )
        values = np.asarray(values, dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("Dataset probabilities contain non-finite values.")
        if self.source is ProbabilitySource.THRESHOLDED_READOUT and (
            np.any(values < 0) or np.any(values > 1)
        ):
            raise ValueError("Thresholded readout must contain binary or probabilistic values in [0, 1].")
        return values
