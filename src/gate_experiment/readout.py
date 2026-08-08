"""Readout-correction strategies managed independently of characterization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .backend.base import BackendSession, CircuitFactory
from .models.acquisition import (
    AcquisitionPlan,
    Locus,
    MeasurementData,
    PlannedCircuit,
)
from .models.results import CalibrationResult


class ReadoutModel(ABC):
    """A fitted model that can correct normalized probability measurements."""

    @abstractmethod
    def apply(self, measurements: MeasurementData) -> MeasurementData:
        """Return measurements containing corrected P(1) values."""


@dataclass(frozen=True)
class ReadoutPreparation:
    """A model and any calibration information retained in final results."""

    model: ReadoutModel
    calibration: CalibrationResult | None = None


class ReadoutStrategy(ABC):
    """Policy for acquiring or selecting a readout model for one run."""

    @abstractmethod
    def prepare(self, session: BackendSession) -> ReadoutPreparation:
        """Prepare a model before characterization acquisitions begin."""


class IdentityReadoutModel(ReadoutModel):
    """Leave backend-provided probabilities unchanged."""

    def apply(self, measurements: MeasurementData) -> MeasurementData:
        return measurements


class NoReadoutCorrection(ReadoutStrategy):
    """Use probabilities exactly as returned by the backend."""

    def prepare(self, session: BackendSession) -> ReadoutPreparation:
        return ReadoutPreparation(IdentityReadoutModel())


@dataclass(frozen=True)
class BinaryReadoutCalibration:
    """Independent binary confusion parameters for one measured locus."""

    p1_given_0: float
    p0_given_1: float

    def correct(self, p1: np.ndarray) -> np.ndarray:
        """Invert the two-state confusion model and clip to physical bounds."""

        denominator = 1 - self.p1_given_0 - self.p0_given_1
        if abs(denominator) < 1e-9:
            raise ValueError("The readout-confusion matrix is singular.")
        return np.clip((np.asarray(p1) - self.p1_given_0) / denominator, 0, 1)


class IndependentBinaryReadoutModel(ReadoutModel):
    """Apply one independent binary confusion model per locus."""

    def __init__(self, calibrations: dict[Locus, BinaryReadoutCalibration]) -> None:
        self.calibrations = dict(calibrations)

    def apply(self, measurements: MeasurementData) -> MeasurementData:
        corrected = {}
        for locus, probability_data in measurements.probabilities.items():
            if locus not in self.calibrations:
                raise KeyError(f"No readout calibration is available for locus {locus!r}.")
            corrected[locus] = self.calibrations[locus].correct(probability_data.raw_p1)
        return measurements.with_corrected(corrected)


@dataclass(frozen=True)
class IndependentReadoutSettings:
    """Acquisition settings for independent ground/excited calibration."""

    shots: int = 1024

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ValueError("Readout-calibration shots must be positive.")


class IndependentReadoutCorrection(ReadoutStrategy):
    """Calibrate each selected qubit from parallel |0> and |1> experiments.

    The backend must return uncorrected thresholded probabilities.  In
    particular, configure :class:`IQMBackendSettings` with
    ``ProbabilitySource.THRESHOLDED_READOUT`` to avoid applying correction twice.
    """

    def __init__(self, settings: IndependentReadoutSettings | None = None) -> None:
        self.settings = settings or IndependentReadoutSettings()

    def prepare(self, session: BackendSession) -> ReadoutPreparation:
        def ground(factory: CircuitFactory) -> object:
            return factory.measured_parallel({locus: [] for locus in session.loci}, "readout_cal")

        def excited(factory: CircuitFactory) -> object:
            operations = {
                locus: [factory.calibrated_prx(locus, np.pi, 0)] for locus in session.loci
            }
            return factory.measured_parallel(operations, "readout_cal")

        plan = AcquisitionPlan(
            technique_id="readout_calibration",
            measurement_key="readout_cal",
            shots=self.settings.shots,
            circuits=(
                PlannedCircuit("readout:ground", ground, {"prepared_state": 0}),
                PlannedCircuit("readout:excited", excited, {"prepared_state": 1}),
            ),
        )
        measurements = session.execute(plan)
        models = {}
        metrics = {}
        for locus, probability_data in measurements.probabilities.items():
            p1 = probability_data.raw_p1
            models[locus] = BinaryReadoutCalibration(
                p1_given_0=float(p1[0]),
                p0_given_1=float(1 - p1[1]),
            )
            metrics[locus] = {
                "p1_given_0": float(p1[0]),
                "p0_given_1": float(1 - p1[1]),
            }
        return ReadoutPreparation(
            model=IndependentBinaryReadoutModel(models),
            calibration=CalibrationResult(
                calibration_id="independent_binary_readout",
                metrics_by_locus=metrics,
            ),
        )
