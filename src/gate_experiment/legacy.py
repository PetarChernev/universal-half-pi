"""Compatibility adapters for the original procedural experiment API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from iqm.pulla.pulla import Pulla

from .backend.iqm import IQMBackend, IQMBackendSettings, ProbabilitySource
from .experiment import CharacterizationExperiment, ExperimentRunner, ExperimentSettings
from .models.points import PointSet, SweepAxis
from .readout import (
    IndependentReadoutCorrection,
    IndependentReadoutSettings,
    NoReadoutCorrection,
)
from .targets.composite import UniversalCompositePulse
from .targets.universal import universal_composite_pulses
from .techniques import (
    InterleavedRandomizedBenchmarking,
    ProcessTomography,
    RBSettings,
    RamseySettings,
    TomographySettings,
    TransitionPhaseRamsey,
)


@dataclass(frozen=True)
class SequenceSpec:
    """Legacy phase-sequence description retained for notebook compatibility."""

    name: str
    target_angle: float
    constituent_angle: float
    phases: tuple[float, ...]

    def to_target(self) -> UniversalCompositePulse:
        """Convert this legacy value to the pluggable target interface."""

        return UniversalCompositePulse.for_x_rotation(
            self.name,
            target_angle=self.target_angle,
            constituent_angle=self.constituent_angle,
            phases=self.phases,
        )


@dataclass(frozen=True)
class Config:
    """Legacy monolithic settings translated into the new object model."""

    qubits: tuple[str, ...] | None = None
    amplitude_errors: tuple[float, ...] = (-0.02, 0.0, 0.02)
    detunings_hz: tuple[float, ...] = (-2e6, 0.0, 2e6)
    ramsey_phases: tuple[float, ...] = tuple(np.linspace(0, 2 * np.pi, 17)[:-1])
    tomography_shots: int = 1024
    ramsey_shots: int = 1024
    rb_shots: int = 512
    rb_lengths: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    rb_samples: int = 10
    seed: int = 7
    prx_implementation: str | None = None
    readout_correction: bool = True


def built_in_sequences() -> dict[str, SequenceSpec]:
    """Return built-in pulses using the original ``SequenceSpec`` representation."""

    targets = universal_composite_pulses()
    return {
        name: SequenceSpec(
            name=name,
            target_angle=np.pi / 2 if name == "H1" else np.pi,
            constituent_angle=target.constituent_angle,
            phases=target.phases,
        )
        for name, target in targets.items()
    }


def run_experiment(
    pulla: Pulla,
    sequences: Sequence[SequenceSpec | UniversalCompositePulse],
    config: Config,
) -> pd.DataFrame:
    """Run the legacy all-techniques Cartesian experiment and return a DataFrame."""

    targets = tuple(
        sequence.to_target() if isinstance(sequence, SequenceSpec) else sequence
        for sequence in sequences
    )
    points = PointSet.cartesian(
        targets=targets,
        axes=(
            SweepAxis("amplitude_error", config.amplitude_errors),
            SweepAxis("detuning_hz", config.detunings_hz, unit="Hz"),
        ),
    )
    techniques = (
        ProcessTomography(TomographySettings(shots=config.tomography_shots)),
        TransitionPhaseRamsey(
            RamseySettings(phases=config.ramsey_phases, shots=config.ramsey_shots)
        ),
        InterleavedRandomizedBenchmarking(
            RBSettings(
                lengths=config.rb_lengths,
                samples=config.rb_samples,
                shots=config.rb_shots,
                seed=config.seed,
            )
        ),
    )
    probability_source = (
        ProbabilitySource.THRESHOLDED_READOUT
        if config.readout_correction
        else ProbabilitySource.EXA_PROBABILITY
    )
    backend = IQMBackend(
        pulla,
        IQMBackendSettings(
            prx_implementation=config.prx_implementation,
            probability_source=probability_source,
        ),
    )
    experiment = CharacterizationExperiment(
        points=points,
        techniques=techniques,
        settings=ExperimentSettings(qubits=config.qubits),
        readout=(
            IndependentReadoutCorrection(
                IndependentReadoutSettings(shots=config.tomography_shots)
            )
            if config.readout_correction
            else NoReadoutCorrection()
        ),
    )
    return ExperimentRunner(backend).run(experiment).legacy_dataframe()
