"""Ramsey transition-phase acquisition and analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from iqm.cpc.compiler.compiler import Compiler
from iqm.pulse.timebox import TimeBox
from xarray import Dataset

from common import Config, Pulla, ReadoutMap, SequenceSpec, calibrated_prx, composite_gate, correct, measured_parallel_circuit, paper_phase_to_iqm, p1_from_dataset, prx_matrix, readout_calibration_metadata, run_batch, wrap
from dataset_persistence import persist_dataset


ResultValue: TypeAlias = str | int | float | bool


def acquire(pulla: Pulla, compiler: Compiler, qubits: Sequence[str], sequence: SequenceSpec, amplitude_error: float, detuning_hz: float, config: Config) -> Dataset:
    """Acquire a phase-scanned Ramsey fringe for one sequence/error setting.

    Each circuit applies the composite gate followed by a calibrated pi/2
    analysis pulse whose phase is taken from ``config.ramsey_phases``.  A pi
    target receives an initial calibrated pi/2 pulse so that the gate acts on
    an equatorial state and its transition phase remains observable.

    The returned dataset contains one circuit result per analysis phase, in
    the same order as ``config.ramsey_phases``.
    """
    builder = compiler.get_schedule_builder()
    circuits: list[TimeBox] = []
    for phase in config.ramsey_phases:
        operations: dict[str, list[TimeBox]] = {}
        for q in qubits:
            ops: list[TimeBox] = []
            # A pi rotation maps |0> to a pole, where an azimuthal phase cannot
            # be observed.  The preparation pulse moves the input to the equator.
            if abs(sequence.target_angle - np.pi) < 1e-6:
                ops.append(calibrated_prx(builder, q, np.pi / 2, 0, config))
            ops.append(composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config))
            ops.append(calibrated_prx(builder, q, np.pi / 2, phase, config))
            operations[q] = ops
        circuits.append(measured_parallel_circuit(builder, qubits, operations, "ramsey"))
    return run_batch(pulla, compiler, circuits, qubits, config.ramsey_shots)


def run(pulla: Pulla, compiler: Compiler, qubits: Sequence[str], sequences: Sequence[SequenceSpec], config: Config, readout: ReadoutMap, output_directory: Path) -> pd.DataFrame:
    """Run Ramsey characterization over all configured sweeps and qubits.

    Raw datasets and acquisition metadata are persisted before readout-corrected
    fringes are fitted.  The returned table has one row per sequence, amplitude
    error, detuning, and qubit combination.
    """
    rows: list[dict[str, ResultValue]] = []
    acquisition_index = 0
    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                dataset = acquire(pulla, compiler, qubits, sequence, amplitude_error, detuning_hz, config)
                persist_dataset(
                    dataset,
                    output_directory / f"raw_ramsey_{acquisition_index:04d}.nc",
                    {
                        "technique": "transition_phase_ramsey",
                        "acquisition_index": acquisition_index,
                        "sequence": sequence.name,
                        "target_angle_rad": float(sequence.target_angle),
                        "constituent_angle_rad": float(sequence.constituent_angle),
                        "paper_phases_rad": [float(phase) for phase in sequence.phases],
                        "iqm_phases_rad": [paper_phase_to_iqm(phase) for phase in sequence.phases],
                        "analysis_phases_rad": [float(phase) for phase in config.ramsey_phases],
                        "amplitude_error": amplitude_error,
                        "detuning_hz": detuning_hz,
                        "qubits": list(qubits),
                        "shots": config.ramsey_shots,
                        "readout_calibration_shots": config.tomography_shots,
                        "prx_implementation": config.prx_implementation,
                        "fit_model": "offset + cosine*cos(phase) + sine*sin(phase)",
                        "readout_calibration": readout_calibration_metadata(readout),
                    },
                )
                acquisition_index += 1
                for q in qubits:
                    row: dict[str, ResultValue] = {"sequence": sequence.name, "pulse_count": len(sequence.phases), "qubit": q, "amplitude_error": amplitude_error, "detuning_hz": detuning_hz}
                    calibration = readout[q]
                    # Readout correction amplifies binomial noise by the inverse
                    # confusion-matrix contrast; carry that factor into the
                    # conservative fringe-identifiability estimate.
                    readout_scale = 1.0 if calibration is None else 1.0 / abs(
                        1 - calibration.p1_given_0 - calibration.p0_given_1
                    )
                    row.update(metrics(
                        correct(p1_from_dataset(dataset, q, "ramsey"), calibration),
                        sequence,
                        config.ramsey_phases,
                        config.ramsey_shots,
                        readout_scale,
                    ))
                    rows.append(row)
    return pd.DataFrame(rows)



def fit_fringe(phases: Sequence[float], p1: Sequence[float]) -> dict[str, float]:
    """Fit ``P(1) = offset + cosine*cos(phase) + sine*sin(phase)``.

    The model is linear in its three coefficients, so ordinary least squares
    gives the fringe amplitude, phase, visibility, and fit residual without a
    nonlinear optimizer.  At least three linearly independent phase samples
    are required.
    """
    phases = np.asarray(phases)
    design = np.column_stack([np.ones(len(phases)), np.cos(phases), np.sin(phases)])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Ramsey phases do not identify offset, cosine, and sine terms.")
    offset, cosine, sine = np.linalg.lstsq(design, np.asarray(p1), rcond=None)[0]
    fitted = design @ np.array([offset, cosine, sine])
    amplitude = float(math.hypot(cosine, sine))
    visibility = amplitude / float(offset) if offset > 0 else float("nan")
    return {
        "offset": float(offset),
        "cosine_coefficient": float(cosine),
        "sine_coefficient": float(sine),
        "fringe_amplitude": amplitude,
        "fringe_visibility": visibility,
        "phase": float(math.atan2(sine, cosine)),
        "fit_rmse": float(np.sqrt(np.mean((np.asarray(p1) - fitted) ** 2))),
    }


def ideal_p1(
    sequence: SequenceSpec,
    phases: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Calculate the ideal target-gate Ramsey probabilities.

    This mirrors the acquisition circuit but replaces the physical composite
    gate by its ideal x-axis rotation.  The result establishes the zero-error
    fringe phase in the IQM analysis-pulse convention.
    """
    state = np.array([1.0, 0.0], dtype=complex)
    if abs(sequence.target_angle - np.pi) < 1e-6:
        state = prx_matrix(np.pi / 2, 0) @ state
    state = prx_matrix(sequence.target_angle, 0) @ state
    return np.array([abs((prx_matrix(np.pi / 2, phase) @ state)[1]) ** 2 for phase in phases])


def metrics(
    p1: npt.NDArray[np.float64],
    sequence: SequenceSpec,
    phases: Sequence[float],
    shots: int,
    readout_scale: float = 1.0,
) -> dict[str, float | bool]:
    """Extract Ramsey fringe diagnostics and transition-phase error.

    The measured fringe phase is compared with the ideal circuit phase and
    wrapped to ``[-pi, pi)``.  A pi target divides the displacement by two;
    other supported targets use it directly.  Transition phase is reported
    only when the fitted quadrature amplitude exceeds three times a
    conservative shot-noise bound.

    Args:
        p1: Readout-corrected excited-state probabilities, ordered by ``phases``.
        sequence: Composite sequence whose ideal target defines the reference.
        phases: Analysis-pulse phases in radians.
        shots: Number of shots used for each phase point.
        readout_scale: Noise amplification caused by readout correction.

    Returns:
        Fitted fringe values, phase uncertainty diagnostics, and transition
        phase error in radians.
    """
    measured = fit_fringe(phases, p1)
    ideal = fit_fringe(phases, ideal_p1(sequence, phases))
    shift = wrap(measured["phase"] - ideal["phase"])
    divisor = 2 if abs(sequence.target_angle - np.pi) < 1e-6 else 1
    design = np.column_stack([np.ones(len(phases)), np.cos(phases), np.sin(phases)])
    # P(1) has binomial variance no greater than 1/4 per shot.  Propagating
    # that worst-case variance through linear least squares bounds the two
    # fitted quadrature coefficients' covariance.
    covariance_bound = (
        (0.25 * readout_scale**2 / shots)
        * np.linalg.inv(design.T @ design)
    )
    coefficient_noise = float(np.sqrt(np.linalg.eigvalsh(covariance_bound[1:, 1:]).max()))
    phase_identifiable = measured["fringe_amplitude"] > 3 * coefficient_noise
    if measured["fringe_amplitude"] > 0:
        # Delta-method propagation through atan2(sine, cosine).
        gradient = np.array([
            -measured["sine_coefficient"],
            measured["cosine_coefficient"],
        ]) / measured["fringe_amplitude"] ** 2
        phase_standard_error_bound = float(
            np.sqrt(gradient @ covariance_bound[1:, 1:] @ gradient)
        )
    else:
        phase_standard_error_bound = float("inf")
    return {
        "ramsey_offset": measured["offset"],
        "ramsey_cosine_coefficient": measured["cosine_coefficient"],
        "ramsey_sine_coefficient": measured["sine_coefficient"],
        "ramsey_fringe_amplitude": measured["fringe_amplitude"],
        "ramsey_fringe_visibility": measured["fringe_visibility"],
        "ramsey_fringe_phase": measured["phase"],
        "ramsey_fit_rmse": measured["fit_rmse"],
        "ramsey_fringe_shift": shift,
        "ramsey_phase_shot_noise_bound": phase_standard_error_bound,
        "ramsey_phase_identifiable": phase_identifiable,
        # The IQM analysis-pulse convention has the opposite phase sign to the paper.
        "transition_phase_error": -shift / divisor if phase_identifiable else float("nan"),
    }
