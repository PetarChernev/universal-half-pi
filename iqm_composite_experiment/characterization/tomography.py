"""Tomography acquisition and analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.timebox import TimeBox
from scipy.spatial.transform import Rotation

from common import (
    Config,
    ReadoutMap,
    SequenceSpec,
    calibrated_prx,
    composite_gate,
    correct,
    measured_parallel_circuit,
    paper_phase_to_iqm,
    p1_from_dataset,
)
from execution import AcquiredBatch, PlannedBatch

TomographyInput: TypeAlias = Literal["0", "1", "+x", "+y"]
TomographyBasis: TypeAlias = Literal["x", "y", "z"]
TomographyCoordinate: TypeAlias = tuple[TomographyInput, TomographyBasis]
ResultValue: TypeAlias = str | int | float | bool

TOMO_INPUTS: tuple[TomographyInput, ...] = ("0", "1", "+x", "+y")
TOMO_BASES: tuple[TomographyBasis, ...] = ("x", "y", "z")


def build_batch(
    builder: ScheduleBuilder,
    qubits: Sequence[str],
    sequence: SequenceSpec,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
    acquisition_index: int,
) -> PlannedBatch:
    """Build the 12 circuits needed for one tomography sweep point.

    For each of the four input states ``|0>``, ``|1>``, ``|+x>``, and ``|+y>``,
    the circuit applies the composite gate and measures its output along x, y,
    and z. The manifest records coordinates in circuit-result order.
    """
    circuits: list[TimeBox] = []
    coordinates: list[TomographyCoordinate] = []
    for state in TOMO_INPUTS:
        for basis in TOMO_BASES:
            operations: dict[str, list[TimeBox]] = {
                q: (
                    prep_boxes(builder, q, state, config)
                    + [composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config)]
                    + analysis_boxes(builder, q, basis, config)
                )
                for q in qubits
            }
            circuits.append(measured_parallel_circuit(builder, qubits, operations, "tomo"))
            coordinates.append((state, basis))
    return PlannedBatch(
        batch_id=f"tomography_{acquisition_index:04d}",
        technique="process_tomography",
        acquisition_index=acquisition_index,
        circuits=tuple(circuits),
        qubits=tuple(qubits),
        shots=config.tomography_shots,
        measurement_key="tomo",
        metadata={
            "technique": "process_tomography",
            "acquisition_index": acquisition_index,
            "sequence": sequence.name,
            "target_angle_rad": float(sequence.target_angle),
            "constituent_angle_rad": float(sequence.constituent_angle),
            "paper_phases_rad": [float(phase) for phase in sequence.phases],
            "iqm_phases_rad": [paper_phase_to_iqm(phase) for phase in sequence.phases],
            "amplitude_error": amplitude_error,
            "detuning_hz": detuning_hz,
            "qubits": list(qubits),
            "shots": config.tomography_shots,
            "readout_calibration_shots": (
                config.tomography_shots
                if config.pre_calibration or config.post_calibration
                else None
            ),
            "pre_calibration": config.pre_calibration,
            "post_calibration": config.post_calibration,
            "prx_implementation": config.prx_implementation,
            "reconstruction": "trace_preserving_linear_inversion",
            "circuit_coordinates": [
                {"input_state": state, "measurement_basis": basis}
                for state, basis in coordinates
            ],
        },
        manifest=tuple(coordinates),
        sequence=sequence,
        amplitude_error=amplitude_error,
        detuning_hz=detuning_hz,
    )


def build_batches(
    builder: ScheduleBuilder,
    qubits: Sequence[str],
    sequences: Sequence[SequenceSpec],
    config: Config,
) -> list[PlannedBatch]:
    """Build every configured tomography batch without submitting work."""
    batches: list[PlannedBatch] = []
    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                batches.append(build_batch(
                    builder,
                    qubits,
                    sequence,
                    amplitude_error,
                    detuning_hz,
                    config,
                    len(batches),
                ))
    return batches


def analyze(
    acquisitions: Sequence[AcquiredBatch],
    readout: ReadoutMap,
) -> pd.DataFrame:
    """Analyze process-tomography batches after raw persistence.

    Corrected probabilities are converted into an affine Bloch map and
    Pauli-transfer matrix (PTM). The
    returned table contains the PTM, translation, fidelity, effective rotation,
    and non-unitary-distortion diagnostics for each qubit and setting.
    """
    rows: list[dict[str, ResultValue]] = []
    for acquisition in acquisitions:
        plan = acquisition.plan
        sequence = plan.sequence
        if plan.technique != "process_tomography" or sequence is None:
            raise ValueError(f"Unexpected tomography batch: {plan.batch_id}")
        coordinates: Sequence[TomographyCoordinate] = plan.manifest
        for q in plan.qubits:
            p1 = correct(
                p1_from_dataset(acquisition.dataset, q, plan.measurement_key),
                readout[q],
            )
            ptm, bloch, translation = reconstruct_ptm(p1, coordinates)
            row: dict[str, ResultValue] = {"sequence": sequence.name, "pulse_count": len(sequence.phases), "qubit": q, "amplitude_error": float(plan.amplitude_error), "detuning_hz": float(plan.detuning_hz), "translation_x": float(translation[0]), "translation_y": float(translation[1]), "translation_z": float(translation[2])}
            row.update({
                f"ptm_{i}{j}": float(ptm[i, j])
                for i in range(4)
                for j in range(4)
            })
            row.update(metrics(ptm, bloch, sequence.target_angle))
            rows.append(row)
    return pd.DataFrame(rows)


def prep_boxes(
    builder: ScheduleBuilder,
    q: str,
    state: TomographyInput,
    config: Config,
) -> list[TimeBox]:
    """Return calibrated pulses that prepare one tomography input state.

    ``|0>`` needs no pulse, while ``|1>``, ``|+x>``, and ``|+y>`` are prepared
    with calibrated PRX rotations using the IQM phase convention.

    Raises:
        ValueError: If ``state`` is not a supported tomography input.
    """
    if state == "0":
        return []
    if state == "1":
        return [calibrated_prx(builder, q, np.pi, 0, config)]
    if state == "+x":
        return [calibrated_prx(builder, q, np.pi / 2, np.pi / 2, config)]
    if state == "+y":
        return [calibrated_prx(builder, q, np.pi / 2, np.pi, config)]
    raise ValueError(state)


def analysis_boxes(
    builder: ScheduleBuilder,
    q: str,
    basis: TomographyBasis,
    config: Config,
) -> list[TimeBox]:
    """Return the calibrated basis rotation for a Pauli measurement.

    Readout directly measures z.  For x and y, a calibrated pi/2 PRX rotates
    the requested Bloch component onto z before computational-basis readout.

    Raises:
        ValueError: If ``basis`` is not x, y, or z.
    """
    if basis == "z":
        return []
    if basis == "x":
        return [calibrated_prx(builder, q, np.pi / 2, -np.pi / 2, config)]
    if basis == "y":
        return [calibrated_prx(builder, q, np.pi / 2, 0, config)]
    raise ValueError(basis)



def reconstruct_ptm(
    p1: npt.NDArray[np.float64],
    metadata: Sequence[TomographyCoordinate],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Reconstruct a trace-preserving affine Bloch map by linear inversion.

    Excited-state probabilities are converted to Pauli expectations using
    ``<sigma> = 1 - 2*P(1)``.  If the channel is ``r_out = M*r_in + t``, the
    opposite z-pole inputs determine ``t`` and the z column of ``M``; the +x
    and +y inputs determine its first two columns.

    Args:
        p1: Corrected probabilities ordered consistently with ``metadata``.
        metadata: Input-state and measurement-basis coordinate for each value.

    Returns:
        The 4x4 PTM, its 3x3 Bloch block ``M``, and translation ``t``.
    """
    outputs: dict[TomographyInput, npt.NDArray[np.float64]] = {
        state: np.zeros(3) for state in TOMO_INPUTS
    }
    axis: dict[TomographyBasis, int] = {"x": 0, "y": 1, "z": 2}
    for probability, (state, basis) in zip(p1, metadata, strict=True):
        outputs[state][axis[basis]] = 1 - 2 * probability
    # For affine maps, the midpoint of the +z and -z outputs is t, and their
    # half-difference is M's z column.
    translation = (outputs["0"] + outputs["1"]) / 2
    bloch = np.column_stack([outputs["+x"] - translation, outputs["+y"] - translation, (outputs["0"] - outputs["1"]) / 2])
    ptm = np.zeros((4, 4))
    ptm[0, 0] = 1
    ptm[1:, 0] = translation
    ptm[1:, 1:] = bloch
    return ptm, bloch, translation


def metrics(
    ptm: npt.NDArray[np.float64],
    bloch: npt.NDArray[np.float64],
    target_angle: float,
) -> dict[str, float]:
    """Summarize fidelity, coherent rotation, and Bloch-map distortion.

    Average gate fidelity compares the reconstructed PTM with the ideal x-axis
    target.  An SVD-based polar decomposition projects the measured Bloch block
    onto the nearest proper rotation; its rotation vector supplies the
    effective angle and axis.  Singular values and the projection residual
    quantify behavior that cannot be represented by a pure rotation.
    """
    target = np.zeros((4, 4))
    target[0, 0] = 1
    target[1:, 1:] = Rotation.from_rotvec([target_angle, 0, 0]).as_matrix()
    fidelity = float((np.trace(target.T @ ptm) + 2) / 6)
    # Correct the orthogonal polar factor when necessary so its determinant is
    # +1 and scipy can interpret it as a physical three-dimensional rotation.
    u, singular_values, vt = np.linalg.svd(bloch)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    nearest_rotation = u @ correction @ vt
    rotvec = Rotation.from_matrix(nearest_rotation).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    axis = np.array([1.0, 0.0, 0.0]) if angle < 1e-12 else rotvec / angle
    # At pi the axis-angle representation is sign-degenerate.  Choose +x so a
    # signed x overrotation remains visible instead of flipping the axis.
    if abs(target_angle - np.pi) < 1e-6 and axis[0] < 0:
        axis = -axis
        angle = 2 * np.pi - angle
    wrap = lambda value: float((value + np.pi) % (2 * np.pi) - np.pi)
    return {
        "average_gate_fidelity": fidelity,
        "average_gate_infidelity": 1 - fidelity,
        "effective_angle": angle,
        "angle_error": wrap(angle - target_angle),
        "axis_x": float(axis[0]), "axis_y": float(axis[1]), "axis_z": float(axis[2]),
        "axis_azimuth": float(math.atan2(axis[1], axis[0])),
        "axis_tilt": float(math.asin(np.clip(axis[2], -1, 1))),
        "axis_distance": float(math.acos(np.clip(axis[0], -1, 1))),
        "bloch_singular_value_1": float(singular_values[0]),
        "bloch_singular_value_2": float(singular_values[1]),
        "bloch_singular_value_3": float(singular_values[2]),
        "polar_rotation_residual": float(np.linalg.norm(bloch - nearest_rotation)),
    }
