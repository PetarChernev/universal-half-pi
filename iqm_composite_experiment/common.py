"""Shared pulse construction and execution helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.gates.prx import PrxGateImplementation
from iqm.pulse.playlist.instructions import IQPulse, Instruction
from iqm.pulse.playlist.schedule import Schedule, Segment
from iqm.pulse.timebox import TimeBox
from xarray import Dataset

from config import Config
from execution import PlannedBatch
from sequences import SequenceSpec, built_in_sequences


TimeBoxLike: TypeAlias = TimeBox | Iterable[TimeBox]

@dataclass(frozen=True)
class ReadoutCalibration:
    p1_given_0: float
    p0_given_1: float

    def correct(
        self,
        p1: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        denominator = 1 - self.p1_given_0 - self.p0_given_1
        if abs(denominator) < 1e-9:
            raise ValueError("Singular readout-confusion matrix.")
        return np.clip((p1 - self.p1_given_0) / denominator, 0, 1)


ReadoutMap: TypeAlias = Mapping[str, ReadoutCalibration | None]


def readout_calibration_metadata(
    calibrations: ReadoutMap,
) -> dict[str, dict[str, float] | None]:
    """Return JSON-compatible readout-calibration parameters."""
    return {
        qubit: None if calibration is None else {
            "p1_given_0": calibration.p1_given_0,
            "p0_given_1": calibration.p0_given_1,
        }
        for qubit, calibration in calibrations.items()
    }


@dataclass(frozen=True)
class Clifford:
    unitary: npt.NDArray[np.complex128]
    pulses: tuple[tuple[float, float], ...]


def prx_matrix(angle: float, phase: float) -> npt.NDArray[np.complex128]:
    """Return the IQM PRX unitary for an angle and hardware phase."""
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return np.array(
        [[c, -1j*s*np.exp(-1j*phase)], [-1j*s*np.exp(1j*phase), c]],
        dtype=complex,
    )


def paper_phase_to_iqm(phase: float) -> float:
    """Convert the paper's Cayley-Klein phase convention to IQM PRX."""
    return -float(phase)


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2*np.pi) - np.pi)


def same_unitary(
    left: npt.NDArray[np.complex128],
    right: npt.NDArray[np.complex128],
    atol: float = 1e-8,
) -> bool:
    return abs(abs(np.trace(left.conj().T @ right)) / 2 - 1) < atol


def serial(boxes: Iterable[TimeBox]) -> TimeBox:
    boxes = list(boxes)
    if not boxes:
        raise ValueError("Cannot serialize an empty list.")
    result = boxes[0]
    for box in boxes[1:]:
        result = result | box
    return result


def parallel(boxes: Iterable[TimeBox]) -> TimeBox:
    boxes = list(boxes)
    if not boxes:
        raise ValueError("Cannot parallelize an empty list.")
    return TimeBox.composite(boxes)


def p1_from_dataset(
    dataset: Dataset,
    qubit: str,
    key: str,
) -> npt.NDArray[np.float64]:
    exact = f"{qubit}__{key}_excited_state_probability"
    if exact in dataset:
        return np.asarray(dataset[exact].data, dtype=float).reshape(-1)
    matches = [
        name for name in dataset.data_vars
        if name.startswith(f"{qubit}__")
        and key in name
        and name.endswith("_excited_state_probability")
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Could not find P(1) for {qubit=} {key=}. "
            f"Variables: {list(dataset.data_vars)}"
        )
    return np.asarray(dataset[matches[0]].data, dtype=float).reshape(-1)


def prx_impl(
    builder: ScheduleBuilder,
    qubit: str,
    config: Config,
) -> PrxGateImplementation:
    if config.prx_implementation is None:
        return builder.prx([qubit])
    return builder.get_implementation(
        "prx", [qubit], impl_name=config.prx_implementation,
    )

def _single_iq_pulse(
    builder: ScheduleBuilder,
    qubit: str,
    box: TimeBox,
) -> tuple[str, Schedule, list[Instruction], int]:
    """Extract the single IQPulse from a calibrated PRX TimeBox."""
    drive = builder.get_drive_channel(qubit)
    atom = box.atom
    if atom is None:
        raise ValueError("PRX implementation did not produce an atomic schedule.")
    schedule = deepcopy(atom)
    instructions = list(schedule[drive])

    indices = [
        i
        for i, instruction in enumerate(instructions)
        if isinstance(instruction, IQPulse)
    ]

    if len(indices) != 1:
        raise ValueError(
            f"PRX on {qubit} contains {len(indices)} IQPulse instructions. "
            "Choose a calibrated single-pulse PRX implementation in Config."
        )

    return drive, schedule, instructions, indices[0]


def add_amplitude_error(
    builder: ScheduleBuilder,
    qubit: str,
    box: TimeBox,
    amplitude_error: float,
) -> TimeBox:
    """
    Apply a relative amplitude error directly to the calibrated IQ pulse.

    amplitude_error = 0.1  -> +10% amplitude
    amplitude_error = -0.1 -> -10% amplitude
    """
    drive, schedule, instructions, i = _single_iq_pulse(
        builder,
        qubit,
        box,
    )

    pulse = instructions[i]
    if not isinstance(pulse, IQPulse):
        raise RuntimeError("Selected PRX instruction is not an IQPulse.")

    instructions[i] = replace(
        pulse,
        scale_i=pulse.scale_i * (1.0 + amplitude_error),
        scale_q=pulse.scale_q * (1.0 + amplitude_error),
    )

    schedule[drive] = Segment(instructions)

    return TimeBox.atomic(
        schedule,
        locus_components=(qubit,),
        label="amplitude_error_prx",
    )


def add_detuning(
    builder: ScheduleBuilder,
    qubit: str,
    box: TimeBox,
    detuning_hz: float,
) -> TimeBox:
    """Apply detuning only to the IQ pulse in this PRX TimeBox."""
    drive, schedule, instructions, i = _single_iq_pulse(
        builder,
        qubit,
        box,
    )

    pulse = instructions[i]
    if not isinstance(pulse, IQPulse):
        raise RuntimeError("Selected PRX instruction is not an IQPulse.")

    offset = detuning_hz / float(builder.channels[drive].sample_rate)

    instructions[i] = replace(
        pulse,
        modulation_frequency=pulse.modulation_frequency + offset,
    )

    schedule[drive] = Segment(instructions)

    return TimeBox.atomic(
        schedule,
        locus_components=(qubit,),
        label="detuned_prx",
    )


def calibrated_prx(
    builder: ScheduleBuilder,
    qubit: str,
    angle: float,
    phase: float,
    config: Config,
) -> TimeBox:
    return prx_impl(builder, qubit, config)(
        float(angle),
        float(phase),
    )


def erroneous_prx(
    builder: ScheduleBuilder,
    qubit: str,
    angle: float,
    phase: float,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
) -> TimeBox:
    # Always construct the nominal calibrated rotation.
    box = calibrated_prx(
        builder,
        qubit,
        angle,
        phase,
        config,
    )

    # Inject errors into the calibrated physical pulse.
    box = add_amplitude_error(
        builder,
        qubit,
        box,
        amplitude_error,
    )

    box = add_detuning(
        builder,
        qubit,
        box,
        detuning_hz,
    )

    return box


def composite_gate(
    builder: ScheduleBuilder,
    qubit: str,
    sequence: SequenceSpec,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
) -> TimeBox:
    return serial(
        erroneous_prx(
            builder=builder,
            qubit=qubit,
            angle=sequence.constituent_angle,
            phase=paper_phase_to_iqm(phase),
            amplitude_error=amplitude_error,
            detuning_hz=detuning_hz,
            config=config,
        )
        for phase in sequence.phases
    )

def as_timebox(box_or_boxes: TimeBoxLike) -> TimeBox:
    if isinstance(box_or_boxes, TimeBox):
        return box_or_boxes
    boxes = list(box_or_boxes)
    if not boxes:
        raise ValueError("Gate implementation returned no TimeBoxes.")
    if not all(isinstance(box, TimeBox) for box in boxes):
        raise TypeError(
            "Expected TimeBox or iterable of TimeBoxes, got "
            f"{type(box_or_boxes).__name__}"
        )
    return TimeBox.composite(boxes)


def measured_parallel_circuit(
    builder: ScheduleBuilder,
    qubits: Sequence[str],
    operations: dict[str, list[TimeBox]],
    key: str,
) -> TimeBox:
    active = [serial(operations[q]) for q in qubits if operations[q]]
    measurement = as_timebox(builder.measure(qubits)(key=key))
    if active:
        return parallel(active) | measurement
    return measurement


def build_readout_batch(
    builder: ScheduleBuilder,
    qubits: Sequence[str],
    config: Config,
    *,
    batch_id: str,
    characterization_technique: str,
    position: str,
    shots: int,
    measurement_key: str,
) -> PlannedBatch:
    """Build readout-calibration circuits without compiling or submitting."""
    ground: dict[str, list[TimeBox]] = {q: [] for q in qubits}
    excited: dict[str, list[TimeBox]] = {
        q: [calibrated_prx(builder, q, np.pi, 0, config)] for q in qubits
    }
    return PlannedBatch(
        batch_id=batch_id,
        technique="readout_calibration",
        acquisition_index=0,
        circuits=(
            measured_parallel_circuit(builder, qubits, ground, measurement_key),
            measured_parallel_circuit(builder, qubits, excited, measurement_key),
        ),
        qubits=tuple(qubits),
        shots=shots,
        measurement_key=measurement_key,
        metadata={
            "technique": "readout_calibration",
            "acquisition_index": 0,
            "characterization_technique": characterization_technique,
            "calibration_position": position,
            "qubits": list(qubits),
            "shots": shots,
            "states": ["ground", "excited"],
            "prx_implementation": config.prx_implementation,
        },
    )


def readout_calibration_from_datasets(
    calibrations: Sequence[tuple[Dataset, str]],
    qubits: Sequence[str],
) -> dict[str, ReadoutCalibration | None]:
    """Average readout-confusion parameters from pre/post calibrations."""
    if not calibrations:
        return {q: None for q in qubits}
    result: dict[str, ReadoutCalibration | None] = {}
    for q in qubits:
        probabilities = [
            p1_from_dataset(dataset, q, measurement_key)
            for dataset, measurement_key in calibrations
        ]
        if any(len(p1) != 2 for p1 in probabilities):
            raise ValueError(
                f"Readout calibration for {q} must contain ground and excited states."
            )
        result[q] = ReadoutCalibration(
            float(np.mean([p1[0] for p1 in probabilities])),
            float(np.mean([1 - p1[1] for p1 in probabilities])),
        )
    return result


def correct(
    p1: npt.NDArray[np.float64],
    calibration: ReadoutCalibration | None,
) -> npt.NDArray[np.float64]:
    return p1 if calibration is None else calibration.correct(p1)
