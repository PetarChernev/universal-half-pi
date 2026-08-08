"""Composite-gate validation on IQM hardware with Pulla.

For each selected qubit, composite sequence, amplitude error, and detuning, this
runs in parallel and computes:
    1. process infidelity,
    2. effective rotation angle and axis,
    3. Ramsey transition-phase error,
    4. interleaved-RB infidelity.

Set Config.qubits=None to use all available qubits.

Target API: iqm-pulla 14.x / iqm-pulse 14.x.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation

from iqm.pulla.pulla import Pulla
from iqm.pulse.playlist.instructions import IQPulse
from iqm.pulse.playlist.schedule import Segment
from iqm.pulse.timebox import TimeBox


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    target_angle: float
    constituent_angle: float
    phases: tuple[float, ...]


@dataclass(frozen=True)
class Config:
    qubits: tuple[str, ...] | None = None
    amplitude_errors: tuple[float, ...] = (0.0,)
    detunings_hz: tuple[float, ...] = (-2e6, 0.0, 2e6)
    ramsey_phases: tuple[float, ...] = tuple(np.linspace(0, 2 * np.pi, 5)[:-1])
    tomography_shots: int = 50
    ramsey_shots: int = 50
    rb_shots: int = 50
    rb_lengths: tuple[int, ...] = (1, 2, 4, 8)
    rb_samples: int = 2
    seed: int = 7
    prx_implementation: str | None = None
    readout_correction: bool = True


@dataclass(frozen=True)
class ReadoutCalibration:
    p1_given_0: float
    p0_given_1: float

    def correct(self, p1: np.ndarray) -> np.ndarray:
        denominator = 1 - self.p1_given_0 - self.p0_given_1
        if abs(denominator) < 1e-9:
            raise ValueError("Singular readout-confusion matrix.")
        return np.clip((p1 - self.p1_given_0) / denominator, 0, 1)


@dataclass(frozen=True)
class Clifford:
    unitary: np.ndarray
    pulses: tuple[tuple[float, float], ...]


def built_in_sequences() -> dict[str, SequenceSpec]:
    """Phase lists available in the supplied manuscript excerpt.

    Add X17, H9, and H18 here after copying their appendix phase vectors.
    """
    theta = math.acos(-math.sqrt(3) / 4)

    return {
        "X1": SequenceSpec("X1", np.pi, np.pi, (0.0,)),
        "X5a": SequenceSpec(
            "X5a",
            np.pi,
            np.pi,
            (2*np.pi/3, 11*np.pi/6, np.pi/3, 11*np.pi/6, 2*np.pi/3),
        ),
        "X5b": SequenceSpec(
            "X5b",
            np.pi,
            np.pi,
            (2*np.pi/3, 5*np.pi/6, np.pi/3, 5*np.pi/6, 2*np.pi/3),
        ),
        "X9a": SequenceSpec(
            "X9a",
            np.pi,
            np.pi,
            (
                np.pi/3,
                5*np.pi/12-theta/2,
                7*np.pi/6-theta,
                17*np.pi/12-theta/2,
                2*np.pi/3,
                17*np.pi/12-theta/2,
                7*np.pi/6-theta,
                5*np.pi/12-theta/2,
                np.pi/3,
            ),
        ),
        "X9b": SequenceSpec(
            "X9b",
            np.pi,
            np.pi,
            (
                np.pi/3,
                5*np.pi/12+theta/2,
                7*np.pi/6+theta,
                17*np.pi/12+theta/2,
                2*np.pi/3,
                17*np.pi/12+theta/2,
                7*np.pi/6+theta,
                5*np.pi/12+theta/2,
                np.pi/3,
            ),
        ),
        "H1": SequenceSpec("H1", np.pi/2, np.pi/2, (0.0,)),
        "H11a": SequenceSpec("H11a", np.pi/2, np.pi/2, (0.777941, 0.518699, 0.660394, 0.877660, 0.525243, 0.273513, 0.525243, 0.877660, 0.660394,
0.518699, 0.777941))
    }


# =============================================================================
# Small math helpers
# =============================================================================


def prx_matrix(angle: float, phase: float) -> np.ndarray:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return np.array(
        [[c, -1j*s*np.exp(1j*phase)], [-1j*s*np.exp(-1j*phase), c]],
        dtype=complex,
    )


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2*np.pi) - np.pi)


def same_unitary(left: np.ndarray, right: np.ndarray, atol: float = 1e-8) -> bool:
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


# =============================================================================
# Pulla execution and pulse construction
# =============================================================================


def select_qubits(pulla: Pulla, requested: Sequence[str] | None) -> tuple[str, ...]:
    topology = pulla.get_chip_topology()
    all_qubits = tuple(getattr(topology, "qubits_sorted", sorted(topology.qubits)))
    if requested is None:
        return all_qubits
    missing = sorted(set(requested) - set(all_qubits))
    if missing:
        raise ValueError(f"Unknown qubits: {missing}")
    return tuple(requested)


def run_batch(
    pulla: Pulla,
    compiler: Any,
    circuits: Sequence[TimeBox],
    qubits: Sequence[str],
    shots: int,
) -> Any:
    settings = compiler.get_settings(timeboxes=list(circuits))
    settings.set_shots(shots)
    run_definition, context = compiler.compile(
        timeboxes=list(circuits),
        components=list(qubits),
        settings=settings,
    )
    job = pulla.submit_playlist(run_definition, context=context)
    job.wait_for_completion()
    return job.result(compiler=compiler).dataset


def p1_from_dataset(dataset: Any, qubit: str, key: str) -> np.ndarray:
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


def prx_impl(builder: Any, qubit: str, config: Config) -> Any:
    if config.prx_implementation is None:
        return builder.prx([qubit])
    return builder.get_implementation(
        "prx", [qubit], impl_name=config.prx_implementation
    )


def add_detuning(
    builder: Any,
    qubit: str,
    box: TimeBox,
    detuning_hz: float,
) -> TimeBox:
    """Apply detuning only to the IQ pulse(s) in this PRX TimeBox."""
    drive = builder.get_drive_channel(qubit)
    schedule = deepcopy(box.atom)
    instructions = list(schedule[drive])
    indices = [i for i, instruction in enumerate(instructions) if isinstance(instruction, IQPulse)]

    if len(indices) != 1:
        raise ValueError(
            f"PRX on {qubit} contains {len(indices)} IQPulse instructions. "
            "Choose a calibrated single-pulse PRX implementation in Config."
        )

    offset = detuning_hz / float(builder.channels[drive].sample_rate)
    i = indices[0]
    instructions[i] = replace(
        instructions[i],
        modulation_frequency=instructions[i].modulation_frequency + offset,
    )
    schedule[drive] = Segment(instructions)
    return TimeBox.atomic(schedule, locus_components=(qubit,), label="detuned_prx")


def calibrated_prx(
    builder: Any,
    qubit: str,
    angle: float,
    phase: float,
    config: Config,
) -> TimeBox:
    return prx_impl(builder, qubit, config)(float(angle), float(phase))


def erroneous_prx(
    builder: Any,
    qubit: str,
    angle: float,
    phase: float,
    detuning_hz: float,
    config: Config,
) -> TimeBox:
    box = calibrated_prx(builder, qubit, angle, phase, config)
    return add_detuning(builder, qubit, box, detuning_hz)


def composite_gate(
    builder: Any,
    qubit: str,
    sequence: SequenceSpec,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
) -> TimeBox:
    angle = sequence.constituent_angle * (1 + amplitude_error)
    return serial(
        erroneous_prx(builder, qubit, angle, phase, detuning_hz, config)
        for phase in sequence.phases
    )

def as_timebox(box_or_boxes: Any) -> TimeBox:
    """Convert a gate implementation result into one TimeBox."""
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
    builder: Any,
    qubits: Sequence[str],
    operations: dict[str, list[TimeBox]],
    key: str,
) -> TimeBox:
    active = [
        serial(operations[q])
        for q in qubits
        if operations[q]
    ]

    measurement = as_timebox(
        builder.measure(qubits)(key=key)
    )

    if active:
        return parallel(active) | measurement

    return measurement

# =============================================================================
# Readout calibration
# =============================================================================


def readout_calibration(
    pulla: Pulla,
    compiler: Any,
    qubits: Sequence[str],
    config: Config,
) -> dict[str, ReadoutCalibration | None]:
    if not config.readout_correction:
        return {q: None for q in qubits}

    builder = compiler.get_schedule_builder()
    ground = {q: [] for q in qubits}
    excited = {q: [calibrated_prx(builder, q, np.pi, 0, config)] for q in qubits}
    dataset = run_batch(
        pulla,
        compiler,
        [
            measured_parallel_circuit(builder, qubits, ground, "ro"),
            measured_parallel_circuit(builder, qubits, excited, "ro"),
        ],
        qubits,
        config.tomography_shots,
    )

    result: dict[str, ReadoutCalibration | None] = {}
    for q in qubits:
        p1 = p1_from_dataset(dataset, q, "ro")
        result[q] = ReadoutCalibration(float(p1[0]), float(1-p1[1]))
    return result


def correct(p1: np.ndarray, calibration: ReadoutCalibration | None) -> np.ndarray:
    return p1 if calibration is None else calibration.correct(p1)


# =============================================================================
# 1 + 2. Tomography, process fidelity, angle and axis
# =============================================================================


TOMO_INPUTS = ("0", "1", "+x", "+y")
TOMO_BASES = ("x", "y", "z")


def prep_boxes(builder: Any, q: str, state: str, config: Config) -> list[TimeBox]:
    if state == "0":
        return []
    if state == "1":
        return [calibrated_prx(builder, q, np.pi, 0, config)]
    if state == "+x":
        return [calibrated_prx(builder, q, np.pi/2, np.pi/2, config)]
    if state == "+y":
        return [calibrated_prx(builder, q, np.pi/2, np.pi, config)]
    raise ValueError(state)


def analysis_boxes(builder: Any, q: str, basis: str, config: Config) -> list[TimeBox]:
    if basis == "z":
        return []
    if basis == "x":
        return [calibrated_prx(builder, q, np.pi/2, -np.pi/2, config)]
    if basis == "y":
        return [calibrated_prx(builder, q, np.pi/2, 0, config)]
    raise ValueError(basis)


def acquire_tomography(
    pulla: Pulla,
    compiler: Any,
    qubits: Sequence[str],
    sequence: SequenceSpec,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
) -> tuple[Any, list[tuple[str, str]]]:
    builder = compiler.get_schedule_builder()
    circuits, metadata = [], []

    for state in TOMO_INPUTS:
        for basis in TOMO_BASES:
            operations = {}
            for q in qubits:
                operations[q] = (
                    prep_boxes(builder, q, state, config)
                    + [composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config)]
                    + analysis_boxes(builder, q, basis, config)
                )
            circuits.append(measured_parallel_circuit(builder, qubits, operations, "tomo"))
            metadata.append((state, basis))

    return (
        run_batch(pulla, compiler, circuits, qubits, config.tomography_shots),
        metadata,
    )


def reconstruct_ptm(
    p1: np.ndarray,
    metadata: Sequence[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = {state: np.zeros(3) for state in TOMO_INPUTS}
    axis = {"x": 0, "y": 1, "z": 2}

    for probability, (state, basis) in zip(p1, metadata, strict=True):
        outputs[state][axis[basis]] = 1 - 2*probability

    translation = (outputs["0"] + outputs["1"]) / 2
    bloch = np.column_stack(
        [
            outputs["+x"] - translation,
            outputs["+y"] - translation,
            (outputs["0"] - outputs["1"]) / 2,
        ]
    )
    ptm = np.zeros((4, 4))
    ptm[0, 0] = 1
    ptm[1:, 0] = translation
    ptm[1:, 1:] = bloch
    return ptm, bloch, translation


def tomography_metrics(
    ptm: np.ndarray,
    bloch: np.ndarray,
    target_angle: float,
) -> dict[str, float]:
    target = np.zeros((4, 4))
    target[0, 0] = 1
    target[1:, 1:] = Rotation.from_rotvec([target_angle, 0, 0]).as_matrix()
    fidelity = float((np.trace(target.T @ ptm) + 2) / 6)

    u, _, vt = np.linalg.svd(bloch)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    rotation_axis = np.array([1.0, 0.0, 0.0]) if angle < 1e-12 else rotvec/angle

    if abs(target_angle-np.pi) < 1e-6 and rotation_axis[0] < 0:
        rotation_axis = -rotation_axis

    return {
        "process_fidelity": fidelity,
        "process_infidelity": 1-fidelity,
        "effective_angle": angle,
        "angle_error": wrap(angle-target_angle),
        "axis_x": float(rotation_axis[0]),
        "axis_y": float(rotation_axis[1]),
        "axis_z": float(rotation_axis[2]),
        "axis_azimuth": float(math.atan2(rotation_axis[1], rotation_axis[0])),
        "axis_tilt": float(math.asin(np.clip(rotation_axis[2], -1, 1))),
        "axis_distance": float(math.acos(np.clip(rotation_axis[0], -1, 1))),
    }


# =============================================================================
# 3. Ramsey transition phase
# =============================================================================


def acquire_ramsey(
    pulla: Pulla,
    compiler: Any,
    qubits: Sequence[str],
    sequence: SequenceSpec,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
) -> Any:
    builder = compiler.get_schedule_builder()
    circuits = []

    for phase in config.ramsey_phases:
        operations = {}
        for q in qubits:
            ops = []
            if abs(sequence.target_angle-np.pi) < 1e-6:
                ops.append(calibrated_prx(builder, q, np.pi/2, 0, config))
            ops.append(composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config))
            ops.append(calibrated_prx(builder, q, np.pi/2, phase, config))
            operations[q] = ops
        circuits.append(measured_parallel_circuit(builder, qubits, operations, "ramsey"))

    return run_batch(pulla, compiler, circuits, qubits, config.ramsey_shots)


def fit_fringe(phases: Sequence[float], p1: Sequence[float]) -> dict[str, float]:
    phases = np.asarray(phases)
    design = np.column_stack([np.ones(len(phases)), np.cos(phases), np.sin(phases)])
    offset, cosine, sine = np.linalg.lstsq(design, np.asarray(p1), rcond=None)[0]
    return {
        "offset": float(offset),
        "contrast": float(math.hypot(cosine, sine)),
        "phase": float(math.atan2(sine, cosine)),
    }


def ideal_ramsey_p1(sequence: SequenceSpec, phases: Sequence[float]) -> np.ndarray:
    state = np.array([1.0, 0.0], dtype=complex)
    if abs(sequence.target_angle-np.pi) < 1e-6:
        state = prx_matrix(np.pi/2, 0) @ state
    state = prx_matrix(sequence.target_angle, 0) @ state
    return np.array([
        abs((prx_matrix(np.pi/2, phase) @ state)[1])**2
        for phase in phases
    ])


def ramsey_metrics(
    p1: np.ndarray,
    sequence: SequenceSpec,
    phases: Sequence[float],
) -> dict[str, float]:
    measured = fit_fringe(phases, p1)
    ideal = fit_fringe(phases, ideal_ramsey_p1(sequence, phases))
    shift = wrap(measured["phase"]-ideal["phase"])
    divisor = 2 if abs(sequence.target_angle-np.pi) < 1e-6 else 1
    return {
        "ramsey_contrast": measured["contrast"],
        "ramsey_fringe_phase": measured["phase"],
        "ramsey_fringe_shift": shift,
        "transition_phase_error": shift/divisor,
    }


# =============================================================================
# 4. Interleaved randomized benchmarking
# =============================================================================


def clifford_group() -> tuple[Clifford, ...]:
    generators = (
        (prx_matrix(np.pi/2, 0), (np.pi/2, 0)),
        (prx_matrix(np.pi/2, np.pi/2), (np.pi/2, np.pi/2)),
    )
    group = [Clifford(np.eye(2, dtype=complex), tuple())]
    queue = [0]

    while queue and len(group) < 24:
        current = group[queue.pop(0)]
        for unitary, pulse in generators:
            candidate = unitary @ current.unitary
            if any(same_unitary(candidate, element.unitary) for element in group):
                continue
            group.append(Clifford(candidate, current.pulses+(pulse,)))
            queue.append(len(group)-1)

    if len(group) != 24:
        raise RuntimeError(f"Generated {len(group)} Cliffords instead of 24.")
    return tuple(group)


def clifford_index(unitary: np.ndarray, group: Sequence[Clifford]) -> int:
    for i, element in enumerate(group):
        if same_unitary(unitary, element.unitary, atol=1e-7):
            return i
    raise ValueError("Unitary is not a single-qubit Clifford.")


def clifford_pulses(
    builder: Any,
    q: str,
    element: Clifford,
    config: Config,
) -> list[TimeBox]:
    return [calibrated_prx(builder, q, angle, phase, config) for angle, phase in element.pulses]


def rb_plan(sequence: SequenceSpec, config: Config) -> tuple[list[dict[str, Any]], tuple[Clifford, ...]]:
    group = clifford_group()
    target = prx_matrix(sequence.target_angle, 0)
    clifford_index(target, group)
    rng = np.random.default_rng(config.seed)
    plan = []

    for length in config.rb_lengths:
        for sample in range(config.rb_samples):
            indices = rng.integers(0, 24, size=length).tolist()

            total = np.eye(2, dtype=complex)
            for i in indices:
                total = group[i].unitary @ total
            reference_recovery = clifford_index(total.conj().T, group)

            total = np.eye(2, dtype=complex)
            for i in indices:
                total = group[i].unitary @ total
                total = target @ total
            interleaved_recovery = clifford_index(total.conj().T, group)

            plan += [
                {
                    "kind": "reference",
                    "length": length,
                    "sample": sample,
                    "indices": indices,
                    "recovery": reference_recovery,
                },
                {
                    "kind": "interleaved",
                    "length": length,
                    "sample": sample,
                    "indices": indices,
                    "recovery": interleaved_recovery,
                },
            ]
    return plan, group


def acquire_rb(
    pulla: Pulla,
    compiler: Any,
    qubits: Sequence[str],
    sequence: SequenceSpec,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
) -> tuple[Any, list[dict[str, Any]]]:
    builder = compiler.get_schedule_builder()
    plan, group = rb_plan(sequence, config)
    circuits = []

    for item in plan:
        operations = {}
        for q in qubits:
            ops = []
            for i in item["indices"]:
                ops += clifford_pulses(builder, q, group[i], config)
                if item["kind"] == "interleaved":
                    ops.append(composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config))
            ops += clifford_pulses(builder, q, group[item["recovery"]], config)
            operations[q] = ops
        circuits.append(measured_parallel_circuit(builder, qubits, operations, "rb"))

    return run_batch(pulla, compiler, circuits, qubits, config.rb_shots), plan


def decay_model(length: np.ndarray, a: float, p: float, b: float) -> np.ndarray:
    return a*p**length+b


def fit_decay(lengths: Sequence[int], survival: Sequence[float]) -> float:
    parameters, _ = curve_fit(
        decay_model,
        np.asarray(lengths, dtype=float),
        np.asarray(survival, dtype=float),
        p0=(0.45, 0.99, 0.5),
        bounds=((0, 0, 0), (1, 1, 1)),
        maxfev=20_000,
    )
    return float(parameters[1])


def rb_metrics(p1: np.ndarray, plan: Sequence[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for probability, item in zip(p1, plan, strict=True):
        grouped.setdefault((item["kind"], item["length"]), []).append(1-probability)

    lengths = sorted({item["length"] for item in plan})
    reference = [np.mean(grouped[("reference", length)]) for length in lengths]
    interleaved = [np.mean(grouped[("interleaved", length)]) for length in lengths]
    p_reference = fit_decay(lengths, reference)
    p_interleaved = fit_decay(lengths, interleaved)
    p_gate = p_interleaved/p_reference

    return {
        "rb_reference_decay": p_reference,
        "rb_interleaved_decay": p_interleaved,
        "rb_gate_decay": p_gate,
        "rb_infidelity": 0.5*(1-p_gate),
    }


# =============================================================================
# End-to-end experiment
# =============================================================================


def run_experiment(
    pulla: Pulla,
    sequences: Sequence[SequenceSpec],
    config: Config,
) -> pd.DataFrame:
    compiler = pulla.get_standard_compiler()
    qubits = select_qubits(pulla, config.qubits)
    readout = readout_calibration(pulla, compiler, qubits, config)
    rows = []

    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                tomo_dataset, tomo_metadata = acquire_tomography(
                    pulla, compiler, qubits, sequence,
                    amplitude_error, detuning_hz, config,
                )
                ramsey_dataset = acquire_ramsey(
                    pulla, compiler, qubits, sequence,
                    amplitude_error, detuning_hz, config,
                )
                rb_dataset, plan = acquire_rb(
                    pulla, compiler, qubits, sequence,
                    amplitude_error, detuning_hz, config,
                )

                for q in qubits:
                    tomo_p1 = correct(p1_from_dataset(tomo_dataset, q, "tomo"), readout[q])
                    ramsey_p1 = correct(p1_from_dataset(ramsey_dataset, q, "ramsey"), readout[q])
                    rb_p1 = correct(p1_from_dataset(rb_dataset, q, "rb"), readout[q])
                    ptm, bloch, translation = reconstruct_ptm(tomo_p1, tomo_metadata)

                    row = {
                        "sequence": sequence.name,
                        "pulse_count": len(sequence.phases),
                        "qubit": q,
                        "amplitude_error": amplitude_error,
                        "detuning_hz": detuning_hz,
                        "translation_x": translation[0],
                        "translation_y": translation[1],
                        "translation_z": translation[2],
                    }
                    row.update(tomography_metrics(ptm, bloch, sequence.target_angle))
                    row.update(ramsey_metrics(ramsey_p1, sequence, config.ramsey_phases))
                    row.update(rb_metrics(rb_p1, plan))
                    rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Minimal use
# =============================================================================


if __name__ == "__main__":
    registry = built_in_sequences()


    # None means all qubits. Example subset: ("QB1", "QB3", "QB5")
    config = Config(qubits=None)

    # Add X17, H9, H18 to built_in_sequences() when their phase lists are available.
    sequences = [
        registry["H11a"],
    ]

    results = run_experiment(Pulla(quantum_computer="garnet"), sequences, config)
    results.to_csv("composite_experiment_results.csv", index=False)
    print(results)
