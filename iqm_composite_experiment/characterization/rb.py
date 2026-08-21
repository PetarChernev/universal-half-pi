"""Interleaved randomized benchmarking acquisition and analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.timebox import TimeBox
from scipy.optimize import curve_fit


from common import (
    Clifford,
    ReadoutMap,
    SequenceSpec,
    calibrated_prx,
    composite_gate,
    correct,
    measured_parallel_circuit,
    paper_phase_to_iqm,
    p1_from_dataset,
    prx_matrix,
    same_unitary,
)
from config import Config
from execution import AcquiredBatch, PlannedBatch


class RBPlanItem(TypedDict):
    """Description of one reference or interleaved RB circuit."""

    kind: Literal["reference", "interleaved"]
    length: int
    sample: int
    indices: list[int]
    recovery: int


ResultValue: TypeAlias = str | int | float | bool


def build_batch(builder: ScheduleBuilder, qubits: Sequence[str], sequence: SequenceSpec, amplitude_error: float, detuning_hz: float, config: Config, acquisition_index: int) -> PlannedBatch:
    """Build reference and interleaved RB circuits for one sweep point.

    Reference circuits contain random single-qubit Cliffords followed by their
    recovery.  Interleaved circuits use the same random Cliffords, insert the
    physical composite gate after each one, and use a recovery computed from
    the sequence's ideal target. The manifest maps every circuit result to its
    kind, length, sample, random indices, and recovery.
    """
    rb_plan, group = plan(sequence, config)
    circuits: list[TimeBox] = []
    for item in rb_plan:
        operations: dict[str, list[TimeBox]] = {}
        for q in qubits:
            ops: list[TimeBox] = []
            for i in item["indices"]:
                ops += clifford_pulses(builder, q, group[i], config)
                if item["kind"] == "interleaved":
                    # Only the gate under test receives the requested systematic
                    # error; the surrounding Clifford pulses stay calibrated.
                    ops.append(composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config))
            ops += clifford_pulses(builder, q, group[item["recovery"]], config)
            operations[q] = ops
        circuits.append(measured_parallel_circuit(builder, qubits, operations, "rb"))
    return PlannedBatch(
        batch_id=f"rb_{acquisition_index:04d}",
        technique="interleaved_randomized_benchmarking",
        acquisition_index=acquisition_index,
        circuits=tuple(circuits),
        qubits=tuple(qubits),
        shots=config.rb_shots,
        measurement_key="rb",
        metadata={
            "technique": "interleaved_randomized_benchmarking",
            "acquisition_index": acquisition_index,
            "sequence": sequence.name,
            "target_angle_rad": float(sequence.target_angle),
            "constituent_angle_rad": float(sequence.constituent_angle),
            "paper_phases_rad": [float(phase) for phase in sequence.phases],
            "iqm_phases_rad": [paper_phase_to_iqm(phase) for phase in sequence.phases],
            "amplitude_error": amplitude_error,
            "detuning_hz": detuning_hz,
            "qubits": list(qubits),
            "shots": config.rb_shots,
            "readout_calibration_shots": (
                config.rb_shots
                if config.pre_calibration or config.post_calibration
                else None
            ),
            "pre_calibration": config.pre_calibration,
            "post_calibration": config.post_calibration,
            "seed": config.seed,
            "prx_implementation": config.prx_implementation,
            "fit_model": "amplitude * decay**length + offset",
            "rb_plan": rb_plan,
        },
        manifest=tuple(rb_plan),
        sequence=sequence,
        amplitude_error=amplitude_error,
        detuning_hz=detuning_hz,
    )


def build_batches(builder: ScheduleBuilder, qubits: Sequence[str], sequences: Sequence[SequenceSpec], config: Config) -> list[PlannedBatch]:
    """Build every configured RB batch without submitting work."""
    batches: list[PlannedBatch] = []
    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                batches.append(build_batch(builder, qubits, sequence, amplitude_error, detuning_hz, config, len(batches)))
    return batches


def analyze(acquisitions: Sequence[AcquiredBatch], readout: ReadoutMap) -> pd.DataFrame:
    """Analyze acquired interleaved-RB batches after raw persistence.

    Readout-corrected results are reduced to one row of
    reference, interleaved, and inferred gate-decay metrics per qubit.
    """
    rows: list[dict[str, ResultValue]] = []
    for acquisition in acquisitions:
        plan = acquisition.plan
        sequence = plan.sequence
        if plan.technique != "interleaved_randomized_benchmarking" or sequence is None:
            raise ValueError(f"Unexpected RB batch: {plan.batch_id}")
        rb_plan: Sequence[RBPlanItem] = plan.manifest
        for q in plan.qubits:
            row: dict[str, ResultValue] = {"sequence": sequence.name, "pulse_count": len(sequence.phases), "qubit": q, "amplitude_error": float(plan.amplitude_error), "detuning_hz": float(plan.detuning_hz)}
            row.update(metrics(
                correct(p1_from_dataset(acquisition.dataset, q, plan.measurement_key), readout[q]),
                rb_plan,
            ))
            rows.append(row)
    return pd.DataFrame(rows)


def clifford_group() -> tuple[Clifford, ...]:
    """Generate the 24 single-qubit Cliffords and calibrated pulse words.

    Breadth-first expansion by positive x and y pi/2 generators gives a short
    PRX decomposition for every group element.  Unitaries that differ only by
    global phase are treated as the same Clifford.
    """
    generators = (
        (prx_matrix(np.pi / 2, 0), (np.pi / 2, 0)),
        (prx_matrix(np.pi / 2, np.pi / 2), (np.pi / 2, np.pi / 2)),
    )
    group = [Clifford(np.eye(2, dtype=complex), tuple())]
    queue = [0]
    # Breadth-first traversal preserves the first, and therefore shortest,
    # generator word found for each Clifford.
    while queue and len(group) < 24:
        current = group[queue.pop(0)]
        for unitary, pulse in generators:
            candidate = unitary @ current.unitary
            if any(same_unitary(candidate, element.unitary) for element in group):
                continue
            group.append(Clifford(candidate, current.pulses + (pulse,)))
            queue.append(len(group) - 1)
    if len(group) != 24:
        raise RuntimeError(f"Generated {len(group)} Cliffords instead of 24.")
    return tuple(group)


def clifford_index(
    unitary: npt.NDArray[np.complex128],
    group: Sequence[Clifford],
) -> int:
    """Return the index of a unitary in ``group``, ignoring global phase.

    Raises:
        ValueError: If ``unitary`` is not represented by the single-qubit
            Clifford group.
    """
    for i, element in enumerate(group):
        if same_unitary(unitary, element.unitary, atol=1e-7):
            return i
    raise ValueError("Unitary is not a single-qubit Clifford.")


def clifford_pulses(
    builder: ScheduleBuilder,
    q: str,
    element: Clifford,
    config: Config,
) -> list[TimeBox]:
    """Build the calibrated PRX pulse decomposition of one Clifford."""
    return [calibrated_prx(builder, q, angle, phase, config) for angle, phase in element.pulses]


def plan(
    sequence: SequenceSpec,
    config: Config,
) -> tuple[list[RBPlanItem], tuple[Clifford, ...]]:
    """Create reproducible reference/interleaved RB circuit descriptions.

    For each configured length and random sample, both variants share the same
    random Clifford indices.  Their recovery Cliffords invert the respective
    ideal accumulated unitary.  The composite target must itself be a
    single-qubit Clifford so that every interleaved recovery belongs to the
    generated group.

    Returns:
        The ordered circuit plan and the Clifford group indexed by that plan.
    """
    group = clifford_group()
    target = prx_matrix(sequence.target_angle, 0)
    # Fail before generating circuits if interleaved Clifford recovery is not
    # defined for the selected ideal target.
    clifford_index(target, group)
    rng = np.random.default_rng(config.seed)
    result: list[RBPlanItem] = []
    for length in config.rb_lengths:
        for sample in range(config.rb_samples):
            indices = [int(index) for index in rng.integers(0, 24, size=length)]
            total = np.eye(2, dtype=complex)
            # Pulses execute left-to-right, so each new unitary multiplies the
            # accumulated operation from the left.
            for i in indices:
                total = group[i].unitary @ total
            reference_recovery = clifford_index(total.conj().T, group)
            total = np.eye(2, dtype=complex)
            for i in indices:
                total = group[i].unitary @ total
                # Acquisition inserts the gate immediately after this Clifford.
                total = target @ total
            interleaved_recovery = clifford_index(total.conj().T, group)
            result += [
                {"kind": "reference", "length": length, "sample": sample, "indices": indices, "recovery": reference_recovery},
                {"kind": "interleaved", "length": length, "sample": sample, "indices": indices, "recovery": interleaved_recovery},
            ]
    return result, group



def decay_model(
    length: npt.NDArray[np.float64],
    a: float,
    p: float,
    b: float,
) -> npt.NDArray[np.float64]:
    """Evaluate the RB survival model ``A * p**length + B``."""
    return a * p ** length + b


def fit_decay(lengths: Sequence[int], survival: Sequence[float]) -> dict[str, float]:
    """Fit a bounded exponential to mean ground-state survival values.

    The amplitude, decay, and offset are constrained to ``[0, 1]``.  Returned
    diagnostics include covariance-derived parameter standard errors and the
    root-mean-square fit residual.
    """
    x = np.asarray(lengths, dtype=float)
    y = np.asarray(survival, dtype=float)
    parameters, covariance = curve_fit(
        decay_model,
        x,
        y,
        p0=(0.45, 0.99, 0.5),
        bounds=((0, 0, 0), (1, 1, 1)),
        maxfev=20_000,
    )
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return {
        "amplitude": float(parameters[0]),
        "decay": float(parameters[1]),
        "offset": float(parameters[2]),
        "amplitude_standard_error": float(standard_errors[0]),
        "decay_standard_error": float(standard_errors[1]),
        "offset_standard_error": float(standard_errors[2]),
        "fit_rmse": float(np.sqrt(np.mean((y - decay_model(x, *parameters)) ** 2))),
    }


def metrics(
    p1: npt.NDArray[np.float64],
    rb_plan: Sequence[RBPlanItem],
) -> dict[str, float | bool]:
    """Infer composite-gate error from reference and interleaved RB decays.

    ``p1`` is converted to ground-state survival and averaged across random
    samples at each length.  Independent exponential fits produce
    ``p_reference`` and ``p_interleaved``; their ratio estimates the decay of
    one composite-gate application, and ``(1 - ratio) / 2`` is the reported
    single-qubit RB infidelity.
    """
    grouped: dict[tuple[str, int], list[float]] = {}
    for probability, item in zip(p1, rb_plan, strict=True):
        grouped.setdefault((item["kind"], item["length"]), []).append(
            float(1 - probability)
        )
    lengths = sorted({item["length"] for item in rb_plan})
    reference = [float(np.mean(grouped[("reference", length)])) for length in lengths]
    interleaved = [float(np.mean(grouped[("interleaved", length)])) for length in lengths]
    reference_fit = fit_decay(lengths, reference)
    interleaved_fit = fit_decay(lengths, interleaved)
    p_reference = reference_fit["decay"]
    p_interleaved = interleaved_fit["decay"]
    if p_reference <= 0:
        p_gate = float("nan")
        p_gate_standard_error = float("nan")
    else:
        p_gate = p_interleaved / p_reference
        # First-order propagation for a ratio of independent fitted decays.
        p_gate_standard_error = float(np.hypot(
            interleaved_fit["decay_standard_error"] / p_reference,
            p_interleaved * reference_fit["decay_standard_error"] / p_reference**2,
        ))
    result: dict[str, float | bool] = {
        "rb_gate_decay": p_gate,
        "rb_gate_decay_standard_error": p_gate_standard_error,
        "rb_infidelity": 0.5 * (1 - p_gate),
        "rb_estimate_physical": bool(0 <= p_gate <= 1),
    }
    result.update({f"rb_reference_{key}": value for key, value in reference_fit.items()})
    result.update({f"rb_interleaved_{key}": value for key, value in interleaved_fit.items()})
    return result
