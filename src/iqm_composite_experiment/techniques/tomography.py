"""Single-qubit process tomography as a pluggable technique."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..backend.base import CircuitFactory
from ..models.acquisition import AcquisitionPlan, MeasurementData, PlannedCircuit
from ..models.points import ExperimentPoint
from ..models.results import TechniqueResult
from ..targets.base import OperationUnderTest, require_single_qubit
from .base import CharacterizationTechnique, TechniqueContext
from .math import unitary_bloch_rotation, wrap_angle

TOMOGRAPHY_INPUTS = ("0", "1", "+x", "+y")
TOMOGRAPHY_BASES = ("x", "y", "z")


@dataclass(frozen=True)
class TomographySettings:
    """Acquisition settings for linear-inversion process tomography."""

    shots: int = 1024
    input_states: tuple[str, ...] = TOMOGRAPHY_INPUTS
    bases: tuple[str, ...] = TOMOGRAPHY_BASES

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ValueError("Tomography shots must be positive.")
        if tuple(self.input_states) != TOMOGRAPHY_INPUTS:
            raise ValueError(
                "The current linear inversion requires input states "
                f"{TOMOGRAPHY_INPUTS!r}."
            )
        if tuple(self.bases) != TOMOGRAPHY_BASES:
            raise ValueError(
                f"The current linear inversion requires bases {TOMOGRAPHY_BASES!r}."
            )


def preparation_operations(
    factory: CircuitFactory,
    locus: tuple[str, ...],
    state: str,
) -> list[object]:
    """Build calibrated preparation operations for one tomography input."""

    if state == "0":
        return []
    if state == "1":
        return [factory.calibrated_prx(locus, np.pi, 0)]
    if state == "+x":
        return [factory.calibrated_prx(locus, np.pi / 2, np.pi / 2)]
    if state == "+y":
        return [factory.calibrated_prx(locus, np.pi / 2, np.pi)]
    raise ValueError(f"Unsupported tomography input state {state!r}.")


def analysis_operations(
    factory: CircuitFactory,
    locus: tuple[str, ...],
    basis: str,
) -> list[object]:
    """Build the calibrated basis change preceding Z measurement."""

    if basis == "z":
        return []
    if basis == "x":
        return [factory.calibrated_prx(locus, np.pi / 2, -np.pi / 2)]
    if basis == "y":
        return [factory.calibrated_prx(locus, np.pi / 2, 0)]
    raise ValueError(f"Unsupported tomography basis {basis!r}.")


def reconstruct_ptm(
    p1: np.ndarray,
    coordinates: tuple[tuple[str, str], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct an affine Pauli-transfer matrix from measured probabilities."""

    outputs = {state: np.zeros(3) for state in TOMOGRAPHY_INPUTS}
    axis = {"x": 0, "y": 1, "z": 2}
    for probability, (state, basis) in zip(p1, coordinates, strict=True):
        outputs[state][axis[basis]] = 1 - 2 * probability
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
    ideal_unitary: np.ndarray,
) -> dict[str, float]:
    """Compute process fidelity and closest-rotation angle/axis metrics."""

    target_rotation = unitary_bloch_rotation(ideal_unitary)
    target_ptm = np.zeros((4, 4))
    target_ptm[0, 0] = 1
    target_ptm[1:, 1:] = target_rotation
    fidelity = float((np.trace(target_ptm.T @ ptm) + 2) / 6)

    left, _, right_transpose = np.linalg.svd(bloch)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_transpose)
    measured_rotation = left @ correction @ right_transpose
    measured_rotvec = Rotation.from_matrix(measured_rotation).as_rotvec()
    measured_angle = float(np.linalg.norm(measured_rotvec))
    measured_axis = (
        np.array([1.0, 0.0, 0.0])
        if measured_angle < 1e-12
        else measured_rotvec / measured_angle
    )

    target_rotvec = Rotation.from_matrix(target_rotation).as_rotvec()
    target_angle = float(np.linalg.norm(target_rotvec))
    target_axis = (
        np.array([1.0, 0.0, 0.0])
        if target_angle < 1e-12
        else target_rotvec / target_angle
    )
    if abs(target_angle - np.pi) < 1e-6 and np.dot(measured_axis, target_axis) < 0:
        measured_axis = -measured_axis

    return {
        "process_fidelity": fidelity,
        "process_infidelity": 1 - fidelity,
        "effective_angle": measured_angle,
        "angle_error": wrap_angle(measured_angle - target_angle),
        "axis_x": float(measured_axis[0]),
        "axis_y": float(measured_axis[1]),
        "axis_z": float(measured_axis[2]),
        "axis_azimuth": float(math.atan2(measured_axis[1], measured_axis[0])),
        "axis_tilt": float(math.asin(np.clip(measured_axis[2], -1, 1))),
        "axis_distance": float(
            math.acos(np.clip(np.dot(measured_axis, target_axis), -1, 1))
        ),
    }


class ProcessTomography(CharacterizationTechnique):
    """Linear-inversion process tomography for arbitrary one-qubit targets."""

    technique_id = "tomography"

    def __init__(self, settings: TomographySettings | None = None) -> None:
        self.settings = settings or TomographySettings()

    def supports(self, target: OperationUnderTest) -> bool:
        return target.ideal_operation.arity == 1

    def validate(self, point: ExperimentPoint, context: TechniqueContext) -> None:
        require_single_qubit(point.target, "Process tomography")
        point.target.validate(point.parameters)
        if not context.loci:
            raise ValueError("Process tomography needs at least one locus.")

    def build_plan(
        self,
        point: ExperimentPoint,
        context: TechniqueContext,
    ) -> AcquisitionPlan:
        circuits = []
        for state in self.settings.input_states:
            for basis in self.settings.bases:

                def build(
                    factory: CircuitFactory,
                    state: str = state,
                    basis: str = basis,
                ) -> object:
                    operations = {}
                    for locus in context.loci:
                        operations[locus] = [
                            *preparation_operations(factory, locus, state),
                            factory.target_operation(point.target, point.parameters, locus),
                            *analysis_operations(factory, locus, basis),
                        ]
                    return factory.measured_parallel(operations, "tomo")

                circuits.append(
                    PlannedCircuit(
                        circuit_id=f"tomo:{state}:{basis}",
                        build=build,
                        coordinates={"input_state": state, "basis": basis},
                    )
                )
        return AcquisitionPlan(
            technique_id=self.technique_id,
            measurement_key="tomo",
            shots=self.settings.shots,
            circuits=tuple(circuits),
        )

    def analyze(
        self,
        point: ExperimentPoint,
        plan: AcquisitionPlan,
        measurements: MeasurementData,
    ) -> tuple[TechniqueResult, ...]:
        coordinates = tuple(
            (str(circuit.coordinates["input_state"]), str(circuit.coordinates["basis"]))
            for circuit in plan.circuits
        )
        results = []
        for locus, probability_data in measurements.probabilities.items():
            ptm, bloch, translation = reconstruct_ptm(probability_data.p1, coordinates)
            metrics = {
                "translation_x": float(translation[0]),
                "translation_y": float(translation[1]),
                "translation_z": float(translation[2]),
                **tomography_metrics(ptm, bloch, point.target.ideal_operation.unitary),
            }
            results.append(
                TechniqueResult(
                    point_id=point.point_id,
                    target_id=point.target.operation_id,
                    technique_id=self.technique_id,
                    locus=locus,
                    metrics=metrics,
                    artifacts={
                        "raw_p1": probability_data.raw_p1,
                        "analyzed_p1": probability_data.p1,
                        "ptm": ptm,
                        "bloch_matrix": bloch,
                        "translation": translation,
                    },
                )
            )
        return tuple(results)
