"""Application-level experiment definition and generic execution runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .backend.base import BackendSession, ExperimentBackend
from .models.points import PointSet
from .models.results import ExperimentResult, PointResult, TechniqueResult
from .readout import NoReadoutCorrection, ReadoutStrategy
from .techniques.base import CharacterizationTechnique, TechniqueContext


class FailurePolicy(str, Enum):
    """How the runner handles a technique failure at one experimental point."""

    ABORT = "abort"
    RECORD_AND_CONTINUE = "record-and-continue"


@dataclass(frozen=True)
class ExperimentSettings:
    """Run-wide settings that do not belong to a target or technique."""

    qubits: tuple[str, ...] | None = None
    failure_policy: FailurePolicy = FailurePolicy.ABORT

    def __post_init__(self) -> None:
        if self.qubits is not None:
            qubits = tuple(self.qubits)
            if not qubits or any(not qubit for qubit in qubits):
                raise ValueError("Configured qubit names must be non-empty.")
            if len(set(qubits)) != len(qubits):
                raise ValueError("Configured qubits must be unique.")
            object.__setattr__(self, "qubits", qubits)


@dataclass(frozen=True)
class CharacterizationExperiment:
    """A declarative collection of points and characterization techniques.

    The definition contains no hardware client and performs no I/O.  It is safe
    to construct, inspect, and serialize from a notebook before choosing a
    backend or opening a service connection.
    """

    points: PointSet
    techniques: tuple[CharacterizationTechnique, ...]
    settings: ExperimentSettings = field(default_factory=ExperimentSettings)
    readout: ReadoutStrategy = field(default_factory=NoReadoutCorrection)

    def __post_init__(self) -> None:
        object.__setattr__(self, "techniques", tuple(self.techniques))
        if not self.points:
            raise ValueError("An experiment needs at least one point.")
        if not self.techniques:
            raise ValueError("An experiment needs at least one characterization technique.")
        identifiers = [technique.technique_id for technique in self.techniques]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Technique IDs must be unique within an experiment.")


class ExperimentRunner:
    """Execute any valid experiment definition against an injected backend."""

    def __init__(self, backend: ExperimentBackend) -> None:
        self.backend = backend

    def run(self, experiment: CharacterizationExperiment) -> ExperimentResult:
        """Validate, acquire, correct, analyze, and return structured results."""

        with self.backend.open_session(experiment.settings.qubits) as session:
            context = TechniqueContext(session.loci)
            self._validate(experiment, session, context)
            readout = experiment.readout.prepare(session)
            point_results = []
            for point in experiment.points:
                technique_results = []
                for technique in experiment.techniques:
                    try:
                        plan = technique.build_plan(point, context)
                        measurements = session.execute(plan)
                        corrected = readout.model.apply(measurements)
                        technique_results.extend(
                            technique.analyze(point, plan, corrected)
                        )
                    except Exception as error:
                        if experiment.settings.failure_policy is FailurePolicy.ABORT:
                            raise
                        technique_results.extend(
                            TechniqueResult.failed(
                                point=point,
                                technique_id=technique.technique_id,
                                locus=locus,
                                error=error,
                            )
                            for locus in session.loci
                        )
                point_results.append(PointResult(point, tuple(technique_results)))
            calibrations = () if readout.calibration is None else (readout.calibration,)
            return ExperimentResult(tuple(point_results), calibrations)

    @staticmethod
    def _validate(
        experiment: CharacterizationExperiment,
        session: BackendSession,
        context: TechniqueContext,
    ) -> None:
        """Validate all combinations before the first acquisition is submitted."""

        for point in experiment.points:
            point.target.validate(point.parameters)
            if not session.supports_target(point.target):
                raise TypeError(
                    f"The selected backend cannot compile target "
                    f"{point.target.operation_id!r} ({type(point.target).__name__})."
                )
            for technique in experiment.techniques:
                if not technique.supports(point.target):
                    raise ValueError(
                        f"Technique {technique.technique_id!r} does not support target "
                        f"{point.target.operation_id!r}."
                    )
                technique.validate(point, context)


# Kept for notebooks written against the first class-based draft. New code should
# use the backend-neutral name CharacterizationExperiment.
CompositeGateExperiment = CharacterizationExperiment
