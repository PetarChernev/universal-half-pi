"""Common interface implemented by all characterization techniques."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..models.acquisition import AcquisitionPlan, Locus, MeasurementData
from ..models.points import ExperimentPoint
from ..models.results import TechniqueResult
from ..targets.base import OperationUnderTest


@dataclass(frozen=True)
class TechniqueContext:
    """Backend capabilities available while a technique creates its plan."""

    loci: tuple[Locus, ...]


class CharacterizationTechnique(ABC):
    """Plan and analyze one reusable method of characterizing an operation.

    A technique never submits jobs itself.  It creates a deferred acquisition
    plan, and the experiment runner gives the normalized measurements back to
    ``analyze``.  Consequently the same technique can run against IQM hardware,
    a simulator, or an in-memory test backend.
    """

    technique_id: str

    @abstractmethod
    def supports(self, target: OperationUnderTest) -> bool:
        """Return whether the technique's scientific assumptions fit ``target``."""

    @abstractmethod
    def validate(self, point: ExperimentPoint, context: TechniqueContext) -> None:
        """Raise before acquisition if this point cannot be characterized."""

    @abstractmethod
    def build_plan(
        self,
        point: ExperimentPoint,
        context: TechniqueContext,
    ) -> AcquisitionPlan:
        """Build deferred circuits and their technique-specific coordinates."""

    @abstractmethod
    def analyze(
        self,
        point: ExperimentPoint,
        plan: AcquisitionPlan,
        measurements: MeasurementData,
    ) -> tuple[TechniqueResult, ...]:
        """Convert normalized observations into metrics and artifacts."""
