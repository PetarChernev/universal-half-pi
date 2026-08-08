"""Universal composite pulses built from calibrated equatorial rotations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np

from ..models.parameters import ParameterSpec, Scalar
from .base import IdealOperation, OperationUnderTest


def prx_unitary(angle: float, phase: float) -> np.ndarray:
    """Return the unitary for an equatorial rotation in IQM PRX convention."""

    cosine = math.cos(angle / 2)
    sine = math.sin(angle / 2)
    return np.array(
        [
            [cosine, -1j * sine * np.exp(1j * phase)],
            [-1j * sine * np.exp(-1j * phase), cosine],
        ],
        dtype=complex,
    )


@dataclass(frozen=True, eq=False)
class UniversalCompositePulse(OperationUnderTest):
    """A target operation implemented as a phase-programmed PRX sequence.

    ``amplitude_error`` scales every constituent rotation angle, while
    ``detuning_hz`` offsets every constituent pulse.  These are deliberately
    point parameters rather than fields, allowing one target instance to be
    reused throughout a sweep.
    """

    operation_id: str
    ideal: IdealOperation
    constituent_angle: float
    phases: tuple[float, ...]

    _PARAMETER_SPECS: ClassVar[tuple[ParameterSpec, ...]] = (
        ParameterSpec(
            "amplitude_error",
            float,
            default=0.0,
            description="Fractional constituent-pulse angle error.",
        ),
        ParameterSpec(
            "detuning_hz",
            float,
            default=0.0,
            unit="Hz",
            description="Drive-frequency offset applied to each constituent pulse.",
        ),
    )

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("Operation IDs cannot be empty.")
        if self.ideal.arity != 1:
            raise ValueError("Universal composite pulses must target one qubit.")
        if not math.isfinite(self.constituent_angle):
            raise ValueError("Constituent angle must be finite.")
        if not self.phases or not all(math.isfinite(phase) for phase in self.phases):
            raise ValueError("A composite pulse needs at least one finite phase.")
        object.__setattr__(self, "phases", tuple(float(phase) for phase in self.phases))

    @property
    def ideal_operation(self) -> IdealOperation:
        return self.ideal

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return self._PARAMETER_SPECS

    def metadata(self) -> dict[str, Scalar]:
        return {
            "pulse_count": len(self.phases),
            "constituent_angle": self.constituent_angle,
        }

    @classmethod
    def for_x_rotation(
        cls,
        operation_id: str,
        *,
        target_angle: float,
        constituent_angle: float,
        phases: tuple[float, ...],
    ) -> UniversalCompositePulse:
        """Construct a composite pulse targeting an ideal X rotation."""

        return cls(
            operation_id=operation_id,
            ideal=IdealOperation(
                unitary=prx_unitary(target_angle, 0.0),
                arity=1,
                name=f"Rx({target_angle:g})",
            ),
            constituent_angle=constituent_angle,
            phases=phases,
        )
