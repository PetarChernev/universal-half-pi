"""IQM TimeBox construction hidden behind the framework circuit protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from iqm.pulse.playlist.instructions import IQPulse
from iqm.pulse.playlist.schedule import Segment
from iqm.pulse.timebox import TimeBox

from ...models.acquisition import Locus
from ...models.parameters import ParameterSet
from ...targets.base import OperationUnderTest
from ..base import OperationCompilerRegistry


@dataclass(frozen=True)
class IQMBuildContext:
    """Concrete resources supplied to IQM operation compilers."""

    builder: Any
    prx_implementation: str | None


def require_single_component(locus: Locus) -> str:
    """Return the component in a one-element locus or raise clearly."""

    if len(locus) != 1:
        raise ValueError(f"The current IQM pulse adapter requires one-component loci, got {locus!r}.")
    return locus[0]


def serial(boxes: Sequence[TimeBox]) -> TimeBox:
    """Compose non-empty TimeBoxes sequentially using IQM's ``|`` operator."""

    if not boxes:
        raise ValueError("Cannot serialize an empty operation list.")
    result = boxes[0]
    for box in boxes[1:]:
        result = result | box
    return result


def parallel(boxes: Sequence[TimeBox]) -> TimeBox:
    """Create a TimeBox composite resolved according to channel conflicts."""

    if not boxes:
        raise ValueError("Cannot parallelize an empty operation list.")
    return TimeBox.composite(list(boxes))


def calibrated_prx(
    builder: Any,
    locus: Locus,
    angle: float,
    phase: float,
    implementation: str | None,
) -> TimeBox:
    """Build a calibrated IQM PRX operation for one locus."""

    qubit = require_single_component(locus)
    if implementation is None:
        gate = builder.prx([qubit])
    else:
        gate = builder.get_implementation("prx", [qubit], impl_name=implementation)
    return gate(float(angle), float(phase))


def add_detuning(builder: Any, locus: Locus, box: TimeBox, detuning_hz: float) -> TimeBox:
    """Offset the only IQ pulse in an atomic PRX TimeBox.

    IQM stores modulation frequency in fractions of channel sample rate.  This
    compiler deliberately rejects composite or multi-pulse PRX implementations,
    because rewriting those silently would not represent a well-defined physical
    detuning model.
    """

    qubit = require_single_component(locus)
    if box.atom is None:
        raise ValueError(
            f"PRX on {qubit} is not atomic; select a calibrated single-pulse implementation."
        )
    drive = builder.get_drive_channel(qubit)
    schedule = deepcopy(box.atom)
    instructions = list(schedule[drive])
    indices = [
        index
        for index, instruction in enumerate(instructions)
        if isinstance(instruction, IQPulse)
    ]
    if len(indices) != 1:
        raise ValueError(
            f"PRX on {qubit} contains {len(indices)} IQPulse instructions; "
            "select a calibrated single-pulse implementation."
        )
    index = indices[0]
    offset = float(detuning_hz) / float(builder.channels[drive].sample_rate)
    instructions[index] = replace(
        instructions[index],
        modulation_frequency=instructions[index].modulation_frequency + offset,
    )
    schedule[drive] = Segment(instructions)
    return TimeBox.atomic(schedule, locus_components=(qubit,), label="detuned_prx")


class IQMCircuitFactory:
    """Concrete circuit factory used inside an IQM compiler callback."""

    def __init__(
        self,
        builder: Any,
        registry: OperationCompilerRegistry,
        *,
        prx_implementation: str | None,
    ) -> None:
        self._context = IQMBuildContext(builder, prx_implementation)
        self._registry = registry

    def calibrated_prx(self, locus: Locus, angle: float, phase: float) -> TimeBox:
        return calibrated_prx(
            self._context.builder,
            locus,
            angle,
            phase,
            self._context.prx_implementation,
        )

    def target_operation(
        self,
        target: OperationUnderTest,
        parameters: ParameterSet,
        locus: Locus,
    ) -> TimeBox:
        operation = self._registry.compile(target, parameters, locus, self._context)
        if not isinstance(operation, TimeBox):
            raise TypeError("IQM operation compilers must return TimeBox objects.")
        return operation

    def measured_parallel(
        self,
        operations: Mapping[Locus, Sequence[object]],
        measurement_key: str,
    ) -> TimeBox:
        boxes = []
        qubits = []
        for locus, locus_operations in operations.items():
            qubits.append(require_single_component(locus))
            typed = []
            for operation in locus_operations:
                if not isinstance(operation, TimeBox):
                    raise TypeError("IQM circuit operations must be TimeBox objects.")
                typed.append(operation)
            if typed:
                boxes.append(serial(typed))
        measurement = self._context.builder.measure(qubits)(key=measurement_key)
        return parallel(boxes) | measurement if boxes else measurement
