"""IQM realizations of the framework's built-in operation targets."""

from __future__ import annotations

from ...models.acquisition import Locus
from ...models.parameters import ParameterSet
from ...targets.composite import UniversalCompositePulse
from ..base import OperationCompiler
from .circuits import IQMBuildContext, add_detuning, calibrated_prx, serial


class IQMUniversalCompositePulseCompiler(OperationCompiler[UniversalCompositePulse]):
    """Compile universal phase sequences into detuned calibrated PRX TimeBoxes."""

    target_type = UniversalCompositePulse

    def compile(
        self,
        target: UniversalCompositePulse,
        parameters: ParameterSet,
        locus: Locus,
        context: IQMBuildContext,
    ) -> object:
        resolved = target.resolve_parameters(parameters)
        amplitude_error = resolved.require_float("amplitude_error")
        detuning_hz = resolved.require_float("detuning_hz")
        angle = target.constituent_angle * (1 + amplitude_error)
        boxes = []
        for phase in target.phases:
            box = calibrated_prx(
                context.builder,
                locus,
                angle,
                phase,
                context.prx_implementation,
            )
            boxes.append(add_detuning(context.builder, locus, box, detuning_hz))
        return serial(boxes)
