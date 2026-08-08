"""Preset universal composite-pulse targets from the supplied manuscript."""

from __future__ import annotations

import math

import numpy as np

from .composite import UniversalCompositePulse


def universal_composite_pulses() -> dict[str, UniversalCompositePulse]:
    """Return fresh target objects for the currently available phase vectors.

    X17, H9, and H18 can be added here when their appendix phase vectors are
    available.  Returning a new dictionary keeps notebook modifications local.
    """

    theta = math.acos(-math.sqrt(3) / 4)
    specs = {
        "X1": (np.pi, np.pi, (0.0,)),
        "X5a": (
            np.pi,
            np.pi,
            (2 * np.pi / 3, 11 * np.pi / 6, np.pi / 3, 11 * np.pi / 6, 2 * np.pi / 3),
        ),
        "X5b": (
            np.pi,
            np.pi,
            (2 * np.pi / 3, 5 * np.pi / 6, np.pi / 3, 5 * np.pi / 6, 2 * np.pi / 3),
        ),
        "X9a": (
            np.pi,
            np.pi,
            (
                np.pi / 3,
                5 * np.pi / 12 - theta / 2,
                7 * np.pi / 6 - theta,
                17 * np.pi / 12 - theta / 2,
                2 * np.pi / 3,
                17 * np.pi / 12 - theta / 2,
                7 * np.pi / 6 - theta,
                5 * np.pi / 12 - theta / 2,
                np.pi / 3,
            ),
        ),
        "X9b": (
            np.pi,
            np.pi,
            (
                np.pi / 3,
                5 * np.pi / 12 + theta / 2,
                7 * np.pi / 6 + theta,
                17 * np.pi / 12 + theta / 2,
                2 * np.pi / 3,
                17 * np.pi / 12 + theta / 2,
                7 * np.pi / 6 + theta,
                5 * np.pi / 12 + theta / 2,
                np.pi / 3,
            ),
        ),
        "H1": (np.pi / 2, np.pi / 2, (0.0,)),
    }
    return {
        name: UniversalCompositePulse.for_x_rotation(
            name,
            target_angle=float(target_angle),
            constituent_angle=float(constituent_angle),
            phases=tuple(float(phase) for phase in phases),
        )
        for name, (target_angle, constituent_angle, phases) in specs.items()
    }
