"""Immutable, validated parameters for an experimental operating point.

The framework deliberately represents point parameters by name instead of adding
a field for every physical control.  A new target can therefore introduce, for
example, ``drag_beta`` or ``pulse_duration`` without changing the experiment
runner or any unrelated characterization technique.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import TypeAlias

Scalar: TypeAlias = bool | int | float | str | None


def _is_value_of_type(value: Scalar, expected: type | tuple[type, ...]) -> bool:
    """Return whether ``value`` matches a declared parameter type.

    Integer values are accepted for floating-point parameters because notebook
    users commonly write ``2`` where ``2.0`` is intended.  Booleans are not
    treated as integers in this scientific configuration layer.
    """

    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if bool not in expected_types and isinstance(value, bool):
        return False
    if float in expected_types and isinstance(value, Real):
        return True
    return isinstance(value, expected_types)


@dataclass(frozen=True)
class ParameterSpec:
    """Schema entry for one parameter accepted by an operation under test."""

    name: str
    value_type: type | tuple[type, ...]
    default: Scalar = None
    required: bool = False
    unit: str | None = None
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter names cannot be empty.")
        if self.required and self.default is not None:
            raise ValueError(f"Required parameter {self.name!r} cannot also have a default.")

    def validate(self, value: Scalar) -> None:
        """Validate one value and raise an informative error on mismatch."""

        if not _is_value_of_type(value, self.value_type):
            raise TypeError(
                f"Parameter {self.name!r} must be {self.value_type!r}; "
                f"received {type(value).__name__}."
            )
        if isinstance(value, Real) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Parameter {self.name!r} must be finite.")
            if self.minimum is not None and numeric < self.minimum:
                raise ValueError(f"Parameter {self.name!r} must be >= {self.minimum}.")
            if self.maximum is not None and numeric > self.maximum:
                raise ValueError(f"Parameter {self.name!r} must be <= {self.maximum}.")


@dataclass(frozen=True, init=False)
class ParameterSet(Mapping[str, Scalar]):
    """An immutable mapping of parameter names to scalar values.

    Values are stored in sorted order so point generation and result export are
    deterministic.  ``ParameterSet`` implements the standard mapping interface,
    so it remains convenient in notebooks: ``dict(point.parameters)`` works as
    expected.
    """

    _items: tuple[tuple[str, Scalar], ...]

    def __init__(
        self,
        values: Mapping[str, Scalar] | None = None,
        /,
        **kwargs: Scalar,
    ) -> None:
        combined = dict(values or {})
        combined.update(kwargs)
        normalized: dict[str, Scalar] = {}
        for name, value in combined.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter names must be non-empty strings.")
            if isinstance(value, bool):
                normalized[name] = value
            elif isinstance(value, Integral):
                normalized[name] = int(value)
            elif isinstance(value, Real):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"Parameter {name!r} must be finite.")
                normalized[name] = numeric
            elif isinstance(value, (str, type(None))):
                normalized[name] = value
            else:
                raise TypeError(
                    f"Parameter {name!r} is not a supported scalar value: {value!r}."
                )
        object.__setattr__(self, "_items", tuple(sorted(normalized.items())))

    def __getitem__(self, key: str) -> Scalar:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def merged(self, overrides: Mapping[str, Scalar]) -> ParameterSet:
        """Return a new set with ``overrides`` applied."""

        values = dict(self)
        values.update(overrides)
        return ParameterSet(values)

    def require_float(self, name: str) -> float:
        """Return a required real-valued parameter as ``float``."""

        value = self[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Parameter {name!r} must be a real number.")
        return float(value)

    def get_float(self, name: str, default: float = 0.0) -> float:
        """Return an optional real-valued parameter as ``float``."""

        if name not in self:
            return float(default)
        return self.require_float(name)

    def validate_against(
        self,
        specs: tuple[ParameterSpec, ...],
        *,
        reject_unknown: bool = True,
    ) -> ParameterSet:
        """Validate this set against ``specs`` and fill declared defaults."""

        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ValueError("Parameter specifications contain duplicate names.")
        unknown = sorted(set(self) - set(by_name))
        if reject_unknown and unknown:
            raise ValueError(f"Unsupported parameters: {unknown}.")

        resolved = dict(self)
        for spec in specs:
            if spec.name not in resolved:
                if spec.required:
                    raise ValueError(f"Missing required parameter {spec.name!r}.")
                resolved[spec.name] = spec.default
            spec.validate(resolved[spec.name])
        return ParameterSet(resolved)
