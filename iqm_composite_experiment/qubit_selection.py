"""Qubit selection using topology and IQM quality metrics."""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Mapping, Sequence
from itertools import combinations
from numbers import Integral

import numpy as np
import numpy.typing as npt
from exa.common.errors.iqm_error import IQMError
from iqm.iqm_client import IQMClient
from iqm.pulla.pulla import Pulla
from iqm.station_control.client.qon import ObservationFinder
from requests import RequestException
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp
from scipy.sparse import coo_matrix

from config import QUBIT_SELECTION_METRICS


logger = logging.getLogger(__name__)

# Differences below one part per million or one nanosecond are treated as ties.
_FIDELITY_SCORE_SCALE = 1_000_000
_COHERENCE_SCORE_SCALE = 1_000_000_000


def _topology_edges(topology: object, qubits: Sequence[str]) -> tuple[tuple[int, int], ...]:
    positions = {qubit: index for index, qubit in enumerate(qubits)}
    resonators = set(getattr(topology, "computational_resonators", ()))
    resonator_qubits: dict[str, set[int]] = {
        resonator: set()
        for resonator in resonators
    }
    edges: set[tuple[int, int]] = set()
    for components in topology.coupler_to_components.values():  # type: ignore[attr-defined]
        connected = sorted(positions[q] for q in components if q in positions)
        edges.update(combinations(connected, 2))
        for resonator in resonators.intersection(components):
            resonator_qubits[resonator].update(connected)
    for connected in resonator_qubits.values():
        edges.update(combinations(sorted(connected), 2))
    return tuple(sorted(edges))


def _all_pairs_distances(
    qubit_count: int,
    edges: Sequence[tuple[int, int]],
) -> npt.NDArray[np.int64]:
    disconnected = qubit_count + 1
    distances = np.full((qubit_count, qubit_count), disconnected, dtype=np.int64)
    np.fill_diagonal(distances, 0)
    for left, right in edges:
        distances[left, right] = 1
        distances[right, left] = 1
    for intermediate in range(qubit_count):
        distances = np.minimum(
            distances,
            distances[:, intermediate, None] + distances[None, intermediate, :],
        )
    return distances


def _base_selection_constraint(
    qubit_count: int,
    pairs: Sequence[tuple[int, int]],
    selected_count: int,
) -> LinearConstraint:
    variable_count = qubit_count + len(pairs)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower = [float(selected_count)]
    upper = [float(selected_count)]

    for qubit in range(qubit_count):
        rows.append(0)
        columns.append(qubit)
        values.append(1.0)

    row = 1
    for pair_index, (left, right) in enumerate(pairs):
        pair_variable = qubit_count + pair_index
        for qubit in (left, right):
            rows.extend((row, row))
            columns.extend((pair_variable, qubit))
            values.extend((1.0, -1.0))
            lower.append(-np.inf)
            upper.append(0.0)
            row += 1

        rows.extend((row, row, row))
        columns.extend((pair_variable, left, right))
        values.extend((1.0, -1.0, -1.0))
        lower.append(-1.0)
        upper.append(np.inf)
        row += 1

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(row, variable_count),
    ).tocsr()
    return LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))


def _row_constraint(
    variable_count: int,
    coefficients: Mapping[int, int | float],
    lower: int | float,
    upper: int | float,
) -> LinearConstraint:
    columns = np.fromiter(coefficients, dtype=int, count=len(coefficients))
    values = np.fromiter(coefficients.values(), dtype=float, count=len(coefficients))
    matrix = coo_matrix(
        (values, (np.zeros(len(columns), dtype=int), columns)),
        shape=(1, variable_count),
    ).tocsr()
    return LinearConstraint(matrix, float(lower), float(upper))


def _exclude_qubit_pairs_constraint(
    variable_count: int,
    pairs: Sequence[tuple[int, int]],
) -> LinearConstraint | None:
    if not pairs:
        return None
    rows = np.repeat(np.arange(len(pairs)), 2)
    columns = np.asarray(pairs, dtype=int).reshape(-1)
    matrix = coo_matrix(
        (np.ones(len(columns)), (rows, columns)),
        shape=(len(pairs), variable_count),
    ).tocsr()
    return LinearConstraint(matrix, -np.inf, 1.0)


def _solve_selection_milp(
    objective: npt.NDArray[np.float64],
    integrality: npt.NDArray[np.int64],
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64],
    constraints: Sequence[LinearConstraint],
    *,
    allow_infeasible: bool = False,
) -> OptimizeResult | None:
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=tuple(constraints),
        options={"mip_rel_gap": 0.0, "presolve": True},
    )
    if result.status == 2 and allow_infeasible:
        return None
    if not result.success or result.x is None:
        raise RuntimeError(f"Qubit-selection MILP failed: {result.message}")
    return result


def _valid_observations(observation_set: object) -> tuple[object, ...]:
    if observation_set.invalid:  # type: ignore[attr-defined]
        return ()
    return tuple(
        observation
        for observation in observation_set.observations  # type: ignore[attr-defined]
        if not observation.invalid
    )


def _gate_is_available(
    gate: object | None,
    locus: tuple[str, ...],
    implementation: str | None = None,
) -> bool:
    if gate is None:
        return False
    implementation = (
        implementation
        if implementation is not None
        else gate.get_default_implementation(locus)  # type: ignore[attr-defined]
    )
    implementation_info = gate.implementations.get(implementation)  # type: ignore[attr-defined]
    return implementation_info is not None and locus in implementation_info.loci


def _qubit_selection_data(
    client: IQMClient,
    qubits: Sequence[str],
    metric: str,
    prx_implementation: str | None,
) -> tuple[dict[str, float], frozenset[str]]:
    architecture = client.get_dynamic_quantum_architecture()
    calibration_set_id = architecture.calibration_set_id
    prx_gate = architecture.gates.get("prx")
    measure_gate = architecture.gates.get("measure")
    available_qubits = frozenset(
        qubit
        for qubit in qubits
        if _gate_is_available(prx_gate, (qubit,), prx_implementation)
        and _gate_is_available(measure_gate, (qubit,))
    )

    try:
        if metric in {"t1", "t2"}:
            observation_set = client.get_calibration_set(calibration_set_id)
        else:
            observation_set = client.get_quality_metric_set(calibration_set_id)
    except (IQMError, RequestException, ValueError) as error:
        logger.warning(
            "Could not retrieve %s metrics; using graph separation: %s",
            metric,
            error,
        )
        return {}, available_qubits

    try:
        observations = _valid_observations(observation_set)
        if not observations:
            logger.warning(
                "No valid %s observations were returned; using graph separation",
                metric,
            )
            return {}, available_qubits
        finder = ObservationFinder(observations, skip_unparseable=True)

        if metric in {"t1", "t2"}:
            t1_values, t2_values = finder.get_coherence_times(qubits)
            values = t1_values if metric == "t1" else t2_values
            if not values:
                logger.warning(
                    "No usable %s metrics were returned; using graph separation",
                    metric,
                )
            return values, available_qubits

        gate_name = "prx" if metric == "prx_fidelity" else "measure"
        gate = prx_gate if gate_name == "prx" else measure_gate
        if gate is None:
            logger.warning(
                "The dynamic architecture has no %s gate; using graph separation",
                gate_name,
            )
            return {}, available_qubits

        values: dict[str, float] = {}
        for qubit in qubits:
            locus = (qubit,)
            implementation = (
                prx_implementation
                if gate_name == "prx" and prx_implementation is not None
                else gate.get_default_implementation(locus)
            )
            implementation_info = gate.implementations.get(implementation)
            if implementation_info is None or locus not in implementation_info.loci:
                continue
            fidelity = finder.get_gate_fidelity(gate_name, implementation, locus)
            if fidelity is not None:
                values[qubit] = fidelity
        if not values:
            logger.warning(
                "No usable %s metrics were returned; using graph separation",
                metric,
            )
        return values, available_qubits
    except (IQMError, RequestException, ValueError) as error:
        logger.warning(
            "Could not parse %s metrics; using graph separation: %s",
            metric,
            error,
        )
        return {}, available_qubits


def _metric_scores(
    metric: str,
    values: Mapping[str, float],
) -> dict[str, int]:
    fidelity = metric in {"prx_fidelity", "readout_fidelity"}
    scale = _FIDELITY_SCORE_SCALE if fidelity else _COHERENCE_SCORE_SCALE
    scores: dict[str, int] = {}
    for qubit, raw_value in values.items():
        value = float(raw_value)
        if not math.isfinite(value):
            continue
        if fidelity and not 0 <= value <= 1:
            continue
        if not fidelity and value <= 0:
            continue
        scores[qubit] = round(value * scale)
    return scores


def _milp_select_qubits(
    qubits: Sequence[str],
    edges: Sequence[tuple[int, int]],
    selected_count: int,
    metric_scores: Mapping[str, int],
    available_qubits: Collection[str] | None = None,
) -> tuple[str, ...]:
    qubit_count = len(qubits)
    variable_count = qubit_count + len(edges)

    base_constraint = _base_selection_constraint(
        qubit_count,
        edges,
        selected_count,
    )
    constraints: list[LinearConstraint] = [base_constraint]
    integrality = np.zeros(variable_count, dtype=np.int64)
    integrality[:qubit_count] = 1
    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.ones(variable_count)
    if available_qubits is not None:
        available_qubits = set(available_qubits)
        for index, qubit in enumerate(qubits):
            if qubit not in available_qubits:
                upper_bounds[index] = 0
    zero_objective = np.zeros(variable_count)

    edge_objective = np.zeros(variable_count)
    edge_objective[qubit_count:] = 1
    edge_result = _solve_selection_milp(
        edge_objective,
        integrality,
        lower_bounds,
        upper_bounds,
        constraints,
    )
    assert edge_result is not None
    minimum_edges = round(float(edge_result.fun))
    if edges:
        constraints.append(_row_constraint(
            variable_count,
            {qubit_count + edge: 1 for edge in range(len(edges))},
            minimum_edges,
            minimum_edges,
        ))

    score_by_index = np.asarray(
        [metric_scores.get(qubit, -1) for qubit in qubits],
        dtype=np.int64,
    )
    metric_bounds = upper_bounds.copy()
    metric_bounds[:qubit_count][score_by_index < 0] = 0
    metric_result = None
    if np.count_nonzero(score_by_index >= 0) >= selected_count:
        metric_result = _solve_selection_milp(
            zero_objective,
            integrality,
            lower_bounds,
            metric_bounds,
            constraints,
            allow_infeasible=True,
        )

    if metric_result is not None:
        thresholds = sorted(set(score_by_index[score_by_index >= 0].tolist()))
        low, high = 0, len(thresholds) - 1
        best_threshold = thresholds[0]
        while low <= high:
            middle = (low + high) // 2
            threshold = thresholds[middle]
            trial_bounds = metric_bounds.copy()
            trial_bounds[:qubit_count][score_by_index < threshold] = 0
            feasible = _solve_selection_milp(
                zero_objective,
                integrality,
                lower_bounds,
                trial_bounds,
                constraints,
                allow_infeasible=True,
            )
            if feasible is None:
                high = middle - 1
            else:
                best_threshold = threshold
                low = middle + 1

        upper_bounds = metric_bounds
        upper_bounds[:qubit_count][score_by_index < best_threshold] = 0
        metric_objective = np.zeros(variable_count)
        metric_objective[:qubit_count] = -score_by_index
        metric_result = _solve_selection_milp(
            metric_objective,
            integrality,
            lower_bounds,
            upper_bounds,
            constraints,
        )
        assert metric_result is not None
        total_metric = round(-float(metric_result.fun))
        metric_coefficients = {
            index: int(score)
            for index, score in enumerate(score_by_index)
            if score >= 0
        }
        if any(metric_coefficients.values()):
            constraints.append(_row_constraint(
                variable_count,
                metric_coefficients,
                total_metric,
                total_metric,
            ))
    elif metric_scores:
        logger.warning(
            "No minimum-edge subset has complete metric data; "
            "using graph separation for qubit selection"
        )

    distances = _all_pairs_distances(qubit_count, edges)
    all_pairs = tuple(combinations(range(qubit_count), 2))
    pair_distances = np.asarray(
        [distances[left, right] for left, right in all_pairs],
        dtype=np.int64,
    )
    if selected_count >= 2:
        thresholds = sorted(set(pair_distances.tolist()))
        low, high = 0, len(thresholds) - 1
        best_threshold = thresholds[0]
        while low <= high:
            middle = (low + high) // 2
            threshold = thresholds[middle]
            excluded_pairs = tuple(
                all_pairs[index]
                for index in np.flatnonzero(pair_distances < threshold)
            )
            exclusion = _exclude_qubit_pairs_constraint(
                variable_count,
                excluded_pairs,
            )
            trial_constraints = constraints + ([] if exclusion is None else [exclusion])
            feasible = _solve_selection_milp(
                zero_objective,
                integrality,
                lower_bounds,
                upper_bounds,
                trial_constraints,
                allow_infeasible=True,
            )
            if feasible is None:
                high = middle - 1
            else:
                best_threshold = threshold
                low = middle + 1

        excluded_pairs = tuple(
            all_pairs[index]
            for index in np.flatnonzero(pair_distances < best_threshold)
        )
        exclusion = _exclude_qubit_pairs_constraint(
            variable_count,
            excluded_pairs,
        )
        if exclusion is not None:
            constraints.append(exclusion)

    # A binary-weighted objective uniquely fixes 16 canonical-order decisions
    # per solve without introducing poorly scaled 54-bit coefficients.
    canonical_result: OptimizeResult | None = None
    canonical_block_size = 16
    for start in range(0, qubit_count, canonical_block_size):
        stop = min(start + canonical_block_size, qubit_count)
        canonical_coefficients = {
            qubit: 1 << (stop - qubit - 1)
            for qubit in range(start, stop)
        }
        canonical_objective = np.zeros(variable_count)
        for qubit, weight in canonical_coefficients.items():
            canonical_objective[qubit] = -weight
        canonical_result = _solve_selection_milp(
            canonical_objective,
            integrality,
            lower_bounds,
            upper_bounds,
            constraints,
        )
        assert canonical_result is not None
        canonical_score = round(-float(canonical_result.fun))
        constraints.append(_row_constraint(
            variable_count,
            canonical_coefficients,
            canonical_score,
            canonical_score,
        ))

    assert canonical_result is not None
    selected = np.flatnonzero(canonical_result.x[:qubit_count] > 0.5).tolist()
    if len(selected) != selected_count:
        raise RuntimeError("Qubit-selection MILP returned the wrong number of qubits.")
    logger.info(
        "Selected %d qubits with %d internal connection%s",
        selected_count,
        minimum_edges,
        "" if minimum_edges == 1 else "s",
    )
    return tuple(qubits[index] for index in selected)


def select_qubits(
    pulla: Pulla,
    requested: Sequence[str] | int | None,
    *,
    selection_metric: str = "prx_fidelity",
    metric_client: IQMClient | None = None,
    prx_implementation: str | None = None,
) -> tuple[str, ...]:
    """Resolve explicit qubits or optimize an integer-sized subset.

    Automatic selection minimizes internal graph edges, maximizes the selected
    quality metric, then maximizes pairwise graph separation.
    """
    topology = pulla.get_chip_topology()
    all_qubits = (
        tuple(topology.qubits_sorted)
        if hasattr(topology, "qubits_sorted")
        else tuple(sorted(topology.qubits))
    )
    if requested is None:
        return all_qubits
    if isinstance(requested, bool):
        raise TypeError("Qubit count must be an integer, not bool.")
    if isinstance(requested, Integral):
        selected_count = int(requested)
        if not 1 <= selected_count <= len(all_qubits):
            raise ValueError(
                f"Qubit count must be between 1 and {len(all_qubits)}, "
                f"got {selected_count}."
            )
        if selection_metric not in QUBIT_SELECTION_METRICS:
            supported = ", ".join(QUBIT_SELECTION_METRICS)
            raise ValueError(
                f"Unknown qubit-selection metric {selection_metric!r}; "
                f"expected one of: {supported}"
            )
        metric_values: dict[str, float] = {}
        available_qubits: frozenset[str] | None = None
        if metric_client is not None:
            try:
                metric_values, available_qubits = _qubit_selection_data(
                    metric_client,
                    all_qubits,
                    selection_metric,
                    prx_implementation,
                )
            except (IQMError, RequestException, ValueError) as error:
                logger.warning(
                    "Could not retrieve %s metrics; using graph separation: %s",
                    selection_metric,
                    error,
                )
            else:
                if len(available_qubits) < selected_count:
                    raise ValueError(
                        f"Only {len(available_qubits)} qubits support the required "
                        f"PRX and measurement implementations; cannot select "
                        f"{selected_count}."
                    )
        else:
            logger.info(
                "No metric client provided; using graph separation for qubit selection"
            )

        metric_scores = _metric_scores(selection_metric, metric_values)
        if metric_values and not metric_scores:
            logger.warning(
                "All returned %s metrics were invalid; using graph separation",
                selection_metric,
            )
        if selected_count == len(all_qubits):
            return all_qubits

        return _milp_select_qubits(
            all_qubits,
            _topology_edges(topology, all_qubits),
            selected_count,
            metric_scores,
            available_qubits,
        )

    if isinstance(requested, (str, bytes)):
        raise TypeError("Explicit qubits must be a sequence of qubit names.")
    missing = sorted(set(requested) - set(all_qubits))
    if missing:
        raise ValueError(f"Unknown qubits: {missing}")
    return tuple(requested)
