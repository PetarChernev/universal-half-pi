from __future__ import annotations

from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1] / "iqm_composite_experiment"
sys.path.insert(0, str(PACKAGE_DIRECTORY))

from config import Config, QUBIT_SELECTION_METRICS  # noqa: E402
import qubit_selection  # noqa: E402


class FakeTopology:
    def __init__(
        self,
        qubits: tuple[str, ...],
        edges: tuple[tuple[str, str], ...],
    ) -> None:
        self.qubits_sorted = qubits
        self.qubits = frozenset(qubits)
        self.coupler_to_components = {
            f"C{index}": edge
            for index, edge in enumerate(edges)
        }


class FakePulla:
    def __init__(self, topology: FakeTopology) -> None:
        self.topology = topology

    def get_chip_topology(self) -> FakeTopology:
        return self.topology


def topology(
    count: int,
    edges: tuple[tuple[int, int], ...] = (),
) -> FakeTopology:
    qubits = tuple(f"Q{index}" for index in range(1, count + 1))
    named_edges = tuple((qubits[left], qubits[right]) for left, right in edges)
    return FakeTopology(qubits, named_edges)


class QubitSelectionTests(unittest.TestCase):
    def test_none_and_explicit_qubits_preserve_existing_behavior(self) -> None:
        pulla = FakePulla(topology(4))

        self.assertEqual(
            qubit_selection.select_qubits(pulla, None),
            ("Q1", "Q2", "Q3", "Q4"),
        )
        self.assertEqual(
            qubit_selection.select_qubits(pulla, ("Q3", "Q1")),
            ("Q3", "Q1"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown qubits"):
            qubit_selection.select_qubits(pulla, ("Q5",))

    def test_integer_count_validation(self) -> None:
        pulla = FakePulla(topology(4))

        for requested in (0, -1, 5):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(ValueError, "between 1 and 4"):
                    qubit_selection.select_qubits(pulla, requested)
        with self.assertRaisesRegex(TypeError, "not bool"):
            qubit_selection.select_qubits(pulla, True)

    def test_milp_minimizes_induced_edges_exactly(self) -> None:
        graph = topology(
            6,
            (
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
            ),
        )
        selected = qubit_selection.select_qubits(FakePulla(graph), 3)

        selected_set = set(selected)
        actual_edges = sum(
            left in selected_set and right in selected_set
            for left, right in graph.coupler_to_components.values()
        )
        optimum = min(
            sum(
                left in subset and right in subset
                for left, right in graph.coupler_to_components.values()
            )
            for subset in map(set, combinations(graph.qubits_sorted, 3))
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(actual_edges, optimum)

    def test_missing_metrics_fall_back_to_maximum_separation(self) -> None:
        graph = topology(5, ((0, 1), (1, 2), (2, 3), (3, 4)))
        with patch.object(
            qubit_selection,
            "_qubit_selection_data",
            return_value=({"Q2": 0.99}, frozenset(graph.qubits_sorted)),
        ):
            selected = qubit_selection.select_qubits(
                FakePulla(graph),
                2,
                metric_client=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(selected, ("Q1", "Q5"))

    def test_metric_maximizes_worst_qubit_before_total(self) -> None:
        # The two zero-edge alternatives are {Q1, Q2} and {Q3, Q4}.
        graph = topology(4, ((0, 2), (0, 3), (1, 2), (1, 3)))
        values = {"Q1": 1.0, "Q2": 0.6, "Q3": 0.79, "Q4": 0.79}
        with patch.object(
            qubit_selection,
            "_qubit_selection_data",
            return_value=(values, frozenset(graph.qubits_sorted)),
        ):
            selected = qubit_selection.select_qubits(
                FakePulla(graph),
                2,
                metric_client=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(selected, ("Q3", "Q4"))

    def test_unavailable_gate_loci_are_not_selected(self) -> None:
        graph = topology(4)
        with patch.object(
            qubit_selection,
            "_qubit_selection_data",
            return_value=(
                {"Q3": 0.8, "Q4": 0.9},
                frozenset(("Q3", "Q4")),
            ),
        ):
            selected = qubit_selection.select_qubits(
                FakePulla(graph),
                2,
                metric_client=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(selected, ("Q3", "Q4"))

        with patch.object(
            qubit_selection,
            "_qubit_selection_data",
            return_value=({}, frozenset(("Q3", "Q4"))),
        ):
            for count in (3, 4):
                with self.subTest(count=count):
                    with self.assertRaisesRegex(ValueError, "Only 2 qubits support"):
                        qubit_selection.select_qubits(
                            FakePulla(graph),
                            count,
                            metric_client=object(),  # type: ignore[arg-type]
                        )

    def test_final_tie_uses_canonical_topology_order(self) -> None:
        selected = qubit_selection.select_qubits(FakePulla(topology(4)), 2)

        self.assertEqual(selected, ("Q1", "Q2"))

    def test_computational_resonator_connections_are_projected_to_qubits(self) -> None:
        graph = topology(3)
        graph.computational_resonators = frozenset(("R1",))
        graph.coupler_to_components = {
            "C1": ("Q1", "R1"),
            "C2": ("Q2", "R1"),
            "C3": ("Q2", "Q3"),
        }

        selected = qubit_selection.select_qubits(FakePulla(graph), 2)

        self.assertEqual(selected, ("Q1", "Q3"))

    def test_milp_handles_a_sparse_54_qubit_graph(self) -> None:
        rows, columns = 6, 9
        edges = []
        for row in range(rows):
            for column in range(columns):
                qubit = row * columns + column
                if row + 1 < rows:
                    edges.append((qubit, qubit + columns))
                if column + 1 < columns:
                    edges.append((qubit, qubit + 1))
        graph = topology(rows * columns, tuple(edges))

        selected = qubit_selection.select_qubits(FakePulla(graph), 8)
        selected_set = set(selected)

        self.assertEqual(len(selected), 8)
        self.assertFalse(any(
            left in selected_set and right in selected_set
            for left, right in graph.coupler_to_components.values()
        ))


class MetricRetrievalTests(unittest.TestCase):
    def test_supported_metrics_are_selected_through_config(self) -> None:
        for metric in QUBIT_SELECTION_METRICS:
            with self.subTest(metric=metric):
                self.assertEqual(
                    Config(qubit_selection_metric=metric).qubit_selection_metric,
                    metric,
                )
        with self.assertRaisesRegex(ValueError, "Unknown qubit-selection metric"):
            Config(qubit_selection_metric="not-a-metric")

    def test_metric_names_dispatch_to_iqm_observations(self) -> None:
        qubits = ("Q1", "Q2")
        loci = tuple((qubit,) for qubit in qubits)

        class FakeGate:
            implementations = {
                "default": SimpleNamespace(loci=loci),
                "custom": SimpleNamespace(loci=loci),
            }

            @staticmethod
            def get_default_implementation(locus: tuple[str, ...]) -> str:
                del locus
                return "default"

        architecture = SimpleNamespace(
            calibration_set_id="calibration-id",
            gates={"prx": FakeGate(), "measure": FakeGate()},
        )
        observation = SimpleNamespace(invalid=False)
        observation_set = SimpleNamespace(
            invalid=False,
            observations=(observation,),
        )

        class FakeClient:
            def get_dynamic_quantum_architecture(self):
                return architecture

            def get_calibration_set(self, calibration_set_id):
                self.assert_calibration_id(calibration_set_id)
                return observation_set

            def get_quality_metric_set(self, calibration_set_id):
                self.assert_calibration_id(calibration_set_id)
                return observation_set

            @staticmethod
            def assert_calibration_id(calibration_set_id) -> None:
                if calibration_set_id != "calibration-id":
                    raise AssertionError("Metric query used the wrong calibration set.")

        class FakeFinder:
            def __init__(self, observations, *, skip_unparseable):
                if tuple(observations) != (observation,) or not skip_unparseable:
                    raise AssertionError("Unexpected observations passed to finder.")

            @staticmethod
            def get_coherence_times(components):
                if tuple(components) != qubits:
                    raise AssertionError("Unexpected coherence-time loci.")
                return ({"Q1": 1.0, "Q2": 2.0}, {"Q1": 3.0, "Q2": 4.0})

            @staticmethod
            def get_gate_fidelity(gate_name, implementation, locus):
                base = 0.9 if gate_name == "prx" else 0.8
                if implementation not in {"default", "custom"}:
                    raise AssertionError("Unexpected gate implementation.")
                return base + (0.01 if locus == ("Q2",) else 0.0)

        expected = {
            "prx_fidelity": {"Q1": 0.9, "Q2": 0.91},
            "readout_fidelity": {"Q1": 0.8, "Q2": 0.81},
            "t1": {"Q1": 1.0, "Q2": 2.0},
            "t2": {"Q1": 3.0, "Q2": 4.0},
        }
        with patch.object(qubit_selection, "ObservationFinder", FakeFinder):
            for metric, values in expected.items():
                with self.subTest(metric=metric):
                    self.assertEqual(
                        qubit_selection._qubit_selection_data(
                            FakeClient(),  # type: ignore[arg-type]
                            qubits,
                            metric,
                            "custom" if metric == "prx_fidelity" else None,
                        ),
                        (values, frozenset(qubits)),
                    )


if __name__ == "__main__":
    unittest.main()
