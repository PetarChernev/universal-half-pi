from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import xarray as xr


PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1] / "iqm_composite_experiment"
sys.path.insert(0, str(PACKAGE_DIRECTORY))

from characterization import ramsey, rb, tomography  # noqa: E402
from common import (  # noqa: E402
    paper_phase_to_iqm,
    prx_matrix,
    readout_calibration_from_datasets,
    same_unitary,
)
from dataset_persistence import persist_dataset  # noqa: E402
from sequences import SequenceSpec, built_in_sequences  # noqa: E402


class PhaseConventionTests(unittest.TestCase):
    def test_negated_iqm_phase_equals_paper_unitary(self) -> None:
        angle = 0.73
        phase = -1.17
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        paper_unitary = np.array([
            [c, -1j * s * np.exp(1j * phase)],
            [-1j * s * np.exp(-1j * phase), c],
        ])

        np.testing.assert_allclose(
            prx_matrix(angle, paper_phase_to_iqm(phase)),
            paper_unitary,
        )

    def test_paper_sequences_reach_their_ideal_targets(self) -> None:
        registry = built_in_sequences()
        for name, sequence in registry.items():
            actual = np.eye(2, dtype=complex)
            for paper_phase in sequence.phases:
                actual = prx_matrix(
                    sequence.constituent_angle,
                    paper_phase_to_iqm(paper_phase),
                ) @ actual
            target = prx_matrix(sequence.target_angle, 0)
            self.assertTrue(same_unitary(actual, target, atol=1e-8), name)


class TomographyMetricTests(unittest.TestCase):
    @staticmethod
    def ptm(rotation: np.ndarray) -> np.ndarray:
        result = np.zeros((4, 4))
        result[0, 0] = 1
        result[1:, 1:] = rotation
        return result

    def test_identity_error_has_unit_average_gate_fidelity(self) -> None:
        angle = np.pi / 2
        bloch = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
        result = tomography.metrics(self.ptm(bloch), bloch, angle)

        self.assertAlmostEqual(result["average_gate_fidelity"], 1.0)
        self.assertAlmostEqual(result["average_gate_infidelity"], 0.0)
        self.assertNotIn("process_fidelity", result)

    def test_x_overrotation_retains_its_sign(self) -> None:
        error = 0.1
        bloch = Rotation.from_rotvec([np.pi + error, 0, 0]).as_matrix()
        result = tomography.metrics(self.ptm(bloch), bloch, np.pi)

        self.assertAlmostEqual(result["effective_angle"], np.pi + error)
        self.assertAlmostEqual(result["angle_error"], error)
        self.assertAlmostEqual(result["axis_x"], 1.0)


class RamseyMetricTests(unittest.TestCase):
    @staticmethod
    def probabilities(target_angle: float, paper_axis_phase: float) -> np.ndarray:
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        state = np.array([1.0, 0.0], dtype=complex)
        if abs(target_angle - np.pi) < 1e-6:
            state = prx_matrix(np.pi / 2, 0) @ state
        state = prx_matrix(target_angle, -paper_axis_phase) @ state
        return np.array([
            abs((prx_matrix(np.pi / 2, phase) @ state)[1]) ** 2
            for phase in phases
        ])

    def test_transition_phase_uses_paper_sign_for_h_and_x(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        paper_axis_phase = 0.12
        for target_angle in (np.pi / 2, np.pi):
            sequence = SequenceSpec("test", target_angle, target_angle, (0.0,))
            result = ramsey.metrics(
                self.probabilities(target_angle, paper_axis_phase),
                sequence,
                phases,
                shots=10_000,
            )
            self.assertTrue(result["ramsey_phase_identifiable"])
            self.assertAlmostEqual(
                result["transition_phase_error"],
                paper_axis_phase,
            )

    def test_transition_phase_is_suppressed_without_contrast(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        sequence = SequenceSpec("test", np.pi / 2, np.pi / 2, (0.0,))
        result = ramsey.metrics(
            np.full(len(phases), 0.5),
            sequence,
            phases,
            shots=100,
        )

        self.assertFalse(result["ramsey_phase_identifiable"])
        self.assertTrue(math.isnan(result["transition_phase_error"]))


class PersistenceTests(unittest.TestCase):
    def test_dataset_metadata_is_written_beside_raw_data(self) -> None:
        dataset = xr.Dataset(
            {"value": ("circuit_index", np.array([1.0, 2.0]))},
            coords={"circuit_index": [0, 1]},
        )
        dataset["value"].attrs["parameter"] = object()
        dataset["circuit_index"].attrs["parameter"] = object()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acquisition.nc"
            metadata = {"technique": "test", "coordinates": [1, 2]}
            persist_dataset(dataset, path, metadata)

            with xr.open_dataset(path) as restored:
                np.testing.assert_array_equal(restored["value"], [1.0, 2.0])
                self.assertEqual(restored["value"].attrs, {})
                self.assertEqual(restored["circuit_index"].attrs, {})
            self.assertEqual(
                json.loads(path.with_suffix(".json").read_text()),
                metadata,
            )
            self.assertIn("parameter", dataset["value"].attrs)


class ReadoutCalibrationTests(unittest.TestCase):
    def test_pre_and_post_calibrations_are_averaged(self) -> None:
        variable = "QB1__tomo_excited_state_probability"
        pre = xr.Dataset({variable: ("circuit_index", [0.1, 0.8])})
        post = xr.Dataset({variable: ("circuit_index", [0.2, 0.9])})

        result = readout_calibration_from_datasets(
            ((pre, "tomo"), (post, "tomo")),
            ("QB1",),
        )

        calibration = result["QB1"]
        self.assertIsNotNone(calibration)
        assert calibration is not None
        self.assertAlmostEqual(calibration.p1_given_0, 0.15)
        self.assertAlmostEqual(calibration.p0_given_1, 0.15)

    def test_no_calibration_disables_readout_correction(self) -> None:
        self.assertEqual(
            readout_calibration_from_datasets((), ("QB1", "QB2")),
            {"QB1": None, "QB2": None},
        )


class RandomizedBenchmarkingMetricTests(unittest.TestCase):
    def test_decay_diagnostics_and_gate_error_are_reported(self) -> None:
        lengths = (1, 2, 4, 8, 16, 32)
        plan = []
        p1 = []
        for length in lengths:
            for kind, decay in (("reference", 0.98), ("interleaved", 0.95)):
                plan.append({"kind": kind, "length": length})
                p1.append(1 - rb.decay_model(np.array(length), 0.4, decay, 0.5))

        result = rb.metrics(np.asarray(p1), plan)

        self.assertAlmostEqual(result["rb_reference_decay"], 0.98, places=6)
        self.assertAlmostEqual(result["rb_interleaved_decay"], 0.95, places=6)
        self.assertAlmostEqual(result["rb_gate_decay"], 0.95 / 0.98, places=6)
        self.assertIn("rb_reference_fit_rmse", result)
        self.assertIn("rb_gate_decay_standard_error", result)
        self.assertTrue(result["rb_estimate_physical"])


if __name__ == "__main__":
    unittest.main()
