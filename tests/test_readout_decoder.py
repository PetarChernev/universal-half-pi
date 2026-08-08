"""Tests for local readout correction and strict IQM dataset decoding."""

from __future__ import annotations

import unittest

import numpy as np

from gate_experiment.backend.iqm.decoder import IQMDatasetDecoder, ProbabilitySource
from gate_experiment.models.acquisition import AcquisitionPlan, PlannedCircuit
from gate_experiment.readout import BinaryReadoutCalibration


class FakeDataArray:
    def __init__(self, data, dims) -> None:
        self.data = np.asarray(data)
        self.dims = tuple(dims)


class FakeDataset(dict):
    @property
    def data_vars(self):
        return tuple(self)


def two_circuit_plan() -> AcquisitionPlan:
    return AcquisitionPlan(
        technique_id="test",
        measurement_key="measure",
        shots=10,
        circuits=(
            PlannedCircuit("c0", lambda factory: object()),
            PlannedCircuit("c1", lambda factory: object()),
        ),
    )


class ReadoutTests(unittest.TestCase):
    def test_binary_correction_inverts_confusion_model(self) -> None:
        calibration = BinaryReadoutCalibration(p1_given_0=0.1, p0_given_1=0.2)

        corrected = calibration.correct(np.array([0.1, 0.45, 0.8]))

        np.testing.assert_allclose(corrected, [0.0, 0.5, 1.0])

    def test_singular_confusion_model_is_rejected(self) -> None:
        calibration = BinaryReadoutCalibration(p1_given_0=0.6, p0_given_1=0.4)

        with self.assertRaisesRegex(ValueError, "singular"):
            calibration.correct(np.array([0.5]))


class IQMDecoderTests(unittest.TestCase):
    def test_exa_probability_preserves_circuit_index_order(self) -> None:
        dataset = FakeDataset(
            {
                "QB1__measure_excited_state_probability": FakeDataArray(
                    [[0.2, 0.8]],
                    ("singleton", "circuit_index"),
                )
            }
        )

        measurements = IQMDatasetDecoder().decode(
            dataset,
            two_circuit_plan(),
            (("QB1",),),
        )

        np.testing.assert_allclose(measurements.for_locus(("QB1",)).raw_p1, [0.2, 0.8])

    def test_thresholded_readout_is_averaged_over_shots(self) -> None:
        dataset = FakeDataset(
            {
                "QB1__measure_readout": FakeDataArray(
                    [[0, 0, 1, 0], [1, 1, 0, 1]],
                    ("circuit_index", "repetitions"),
                )
            }
        )
        decoder = IQMDatasetDecoder(ProbabilitySource.THRESHOLDED_READOUT)

        measurements = decoder.decode(dataset, two_circuit_plan(), (("QB1",),))

        np.testing.assert_allclose(measurements.for_locus(("QB1",)).raw_p1, [0.25, 0.75])

    def test_missing_exact_variable_is_not_fuzzy_matched(self) -> None:
        dataset = FakeDataset(
            {"QB1__similar_measure_excited_state_probability": FakeDataArray([0.2, 0.8], ("circuit_index",))}
        )

        with self.assertRaisesRegex(KeyError, "missing"):
            IQMDatasetDecoder().decode(dataset, two_circuit_plan(), (("QB1",),))


if __name__ == "__main__":
    unittest.main()
