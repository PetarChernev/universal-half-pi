"""Service-free tests for YAML-to-domain composition."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gate_experiment.backend.iqm import ProbabilitySource
from gate_experiment.configuration import load_application_configuration


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_repository_configuration_loads_without_creating_pulla(self) -> None:
        configuration = load_application_configuration(ROOT / "config.yaml")

        self.assertIsNone(configuration.server_url)
        self.assertEqual(len(configuration.experiment.points), 54)
        self.assertEqual(
            [technique.technique_id for technique in configuration.experiment.techniques],
            ["tomography", "transition_phase_ramsey", "interleaved_rb"],
        )
        self.assertIs(
            configuration.backend_settings.probability_source,
            ProbabilitySource.THRESHOLDED_READOUT,
        )

    def test_local_readout_rejects_exa_corrected_probability(self) -> None:
        document = """
iqm:
  server_url: null
  probability_source: exa_probability
experiment:
  targets: [X1]
  sweep:
    kind: cartesian
    axes: []
techniques:
  tomography:
    shots: 8
readout:
  enabled: true
  shots: 8
output:
  csv: output.csv
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(document, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "double correction"):
                load_application_configuration(path)

    def test_disabled_technique_is_omitted(self) -> None:
        document = """
iqm:
  server_url: null
experiment:
  targets: [X1]
  sweep:
    axes: []
techniques:
  tomography:
    enabled: true
    shots: 8
  interleaved_rb:
    enabled: false
readout:
  enabled: false
output:
  csv: output.csv
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(document, encoding="utf-8")
            configuration = load_application_configuration(path)

        self.assertEqual(
            [technique.technique_id for technique in configuration.experiment.techniques],
            ["tomography"],
        )


if __name__ == "__main__":
    unittest.main()
