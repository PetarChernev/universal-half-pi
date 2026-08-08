"""Safety checks for the plain Python production composition script."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import run_experiment


class EntrypointSafetyTests(unittest.TestCase):
    def test_missing_url_fails_before_pulla_is_constructed(self) -> None:
        with patch.object(run_experiment, "Pulla") as pulla:
            with self.assertRaisesRegex(ValueError, "iqm.server_url"):
                run_experiment.main()

        pulla.assert_not_called()


if __name__ == "__main__":
    unittest.main()
