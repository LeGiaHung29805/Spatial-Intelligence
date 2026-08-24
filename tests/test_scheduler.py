import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# The review runtime intentionally installs only the libraries needed by unit tests.
# Scheduler behavior itself is not exercised here, so a minimal module is sufficient.
sys.modules.setdefault("schedule", SimpleNamespace())
import main_scheduler


class SchedulerTests(unittest.TestCase):
    def test_interval_defaults_to_five_minutes(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(5, main_scheduler.get_interval_minutes())

    def test_interval_requires_a_positive_integer(self):
        with patch.dict(os.environ, {"AI_PIPELINE_INTERVAL_MINUTES": "15"}, clear=True):
            self.assertEqual(15, main_scheduler.get_interval_minutes())

        with patch.dict(os.environ, {"AI_PIPELINE_INTERVAL_MINUTES": "0"}, clear=True):
            self.assertIsNone(main_scheduler.get_interval_minutes())


if __name__ == "__main__":
    unittest.main()
