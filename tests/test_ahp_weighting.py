import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "src" / "module3_routing" / "08_ahp_weighting.py"
SPEC = importlib.util.spec_from_file_location("ahp_weighting", MODULE_PATH)
ahp_weighting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ahp_weighting)


class _Result:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class _Connection:
    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_):
        return _Result(self.row)


class _Engine:
    def __init__(self, row):
        self.row = row

    def connect(self):
        return _Connection(self.row)


class AhpWeightingTests(unittest.TestCase):
    def test_reads_valid_admin_weights_from_database(self):
        row = {
            "w_distance": 0.10,
            "w_flood": 0.20,
            "w_landslide": 0.25,
            "w_capacity": 0.15,
            "w_bridge": 0.10,
            "w_report": 0.20,
        }

        self.assertEqual(row, ahp_weighting.get_ahp_weights(_Engine(row), "safety"))

    def test_uses_default_when_database_has_no_configuration(self):
        weights = ahp_weighting.get_ahp_weights(_Engine(None), "rescue")

        self.assertEqual(set(ahp_weighting.WEIGHT_COLUMNS), set(weights))
        self.assertAlmostEqual(1.0, sum(weights.values()), places=6)


if __name__ == "__main__":
    unittest.main()
