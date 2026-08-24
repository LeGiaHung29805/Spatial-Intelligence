import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import networkx as nx


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


class RoutingGraphSourceTests(unittest.TestCase):
    @staticmethod
    def _graph_with(columns):
        graph = nx.MultiDiGraph()
        graph.add_edge(1, 2, key=0, **{column: 1 for column in columns})
        return graph

    def test_prefers_current_graph_when_it_has_mcdm_schema(self):
        current_graph = self._graph_with(ahp_weighting.REQUIRED_EDGE_COLUMNS)
        legacy_graph = self._graph_with(set())

        with TemporaryDirectory() as temporary_directory:
            current_path = Path(temporary_directory) / "current.graphml"
            legacy_path = Path(temporary_directory) / "legacy.graphml"
            current_path.touch()
            legacy_path.touch()

            with patch.object(ahp_weighting, "GRAPH_SOURCE_PATHS", (current_path, legacy_path)), \
                 patch.object(ahp_weighting.ox, "load_graphml", side_effect=[current_graph, legacy_graph]) as load_graph:
                selected = ahp_weighting.load_compatible_graph()

        self.assertIs(selected, current_graph)
        load_graph.assert_called_once_with(current_path)

    def test_rejects_graphs_missing_required_mcdm_attributes(self):
        incomplete_graph = self._graph_with({"length_m"})

        with TemporaryDirectory() as temporary_directory:
            current_path = Path(temporary_directory) / "current.graphml"
            current_path.touch()

            with patch.object(ahp_weighting, "GRAPH_SOURCE_PATHS", (current_path,)), \
                 patch.object(ahp_weighting.ox, "load_graphml", return_value=incomplete_graph):
                selected = ahp_weighting.load_compatible_graph()

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
