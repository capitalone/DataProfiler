from __future__ import print_function

import os
import unittest
from cgi import test
from collections import defaultdict

import networkx as nx

from dataprofiler.data_readers.graph_data import GraphData
from dataprofiler.profilers.graph_profiler import GraphProfile

from . import utils

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestGraphProfiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = nx.Graph()
        cls.graph.add_nodes_from([1, 2, 3, 4])
        cls.graph.add_edges_from(
            [
                (1, 2, {"id": 1, "weight": 2.5}),
                (2, 3, {"id": 2, "weight": 1.7}),
                (3, 4, {"id": 3, "weight": 1.8}),
                (4, 1, {"id": 4, "weight": 4.1}),
                (1, 3, {"id": 4, "weight": 2.1}),
            ]
        )

        cls.expected_profile = dict(
            num_nodes=4,
            num_edges=5,
            categorical_attributes=["id"],
            continuous_attributes=["weight"],
            avg_node_degree=2.5,
            global_max_component_size=5,
            continuous_distribution={
                "id": None,
                "weight": {"name": "lognorm", "scale": -15.250985118262854},
            },
            categorical_distribution={
                "id": {
                    "bin_counts": [1, 1, 1, 2],
                    "bin_edges": [1.0, 1.75, 2.5, 3.25, 4.0],
                },
                "weight": None,
            },
            times=defaultdict(
                float,
                dict(
                    num_nodes=1.0,
                    num_edges=1.0,
                    categorical_attributes=1.0,
                    continuous_attributes=1.0,
                    avg_node_degree=1.0,
                    global_max_component_size=1.0,
                    continuous_distribution=1.0,
                    categorical_distribution=1.0,
                ),
            ),
        )

    def test_profile(self):
        graph_profile = GraphProfile("test_update")
        with utils.mock_timeit():
            profile = graph_profile.update(self.graph)
        self.assertDictEqual(self.expected_profile, profile.profile)

    def test_report(self):
        graph_profile = GraphProfile("test_report")
        with utils.mock_timeit():
            profile = graph_profile.update(self.graph)
        self.maxDiff = None
        self.assertDictEqual(self.expected_profile, graph_profile.report())

    def test_graph_data_object(self):
        data = GraphData(input_file_path=None, options=None, data=self.graph)
        graph_profile = GraphProfile("test_graph_data_object_update")
        with utils.mock_timeit():
            profile = graph_profile.update(data)
        self.assertDictEqual(self.expected_profile, profile.profile)


if __name__ == "__main__":
    unittest.main()
