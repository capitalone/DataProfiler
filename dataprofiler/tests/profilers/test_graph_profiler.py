from __future__ import print_function

import os
import unittest
from io import BytesIO
from unittest import mock
from cgi import test
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

import dataprofiler as dp
from dataprofiler.data_readers.graph_data import GraphData
from dataprofiler.profilers.graph_profiler import GraphProfiler

from dataprofiler.profilers.profiler_options import (
    ProfilerOptions,
)

from . import utils

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def setup_save_mock_open(mock_open):
    mock_file = BytesIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file

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
                "weight": {"name": "lognorm"},
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

        cls.expected_props = [
            8.646041719759628,
            1.6999999999999997,
            0.19403886939727638,
            np.array([8.64604172, 1.7, 0.19403887]),
            np.array([8.64604172, 1.7, 0.19403887]),
            np.array([0.68017604, 1.53392998, 4.54031127]),
            np.array([0.69395918, 3.52941176, 30.92163966]),
        ]

    def check_continuous_properties(self, continuous_distribution_props):
        """Tests the properties array for continuous distribution"""
        for index, property in enumerate(continuous_distribution_props):
            if isinstance(property, np.ndarray):
                np.testing.assert_array_almost_equal(
                    self.expected_props[index], property
                )
            else:
                self.assertAlmostEqual(self.expected_props[index], property)

    def test_profile(self):
        graph_profile = GraphProfiler("test_update")
        with utils.mock_timeit():
            profile = graph_profile.update(self.graph)
        scale = profile.profile["continuous_distribution"]["weight"].pop("scale")
        continuous_distribution_props = profile.profile["continuous_distribution"][
            "weight"
        ].pop("properties")
        self.assertAlmostEqual(scale, -15.250985118262854)
        self.check_continuous_properties(continuous_distribution_props)
        self.assertDictEqual(self.expected_profile, profile.profile)

    def test_report(self):
        graph_profile = GraphProfiler("test_report")
        with utils.mock_timeit():
            profile = graph_profile.update(self.graph)
        scale = profile.profile["continuous_distribution"]["weight"].pop("scale")
        continuous_distribution_props = profile.profile["continuous_distribution"][
            "weight"
        ].pop("properties")
        self.assertAlmostEqual(scale, -15.250985118262854)
        self.check_continuous_properties(continuous_distribution_props)
        self.assertDictEqual(self.expected_profile, graph_profile.report())

    def test_graph_data_object(self):
        data = GraphData(input_file_path=None, data=self.graph)
        graph_profile = GraphProfiler("test_graph_data_object_update")

        with utils.mock_timeit():
            profile = graph_profile.update(data)
        scale = profile.profile["continuous_distribution"]["weight"].pop("scale")
        continuous_distribution_props = profile.profile["continuous_distribution"][
            "weight"
        ].pop("properties")
        self.assertAlmostEqual(scale, -15.250985118262854)
        self.check_continuous_properties(continuous_distribution_props)
        self.assertDictEqual(self.expected_profile, profile.profile)

    def test_save_and_load(self):
        data = GraphData(input_file_path=None, data=self.graph)
        save_profile = dp.GraphProfiler("test_save_and_load")
        save_profile = save_profile.update(data)

        # Save and Load profile with Mock IO
        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_open(m)
            save_profile.save()

            mock_file.seek(0)
            load_profile = dp.GraphProfiler.load("mock.pkl")

        # Removed to avoid dict equality ambiguity
        save_profile.profile["continuous_distribution"]["weight"].pop("properties")
        load_profile.profile["continuous_distribution"]["weight"].pop("properties")

        # Check that reports are equivalent
        save_report = save_profile.report()
        load_report = load_profile.report()
        self.assertDictEqual(save_report, load_report)

        # adding new data and updating profiles
        self.graph.add_edges_from(
            [
                (2, 4, {"id": 6, "weight": 1.2}),
            ]
        )
        data = GraphData(input_file_path=None, data=self.graph)

        # validate both are still usable after
        save_profile.update(data)
        load_profile.update(data)


if __name__ == "__main__":
    unittest.main()
