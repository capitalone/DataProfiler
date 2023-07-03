import os
import unittest
from collections import defaultdict
from io import BytesIO
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd

import dataprofiler as dp
from dataprofiler.data_readers.graph_data import GraphData
from dataprofiler.profilers.graph_profiler import GraphProfiler
from dataprofiler.profilers.profiler_options import ProfilerOptions

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
        cls.graph_1 = nx.Graph()
        cls.graph_1.add_nodes_from([1, 2, 3, 4])
        cls.graph_1.add_edges_from(
            [
                (1, 2, {"id": 1, "weight": 2.5}),
                (2, 3, {"id": 2, "weight": 1.7}),
                (3, 4, {"id": 3, "weight": 1.8}),
                (4, 1, {"id": 4, "weight": 4.1}),
                (1, 3, {"id": 5, "weight": 2.1}),
            ]
        )

        cls.graph_2 = nx.Graph()
        cls.graph_2.add_nodes_from([1, 2, 3, 4])
        cls.graph_2.add_edges_from(
            [
                (1, 2, {"id": 1, "weight": 3.8, "value": 10}),
                (2, 3, {"id": 2, "weight": 1.7, "value": 7}),
                (3, 4, {"id": 3, "weight": 2.9, "value": 9}),
                (3, 1, {"id": 4, "weight": 2.5, "value": 4}),
            ]
        )

        cls.graph_3 = nx.Graph()
        cls.graph_3.add_nodes_from([1, 2, 3, 4, 5])
        cls.graph_3.add_edges_from(
            [
                (1, 3, {"id": 1, "weight": 2.7, "value": 2.3}),
                (4, 3, {"id": 2, "weight": 1.2, "value": 6.8}),
                (1, 4, {"id": 3, "weight": 2.4, "value": 6.5}),
                (2, 1, {"id": 4, "weight": 5.8, "value": 7.3}),
                (2, 5, {"id": 5, "weight": 3.9, "value": 4.5}),
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
                "weight": {
                    "name": "lognorm",
                },
            },
            categorical_distribution={
                "id": {
                    "bin_counts": [1, 1, 1, 2],
                    "bin_edges": [1.0, 2.0, 3.0, 4.0, 5.0],
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

        cls.expected_properties = {
            "best_fit_properties": [
                8.646041719759628,
                1.6999999999999997,
                0.19403886939727638,
            ],
            "mean": [1.7085707836543698e16, 4.241852142820433, 1.0190038591415866],
            "variance": [8.521811094505713e64, 305.76588081569196, 0.03984103474264823],
            "skew": [4.987683961374356e48, 82.41830452500491, 0.5951548443693909],
            "kurtosis": [7.262126433044066e129, 117436.2896499293, 0.6363254662738349],
        }

        cls.expected_diff_1 = {
            "num_nodes": "unchanged",
            "num_edges": 1,
            "categorical_attributes": [[], ["id"], ["value"]],
            "continuous_attributes": "unchanged",
            "avg_node_degree": 0.5,
            "global_max_component_size": 1,
            "continuous_distribution": [
                {},
                {
                    "id": "unchanged",
                    "weight": [
                        {},
                        {
                            "name": ["lognorm", "uniform"],
                        },
                        {},
                    ],
                },
                {"value": None},
            ],
            "categorical_distribution": [
                {},
                {
                    "id": [
                        {},
                        {
                            "bin_counts": [[1], [1, 1, 2], []],
                            "bin_edges": [[5.0], [1.0, 2.0, 3.0, 4.0], []],
                        },
                        {},
                    ],
                    "weight": "unchanged",
                },
                {
                    "value": {
                        "bin_counts": [1, 1, 2],
                        "bin_edges": [4.0, 6.0, 8.0, 10.0],
                    }
                },
            ],
            "times": {
                "num_nodes": "unchanged",
                "num_edges": "unchanged",
                "categorical_attributes": "unchanged",
                "continuous_attributes": "unchanged",
                "avg_node_degree": "unchanged",
                "global_max_component_size": "unchanged",
                "continuous_distribution": "unchanged",
                "categorical_distribution": "unchanged",
            },
        }
        cls.expected_diff_2 = {
            "num_nodes": -1,
            "num_edges": -1,
            "categorical_attributes": [["value"], ["id"], []],
            "continuous_attributes": [[], ["weight"], ["value"]],
            "avg_node_degree": "unchanged",
            "global_max_component_size": -1,
            "continuous_distribution": [
                {},
                {
                    "weight": [
                        {},
                        {
                            "name": ["uniform", "gamma"],
                        },
                        {},
                    ],
                    "id": "unchanged",
                    "value": [
                        None,
                        {
                            "name": "uniform",
                        },
                    ],
                },
                {},
            ],
            "categorical_distribution": [
                {},
                {
                    "weight": "unchanged",
                    "id": [
                        {},
                        {
                            "bin_counts": [[], [1, 1, 2], [1]],
                            "bin_edges": [[], [1.0, 2.0, 3.0, 4.0], [5.0]],
                        },
                        {},
                    ],
                    "value": [
                        {"bin_counts": [1, 1, 2], "bin_edges": [4.0, 6.0, 8.0, 10.0]},
                        None,
                    ],
                },
                {},
            ],
            "times": {
                "num_nodes": "unchanged",
                "num_edges": "unchanged",
                "categorical_attributes": "unchanged",
                "continuous_attributes": "unchanged",
                "avg_node_degree": "unchanged",
                "global_max_component_size": "unchanged",
                "continuous_distribution": "unchanged",
                "categorical_distribution": "unchanged",
            },
        }

    def check_continuous_properties(self, continuous_distribution_props):
        """
        NOTE: this function is needed because github tests often lead result in
        slightly different property values. Hence why assertAlmostEqual is used.

        """
        for key in continuous_distribution_props:
            for x, y in zip(
                self.expected_properties[key], continuous_distribution_props[key]
            ):
                self.assertAlmostEqual(x, y)

    def test_add(self):
        profile_1 = GraphProfiler(self.graph_1)
        profile_2 = GraphProfiler(self.graph_2)

        with self.assertRaises(
            NotImplementedError,
            msg="profile adding is not currently supported for the GraphProfiler",
        ):
            profile_1 + profile_2

    def test_profile(self):
        # test_update
        graph_profile = GraphProfiler(self.graph_1)
        with utils.mock_timeit():
            profile = graph_profile.update(self.graph_1)

        # check that scale is almost equal
        scale = profile.profile["continuous_distribution"]["weight"].pop("scale")
        self.assertAlmostEqual(scale, -15.250985118262854)

        # check that properties are almost equal
        properties = profile.profile["continuous_distribution"]["weight"].pop(
            "properties"
        )
        self.check_continuous_properties(properties)

        self.assertDictEqual(self.expected_profile, profile.profile)

    def test_report(self):
        # test_report
        profile = GraphProfiler(self.graph_1)
        with utils.mock_timeit():
            profile.update(self.graph_1)

        # check that scale is almost equal
        scale = profile.profile["continuous_distribution"]["weight"].pop("scale")
        self.assertAlmostEqual(scale, -15.250985118262854)

        # check that properties are almost equal
        properties = profile.profile["continuous_distribution"]["weight"].pop(
            "properties"
        )
        self.check_continuous_properties(properties)

        self.assertDictEqual(self.expected_profile, profile.report())

    def test_graph_data_object(self):
        data = GraphData(data=self.graph_1)
        graph_profile = GraphProfiler("test_graph_data_object_update")

        with utils.mock_timeit():
            profile = graph_profile.update(data)

        # check that scale is almost equal
        scale = profile.profile["continuous_distribution"]["weight"].pop("scale")
        self.assertAlmostEqual(scale, -15.250985118262854)

        # check that properties are almost equal
        properties = profile.profile["continuous_distribution"]["weight"].pop(
            "properties"
        )
        self.check_continuous_properties(properties)

        self.assertDictEqual(self.expected_profile, profile.profile)

    def test_diff(self):
        profile_1 = dp.GraphProfiler(self.graph_1)
        profile_2 = dp.GraphProfiler(self.graph_2)
        profile_3 = dp.GraphProfiler(self.graph_3)

        with utils.mock_timeit():
            profile_1 = profile_1.update(self.graph_1)
            profile_2 = profile_2.update(self.graph_2)
            profile_3 = profile_3.update(self.graph_3)

        # Remove scale because it causes rounding issues during the test
        profile_1.profile["continuous_distribution"]["weight"].pop("scale")
        profile_1.profile["continuous_distribution"]["weight"].pop("properties")
        profile_2.profile["continuous_distribution"]["weight"].pop("scale")
        profile_2.profile["continuous_distribution"]["weight"].pop("properties")
        profile_3.profile["continuous_distribution"]["weight"].pop("scale")
        profile_3.profile["continuous_distribution"]["weight"].pop("properties")
        profile_3.profile["continuous_distribution"]["value"].pop("scale")
        profile_3.profile["continuous_distribution"]["value"].pop("properties")

        diff_1 = profile_1.diff(profile_2)
        self.assertDictEqual(diff_1, self.expected_diff_1)

        # Tests diffs between profiles with different # nodes and different continuous/categorical attributes
        diff_2 = profile_2.diff(profile_3)
        self.assertDictEqual(diff_2, self.expected_diff_2)

    def test_save_and_load(self):
        data = GraphData(data=self.graph_1)
        # test_save_and_load
        save_profile = dp.GraphProfiler(self.graph_1)
        save_profile = save_profile.update(data)

        # Save and Load profile with Mock IO
        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_open(m)
            save_profile.save()

            mock_file.seek(0)
            load_profile = dp.GraphProfiler.load("mock.pkl")

        # Check that reports are equivalent
        save_report = save_profile.report()
        load_report = load_profile.report()
        self.assertDictEqual(save_report, load_report)

        # adding new data and updating profiles
        data = GraphData(data=self.graph_1)

        # validate both are still usable after
        save_profile.update(data)
        load_profile.update(data)


if __name__ == "__main__":
    unittest.main()
