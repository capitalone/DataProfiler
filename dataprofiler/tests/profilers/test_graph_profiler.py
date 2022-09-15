from __future__ import print_function

import os
import unittest
from cgi import test
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
                "weight": {
                    "name": "lognorm",
                    "scale": -15.250985118262854,
                    "properties": {
                        "best_fit_properties": [
                            8.646041719759628,
                            1.6999999999999997,
                            0.19403886939727638,
                        ],
                        "mean": [
                            8.646041719759628,
                            1.6999999999999997,
                            0.19403886939727638,
                        ],
                        "variance": [
                            8.646041719759628,
                            1.6999999999999997,
                            0.19403886939727638,
                        ],
                        "skew": [
                            0.6801760445870136,
                            1.5339299776947408,
                            4.540311271443057,
                        ],
                        "kurtosis": [
                            0.6939591774450526,
                            3.5294117647058827,
                            30.921639662389307,
                        ],
                    },
                },
            },
            categorical_distribution={
                "id": {
                    "bin_counts": [1, 1, 1, 2],
                    "bin_edges": [1.0, 2.0, 3.0, 4.0, 5.0],
                },
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

        cls.expected_props = dict()

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
                    "weight": [
                        {},
                        {
                            "name": ["lognorm", "uniform"],
                            "scale": -18.218734497180364,
                            "properties": [
                                {},
                                {
                                    "best_fit_properties": [
                                        [
                                            8.646041719759628,
                                            1.6999999999999997,
                                            0.19403886939727638,
                                        ],
                                        [],
                                        [1.7, 2.0999999999999996],
                                    ],
                                    "mean": [
                                        [
                                            8.646041719759628,
                                            1.6999999999999997,
                                            0.19403886939727638,
                                        ],
                                        [],
                                        [1.7, 2.0999999999999996],
                                    ],
                                    "variance": [
                                        [
                                            8.646041719759628,
                                            1.6999999999999997,
                                            0.19403886939727638,
                                        ],
                                        [],
                                        [1.7, 2.0999999999999996],
                                    ],
                                    "skew": [
                                        [0.6801760445870136, 4.540311271443057],
                                        [1.5339299776947408],
                                        [1.3801311186847085],
                                    ],
                                    "kurtosis": [
                                        [
                                            0.6939591774450526,
                                            3.5294117647058827,
                                            30.921639662389307,
                                        ],
                                        [],
                                        [3.5294117647058822, 2.8571428571428577],
                                    ],
                                },
                                {},
                            ],
                        },
                        {},
                    ]
                },
                {},
            ],
            "categorical_distribution": [
                {},
                {
                    "id": [
                        {},
                        {
                            "bin_counts": "unchanged",
                            "bin_edges": [[5.0], [1.0, 2.0, 3.0, 4.0], []],
                        },
                        {},
                    ]
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
                            "scale": 17.316888337101123,
                            "properties": [
                                {},
                                {
                                    "best_fit_properties": [
                                        [1.7, 2.0999999999999996],
                                        [],
                                        [
                                            0.24894970623537693,
                                            1.1999999999999997,
                                            11.973583222228775,
                                        ],
                                    ],
                                    "mean": [
                                        [1.7, 2.0999999999999996],
                                        [],
                                        [
                                            0.24894970623537693,
                                            1.1999999999999997,
                                            11.973583222228775,
                                        ],
                                    ],
                                    "variance": [
                                        [1.7, 2.0999999999999996],
                                        [],
                                        [
                                            0.24894970623537693,
                                            1.1999999999999997,
                                            11.973583222228775,
                                        ],
                                    ],
                                    "skew": [
                                        [1.5339299776947408, 1.3801311186847085],
                                        [],
                                        [
                                            4.008428917954561,
                                            1.8257418583505538,
                                            0.5779868092477709,
                                        ],
                                    ],
                                    "kurtosis": [
                                        [3.5294117647058822, 2.8571428571428577],
                                        [],
                                        [
                                            24.101253585441555,
                                            5.000000000000001,
                                            0.5011031274966287,
                                        ],
                                    ],
                                },
                                {},
                            ],
                        },
                        {},
                    ]
                },
                {
                    "value": {
                        "name": "uniform",
                        "scale": 8.047189562170502,
                        "properties": {
                            "best_fit_properties": [2.3, 5.0],
                            "mean": [2.3, 5.0],
                            "variance": [2.3, 5.0],
                            "skew": [1.3187609467915742, 0.8944271909999159],
                            "kurtosis": [2.608695652173913, 1.2],
                        },
                    }
                },
            ],
            "categorical_distribution": [
                {
                    "value": {
                        "bin_counts": [1, 1, 2],
                        "bin_edges": [4.0, 6.0, 8.0, 10.0],
                    }
                },
                {
                    "id": [
                        {},
                        {
                            "bin_counts": "unchanged",
                            "bin_edges": [[], [1.0, 2.0, 3.0, 4.0], [5.0]],
                        },
                        {},
                    ]
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

    def test_profile(self):
        # test_update
        graph_profile = GraphProfiler(self.graph_1)
        with utils.mock_timeit():
            profile = graph_profile.update(self.graph_1)
        self.assertDictEqual(self.expected_profile, profile.profile)

    def test_report(self):
        # test_report
        graph_profile = GraphProfiler(self.graph_1)
        with utils.mock_timeit():
            profile = graph_profile.update(self.graph_1)
        self.assertDictEqual(self.expected_profile, graph_profile.report())

    def test_graph_data_object(self):
        data = GraphData(input_file_path=None, data=self.graph_1)
        graph_profile = GraphProfiler("test_graph_data_object_update")

        with utils.mock_timeit():
            profile = graph_profile.update(data)
        self.assertDictEqual(self.expected_profile, profile.profile)

    def test_diff(self):
        data_1 = GraphData(input_file_path=None, data=self.graph_1)
        data_2 = GraphData(input_file_path=None, data=self.graph_2)

        profile_1 = dp.GraphProfiler(self.graph_1)
        profile_2 = dp.GraphProfiler(self.graph_2)
        profile_3 = dp.GraphProfiler(self.graph_3)

        with utils.mock_timeit():
            profile_1 = profile_1.update(self.graph_1)
            profile_2 = profile_2.update(self.graph_2)
            profile_3 = profile_3.update(self.graph_3)

        diff_1 = profile_1.diff(profile_2)
        self.assertDictEqual(diff_1, self.expected_diff_1)

        # Tests diffs between profiles with different # nodes and different continuous/categorical attributes
        diff_2 = profile_2.diff(profile_3)
        self.assertEqual(diff_2, self.expected_diff_2)

    def test_save_and_load(self):
        data = GraphData(input_file_path=None, data=self.graph_1)
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
        self.graph_1.add_edges_from(
            [
                (2, 4, {"id": 6, "weight": 1.2}),
            ]
        )
        data = GraphData(input_file_path=None, data=self.graph_1)

        # validate both are still usable after
        save_profile.update(data)
        load_profile.update(data)


if __name__ == "__main__":
    unittest.main()
