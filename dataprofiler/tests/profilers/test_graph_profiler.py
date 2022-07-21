from __future__ import print_function

import os
import unittest
from cgi import test

import networkx as nx

from dataprofiler.profilers.graph_profiler import GraphProfile

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestGraphProfiler(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([1, 2, 3, 4])
        self.graph.add_edges_from(
            [
                (1, 2, {"id": 1, "weight": 2.5}),
                (2, 3, {"id": 2, "weight": 1.7}),
                (3, 4, {"id": 3, "weight": 1.8}),
                (4, 1, {"id": 4, "weight": 4.1}),
                (1, 3, {"id": 4, "weight": 2.1}),
            ]
        )
        self.graph_profile = GraphProfile("graph")

    @classmethod
    def setUpClass(cls):
        super(TestGraphProfiler, cls).setUpClass()

    # WIP
    def test_init(self):
        self.assertTrue(True, True)

    # WIP
    def test_diff(self):
        self.assertTrue(True, True)

    # WIP
    def test_report(self):
        self.assertTrue(True, True)

    def test_num_nodes(self):
        self.assertEqual(4, GraphProfile._get_num_nodes(self.graph))

    def test_num_edges(self):
        self.assertEqual(5, GraphProfile._get_num_edges(self.graph))

    def test_find_all_attributes(self):
        attribute_list = ["weight", "id"]
        test_list = self.graph_profile._find_all_attributes(self.graph)
        is_list_same = True
        for attribute in attribute_list:
            if attribute not in test_list:
                is_list_same = False
        self.assertTrue(is_list_same)

    def test_find_attribute_categories(self):
        self.assertEqual(
            (["id"], ["weight"]),
            self.graph_profile._get_categorical_and_continuous_attributes(self.graph),
        )

    def test_attribute_to_list(self):
        data_list = [4.1, 2.5, 2.1, 1.7, 1.8]
        data_list_test = self.graph_profile._attribute_data_as_list(
            self.graph, "weight"
        )
        is_equal = True
        for data in data_list_test:
            if data not in data_list:
                is_equal = False
        self.assertTrue(is_equal)

    def test_avg_node_degree(self):
        avg_node_degree_test = GraphProfile._get_avg_node_degree(self.graph)
        self.assertEqual(2.5, avg_node_degree_test)

    def test_global_max_component_size(self):
        self.assertEqual(5, GraphProfile._get_global_max_component_size(self.graph))

    # WIP
    def test_continuous_distribution(self):
        self.assertTrue(True)

    # WIP
    def test_categorical_distribution(self):
        self.assertTrue(True)

    """
    Update properties tests
    """

    def test_update_num_nodes(self):
        graph_profiler = GraphProfile("update_test_graph")
        graph_profiler._num_nodes = graph_profiler._update_num_nodes(self.graph)
        self.assertEqual(4, graph_profiler._num_nodes)

    def test_update_num_edges(self):
        graph_profiler = GraphProfile("update_test_graph")
        graph_profiler._num_edges = graph_profiler._update_num_edges(self.graph)
        self.assertEqual(5, graph_profiler._num_edges)

    def test_update_avg_node_degree(self):
        graph_profiler = GraphProfile("update_test_graph")
        graph_profiler._avg_node_degree = graph_profiler._update_avg_node_degree(
            self.graph
        )
        self.assertEqual(2.5, graph_profiler._avg_node_degree)

    def test_update_global_max_component_size(self):
        graph_profiler = GraphProfile("update_test_graph")
        graph_profiler._global_max_component_size = (
            graph_profiler._update_global_max_component_size(self.graph)
        )
        self.assertEqual(5, graph_profiler._global_max_component_size)

    def test_update_categorical_attributes(self):
        graph_profiler = GraphProfile("update_test_graph")
        graph_profiler._categorical_attributes = (
            graph_profiler._update_categorical_attributes(self.graph)
        )
        self.assertEqual(["id"], graph_profiler._categorical_attributes)

    def test_update_continuous_attributes(self):
        graph_profiler = GraphProfile("update_test_graph")
        graph_profiler._continuous_attributes = (
            graph_profiler._update_continuous_attributes(self.graph)
        )
        self.assertEqual(["weight"], graph_profiler._continuous_attributes)

    # WIP
    def test_update_categorical_distribution(self):
        self.assertTrue(True)

    # WIP
    def test_update_continuous_distribution(self):
        graph_profiler = GraphProfile("update_test_graph")
        graph_profiler._update_num_nodes(self.graph)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
