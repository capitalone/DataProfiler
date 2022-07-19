from __future__ import print_function
from cgi import test

import os
import profile
import unittest

import networkx as nx

from dataprofiler.profilers.graph_profiler import GraphProfile

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestGraphProfiler(unittest.TestCase):

    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([1,2,3,4])
        self.graph.add_edges_from([
            (1,2, {"id":1, "weight":2.5}),
            (2,3, {"id":2, "weight":1.7}),
            (3,4, {"id":3, "weight":1.8}),
            (4,1, {"id":4, "weight":4.1}),
            (1,3, {"id":4, "weight":2.1})
        ])
        self.graph_profile = GraphProfile("graph", self.graph)

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
        self.assertEqual(4, self.graph_profile._compute_num_nodes())

    def test_num_edges(self):
        self.assertEqual(5, self.graph_profile._compute_num_edges())

    def test_find_all_attributes(self):
        attribute_list = ['weight', 'id']
        test_list = self.graph_profile._find_all_attributes()
        is_list_same = True
        for attribute in attribute_list:
            if attribute not in test_list:
                is_list_same = False
        self.assertTrue(is_list_same)

    def test_find_attribute_categories(self):
        self.assertEqual((["id"], ["weight"]), self.graph_profile._find_categorical_and_continuous_attributes())
    
    def test_attribute_to_list(self):
        data_list = [4.1, 2.5, 2.1, 1.7, 1.8]
        data_list_test = self.graph_profile._attribute_data_as_list('weight')
        is_equal = True
        for data in data_list_test:
            if data not in data_list:
                is_equal = False
        self.assertTrue(is_equal)

    def test_avg_node_degree(self):
        avg_node_degree_test = self.graph_profile._compute_avg_node_degree()
        self.assertEqual(2.5, avg_node_degree_test)
    
    def test_global_max_component_size(self):
        self.assertEqual(5, self.graph_profile._compute_global_max_component_size())
    
    # WIP
    def test_continuous_distribution(self):
        self.assertTrue(True)
    
    # WIP
    def test_categorical_distribution(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()