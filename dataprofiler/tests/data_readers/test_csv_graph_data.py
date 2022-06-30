import os
import unittest
from io import BytesIO, StringIO, TextIOWrapper

import pandas as pd

from dataprofiler.data_readers.data import CSVData, Data
from dataprofiler.data_readers.data_utils import is_stream_buffer
from dataprofiler.data_readers import graph_differentiator

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestGraphDifferentiatorClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'csv/graph-differentiator-input-standard.csv'),
                 count=7, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/graph-differentiator-input-positive.csv'),
                 count=7, delimiter=',', has_header=[0],
                 num_columns=11, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/guns.csv'),
                 count=100799, delimiter=',', has_header=[None],
                 num_columns=10, encoding='utf-8'),    
            dict(path=os.path.join(test_dir, 'csv/graph-differentiator-input-subset-standard-1.csv'),
                 count=4, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/graph-differentiator-input-subset-standard-2.csv'),
                 count=6, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
        ]

        cls.buffer_list = []
        for input_file in cls.input_file_names:
            # add StringIO
            buffer_info = input_file.copy()
            with open(input_file['path'], 'r', encoding=input_file['encoding']) as fp:
                buffer_info['path'] = StringIO(fp.read())
            cls.buffer_list.append(buffer_info)
            
            # add BytesIO
            buffer_info = input_file.copy()
            with open(input_file['path'], 'rb') as fp:
                buffer_info['path'] = BytesIO(fp.read())
            cls.buffer_list.append(buffer_info)

        cls.file_or_buf_list = cls.input_file_names + cls.buffer_list

        cls.output_file_path = None
    
    @classmethod
    def setUp(cls):
        for buffer in cls.buffer_list:
            buffer['path'].seek(0)

    # test taking subsets of csv files
    def test_file_subset_1(self):
        """
        Determine if the the subset function works as intended (test 1)
        """
        input_file = self.input_file_names[0]['path']
        check_file = self.input_file_names[3]['path']

        subset = graph_differentiator.GraphDifferentiator.file_subset(input_file, 4, "csv")
        self.assertEqual(subset, check_file)

    def test_file_subset_2(self):
        """
        Determine if the the subset function works as intended (test 2)
        """
        input_file = self.input_file_names[0]['path']
        check_file = self.input_file_names[4]['path']

        subset = graph_differentiator.GraphDifferentiator.file_subset(input_file, 6, "csv")
        self.assertEqual(subset, check_file)

    # test is_match for true output w/ different options
    def test_is_graph_positive_1(self):
        """
        Determine if the input CSV file can automatically be recognized as being a graph
        """
        input_file = self.input_file_names[1]['path']
        options = '{"format":"none"}'
        test_graph = graph_differentiator.GraphDifferentiator.is_match(input_file, options)
        self.assertTrue(test_graph)

    def test_is_graph_positive_2(self):
        """
        Determine if the input CSV file can automatically be recognized as being a graph w/ adjacency list option selected
        """
        input_file = self.input_file_names[1]['path']
        options = '{"format":"adjacency_list"}'
        test_graph = graph_differentiator.GraphDifferentiator.is_match(input_file, options)
        self.assertTrue(test_graph)

    def test_is_graph_positive_3(self):
        """
        Determine if the input CSV file can automatically be recognized as being a graph w/ edge list option selected
        """
        input_file = self.input_file_names[1]['path']
        options = '{"format":"edge_list"}'
        test_graph = graph_differentiator.GraphDifferentiator.is_match(input_file, options)
        self.assertTrue(test_graph)

    # test is_match for false output w/ different options
    def test_is_graph_negative_1(self):
        """
        Determine if the input CSV file can be automatically recognized as not being a graph w/ no options selected
        """
        input_file = self.input_file_names[2]['path']
        options = '{"format":"none"}'
        test_graph = graph_differentiator.GraphDifferentiator.is_match(input_file, options)
        self.assertFalse(test_graph)

    def test_is_graph_negative_2(self):
        """
        Determine if the input CSV file can be automatically recognized as not being a graph w/ adjacency list option selected
        """
        input_file = self.input_file_names[2]['path']
        options = '{"format":"adjacency_list"}'
        test_graph = graph_differentiator.GraphDifferentiator.is_match(input_file, options)
        self.assertFalse(test_graph)

    def test_is_graph_negative_3(self):
        """
        Determine if the input CSV file can be automatically recognized as not being a graph w/ edge list option selected
        """
        input_file = self.input_file_names[2]['path']
        options = '{"format":"edge_list"}'
        test_graph = graph_differentiator.GraphDifferentiator.is_match(input_file, options)
        self.assertFalse(test_graph)

if __name__ == '__main__':
    unittest.main()
