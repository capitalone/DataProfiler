import os
import unittest
from io import BytesIO, StringIO, TextIOWrapper

import pandas as pd

from dataprofiler.data_readers.data import CSVData, Data
from dataprofiler.data_readers.data_utils import is_stream_buffer

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

    def test_graph_converter_positive(self):
        """
        Determine if the graph converter works at converting a graph file to a standardized graph format
        """
        input_file = self.input_file_names[1]['path']
        standard_file = self.input_file_names[0]['path']
        self.assertEqual(CSVData.convert_graph(input_file), standard_file)

    def test_is_graph_negative(self):
        """
        Determine if the input CSV file can be automatically recognized as not being a graph
        """
        input_file = self.input_file_names[2]['path']
        test_graph = input_file.is_graph()
        self.assertFalse(test_graph)

    def test_is_graph_positive(self):
        """
        Determine if the input CSV file can automatically be recognized as being a graph
        """
        input_file = self.input_file_names[1]['path']
        test_graph = input_file.is_graph()
        self.assertFalse(test_graph)



if __name__ == '__main__':
    unittest.main()
