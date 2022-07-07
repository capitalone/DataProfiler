import os
import unittest
from io import BytesIO, StringIO, TextIOWrapper

from dataprofiler.data_readers.data_utils import is_stream_buffer
from dataprofiler.data_readers.graph_data import GraphData

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestGraphDifferentiatorClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'csv/graph-differentiator-input-positive.csv'),
                 count=7, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/graph-differentiator-input-standard-positive.csv'),
                 count=7, delimiter=',', has_header=[0],
                 num_columns=11, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/graph-differentiator-input-negative.csv'),
                 count=4, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8')
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

    # test find_target_string_in_column
    def test_finding_string_in_column_positive_1(self):
        '''
        Determine whether keywords can be detected with underscore before
        '''
        column_names = ['node_src', 'node_dst', 'attribute1']
        keyword_list = ['src', 'source']
        self.assertTrue(GraphData().find_target_string_in_column(column_names, keyword_list))

    def test_finding_string_in_column_positive_2(self):
        '''
        Determine whether keywords can be detected with underscore after
        '''
        column_names = ['src_node', 'dst_node', 'attribute1']
        keyword_list = ['src', 'source']
        self.assertTrue(GraphData().find_target_string_in_column(column_names, keyword_list))

    def test_finding_string_in_column_negative(self):
        '''
        Determine whether the output is false when keywords are not found
        '''
        column_names = ['movie', 'audience_type', 'audience_source']
        keyword_list = ['dst', 'destination', 'target']
        self.assertFalse(GraphData().find_target_string_in_column(column_names, keyword_list))

    def test_finding_string_in_column_negative(self):
            '''
            Determine whether the output is false when keywords is present but a substring of a word without [_, ., -] delimiters
            '''
            column_names = ['flight_number', 'destination', 'price']
            keyword_list = ['dst', 'destination', 'target']
            self.assertFalse(GraphData().find_target_string_in_column(column_names, keyword_list))

    #test csv_column_name
    def test_csv_column_names(self):
        '''
        Determine if column names are fetched correctly and in the right format
        '''

        column_names = ['node_id_dst', 'node_id_src', 'attrib_id', 'attrib_type', 'edge_date', 'open_date_src', 'open_date_dst']
        input_file = self.input_file_names[1]['path']
        self.assertEqual(GraphData().csv_column_names(input_file, ','), column_names)
        
    # test is_match for true output w/ different options
    def test_is_graph_positive_1(self):
        """
        Determine if the input CSV file can automatically be recognized as being a graph
        """
        input_file = self.input_file_names[0]['path']
        test_graph = GraphData().is_match(input_file, ',')
        self.assertTrue(test_graph)

    def test_is_graph_positive_2(self):
        """
        Determine if the input CSV file can automatically be recognized as being a graph w/ adjacency list option selected
        """
        input_file = self.input_file_names[1]['path']
        test_graph = GraphData().is_match(input_file, ',')
        self.assertTrue(test_graph)

    # test is_match for false output w/ different options
    def test_is_graph_negative_1(self):
        """
        Determine if the input CSV file can be automatically recognized as not being a graph w/ no options selected
        """
        input_file = self.input_file_names[2]['path']
        test_graph = GraphData().is_match(input_file, ',')
        self.assertFalse(test_graph)

if __name__ == '__main__':
    unittest.main()
