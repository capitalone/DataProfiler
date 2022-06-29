import csv
import random
import re
from collections import Counter
from io import BytesIO
from dataprofiler.data_readers.csv_data import CSVData

from six import StringIO

from . import data_utils
from .avro_data import AVROData
from .base_data import BaseData
from .json_data import JSONData
from .parquet_data import ParquetData
from .structured_mixins import SpreadSheetDataMixin
import networkx as nx


class GraphDifferentiator():
    
    data_type = 'csv'
    
    def __init__(self, input_file_path=None, data=None, options=None):
        
        options = self._check_and_return_options(options)
        BaseData.__init__(self, input_file_path, data, options)
        SpreadSheetDataMixin.__init__(self, input_file_path, data, options)

        self._data_formats["records"] = self._get_data_as_records
        self.SAMPLES_PER_LINE_DEFAULT = options.get("record_samples_per_line",
                                                    1)
        self._selected_data_format = options.get("data_format", "dataframe")
        self._delimiter = options.get("delimiter", None)
        self._quotechar = options.get("quotechar", None)
        self._selected_columns = options.get("selected_columns", list())
        self._header = options.get("header", 'auto')
        self._checked_header = "header" in options and self._header != 'auto'
        self._default_delimiter = ','
        self._default_quotechar = '"'

        if data is not None:
            self._load_data(data)
            if not self._delimiter:
                self._delimiter = self._default_delimiter
            if not self._quotechar:
                self._quotechar = self._default_quotechar

    def graph_subset(file):

        return

    def is_match(file, options):
        '''
        Determines whether the file is a graph
        User is able to specify particular format to check through options parameter
        Current formats checked:
            - adjacency list
            - edge list
        '''
        graph = True
        file_handle = open(file, "rb")

        if options.format.__eq__("adjacency_list"):
            try:
                adjacency_list = nx.read_adjlist(file_handle)
            except nx.NetworkXAlgorithmError:
                graph = False
            except nx.NetworkXUnfeasible:
                graph = False
        if options.format.__eq__("edge_list"):
            try:
                edge_list = nx.read_edgelist(file_handle)
            except nx.NetworkXAlgorithmError:
                graph = False
            except nx.NetworkXUnfeasible:
                graph = False
        else:
            try:
                adjacency_list = nx.read_adjlist(file_handle)
                adjacency_list = nx.read_edgelist(file_handle)
            except nx.NetworkXAlgorithmError:
                graph = False
                return
            except nx.NetworkXUnfeasible:
                graph = False
                return
        return graph

    @classmethod
    def convert_graph(file, format):
        '''
        Allows the user to input a desired format
        '''
        file_handle = open(file, "rb")

        if is_graph(file):
            if format.__eq__("adj_list"):
                return nx.read_adjlist(file_handle)
            if format.__eq__("edge_list"):
                return nx.read_edgelist(file_handle)
            else:
                raise ValueError("Need to specify desired format")
    
