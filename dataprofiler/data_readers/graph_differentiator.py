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
        
    def __init__(self, input_file_path=None, data=None, options=None):
        options = self._check_and_return_options(options)
        BaseData.__init__(self, input_file_path, data, options)        
        SpreadSheetDataMixin.__init__(self, input_file_path, data, options)
        self.SAMPLES_PER_LINE_DEFAULT = options.get("record_samples_per_line", 1)
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

    def file_subset(file, number_lines, format):
        '''
        Returns a subset of a file
        '''

        # check for input errors
        # check correct handled format
        if not (format.__eq__("csv") or format.__eq__("json")):
            raise ValueError("input file has to be csv or json")

        # operations for input csv
        if format.__eq__("csv"):
            with open('subset.csv', 'w') as subset:
                csvwriter = csv.writer(subset, delimiter=",")
                with open(file, 'r') as fp:
                    csvreader = csv.reader(fp)
                    line_number = 0
                    
                    for row in csvreader:
                        if line_number > number_lines:
                            break
                        csvwriter.writerow(row)
                        line_number += 1
            return subset

        if format.__eq__("json"):
            # create subset for json
            return

    def is_match(file, options):
        '''
        Determines whether the file is a graph
        Current formats checked:
            - adjacency list
            - edge list
        User is able to specify particular format to check through options parameter
        '''
        graph = True
        file_handle = open(file, "rb")

        # need to take subset first (later implementation)
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
            except nx.NetworkXUnfeasible:
                graph = False
        return graph
    
