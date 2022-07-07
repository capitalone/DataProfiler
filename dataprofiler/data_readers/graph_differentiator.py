import csv
import random
import re
from collections import Counter
from io import BytesIO
from readline import add_history
from dataprofiler.data_readers.csv_data import CSVData
from six import StringIO
import os

#from . import data_utils
#from .avro_data import AVROData
#from .json_data import JSONData
#from .parquet_data import ParquetData
from .base_data import BaseData
#from .structured_mixins import SpreadSheetDataMixin

class GraphDifferentiator():
        
    def __init__(self, input_file_path=None, data=None, options=None):
        BaseData.__init__(self, input_file_path, data, options)        
        #SpreadSheetDataMixin.__init__(self, input_file_path, data, options)
        #self.SAMPLES_PER_LINE_DEFAULT = options.get("record_samples_per_line", 1)
        #self._selected_data_format = options.get("data_format", "dataframe")
        #self._delimiter = options.get("delimiter", None)
        #self._quotechar = options.get("quotechar", None)
        #self._selected_columns = options.get("selected_columns", list())
        #self._header = options.get("header", 'auto')
        #self._checked_header = "header" in options and self._header != 'auto'
        #self._default_delimiter = ','
        #self._default_quotechar = '"'

        #if data is not None:
        #    self._load_data(data)
        #    if not self._delimiter:
        #        self._delimiter = self._default_delimiter
        #    if not self._quotechar:
        #        self._quotechar = self._default_quotechar

    def find_target_string_in_column(self, column_names, keyword_list):
        '''
        Find whether one of the columns names contains a keyword that could refer to a target node column
        '''

        column_name_symbols = ['_', '.', '-']
        has_target = False
        
        # iterate through columns, keywords, and delimiter name symbols to see if any permutation is contained in column names
        for column in column_names:
            for keyword in keyword_list:
                for symbol in column_name_symbols:
                    
                    append_start_word = symbol + keyword
                    append_end_word = keyword + symbol

                    if append_start_word in column or append_end_word in column:
                        has_target = True
                        break
            if has_target:
                break
        
        return has_target

    def csv_column_names(self, file, input_delimiter):
        '''
        fetches a list of column names from the csv file
        '''
        column_names = []

        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = input_delimiter)
            
            # fetch only column names
            for row in csv_reader:
                column_names.append(row)
                break
        
        column_names = column_names[0]

        # replace all whitespaces in the column names
        for index in range(0, len(column_names)):
            column_names[index] = column_names[index].replace(" ", "")

        return column_names
    
    def is_match(self, file, input_delimiter):
        '''
        Determines whether the file is a graph
        Current formats checked:
            - attributed edge list

        This works by finding whether the file contains a target and a source node
        '''

        graph = False
        column_names = self.csv_column_names(file, input_delimiter)

        target_keywords = ['target', 'destination', 'dst']
        source_keywords = ['source', 'src', 'origin']

        has_target = self.find_target_string_in_column(column_names, target_keywords)
        has_source = self.find_target_string_in_column(column_names, source_keywords)

        if has_target and has_source:
            graph = True

        return graph
