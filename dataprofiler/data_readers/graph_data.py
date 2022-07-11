import csv

import networkx as nx
from numpy import source

from .base_data import BaseData
from .csv_data import CSVData
from .filepath_or_buffer import FileOrBufferHandler

class GraphData(BaseData):
        
    def __init__(self, input_file_path=None, data=None, options=None):
        BaseData.__init__(self, input_file_path, data, options)
        if options is None:
            options = dict()

        return self._load_data()


    @classmethod
    def _find_target_string_in_column(self, column_names, keyword_list):
        '''
        Find whether one of the columns names contains a keyword that could refer to a target node column
        '''
        column_name_symbols = ['_', '.', '-']
        has_target = False
        target_index = -1
        
        # iterate through columns, keywords, and delimiter name symbols to see if any permutation is contained in column names
        for column in range(0, len(column_names)):
            for keyword in keyword_list:
                for symbol in column_name_symbols:
                    
                    append_start_word = symbol + keyword
                    append_end_word = keyword + symbol

                    if append_start_word in column_names[column] or append_end_word in column_names[column]:
                        target_index = column
                        has_target = True
                        break
            if has_target:
                break
        
        return target_index


    @classmethod
    def csv_column_names(cls, file_path, options):
        '''
        fetches a list of column names from the csv file
        '''
        column_names = []

        with FileOrBufferHandler(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = options.get("delimiter", ","))
            
            # fetch only column names
            for row in csv_reader:
                column_names.append(row)
                break
        column_names = column_names[0]

        # replace all whitespaces in the column names
        for index in range(0, len(column_names)):
            column_names[index] = column_names[index].replace(" ", "")

        return column_names


    @classmethod
    def is_match(cls, file_path, options=None):
        '''
        Determines whether the file is a graph
        Current formats checked:
            - attributed edge list

        This works by finding whether the file contains a target and a source node
        '''
        if options is None:
            options = dict()
        if not CSVData.is_match(file_path, options):
            return False
        column_names = cls.csv_column_names(file_path, options)
        source_keywords = ['source', 'src', 'origin']
        target_keywords = ['target', 'destination', 'dst']
        source_index = cls._find_target_string_in_column(column_names, source_keywords)
        destination_index = cls._find_target_string_in_column(column_names, target_keywords)
        has_source = True if source_index >= 0 else False
        has_target = True if destination_index >= 0 else False

        if has_target and has_source:
            options.update(source_node = source_index)
            options.update(destination_node = destination_index)
            options.update(destination_list = target_keywords)
            options.update(source_list = source_keywords)
            options.update(column_name = column_names)
            return True

        return False

    def _format_data_networkx(self):
        '''
        Formats the input file into a networkX graph
        '''
        source_keywords = ['source', 'src', 'origin']
        target_keywords = ['target', 'destination', 'dst']

        if self.input_file_path:
            column_names = self.csv_column_names(self.input_file_path, self.options)

            if self.options.get("source_node") is None or self.options.get("destination_node") is None:
                source_node = self._find_target_string_in_column(column_names, source_keywords)
                destination_node = self._find_target_string_in_column(column_names, target_keywords)
            else:
                source_node = self.options.get("source_node")
                destination_node = self.options.get("destination_node")

            # read lines from csv
            csv_as_list = []
            with FileOrBufferHandler(self.input_file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter = self.options.get("delimiter", self.options.get("delimiter", ",")))
                for row in csv_reader:
                    row = [item.replace(" ", "") for item in row]
                    csv_as_list.append(row)
            
            # grab list of edges from source/dest nodes
            list_edges = []
            list_nodes = set()

            for line in range(1, len(csv_as_list)):
                # fetch attributes in columns
                attributes = dict()
                for column in range(0, len(csv_as_list[0])):

                    if csv_as_list[line][column] is None:
                        continue
                    if column is not source_node or column is not destination_node:
                        attributes[column_names[column]] = (csv_as_list[line][column])
                        list_edges.append([csv_as_list[line][destination_node], csv_as_list[line][source_node], attributes])
                    if column is source_node or column is destination_node:
                        list_nodes.add(csv_as_list[line][column])

            # get NetworkX object from list
            networkx_graph = nx.DiGraph()
            networkx_graph.add_nodes_from([*list_nodes, ])
            networkx_graph.add_edges_from(list_edges)
            return networkx_graph
        else:
            raise ValueError("No data to load")
            
    def _load_data(self, data=None):
        '''
        Loads the input data into memory from __init__
        '''
        if self.input_file_path:
            self._data = self._format_data_networkx()
        else:
            raise ValueError("No data to load.")

