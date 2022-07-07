import csv

from .base_data import BaseData
from .csv_data import CSVData

class GraphData(BaseData):
        
    def __init__(self, input_file_path=None, data=None, options=None):
        BaseData.__init__(self, input_file_path, data, options)        

    def _find_target_string_in_column(self, column_names, keyword_list):
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

    def csv_column_names(self, file_path, options):
        '''
        fetches a list of column names from the csv file
        '''

        column_names = []

        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = options["delimiter"])
            
            # fetch only column names
            for row in csv_reader:
                column_names.append(row)
                break
        
        column_names = column_names[0]

        # replace all whitespaces in the column names
        for index in range(0, len(column_names)):
            column_names[index] = column_names[index].replace(" ", "")

        return column_names
    
    def is_match(self, file_path, options):
        '''
        Determines whether the file is a graph
        Current formats checked:
            - attributed edge list

        This works by finding whether the file contains a target and a source node
        '''

        if options is None:
            options = dict()
        if 'delimiter' not in options:
            options["delimiter"] = ','
        if 'header' not in options:
            options["header"] = True

        if options["header"] is False or CSVData.is_match(file_path, options):
            return False

        graph = False

        column_names = self.csv_column_names(file_path, options)

        target_keywords = ['target', 'destination', 'dst']
        source_keywords = ['source', 'src', 'origin']

        has_target = self._find_target_string_in_column(column_names, target_keywords)
        has_source = self._find_target_string_in_column(column_names, source_keywords)

        if has_target and has_source:
            graph = True

        return graph
