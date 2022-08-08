"""Contains class for identifying, reading, and loading graph data."""
import csv

import networkx as nx

from . import data_utils
from .base_data import BaseData
from .csv_data import CSVData
from .filepath_or_buffer import FileOrBufferHandler


class GraphData(BaseData):
    """GraphData class to identify, read, and load graph data."""

    data_type = "graph"

    def __init__(self, input_file_path=None, data=None, options=None):
        """
        Initialize Data class for identifying, reading, and loading graph data.

        Current implementation only accepts file path as input.
        An options parameter is also passed in to specify properties of the
        input file.

        Possible Options::

            options = dict(
                delimiter= type: str
                column_names= type: list(str)
                source_node= type: int
                destination_node= type: int
                target_keywords= type: list(str)
                source_keywords= type: list(str)
                header= type: any
                quotechar= type: str
            )

        delimiter: delimiter used to decipher the csv input file
        column_names: list of column names of the csv
        source_node: index of the source node column, range of (0,n-1)
        target_node: index of the target node column, range of (0,n-1)
        target_keywords: list of keywords to identify target/destination node col
        source_keywords: list of keywords to identify source node col
        header: location o the header in the file
        quotechar: quote character used in the delimited file

        :param input_file_path: path to the file being loaded or None
        :type input_file_path: str
        :param data: data being loaded into the class instead of an input file
        :type data: multiple types
        :param options: options pertaining to the data type
        :type options: dict
        :return: None
        """
        options = self._check_and_return_options(options)
        BaseData.__init__(self, input_file_path, data, options)

        self._source_node = options.get("source_node", None)
        self._destination_node = options.get("destination_node", None)
        self._target_keywords = options.get(
            "target_keywords", ["target", "destination", "dst"]
        )
        self._source_keywords = options.get(
            "source_keywords", ["source", "src", "origin"]
        )
        self._column_names = options.get(
            "column_names", self.csv_column_names(self.input_file_path, self.options)
        )
        self._delimiter = options.get("delimiter", None)
        self._quotechar = options.get("quotechar", None)
        self._header = options.get("header", "auto")

        self._load_data(data)

    @classmethod
    def _find_target_string_in_column(self, column_names, keyword_list):
        """Find out if col name contains keyword that could refer to target node col."""
        column_name_symbols = ["_", ".", "-"]
        has_target = False
        target_index = -1

        # iterate through columns, keywords, and delimiter name symbols to see
        # if any permutation is contained in column names
        for column in range(0, len(column_names)):
            for keyword in keyword_list:
                for symbol in column_name_symbols:

                    append_start_word = symbol + keyword
                    append_end_word = keyword + symbol

                    if (
                        append_start_word in column_names[column]
                        or append_end_word in column_names[column]
                    ):
                        target_index = column
                        has_target = True
                        break
            if has_target:
                break
        return target_index

    @classmethod
    def csv_column_names(cls, file_path, options):
        """Fetch a list of column names from the csv file."""
        column_names = []
        if options.get("header") is None:
            return column_names

        delimiter = options.get("delimiter", None)
        if delimiter is None:
            delimiter = ","

        with FileOrBufferHandler(
            file_path, encoding=options.get("encoding", "utf-8")
        ) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)

            # fetch only column names
            row_count = 0
            for row in csv_reader:
                if row_count is options.get("header"):
                    column_names.append(row)
                    break
                row_count += 1
        column_names = column_names[0]

        # replace all whitespaces in the column names
        for index in range(0, len(column_names)):
            column_names[index] = column_names[index].replace(" ", "")
        return column_names

    @classmethod
    def is_match(cls, file_path, options=None):
        """
        Determine whether the file is a graph.

        Current formats checked:
            - attributed edge list

        This works by finding whether the file contains a target and a source node
        """
        if options is None:
            options = dict()
        if not CSVData.is_match(file_path, options):
            return False
        column_names = cls.csv_column_names(file_path, options)
        source_keywords = ["source", "src", "origin"]
        target_keywords = ["target", "destination", "dst"]
        node_keyword = ["node"]
        source_index = cls._find_target_string_in_column(column_names, source_keywords)
        destination_index = cls._find_target_string_in_column(
            column_names, target_keywords
        )
        node_index = cls._find_target_string_in_column(column_names, node_keyword)

        has_source = True if source_index >= 0 else False
        has_target = True if destination_index >= 0 else False
        has_node = True if node_index >= 0 else False

        if has_target and has_source and has_node:
            options.update(source_node=source_index)
            options.update(destination_node=destination_index)
            options.update(destination_list=target_keywords)
            options.update(source_list=source_keywords)
            options.update(column_names=column_names)
            return True
        return False

    def _format_data_networkx(self):
        """Format the input file into a networkX graph."""
        networkx_graph = nx.Graph()

        # read lines from csv
        csv_as_list = []
        data_as_pd = data_utils.read_csv_df(
            self.input_file_path,
            self._delimiter,
            self._header,
            [],
            read_in_string=True,
            encoding=self.file_encoding,
        )
        data_as_pd = data_as_pd.apply(lambda x: x.str.strip())
        csv_as_list = data_as_pd.values.tolist()

        # grab list of edges from source/dest nodes
        for line in range(0, len(csv_as_list)):
            # fetch attributes in columns
            attributes = dict()
            for column in range(0, len(csv_as_list[0])):
                if csv_as_list[line][column] is None:
                    continue
                if (
                    column is not self._source_node
                    or column is not self._destination_node
                ):
                    attributes[self._column_names[column]] = float(
                        csv_as_list[line][column]
                    )
                elif column is self._source_node or column is self._destination_node:
                    networkx_graph.add_node(
                        self.check_integer(csv_as_list[line][column])
                    )
            networkx_graph.add_edge(
                self.check_integer(csv_as_list[line][self._source_node]),
                self.check_integer(csv_as_list[line][self._destination_node]),
                **attributes
            )

        # get NetworkX object from list
        return networkx_graph

    def _load_data(self, data=None):
        if data is not None:
            if not isinstance(data, nx.Graph):
                raise ValueError("Only NetworkX Graph objects allowed as input data.")
            self._data = data
        else:
            self._data = self._format_data_networkx()

    def check_integer(self, string):
        """Check whether string is integer and output integer."""
        stringVal = string
        if string[0] == ("-", "+"):
            stringVal = string[1:]
        if stringVal.isdigit():
            return int(string)
        else:
            return string
