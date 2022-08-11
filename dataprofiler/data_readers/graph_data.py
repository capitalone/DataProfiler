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
        graph_keywords: list of keywords to identify if data has graph data
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
        self._graph_keywords = options.get("graph_keywords", ["node"])
        self._column_names = options.get("column_names", None)
        self._delimiter = options.get("delimiter", None)
        self._quotechar = options.get("quotechar", None)
        self._header = options.get("header", "auto")
        self._checked_header = "header" in options and self._header != "auto"

        if data is not None:
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
    def csv_column_names(cls, file_path, header, delimiter, encoding="utf-8"):
        """Fetch a list of column names from the csv file."""
        column_names = []
        if delimiter is None:
            delimiter = ","
        if header is None:
            return column_names

        with FileOrBufferHandler(file_path, encoding=encoding) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)

            # fetch only column names
            row_count = 0
            for row in csv_reader:
                if row_count is header:
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
        header = options.get("header", 0)
        delimiter = options.get("delimiter", ",")
        encoding = options.get("encoding", "utf-8")
        column_names = cls.csv_column_names(file_path, header, delimiter, encoding)
        source_keywords = options.get("source_keywords", ["source", "src", "origin"])
        target_keywords = options.get(
            "target_keywords", ["target", "destination", "dst"]
        )
        graph_keywords = options.get("graph_keywords", ["node"])
        source_index = cls._find_target_string_in_column(column_names, source_keywords)
        destination_index = cls._find_target_string_in_column(
            column_names, target_keywords
        )
        graph_index = cls._find_target_string_in_column(column_names, graph_keywords)

        has_source = True if source_index >= 0 else False
        has_target = True if destination_index >= 0 else False
        has_graph_data = True if graph_index >= 0 else False

        if has_target and has_source and has_graph_data:
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
        if not self._checked_header or not self._delimiter:
            delimiter, quotechar = None, None
            data_as_str = data_utils.load_as_str_from_file(
                self.input_file_path, self.file_encoding
            )

            if not self._delimiter or not self._quotechar:
                delimiter, quotechar = CSVData._guess_delimiter_and_quotechar(
                    data_as_str
                )
            if not self._delimiter:
                self._delimiter = delimiter
            if not self._quotechar:
                self._quotechar = quotechar

            if self._header == "auto":
                self._header = CSVData._guess_header_row(
                    data_as_str, self._delimiter, self._quotechar
                )
                self._checked_header = True

            # if there is only one delimiter at the end of each row,
            # set delimiter to None
            if self._delimiter:
                if len(data_as_str) > 0:
                    num_lines_read = 0
                    count_delimiter_last = 0
                    for line in data_as_str.split("\n"):
                        if len(line) > 0:
                            if (
                                line.count(self._delimiter) == 1
                                and line.strip()[-1] == self._delimiter
                            ):
                                count_delimiter_last += 1
                            num_lines_read += 1
                    if count_delimiter_last == num_lines_read:
                        self._delimiter = None

        if self._column_names is None:
            self._column_names = self.csv_column_names(
                self.input_file_path, self._header, self._delimiter, self.file_encoding
            )
        if self._source_node is None:
            self._source_node = self._find_target_string_in_column(
                self._column_names, self._source_keywords
            )
        if self._destination_node is None:
            self._destination_node = self._find_target_string_in_column(
                self._column_names, self._target_keywords
            )

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
