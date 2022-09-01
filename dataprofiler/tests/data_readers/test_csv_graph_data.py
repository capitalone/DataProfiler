import os
import unittest
from io import BytesIO, StringIO, TextIOWrapper

import networkx as nx

from dataprofiler.data_readers.data import Data
from dataprofiler.data_readers.graph_data import GraphData

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestGraphDataClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(test_root_path, "data")

        cls.input_file_names_pos = [
            dict(
                path=os.path.join(
                    test_dir, "csv/graph-differentiator-input-positive.csv"
                ),
                list_nodes=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                list_edges=[
                    (2, 1),
                    (3, 2),
                    (4, 2),
                    (5, 2),
                    (6, 2),
                    (7, 2),
                    (8, 2),
                    (9, 2),
                ],
                options={"header": 0, "delimiter": ","},
                encoding="utf-8",
            ),
            dict(
                path=os.path.join(
                    test_dir, "csv/graph-differentiator-input-standard-positive.csv"
                ),
                list_nodes=[1, 2, 3, 4],
                list_edges=[(2, 1), (1, 3), (2, 4)],
                options={"header": 0, "delimiter": ","},
                encoding="utf-8",
            ),
            dict(
                path=os.path.join(test_dir, "csv/graph-data-input-positive-header.csv"),
                list_nodes=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                list_edges=[
                    (2, 1),
                    (3, 2),
                    (4, 2),
                    (5, 2),
                    (6, 2),
                    (7, 2),
                    (8, 2),
                    (9, 2),
                ],
                options={"header": 2, "delimiter": ","},
                encoding="utf-8",
            ),
            dict(
                path=os.path.join(test_dir, "csv/graph_data_csv_identify.csv"),
                list_nodes=None,  # too long to fit
                list_edges=None,  # too long to fit
                options={"header": 0, "delimiter": ","},
                encoding="utf-8",
            ),
        ]

        cls.input_file_names_neg = [
            dict(
                path=os.path.join(
                    test_dir, "csv/graph-differentiator-input-negative.csv"
                ),
                encoding="utf-8",
            ),
            dict(
                path=os.path.join(test_dir, "csv/graph-data-input-json.json"),
                encoding="utf-8",
            ),
        ]

        cls.buffer_list = []
        for input_file in cls.input_file_names_pos:
            # add StringIO
            buffer_info = input_file.copy()
            with open(input_file["path"], "r", encoding=input_file["encoding"]) as fp:
                buffer_info["path"] = StringIO(fp.read())
            cls.buffer_list.append(buffer_info)

            # add BytesIO
            buffer_info = input_file.copy()
            with open(input_file["path"], "rb") as fp:
                buffer_info["path"] = BytesIO(fp.read())
            cls.buffer_list.append(buffer_info)

        cls.file_or_buf_list = cls.input_file_names_pos + cls.buffer_list

    @classmethod
    def setUp(cls):
        for buffer in cls.buffer_list:
            buffer["path"].seek(0)

    def test_finding_string_in_column_positive(self):
        """
        Determine whether keywords can be detected with underscore before and after
        """
        column_names_after = ["node_src", "node_dst", "attribute1"]
        column_names_before = ["src_node", "dst_node", "attribute1"]
        keyword_list = ["src", "destination"]
        self.assertEqual(
            GraphData._find_target_string_in_column(column_names_after, keyword_list), 0
        )
        self.assertEqual(
            GraphData._find_target_string_in_column(column_names_before, keyword_list),
            0,
        )

    def test_finding_string_in_column_negative(self):
        """
        Determine whether the output is false when keywords are not found or without substring delimiters
        """
        column_names_no_keywords = ["movie", "audience_type", "audience_source"]
        column_names_no_delimiter = ["flight_number", "destination", "price"]
        keyword_list = ["dst", "destination", "target"]
        self.assertEqual(
            GraphData._find_target_string_in_column(
                column_names_no_keywords, keyword_list
            ),
            -1,
        )
        self.assertEqual(
            GraphData._find_target_string_in_column(
                column_names_no_delimiter, keyword_list
            ),
            -1,
        )

    # test csv_column_name
    def test_csv_column_names(self):
        """
        Determine if column names are fetched correctly and in the right format
        """
        column_names = [
            "node_id_dst",
            "node_id_src",
            "attrib_id",
            "attrib_type",
            "edge_date",
            "open_date_src",
            "open_date_dst",
        ]
        for input_file in self.file_or_buf_list:
            if input_file["list_nodes"] is None or input_file["list_edges"] is None:
                continue
            options = input_file["options"]
            self.assertEqual(
                GraphData.csv_column_names(
                    input_file["path"],
                    options.get("header", 0),
                    options.get("delimiter", ","),
                    options.get("encoding", "utf-8"),
                ),
                column_names,
            )

    # test is_match for true output w/ different options
    def test_is_graph_positive_1(self):
        """
        Determine if the input CSV file can automatically be recognized as being a graph
        """
        for input_file in self.file_or_buf_list:
            self.assertTrue(GraphData.is_match(input_file["path"]))

    # test is_match for false output w/ different options
    def test_is_graph_negative_1(self):
        """
        Determine if the input CSV file can be automatically recognized as not being a graph w/ no options selected
        """
        for input_file in self.input_file_names_neg:
            self.assertFalse(GraphData.is_match(input_file["path"]))

    # test loading data
    def test_data_loader_nodes(self):
        """
        Determine whether the data loader works as expected for nodes
        """

        for input_file in self.file_or_buf_list:
            options = dict()
            if not GraphData.is_match(input_file["path"], options):
                return
            if input_file["list_nodes"] is None:
                continue
            data = GraphData(input_file_path=input_file["path"], options=options)
            self.assertEqual(input_file["list_nodes"], sorted(data.nodes))

    def test_data_loader_edges(self):
        """
        Determine whether the data loader works as expected for edges
        """

        for input_file in self.file_or_buf_list:
            options = dict()
            all_edges_present = True
            if not GraphData.is_match(input_file["path"], options):
                return
            if input_file["list_edges"] is None:
                continue
            data = GraphData(input_file_path=input_file["path"], options=options)
            data_edges = list(data.edges)

            for edge in input_file["list_edges"]:
                if edge not in data_edges and (edge[1], edge[0]) not in data_edges:
                    all_edges_present = False
            self.assertTrue(all_edges_present)

    def test_class_load(self):
        for input_file in self.file_or_buf_list:
            data = GraphData(input_file["path"])
            self.assertIsNotNone(data.data)

    def test_factory_load(self):
        """
        Determine whether factory class Data identifies file correctly
        """
        for input_file in self.file_or_buf_list:
            data = Data(input_file["path"])
            self.assertIsInstance(data, GraphData)


if __name__ == "__main__":
    unittest.main()
