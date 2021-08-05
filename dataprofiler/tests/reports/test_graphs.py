import unittest
from unittest import mock

import pandas as pd
from matplotlib import pyplot as plt

import dataprofiler as dp
from dataprofiler.profilers import IntColumn
from dataprofiler.reports import graphs


@mock.patch("dataprofiler.reports.graphs.plt.show")
@mock.patch("dataprofiler.reports.graphs.plot_col_histogram")
class TestPlotHistograms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame(
            [[1, 'a', 1.0, '1/2/2021'],
            [None, 'b', None, '1/2/2020'],
            [3, 'c', 3.5, '1/2/2022'],
            [4, 'd', 4.5, '1/2/2023'],
            [5, 'e', 6.0, '5/2/2020'],
            [None, 'f', None, '1/5/2020'],
            [1, 'g', 1.0, '2/5/2020'],
            [None, 1, 10.0, '3/5/2020']],
            columns=['int', 'str', 'float', 'datetime'])
        cls.options = dp.ProfilerOptions()
        cls.options.set({"data_labeler.is_enabled": False})
        cls.options.set({"multiprocess.is_enabled": False})
        cls.profiler = dp.StructuredProfiler(cls.data, options=cls.options)

    def test_bad_inputs(self, *mocks):
        # columns and column_inds cannot be specified simultaneously
        with self.assertRaisesRegex(ValueError,
                                    "Can only specify either `columns` or "
                                    "`column_inds` but not both "
                                    "simultaneously"):
            graphs.plot_histograms(self.profiler,
                                   columns=['test'],
                                   column_inds=[1])

        # when columns is bad
        bad_columns_input = [-1, [{}, 1], {}, 3.2, [3.2]]
        for bad_input in bad_columns_input:
            with self.assertRaisesRegex(ValueError,
                                        "`columns` must be a list integers or "
                                        "strings matching the names of columns "
                                        "in the profiler."):
                graphs.plot_histograms(self.profiler, columns=bad_input)

        # when column_inds is bad
        bad_columns_inds_input = [-1, [{}, 1], {}, 3.2, [3.2], ['test']]
        for bad_input in bad_columns_inds_input:
            with self.assertRaisesRegex(ValueError,
                                        "`column_inds` must be a list of "
                                        "integers matching column indexes in "
                                        "the profiler"):
                graphs.plot_histograms(self.profiler, column_inds=bad_input)

        # test column name doesn't exist
        with self.assertRaisesRegex(ValueError,
                                    "Column \"a\" is not found as a profiler "
                                    "column"):
            graphs.plot_histograms(self.profiler, ["int", "a"])

    def test_no_columns_specified(self, plot_col_mock, plt_mock):
        graphsplot = graphs.plot_histograms(self.profiler)
        self.assertEqual(2, plot_col_mock.call_count)

        # grabs the first argument passed into the plot col call and validates
        # it is the column profiler and its name matches what we expect it to
        self.assertEqual("int", plot_col_mock.call_args_list[0][0][0].name)

        # grabs the second argument passed into the plot col call and validates
        # it is the column profiler and its name matches what we expect it to
        self.assertEqual("float", plot_col_mock.call_args_list[1][0][0].name)
        self.assertIsInstance(graphsplot, plt.Figure)

    def test_specify_column(self, plot_col_mock, plt_mock):
        graphsplot = graphs.plot_histograms(self.profiler, columns=["float"])
        self.assertEqual(1, plot_col_mock.call_count)
        self.assertEqual("float", plot_col_mock.call_args_list[0][0][0].name)
        self.assertIsInstance(graphsplot, plt.Figure)

    def test_specify_column_inds(self, plot_col_mock, plt_mock):
        graphsplot = graphs.plot_histograms(self.profiler, column_inds=[2])
        self.assertEqual(1, plot_col_mock.call_count)
        self.assertEqual("float", plot_col_mock.call_args_list[0][0][0].name)
        self.assertIsInstance(graphsplot, plt.Figure)

    def test_no_column_plottable(self, plot_col_mock, plt_mock):
        with self.assertWarnsRegex(Warning, "No plots were constructed"
                                            " because no int or float columns "
                                            "were found in columns"):
            fig = graphs.plot_histograms(self.profiler, ["str", "datetime"])
        self.assertIsNone(fig)

    def test_empty_profiler(self, plot_col_mock, plt_mock):
        with self.assertWarnsRegex(Warning, "No plots were constructed"
                                            " because no int or float columns "
                                            "were found in columns"):
            fig = graphs.plot_histograms(
                dp.StructuredProfiler(data=None, options=self.options))
        self.assertIsNone(fig)


@mock.patch("dataprofiler.reports.graphs.plt.show")
class TestPlotColHistogram(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = pd.Series([1, 2, 4, 2, 5, 35, 32], dtype=str)
        cls.profiler = IntColumn('test')
        cls.profiler.update(cls.data)

    def test_normal(self, plt_mock):
        self.assertIsInstance(graphs.plot_col_histogram(self.profiler),
                              plt.Axes)

    def test_empty_data(self, plt_mock):
        data_name = "Fake Name"
        profiler = IntColumn(data_name)
        with self.assertRaisesRegex(ValueError,
                                    f"The column profiler, {data_name}, "
                                    "had no data and therefore could not be "
                                    "plotted."):
            graphs.plot_col_histogram(profiler)
