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
        cls.data = [[1, 'a', 1.0, '1/2/2021'],
                    [None, 'b', None, '1/2/2020'],
                    [3, 'c', 3.5, '1/2/2022'],
                    [4, 'd', 4.5, '1/2/2023'],
                    [5, 'e', 6.0, '5/2/2020'],
                    [None, 'f', None, '1/5/2020'],
                    [1, 'g', 1.0, '2/5/2020'],
                    [None, 1, 10.0, '3/5/2020']]
        cls.options = dp.ProfilerOptions()
        cls.options.set({"data_labeler.is_enabled": False})
        cls.profiler = dp.StructuredProfiler(cls.data, options=cls.options)

    def test_no_columns_specified(self, plot_col_mock, plt_mock):
        graphsplot = graphs.plot_histograms(self.profiler)
        self.assertEqual(2, plot_col_mock.call_count)
        # grabs the first argument passed into the plot col call and validates
        # it is the column profiler and its name matches what we expect it to
        self.assertEqual(0, plot_col_mock.call_args_list[0][0][0].name)
        # grabs the second argument passed into the plot col call and validates
        # it is the column profiler and its name matches what we expect it to
        self.assertEqual(2, plot_col_mock.call_args_list[1][0][0].name)
        self.assertIsInstance(graphsplot, plt.Figure)

    def test_normal(self, plot_col_mock, plt_mock):
        graphsplot = graphs.plot_histograms(self.profiler, [2])
        self.assertEqual(1, plot_col_mock.call_count)
        self.assertEqual(2, plot_col_mock.call_args_list[0][0][0].name)
        self.assertIsInstance(graphsplot, plt.Figure)

    def test_bad_column_name(self, plot_col_mock, plt_mock):
        with self.assertRaisesRegex(ValueError,
                                    "Column \"a\" is not found as a profiler "
                                    "column"):
            graphs.plot_histograms(self.profiler, [0, "a"])

    def test_no_column_plottable(self, plot_col_mock, plt_mock):
        with self.assertWarnsRegex(Warning, "No plots were constructed"
                                            " because no int or float columns "
                                            "were found in columns"):
            graphs.plot_histograms(self.profiler, [1, 3])

    def test_empty_profiler(self, plot_col_mock, plt_mock):
        with self.assertWarnsRegex(Warning, "No plots were constructed"
                                            " because no int or float columns "
                                            "were found in columns"):
            graphs.plot_histograms(
                dp.StructuredProfiler(data=None, options=self.options))


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
        data = pd.Series([], dtype=str)
        profiler = IntColumn(data.name)
        with self.assertRaisesRegex(ValueError, "The column profiler, None, "
                                                "provided had no data and "
                                                "therefore could not be "
                                                "plotted."):
            graphs.plot_col_histogram(profiler)
