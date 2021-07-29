import unittest
import warnings

from dataprofiler.profilers import IntColumn
from dataprofiler.reports import graphs
from unittest import mock
import dataprofiler as dp
import pandas as pd


@mock.patch("dataprofiler.reports.graphs.plt.show")
@mock.patch("dataprofiler.reports.graphs.sns")
class TestPlotHistograms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = [[1, 1.0, 'a', '1/2/2021'],
                    [None, None, 'b', '1/2/2020'],
                    [3, 3.5, 'c', '1/2/2022'],
                    [4, 4.5, 'd', '1/2/2023'],
                    [5, 6.0, 'e', '5/2/2020'],
                    [None, None, 'f', '1/5/2020'],
                    [1, 1.0, 'g', '2/5/2020'],
                    [None, 10.0, 1, '3/5/2020']]
        cls.options = dp.ProfilerOptions()
        cls.options.set({"data_labeler.is_enabled": False})
        cls.profiler = dp.StructuredProfiler(cls.data, options=cls.options)

    @mock.patch("dataprofiler.reports.graphs.plot_col_histogram")
    def test_no_columns_specified(self, plot_col_mock, seaborn_mock, plt_mock):
        x = graphs.plot_histograms(self.profiler)
        print(graphs)
        self.assertEqual(2, plot_col_mock.call_count)
        self.assertEqual(0, plot_col_mock.call_args_list[0].args[0].name)
        self.assertEqual(1, plot_col_mock.call_args_list[1].args[0].name)
        self.assertIsNotNone(x)

    @mock.patch("dataprofiler.reports.graphs.plot_col_histogram")
    def test_normal(self, plot_col_mock, seaborn_mock, plt_mock):
        x = graphs.plot_histograms(self.profiler, [1])
        print(graphs)
        self.assertEqual(1, plot_col_mock.call_count)
        self.assertEqual(1, plot_col_mock.call_args_list[0].args[0].name)
        self.assertIsNotNone(x)

    def test_bad_column_name(self, seaborn_mock, plt_mock):
        with self.assertRaisesRegex(ValueError,
                                    "Column \"a\" is not found as a profiler column"):
            graphs.plot_histograms(self.profiler, [0, "a"])

    def test_no_column_plottable(self, seaborn_mock, plt_mock):
        with self.assertWarnsRegex(Warning, "No plots were constructed"
                                            " because no int or float columns were found in columns"):
            graphs.plot_histograms(self.profiler, [2, 3])

    @mock.patch("dataprofiler.reports.graphs.plot_col_histogram")
    def test_empty_profiler(self, plot_col_mock, seaborn_mock, plt_mock):
        with self.assertWarnsRegex(Warning, "No plots were constructed"
                                            " because no int or float columns were found in columns"):
            graphs.plot_histograms(
                dp.StructuredProfiler(data=None, options=self.options))


@mock.patch("dataprofiler.reports.graphs.plt.show")
class TestPlotColHistogram(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = pd.Series([1, 2, 4, 2, 5, 35, 32], dtype=str)
        cls.profiler = IntColumn(cls.data.name)
        cls.profiler.update(cls.data)

    def test_normal(self, plt_mock):
        graphs.plot_col_histogram(self.profiler)

    def test_empty_data(self, plt_mock):
        data = pd.Series([], dtype=str)
        profiler = IntColumn(data.name)
        with self.assertRaisesRegex(ValueError, "The column profiler, None, "
                                                "provided had no data and "
                                                "therefore could not be "
                                                "plotted."):
            graphs.plot_col_histogram(profiler)
