import unittest
from unittest import mock
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import dataprofiler as dp
from dataprofiler.profilers import IntColumn
from dataprofiler.reports import graphs


class TestGraphImport(unittest.TestCase):

    def missing_module_test(self, graph_func, module_name):
        orig_import = __import__

        # necessary for any wrapper around the library to test if snappy caught
        # as an issue

        def import_mock(name, *args, **kwargs):
            if name.startswith(module_name):
                raise ImportError('test')
            return orig_import(name, *args, **kwargs)

        import re
        warning_regex = re.compile(
            ".*WARNING Graphing Failure.*" + module_name + '.*', re.DOTALL)
        with mock.patch('builtins.__import__', side_effect=import_mock):
            with self.assertWarnsRegex(RuntimeWarning, warning_regex):
                modules_to_remove = [
                    'dataprofiler.reports.graphs',
                    module_name,
                ]
                for module in modules_to_remove:
                    if module in sys.modules:
                        del sys.modules[module]
                # re-add module for testing
                for module in modules_to_remove[:-1]:
                    import importlib
                    importlib.import_module(module)
                graph_func(None)

    def test_import_from_base_repo(self):
        self.assertTrue(hasattr(dp, 'graphs'))

    def test_no_seaborn(self):
        self.missing_module_test(dp.graphs.plot_histograms, 'seaborn')
        self.missing_module_test(dp.graphs.plot_missing_values_matrix, 'seaborn')
        self.missing_module_test(dp.graphs.plot_col_missing_values, 'seaborn')

    def test_no_matplotlib(self):
        self.missing_module_test(dp.graphs.plot_histograms, 'matplotlib')
        self.missing_module_test(dp.graphs.plot_missing_values_matrix, 'matplotlib')
        self.missing_module_test(dp.graphs.plot_col_missing_values, 'matplotlib')



@mock.patch("dataprofiler.graphs.plt.show")
@mock.patch("dataprofiler.graphs.plot_col_histogram")
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
                                    "Can only specify either `column_names` or "
                                    "`column_inds` but not both "
                                    "simultaneously"):
            graphs.plot_histograms(self.profiler,
                                   column_names=['test'],
                                   column_inds=[1])

        # when column_names is bad
        bad_columns_input = [-1, [{}, 1], {}, 3.2, [3.2]]
        for bad_input in bad_columns_input:
            with self.assertRaisesRegex(ValueError,
                                        "`column_names` must be a list integers"
                                        " or strings matching the names of "
                                        "columns in the profiler."):
                graphs.plot_histograms(self.profiler, column_names=bad_input)

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
        graphsplot = graphs.plot_histograms(self.profiler,
                                            column_names=["float"])
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


@mock.patch("dataprofiler.profilers.profile_builder.ColumnStatsProfileCompiler")
@mock.patch("dataprofiler.profilers.profile_builder.ColumnDataLabelerCompiler")
@mock.patch("dataprofiler.profilers.profile_builder."
            "ColumnPrimitiveTypeProfileCompiler")
class TestPlotMissingValuesMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.options = dp.ProfilerOptions()
        cls.options.set({"data_labeler.is_enabled": False})
        cls.options.set({"multiprocess.is_enabled": False})
        cls.options.set({"correlation.is_enabled": False})
        cls.options.set({"chi2_homogeneity.is_enabled": False})

    def test_no_data(self, *mocks):
        profiler = dp.StructuredProfiler([], options=self.options)
        with self.assertWarnsRegex(UserWarning,
                                   'There was no data in the profiles to plot '
                                   'missing column values.'):
            graphs.plot_missing_values_matrix(profiler)

    def test_null_list(self, *mocks):
        data = [None, None, None]

        profiler = dp.StructuredProfiler(data, options=self.options)

        fig = graphs.plot_missing_values_matrix(profiler)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(1, len(fig.axes))

        ax = fig.axes[0]
        patches, labels = ax.get_legend_handles_labels()
        self.assertEqual(['"None"'], labels)

        expected_patch_values = [
            {'xy': (0.1, -0.5), 'width': 0.8, 'height': 3},
        ]

        for patch, expected in zip(patches, expected_patch_values):
            np.testing.assert_almost_equal(expected['xy'], patch.xy)
            self.assertEqual(expected['width'], patch.get_width())
            self.assertEqual(expected['height'], patch.get_height())
        xtick_labels = [xtick.get_text() for xtick in ax.get_xticklabels()]
        self.assertListEqual(['"0"'], xtick_labels)
        self.assertEqual('column name', ax.get_xlabel())
        self.assertEqual('row index', ax.get_ylabel())

    def test_1_null_type_multicol(self, *mocks):
        data = [[None, None, 1.0 , '1/2/2021'],
                [3   , None, 3.5 , None      ],
                [1   , None, 1.0 , '2/5/2020'],
                [None,    1, 10.0, '3/5/2020']]

        profiler = dp.StructuredProfiler(data, options=self.options)

        fig = graphs.plot_missing_values_matrix(profiler)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(1, len(fig.axes))

        ax = fig.axes[0]
        patches, labels = ax.get_legend_handles_labels()
        self.assertEqual(['"None"', '"None"', '"None"', '"None"'], labels)

        expected_patch_values = [
            {'xy': (0.1, -0.5), 'width': 0.8, 'height': 1},
            {'xy': (0.1, 2.5), 'width': 0.8, 'height': 1},
            {'xy': (1.1, -0.5), 'width': 0.8, 'height': 3},
            {'xy': (3.1, 0.5), 'width': 0.8, 'height': 1},
        ]

        for patch, expected in zip(patches, expected_patch_values):
            np.testing.assert_almost_equal(expected['xy'], patch.xy)
            self.assertEqual(expected['width'], patch.get_width())
            self.assertEqual(expected['height'], patch.get_height())

        xtick_labels = [xtick.get_text() for xtick in ax.get_xticklabels()]
        self.assertListEqual(['"0"', '"1"', '"2"', '"3"'], xtick_labels)
        self.assertEqual('column name', ax.get_xlabel())
        self.assertEqual('row index', ax.get_ylabel())

    def test_2_null_types_multicol(self, *mocks):
        data = pd.DataFrame(
            [[None, '', 1.0, '1/2/2021'],
             [3, None, 3.5, ''],
             [1, None, 1.0, '2/5/2020'],
             [None, 1, 10.0, '3/5/2020']],
            columns=['integer', 'str', 'float', 'datetime'],
            dtype=object
        )

        profiler = dp.StructuredProfiler(data, options=self.options)

        fig = graphs.plot_missing_values_matrix(profiler)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(1, len(fig.axes))

        ax = fig.axes[0]
        patches, labels = ax.get_legend_handles_labels()
        self.assertEqual(['"None"', '"None"', '""', '"None"', '""'], labels)

        expected_patch_values = [
            {'xy': (0.1, -0.5), 'width': 0.8, 'height': 1},
            {'xy': (0.1, 2.5), 'width': 0.8, 'height': 1},
            {'xy': (1.1, -0.5), 'width': 0.8, 'height': 1},
            {'xy': (1.1, 0.5), 'width': 0.8, 'height': 2},
            {'xy': (3.1, 0.5), 'width': 0.8, 'height': 1},
        ]

        for patch, expected in zip(patches, expected_patch_values):
            np.testing.assert_almost_equal(expected['xy'], patch.xy)
            self.assertEqual(expected['width'], patch.get_width())
            self.assertEqual(expected['height'], patch.get_height())
        xtick_labels = [xtick.get_text() for xtick in ax.get_xticklabels()]
        self.assertListEqual(
            ['"integer"', '"str"', '"float"', '"datetime"'], xtick_labels)
        self.assertEqual('column name', ax.get_xlabel())
        self.assertEqual('row index', ax.get_ylabel())

    def test_bad_input(self, *mocks):

        with self.assertRaisesRegex(ValueError,
                                    '`col_profiler_list` must be a list of '
                                    'StructuredColProfilers'):
            graphs.plot_col_missing_values(None)

        with self.assertRaisesRegex(ValueError,
                                    '`profiler` must of type '
                                    'StructuredProfiler.'):
            graphs.plot_missing_values_matrix(None)