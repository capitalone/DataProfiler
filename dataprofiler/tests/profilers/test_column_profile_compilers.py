import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from dataprofiler.labelers import BaseDataLabeler
from dataprofiler.profilers import column_profile_compilers as col_pro_compilers
from dataprofiler.profilers.base_column_profilers import BaseColumnProfiler
from dataprofiler.profilers.json_decoder import load_compiler
from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import (
    BaseOption,
    StructuredOptions,
    UnstructuredOptions,
)

from . import utils as test_utils


class TestBaseProfileCompilerClass(unittest.TestCase):
    def test_cannot_instantiate(self):
        """showing we normally can't instantiate an abstract class"""
        with self.assertRaises(TypeError) as e:
            col_pro_compilers.BaseCompiler()
        self.assertRegex(
            str(e.exception),
            "Can't instantiate abstract class BaseCompiler with "
            "abstract methods? report",
        )

    @mock.patch.multiple(
        col_pro_compilers.BaseCompiler,
        __abstractmethods__=set(),
        _profilers=[mock.Mock()],
        _option_class=mock.Mock(spec=BaseOption),
    )
    @mock.patch.multiple(
        col_pro_compilers.ColumnStatsProfileCompiler, _profilers=[mock.Mock()]
    )
    def test_add_profilers(self):
        compiler1 = col_pro_compilers.BaseCompiler(mock.Mock())
        compiler2 = col_pro_compilers.BaseCompiler(mock.Mock())

        # test incorrect type
        with self.assertRaisesRegex(
            TypeError,
            "`BaseCompiler` and `int` are " "not of the same profile compiler type.",
        ):
            compiler1 + 3

        compiler3 = col_pro_compilers.ColumnStatsProfileCompiler(mock.Mock())
        compiler3._profiles = [mock.Mock()]
        with self.assertRaisesRegex(
            TypeError,
            "`BaseCompiler` and "
            "`ColumnStatsProfileCompiler` are "
            "not of the same profile compiler type.",
        ):
            compiler1 + compiler3

        # test mismatched names
        compiler1.name = "compiler1"
        compiler2.name = "compiler2"
        with self.assertRaisesRegex(
            ValueError, "Column profile names are unmatched: " "compiler1 != compiler2"
        ):
            compiler1 + compiler2

        # test mismatched profiles due to options
        compiler2.name = "compiler1"
        compiler1._profiles = dict(test1=mock.Mock())
        compiler2._profiles = dict(test2=mock.Mock())
        with self.assertRaisesRegex(
            ValueError,
            "Column profilers were not setup with the "
            "same options, hence they do not calculate "
            "the same profiles and cannot be added "
            "together.",
        ):
            compiler1 + compiler2

        # test success
        compiler1._profiles = dict(test=1)
        compiler2._profiles = dict(test=2)
        merged_compiler = compiler1 + compiler2
        self.assertEqual(3, merged_compiler._profiles["test"])
        self.assertEqual("compiler1", merged_compiler.name)

    @mock.patch.multiple(col_pro_compilers.BaseCompiler, __abstractmethods__=set())
    def test_no_profilers_error(self):
        with self.assertRaises(NotImplementedError) as e:
            col_pro_compilers.BaseCompiler()
        self.assertEqual("Must add profilers.", str(e.exception))

    @mock.patch.multiple(
        col_pro_compilers.BaseCompiler, __abstractmethods__=set(), _profilers="mock"
    )
    def test_no_options_error(self):
        with self.assertRaisesRegex(
            NotImplementedError, "Must set the expected OptionClass."
        ):
            col_pro_compilers.BaseCompiler()

    def test_update_match_are_abstract(self):
        self.assertCountEqual(
            {"report"}, col_pro_compilers.BaseCompiler.__abstractmethods__
        )

    @mock.patch.multiple(BaseColumnProfiler, __abstractmethods__=set())
    def test_json_encode(self):
        with mock.patch.multiple(
            col_pro_compilers.BaseCompiler,
            __abstractmethods__=set(),
            _profilers=[BaseColumnProfiler],
            _option_class=BaseOption,
        ):
            profile = col_pro_compilers.BaseCompiler()

        base_column_profiler = BaseColumnProfiler(name="test")
        with mock.patch.object(
            profile, "_profiles", {"BaseColumn": base_column_profiler}
        ):
            serialized = json.dumps(profile, cls=ProfileEncoder)

        dict_of_base_column_profiler = json.loads(
            json.dumps(base_column_profiler, cls=ProfileEncoder)
        )
        expected = json.dumps(
            {
                "class": "BaseCompiler",
                "data": {
                    "name": None,
                    "_profiles": {
                        "BaseColumn": dict_of_base_column_profiler,
                    },
                },
            }
        )

        self.assertEqual(expected, serialized)


class TestColumnPrimitiveTypeProfileCompiler(unittest.TestCase):
    def test_primitive_compiler_report(self):
        structured_options = StructuredOptions()
        data1 = pd.Series(["2.6", "-1.8"])
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        structured_options.float.precision.is_enabled = False
        compiler1._create_profile(data1, structured_options)

        report = compiler1.report(remove_disabled_flag=True)
        self.assertNotIn("precision", report["statistics"])

        report = compiler1.report(remove_disabled_flag=False)
        self.assertIn("precision", report["statistics"])

        structured_options2 = StructuredOptions()
        data2 = pd.Series(["abcd", "aa", "abcd", "aa", "b", "dfd"])
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)
        structured_options2.text.vocab.is_enabled = False
        compiler2._create_profile(data2, structured_options2)

        report = compiler2.report(remove_disabled_flag=True)
        self.assertNotIn("vocab", report["statistics"])

        report = compiler2.report(remove_disabled_flag=False)
        self.assertIn("vocab", report["statistics"])

    def test_diff_primitive_compilers(self):
        # Test different data types
        data1 = pd.Series(["-2", "-1", "1", "2"])
        data2 = pd.Series(["YO YO YO", "HELLO"])
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)

        expected_diff = {
            "data_type_representation": {
                "datetime": "unchanged",
                "int": 1.0,
                "float": 1.0,
                "text": "unchanged",
            },
            "data_type": ["int", "text"],
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test different data types with datetime specifically
        data1 = pd.Series(["-2", "-1", "1", "2"])
        data2 = pd.Series(["01/12/1967", "11/9/2024"])
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)

        expected_diff = {
            "data_type_representation": {
                "datetime": -1.0,
                "int": 1.0,
                "float": 1.0,
                "text": "unchanged",
            },
            "data_type": ["int", "datetime"],
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test same data types
        data1 = pd.Series(["-2", "15", "1", "2"])
        data2 = pd.Series(["5", "-1"])

        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)

        expected_diff = {
            "data_type_representation": {
                "datetime": "unchanged",
                "int": "unchanged",
                "float": "unchanged",
                "text": "unchanged",
            },
            "data_type": "unchanged",
            "statistics": {
                "min": -1.0,
                "max": 10.0,
                "sum": 12.0,
                "mean": 2.0,
                "median": -0.5,
                "mode": [[-2, 15, 1, 2], [], [5, -1]],
                "median_absolute_deviation": -1,
                "variance": 38.666666666666664,
                "stddev": 3.285085839971525,
                "t-test": {
                    "t-statistic": 0.4155260166386663,
                    "conservative": {"deg_of_free": 1.0, "p-value": 0.749287157907667},
                    "welch": {
                        "deg_of_free": 3.6288111187629117,
                        "p-value": 0.7011367179395704,
                    },
                },
                "psi": 0.17328679513998632,
            },
        }
        profile_diff = compiler1.diff(compiler2)
        self.assertAlmostEqual(
            expected_diff["statistics"].pop("median"),
            profile_diff["statistics"].pop("median"),
            places=2,
        )
        expected_diff_mode = expected_diff["statistics"].pop("mode")
        diff_mode = profile_diff["statistics"].pop("mode")
        for i in range(len(expected_diff_mode)):
            np.testing.assert_almost_equal(
                sorted(expected_diff_mode[i]), sorted(diff_mode[i]), 2
            )
        self.assertAlmostEqual(
            expected_diff["statistics"].pop("median_absolute_deviation"),
            profile_diff["statistics"].pop("median_absolute_deviation"),
            places=2,
        )
        self.assertDictEqual(expected_diff, profile_diff)

        # Test different compilers
        data1 = pd.Series(["-2", "-1", "1", "2"])
        data2 = pd.Series(["5", "15"])

        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnStatsProfileCompiler(data2)
        # Assert type error is properly called
        with self.assertRaises(TypeError) as exc:
            compiler1.diff(compiler2)
        self.assertEqual(
            str(exc.exception),
            "`ColumnPrimitiveTypeProfileCompiler` and "
            "`ColumnStatsProfileCompiler` are not of the same "
            "profile compiler type.",
        )

    def test_disabling_columns_during_primitive_diff(self):

        data1 = pd.Series(["-2", "-1", "1", "2"])
        data2 = pd.Series(["5", "15"])
        options = StructuredOptions()

        # Test disabled column in one compiler
        options.int.is_enabled = False
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1, options)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)
        expected_diff = {
            "data_type_representation": {
                "datetime": "unchanged",
                "float": "unchanged",
                "text": "unchanged",
                "int": [None, 1.0],
            },
            "data_type": ["float", "int"],
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test disabled column in both compilers
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2, options)
        expected_diff = {
            "data_type_representation": {
                "datetime": "unchanged",
                "float": "unchanged",
                "text": "unchanged",
            },
            "data_type": "unchanged",
            "statistics": {
                "min": -7.0,
                "max": -13.0,
                "sum": -20.0,
                "mean": -10.0,
                "median": -10,
                "mode": [[-2.0, -1.0, 1.0, 2.0], [], [5, 15]],
                "median_absolute_deviation": -3.5,
                "variance": -46.666666666666664,
                "stddev": data1.astype(int).std() - data2.astype(int).std(),
                "precision": {
                    "min": "unchanged",
                    "max": -1,
                    "mean": -0.5,
                    "var": -0.5,
                    "std": -0.71,
                    "sample_size": 2,
                    "margin_of_error": -1.6,
                },
                "t-test": {
                    "t-statistic": -1.9674775073518591,
                    "conservative": {
                        "deg_of_free": 1.0,
                        "p-value": 0.29936264581081673,
                    },
                    "welch": {
                        "deg_of_free": 1.0673824509440946,
                        "p-value": 0.28696889329266506,
                    },
                },
                "psi": 0,
            },
        }
        profile_diff = compiler1.diff(compiler2)
        self.assertAlmostEqual(
            expected_diff["statistics"].pop("median"),
            profile_diff["statistics"].pop("median"),
            places=2,
        )
        expected_diff_mode = expected_diff["statistics"].pop("mode")
        diff_mode = profile_diff["statistics"].pop("mode")
        for i in range(len(expected_diff_mode)):
            np.testing.assert_almost_equal(
                sorted(expected_diff_mode[i]), sorted(diff_mode[i]), 2
            )
        self.assertAlmostEqual(
            expected_diff["statistics"].pop("median_absolute_deviation"),
            profile_diff["statistics"].pop("median_absolute_deviation"),
            places=2,
        )
        self.assertDictEqual(expected_diff, profile_diff)

        # Test disabling all columns in one compiler
        options.float.is_enabled = False
        options.text.is_enabled = False
        options.datetime.is_enabled = False
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1, options)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)
        expected_diff = {
            "data_type_representation": {
                "datetime": [None, 0.0],
                "int": [None, 1.0],
                "float": [None, 1.0],
                "text": [None, 1.0],
            },
            "data_type": [None, "int"],
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test disabling all columns in all compilers
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2, options)
        expected_diff = {}
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

    def test_json_encode(self):

        compiler = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler()

        serialized = json.dumps(compiler, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "class": "ColumnPrimitiveTypeProfileCompiler",
                "data": {
                    "name": None,
                    "_profiles": {},
                },
            }
        )
        self.assertEqual(expected, serialized)

    def test_json_encode_after_update(self):

        data = pd.Series(["-2", "-1", "1", "2"], name="test")
        with test_utils.mock_timeit():
            compiler = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data)

        with mock.patch.object(compiler._profiles["datetime"], "__dict__", {}):
            with mock.patch.object(compiler._profiles["int"], "__dict__", {}):
                with mock.patch.object(compiler._profiles["float"], "__dict__", {}):
                    with mock.patch.object(compiler._profiles["text"], "__dict__", {}):
                        serialized = json.dumps(compiler, cls=ProfileEncoder)

        # pop the data inside primitive column profiler as we just want to make
        # sure generally it is serializing, decode will validate true replication

        expected = json.dumps(
            {
                "class": "ColumnPrimitiveTypeProfileCompiler",
                "data": {
                    "name": "test",
                    "_profiles": {
                        "datetime": {"class": "DateTimeColumn", "data": {}},
                        "int": {"class": "IntColumn", "data": {}},
                        "float": {"class": "FloatColumn", "data": {}},
                        "text": {"class": "TextColumn", "data": {}},
                    },
                },
            }
        )

        self.assertEqual(expected, serialized)

    def test_json_decode(self):
        expected_compiler = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler()
        serialized = json.dumps(expected_compiler, cls=ProfileEncoder)

        deserialized = load_compiler(json.loads(serialized))

        test_utils.assert_profiles_equal(expected_compiler, deserialized)

    def test_json_decode_after_update(self):

        data = pd.Series(["-2", "-1", "1", "2"], name="test")
        with test_utils.mock_timeit():
            expected_compiler = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(
                data
            )

        serialized = json.dumps(expected_compiler, cls=ProfileEncoder)
        deserialized = load_compiler(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_compiler)
        # assert before update
        assert (
            deserialized.report().get("statistics", {}).get("mean")
            == sum([-2, -1, 1, 2]) / 4
        )

        df_float = pd.Series(
            [
                4.0,  # add existing
                15.0,  # add new
            ]
        ).apply(str)

        # validating update after deserialization with a few small tests
        deserialized.update_profile(df_float)

        for profile in deserialized._profiles.values():
            assert profile.sample_size == 6
        assert (
            deserialized.report().get("statistics", {}).get("mean")
            == sum([-2, -1, 1, 2, 4, 15]) / 6
        )


class TestColumnStatsProfileCompiler(unittest.TestCase):
    def test_column_stats_profile_compiler_report(self):
        structured_options = StructuredOptions()
        structured_options.category.is_enabled = False
        data1 = pd.Series(["2.6", "-1.8", "-2.3"])
        compiler1 = col_pro_compilers.ColumnStatsProfileCompiler(
            data1, structured_options
        )
        report = compiler1.report(remove_disabled_flag=True)
        self.assertNotIn("categorical", report)
        self.assertIn("order", report)

        structured_options = StructuredOptions()
        structured_options.order.is_enabled = False
        data1 = pd.Series(["2.6", "-1.8", "-2.3"])
        compiler1 = col_pro_compilers.ColumnStatsProfileCompiler(
            data1, structured_options
        )
        report = compiler1.report(remove_disabled_flag=False)
        self.assertIn("categorical", report)
        self.assertNotIn("order", report)

    def test_column_stats_profile_compiler_stats_diff(self):
        data1 = pd.Series(["1", "9", "9"])
        data2 = pd.Series(["10", "9", "9", "9"])
        options = StructuredOptions()

        # Test normal diff
        compiler1 = col_pro_compilers.ColumnStatsProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnStatsProfileCompiler(data2)
        expected_diff = {
            "order": ["ascending", "descending"],
            "categorical": "unchanged",
            "statistics": {
                "unique_count": "unchanged",
                "unique_ratio": 0.16666666666666663,
                "categories": [["1"], ["9"], ["10"]],
                "gini_impurity": 0.06944444444444448,
                "unalikeability": 0.16666666666666663,
                "categorical_count": {"9": -1, "1": 1, "10": -1},
                "chi2-test": {
                    "chi2-statistic": 2.1,
                    "deg_of_free": 2,
                    "p-value": 0.3499377491111554,
                },
                "psi": 0.009815252971365292,
            },
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test disabled categorical column in one compiler
        options.category.is_enabled = False
        compiler1 = col_pro_compilers.ColumnStatsProfileCompiler(data1, options)
        compiler2 = col_pro_compilers.ColumnStatsProfileCompiler(data2)
        expected_diff = {"order": ["ascending", "descending"]}
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test disabling categorical profile in both compilers
        compiler1 = col_pro_compilers.ColumnStatsProfileCompiler(data1, options)
        compiler2 = col_pro_compilers.ColumnStatsProfileCompiler(data2, options)
        expected_diff = {"order": ["ascending", "descending"]}
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test disabling everything
        options.order.is_enabled = False
        compiler1 = col_pro_compilers.ColumnStatsProfileCompiler(data1, options)
        compiler2 = col_pro_compilers.ColumnStatsProfileCompiler(data2, options)
        expected_diff = {}
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

    def test_json_encode(self):

        compiler = col_pro_compilers.ColumnStatsProfileCompiler()

        serialized = json.dumps(compiler, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "class": "ColumnStatsProfileCompiler",
                "data": {
                    "name": None,
                    "_profiles": {},
                },
            }
        )
        self.assertEqual(expected, serialized)

    def test_json_encode_after_update(self):

        data = pd.Series(["-2", "-1", "1", "2"], name="test")
        with test_utils.mock_timeit():
            compiler = col_pro_compilers.ColumnStatsProfileCompiler(data)

        with mock.patch.object(
            compiler._profiles["order"], "__dict__", {"an": "order"}
        ):
            with mock.patch.object(
                compiler._profiles["category"], "__dict__", {"this": "category"}
            ):
                serialized = json.dumps(compiler, cls=ProfileEncoder)

        expected = json.dumps(
            {
                "class": "ColumnStatsProfileCompiler",
                "data": {
                    "name": "test",
                    "_profiles": {
                        "order": {"class": "OrderColumn", "data": {"an": "order"}},
                        "category": {
                            "class": "CategoricalColumn",
                            "data": {"this": "category"},
                        },
                    },
                },
            }
        )

        self.assertEqual(expected, serialized)

    def test_json_decode(self):
        expected_compiler = col_pro_compilers.ColumnStatsProfileCompiler()
        serialized = json.dumps(expected_compiler, cls=ProfileEncoder)

        deserialized = load_compiler(json.loads(serialized))

        test_utils.assert_profiles_equal(expected_compiler, deserialized)

    def test_json_decode_after_update(self):

        data = pd.Series(["-2", "-1", "1", "15"], name="test")
        with test_utils.mock_timeit():
            expected_compiler = col_pro_compilers.ColumnStatsProfileCompiler(data)

        serialized = json.dumps(expected_compiler, cls=ProfileEncoder)
        deserialized = load_compiler(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_compiler)
        # assert before update
        assert deserialized.report().get("order", None) == "ascending"
        assert deserialized.report().get("categorical", None) == True

        df_float = pd.Series(
            list(range(100))  # make orer random and not categorical
        ).apply(str)

        # validating update after deserialization with a few small tests
        deserialized.update_profile(df_float)
        assert deserialized.report().get("order", None) == "random"
        assert deserialized.report().get("categorical", None) == False


@mock.patch(
    "dataprofiler.profilers.profiler_utils.DataLabeler",
    spec=BaseDataLabeler,
)
@mock.patch(
    "dataprofiler.profilers.data_labeler_column_profile.DataLabeler",
    spec=BaseDataLabeler,
)
class TestColumnDataLabelerCompiler(unittest.TestCase):
    @staticmethod
    def _setup_data_labeler_mock(mock_instance):
        mock_DataLabeler = mock_instance.return_value
        mock_DataLabeler.label_mapping = {"a": 0, "b": 1}
        mock_DataLabeler.reverse_label_mapping = {0: "a", 1: "b"}
        mock_DataLabeler.model.num_labels = 2
        mock_DataLabeler.model.requires_zero_mapping = False
        mock_DataLabeler._default_model_loc = "structured_model"

        mock_instance.load_from_library.side_effect = mock_instance

        def mock_predict(data, *args, **kwargs):
            len_data = len(data)
            output = [[1, 0], [0, 1]] * (len_data // 2)
            if len_data % 2:
                output += [[1, 0]]
            conf = np.array(output)
            if mock_DataLabeler.model.requires_zero_mapping:
                conf = np.concatenate([[[0]] * len_data, conf], axis=1)
            pred = np.argmax(conf, axis=1)
            return {"pred": pred, "conf": conf}

        mock_DataLabeler.predict.side_effect = mock_predict

    def test_column_data_labeler_compiler_report(self, mock_instance, *mocks):
        self._setup_data_labeler_mock(mock_instance)
        structured_options = StructuredOptions()
        data1 = pd.Series(["2.6", "-1.8", "-2.3"])
        compiler1 = col_pro_compilers.ColumnDataLabelerCompiler(
            data1, structured_options
        )
        report = compiler1.report(remove_disabled_flag=True)
        self.assertIn("data_label", report)
        self.assertIn("statistics", report)

    def test_compiler_data_labeler_diff(self, *mocks):
        # Initialize dummy data
        data = pd.Series([])

        # Test normal diff
        compiler1 = col_pro_compilers.ColumnDataLabelerCompiler(data)
        compiler2 = col_pro_compilers.ColumnDataLabelerCompiler(data)

        # Mock out the data_label, avg_predictions, and label_representation
        # properties
        with mock.patch(
            "dataprofiler.profilers.data_labeler_column_profile"
            ".DataLabelerColumn.data_label"
        ), mock.patch(
            "dataprofiler.profilers.data_labeler_column_profile."
            "DataLabelerColumn.avg_predictions"
        ), mock.patch(
            "dataprofiler.profilers.data_labeler_column_profile."
            "DataLabelerColumn.label_representation"
        ):
            compiler1._profiles["data_labeler"].sample_size = 20
            compiler1._profiles["data_labeler"].data_label = "a"
            compiler1._profiles["data_labeler"].avg_predictions = {
                "a": 0.25,
                "b": 0.0,
                "c": 0.75,
            }
            compiler1._profiles["data_labeler"].label_representation = {
                "a": 0.15,
                "b": 0.01,
                "c": 0.84,
            }

            compiler2._profiles["data_labeler"].sample_size = 20
            compiler2._profiles["data_labeler"].data_label = "b"
            compiler2._profiles["data_labeler"].avg_predictions = {
                "a": 0.25,
                "b": 0.70,
                "c": 0.05,
            }
            compiler2._profiles["data_labeler"].label_representation = {
                "a": 0.99,
                "b": 0.01,
                "c": 0.0,
            }

            expected_diff = {
                "statistics": {
                    "avg_predictions": {"a": "unchanged", "b": -0.7, "c": 0.7},
                    "label_representation": {"a": -0.84, "b": "unchanged", "c": 0.84},
                },
                "data_label": [["a"], [], ["b"]],
            }
            self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test disabling one datalabeler profile for compiler diff
        options = StructuredOptions()
        options.data_labeler.is_enabled = False
        compiler1 = col_pro_compilers.ColumnDataLabelerCompiler(data, options)
        expected_diff = {}
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test disabling both datalabeler profiles for compiler diff
        compiler2 = col_pro_compilers.ColumnDataLabelerCompiler(data, options)
        expected_diff = {}
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

    def test_json_encode(self, *mocks):
        compiler = col_pro_compilers.ColumnDataLabelerCompiler()

        serialized = json.dumps(compiler, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "class": "ColumnDataLabelerCompiler",
                "data": {
                    "name": None,
                    "_profiles": {},
                },
            }
        )
        self.assertEqual(expected, serialized)

    def test_json_decode(self, *mocks):
        expected_compiler = col_pro_compilers.ColumnDataLabelerCompiler()
        serialized = json.dumps(expected_compiler, cls=ProfileEncoder)

        deserialized = load_compiler(json.loads(serialized))

        test_utils.assert_profiles_equal(expected_compiler, deserialized)

    def test_json_encode_after_update(self, mock_instance, *mocks):
        self._setup_data_labeler_mock(mock_instance)

        data = pd.Series(["-2", "-1", "1", "2"])
        with test_utils.mock_timeit():
            compiler = col_pro_compilers.ColumnDataLabelerCompiler(data)

        with mock.patch.object(
            compiler._profiles["data_labeler"], "__dict__", {"data_label": "INTEGER"}
        ):
            serialized = json.dumps(compiler, cls=ProfileEncoder)

        expected = json.dumps(
            {
                "class": "ColumnDataLabelerCompiler",
                "data": {
                    "name": None,
                    "_profiles": {
                        "data_labeler": {
                            "class": "DataLabelerColumn",
                            "data": {"data_label": "INTEGER"},
                        },
                    },
                },
            }
        )

        self.assertEqual(expected, serialized)

    def test_json_decode_after_update(self, mock_instance, mock_utils_DataLabeler):

        self._setup_data_labeler_mock(mock_instance)
        mock_instance._default_model_loc = "structured_model"
        mock_utils_DataLabeler.load_from_library = mock_instance

        data = pd.Series(["2", "-1", "1", "2"], name="test")
        with test_utils.mock_timeit():
            expected_compiler = col_pro_compilers.ColumnDataLabelerCompiler(data)

        serialized = json.dumps(expected_compiler, cls=ProfileEncoder)
        deserialized = load_compiler(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_compiler)
        # assert before update
        assert deserialized.report().get("data_label", None) == "a|b"
        assert deserialized.report().get("statistics", {}).get(
            "data_label_representation", None
        ) == {"a": 0.5, "b": 0.5}

        new_data = pd.Series(["100"])

        # validating update after deserialization with a few small tests
        deserialized.update_profile(new_data)
        assert deserialized.report().get("data_label", None) == "a|b"
        assert deserialized.report().get("statistics", {}).get(
            "data_label_representation", None
        ) == {"a": 0.6, "b": 0.4}

    def test_json_decode_with_options(
        self, mock_DataLabeler_cls, mock_utils_DataLabeler
    ):
        self._setup_data_labeler_mock(mock_DataLabeler_cls)
        mock_DataLabeler_cls._default_model_loc = "structured_model"
        mock_utils_DataLabeler.load_from_library = mock_DataLabeler_cls

        data = pd.Series(["2", "-1", "1", "2"], name="test")
        with test_utils.mock_timeit():
            expected_compiler = col_pro_compilers.ColumnDataLabelerCompiler(data)

        serialized = json.dumps(expected_compiler, cls=ProfileEncoder)

        # create a new labeler ot load instead of from_library
        new_mock_data_labeler = mock.Mock(spec=BaseDataLabeler)
        new_mock_data_labeler.name = "new fake data labeler"
        new_mock_data_labeler._default_model_loc = "my/fake/path"
        options = {
            "DataLabelerColumn": {
                "from_library": {"structured_model": new_mock_data_labeler}
            }
        }

        mock_DataLabeler_cls.reset_mock()  # set to 0 calls as option should override
        deserialized = load_compiler(json.loads(serialized), options)

        # ensure doesn't change original, but options updates deserialized labeler
        assert (
            expected_compiler._profiles.get("data_labeler", mock.Mock()).data_labeler
            == mock_DataLabeler_cls.return_value
        )
        assert (
            deserialized._profiles.get("data_labeler", mock.Mock()).data_labeler
            == new_mock_data_labeler
        )
        mock_DataLabeler_cls.assert_not_called()


class TestUnstructuredCompiler(unittest.TestCase):
    @mock.patch("dataprofiler.profilers.unstructured_labeler_profile." "DataLabeler")
    @mock.patch(
        "dataprofiler.profilers.unstructured_labeler_profile." "CharPostprocessor"
    )
    def test_base(self, *mocks):
        from collections import defaultdict

        import pandas as pd

        df_series = pd.Series(["test", "hi my name is John Doe. 123-432-1234"])

        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            compiler = col_pro_compilers.UnstructuredCompiler(df_series)

        expected_dict = {
            "data_label": {
                "entity_counts": {
                    "postprocess_char_level": defaultdict(int),
                    "true_char_level": defaultdict(int),
                    "word_level": defaultdict(int),
                },
                "entity_percentages": {
                    "postprocess_char_level": defaultdict(int),
                    "true_char_level": defaultdict(int),
                    "word_level": defaultdict(int),
                },
                "times": {"data_labeler_predict": 1.0},
            },
            "statistics": {
                "times": {"vocab": 1.0, "words": 1.0},
                "vocab_count": {
                    " ": 6,
                    "-": 2,
                    ".": 1,
                    "1": 2,
                    "2": 3,
                    "3": 3,
                    "4": 2,
                    "D": 1,
                    "J": 1,
                    "a": 1,
                    "e": 3,
                    "h": 2,
                    "i": 2,
                    "m": 2,
                    "n": 2,
                    "o": 2,
                    "s": 2,
                    "t": 2,
                    "y": 1,
                },
                "vocab": [
                    " ",
                    "-",
                    ".",
                    "1",
                    "2",
                    "3",
                    "4",
                    "D",
                    "J",
                    "a",
                    "e",
                    "h",
                    "i",
                    "m",
                    "n",
                    "o",
                    "s",
                    "t",
                    "y",
                ],
                "word_count": {
                    "123-432-1234": 1,
                    "Doe": 1,
                    "John": 1,
                    "hi": 1,
                    "name": 1,
                    "test": 1,
                },
                "words": ["test", "hi", "name", "John", "Doe", "123-432-1234"],
            },
        }

        output_profile = compiler.profile

        # because vocab uses a set, it will be random order every time, hence
        # we need to sort to check exact match between profiles
        if "statistics" in output_profile and "vocab" in output_profile["statistics"]:
            output_profile["statistics"]["vocab"] = sorted(
                output_profile["statistics"]["vocab"]
            )

        self.assertDictEqual(expected_dict, output_profile)

    @mock.patch("dataprofiler.profilers.unstructured_labeler_profile." "DataLabeler")
    @mock.patch(
        "dataprofiler.profilers.unstructured_labeler_profile." "CharPostprocessor"
    )
    def test_compiler_unstructured_reports(self, *mocks):
        data = pd.Series(["Hello Hello", "This is a test grant"])
        compiler = col_pro_compilers.UnstructuredCompiler(data)
        unstructured_options = UnstructuredOptions()
        unstructured_options.text.vocab.is_enabled = False
        compiler._create_profile(data, unstructured_options)

        report = compiler.report(remove_disabled_flag=True)
        self.assertNotIn("vocab", report["statistics"])
        self.assertIn("words", report["statistics"])

        report = compiler.report(remove_disabled_flag=False)
        self.assertIn("vocab", report["statistics"])
        self.assertIn("words", report["statistics"])

        unstructured_options.text.vocab.is_enabled = True
        compiler._create_profile(data, unstructured_options)
        report = compiler.report(remove_disabled_flag=True)
        self.assertIn("vocab", report["statistics"])
        self.assertIn("words", report["statistics"])

        unstructured_options.text.words.is_enabled = False
        compiler._create_profile(data, unstructured_options)
        report = compiler.report(remove_disabled_flag=True)
        self.assertIn("vocab", report["statistics"])
        self.assertNotIn("words", report["statistics"])

    @mock.patch("dataprofiler.profilers.unstructured_labeler_profile." "DataLabeler")
    @mock.patch(
        "dataprofiler.profilers.unstructured_labeler_profile." "CharPostprocessor"
    )
    def test_compiler_stats_diff(self, *mocks):
        data1 = pd.Series(["Hello Hello", "This is a test grant"])
        data2 = pd.Series(["This is unknown", "my name grant", "9", "9"])

        # Test normal diff
        compiler1 = col_pro_compilers.UnstructuredCompiler(data1)
        compiler2 = col_pro_compilers.UnstructuredCompiler(data2)
        labeler_1 = compiler1._profiles["data_labeler"]
        labeler_2 = compiler2._profiles["data_labeler"]

        labeler_1.char_sample_size = 20
        labeler_1.word_sample_size = 15
        entity_counts = {
            "word_level": {"UNKNOWN": 5, "TEST": 5, "UNIQUE1": 5},
            "true_char_level": {"UNKNOWN": 4, "TEST": 8, "UNIQUE1": 8},
            "postprocess_char_level": {"UNKNOWN": 5, "TEST": 10, "UNIQUE1": 5},
        }
        labeler_1.entity_counts = entity_counts
        labeler_1.update(pd.Series(["a"]))

        labeler_2.char_sample_size = 20
        labeler_2.word_sample_size = 10
        entity_counts = {
            "word_level": {"UNKNOWN": 2, "TEST": 4, "UNIQUE2": 4},
            "true_char_level": {"UNKNOWN": 8, "TEST": 8, "UNIQUE2": 4},
            "postprocess_char_level": {"UNKNOWN": 5, "TEST": 10, "UNIQUE2": 5},
        }
        labeler_2.entity_counts = entity_counts
        labeler_2.update(pd.Series(["a"]))

        expected_diff = {
            "statistics": {
                "vocab": [
                    ["H", "l"],
                    ["e", "o", " ", "T", "h", "i", "s", "a", "t", "g", "r", "n"],
                    ["u", "k", "w", "m", "y", "9"],
                ],
                "vocab_count": [
                    {"l": 4, "H": 2},
                    {
                        " ": 1,
                        "e": 2,
                        "s": 1,
                        "t": 2,
                        "o": 1,
                        "i": "unchanged",
                        "a": "unchanged",
                        "T": "unchanged",
                        "h": "unchanged",
                        "g": "unchanged",
                        "r": "unchanged",
                        "n": -4,
                    },
                    {"m": 2, "9": 2, "u": 1, "k": 1, "w": 1, "y": 1},
                ],
                "words": [["Hello", "test"], ["grant"], ["unknown", "name", "9"]],
                "word_count": [
                    {"Hello": 2, "test": 1},
                    {"grant": "unchanged"},
                    {"9": 2, "unknown": 1, "name": 1},
                ],
            },
            "data_label": {
                "entity_counts": {
                    "word_level": {
                        "UNKNOWN": 3,
                        "TEST": 1,
                        "UNIQUE1": [5, None],
                        "UNIQUE2": [None, 4],
                    },
                    "true_char_level": {
                        "UNKNOWN": -4,
                        "TEST": "unchanged",
                        "UNIQUE1": [8, None],
                        "UNIQUE2": [None, 4],
                    },
                    "postprocess_char_level": {
                        "UNKNOWN": "unchanged",
                        "TEST": "unchanged",
                        "UNIQUE1": [5, None],
                        "UNIQUE2": [None, 5],
                    },
                },
                "entity_percentages": {
                    "word_level": {
                        "UNKNOWN": 0.1333333333333333,
                        "TEST": -0.06666666666666671,
                        "UNIQUE1": [0.3333333333333333, None],
                        "UNIQUE2": [None, 0.4],
                    },
                    "true_char_level": {
                        "UNKNOWN": -0.2,
                        "TEST": "unchanged",
                        "UNIQUE1": [0.4, None],
                        "UNIQUE2": [None, 0.2],
                    },
                    "postprocess_char_level": {
                        "UNKNOWN": "unchanged",
                        "TEST": "unchanged",
                        "UNIQUE1": [0.25, None],
                        "UNIQUE2": [None, 0.25],
                    },
                },
            },
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test while disabling a column
        options = UnstructuredOptions()
        options.data_labeler.is_enabled = False
        compiler2 = col_pro_compilers.UnstructuredCompiler(data2, options)
        expected_diff = {
            "statistics": {
                "vocab": [
                    ["H", "l"],
                    ["e", "o", " ", "T", "h", "i", "s", "a", "t", "g", "r", "n"],
                    ["u", "k", "w", "m", "y", "9"],
                ],
                "vocab_count": [
                    {"l": 4, "H": 2},
                    {
                        " ": 1,
                        "e": 2,
                        "s": 1,
                        "t": 2,
                        "o": 1,
                        "i": "unchanged",
                        "a": "unchanged",
                        "T": "unchanged",
                        "h": "unchanged",
                        "g": "unchanged",
                        "r": "unchanged",
                        "n": -4,
                    },
                    {"m": 2, "9": 2, "u": 1, "k": 1, "w": 1, "y": 1},
                ],
                "words": [["Hello", "test"], ["grant"], ["unknown", "name", "9"]],
                "word_count": [
                    {"Hello": 2, "test": 1},
                    {"grant": "unchanged"},
                    {"9": 2, "unknown": 1, "name": 1},
                ],
            }
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test while disabling 2 columns
        options.text.is_enabled = False
        compiler2 = col_pro_compilers.UnstructuredCompiler(data2, options)
        expected_diff = {}
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test while disabling all columns
        compiler1 = col_pro_compilers.UnstructuredCompiler(data1, options)
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))


if __name__ == "__main__":
    unittest.main()
