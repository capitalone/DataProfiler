import json
import unittest

import numpy as np

from dataprofiler.profilers.helpers.report_helpers import _prepare_report


class TestReportHelperClass(unittest.TestCase):
    def test_serializable_report(self):
        test_dict = {
            "key_1": {"nested_key_1": {"nested_key_1_2": np.ndarray(1)}},
            np.int64(1): 0,
        }
        with self.assertRaises(TypeError):
            json.dumps(test_dict)
        with self.assertRaises(TypeError):
            json.dumps(_prepare_report(test_dict))
        with self.assertRaises(TypeError):
            json.dumps(_prepare_report(test_dict, output_format="not a real one"))
        try:
            x = _prepare_report(test_dict, output_format="serIaliZable")
            json.dumps(x)
        except:
            self.fail("serialize_report not json serializable")

    def test_omit_keys_in_report(self):

        report = {
            "test0": 0,
            "test1": 1,
            "test2": {"test3": 3},
            "test4": {"test5": {"test3": 3}},
        }
        output_format = "pretty"
        omit_keys = []

        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, report)

        omit_keys = ["test0"]
        report_test1 = {
            "test1": 1,
            "test2": {"test3": 3},
            "test4": {"test5": {"test3": 3}},
        }
        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, report_test1)

        omit_keys = ["test0", "test4.test5.test3"]
        report_test2 = {"test1": 1, "test2": {"test3": 3}, "test4": {"test5": {}}}
        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, report_test2)

        omit_keys = ["test0", "test4.test5"]
        report_test3 = {"test1": 1, "test2": {"test3": 3}, "test4": {}}
        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, report_test3)

        omit_keys = ["test1", "test4"]
        report_test4 = {"test0": 0, "test2": {"test3": 3}}
        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, report_test4)

        omit_keys = ["test0", "test2.test3"]
        report_test5 = {"test1": 1, "test2": {}, "test4": {"test5": {"test3": 3}}}
        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, report_test5)

        # Keys that don't exist in the report
        omit_keys = ["test6", "test3"]
        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, report)

        # Robustness check
        omit_keys = None
        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, report)

        # Test wildcard removals
        wild_report1 = {
            "test0": 0,
            "test1": 1,
            "test2": {},
            "test4": {"test5": {"test3": 3}},
        }
        omit_keys = ["*.test3"]

        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, wild_report1)

        # Ensure rigid counting
        wild_report2 = {
            "test0": 0,
            "test1": 1,
            "test2": {"test3": 3},
            "test4": {"test5": {"test3": 3}},
        }
        omit_keys = ["*.test3.test3"]

        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, wild_report2)

        # Ensure multiple wildcards
        wild_report3 = {
            "test0": 0,
            "test1": 1,
            "test2": {"test3": 3},
            "test4": {"test5": {}},
        }
        omit_keys = ["*.*.test3"]

        prepared_report = _prepare_report(report, output_format, omit_keys)
        self.assertDictEqual(prepared_report, wild_report3)


if __name__ == "__main__":
    unittest.main()
