import unittest

import numpy as np
import json

from data_profiler.profilers.helpers.report_helpers import _prepare_report


class TestReportHelperClass(unittest.TestCase):
    def test_serializable_report(self):
        test_dict = {
            "key_1": {
                "nested_key_1": {
                    "nested_key_1_2": np.ndarray(1)
                }
            }
        }
        with self.assertRaises(TypeError):
            json.dumps(test_dict)
        with self.assertRaises(TypeError):
            json.dumps(_prepare_report(test_dict))
        with self.assertRaises(TypeError):
            json.dumps(_prepare_report(test_dict, output_format='not a real one'))
        try:
            x = _prepare_report(test_dict, output_format='serIaliZable')
            json.dumps(x)
        except:
            self.fail('serialize_report not json serializable')

if __name__ == '__main__':
    unittest.main()