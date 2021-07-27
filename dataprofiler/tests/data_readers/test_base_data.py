from __future__ import print_function
from __future__ import absolute_import

import os
import unittest

from dataprofiler.data_readers.base_data import BaseData


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestBaseDataClass(unittest.TestCase):

    def test_can_apply_data_functions(self):
        class FakeDataClass:
            # matches the `data_type` value in BaseData for validating priority
            data_type = "FakeData"

            def func1(self):
                return "success"

        # initialize the data class
        data = BaseData(input_file_path="", data=FakeDataClass(), options={})

        # if the function exists in BaseData fail the test because the results
        # may become inaccurate.
        self.assertFalse(hasattr(BaseData, 'func1'))

        with self.assertRaisesRegex(AttributeError,
                                    "Neither 'BaseData' nor 'FakeDataClass' "
                                    "objects have attribute 'test'"):
            data.test

        # validate it will take BaseData attribute over the data attribute
        self.assertIsNone(data.data_type)

        # validate will auto call the data function if it doesn't exist in
        # BaseData
        self.assertEqual("success", data.func1())
