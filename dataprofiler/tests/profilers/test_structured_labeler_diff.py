import os
import unittest

import numpy as np
import pandas as pd

from dataprofiler import Data, Profiler

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestStructDiff(unittest.TestCase):
    def test_diff(self):
        # divide dataset in half
        test_dir = os.path.join(test_root_path, "data")
        data = Data(os.path.join(test_dir, "csv/diamonds.csv"))
        df = data.data
        df1 = df.iloc[: int(len(df) / 2)]
        df2 = df.iloc[int(len(df) / 2) :]

        profile1 = Profiler(df1)
        profile2 = Profiler(df2)

        data_split_differences = profile1.diff(profile2)


test_diff = TestStructDiff()
test_diff.test_diff()
