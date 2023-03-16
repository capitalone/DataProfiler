import json
import unittest
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pandas as pd

from dataprofiler.profilers.base_column_profilers import BaseColumnProfiler
from dataprofiler.profilers.categorical_column_profile import CategoricalColumn
from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.order_column_profile import OrderColumn


class TestJsonEncoder(unittest.TestCase):
    def test_encode_base_column_profiler(self):
        with patch.multiple(BaseColumnProfiler, __abstractmethods__=set()):
            profile = BaseColumnProfiler(name="0")

        serialized = json.dumps(profile, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "name": "0",
                "col_index": np.nan,
                "sample_size": 0,
                "metadata": dict(),
                "times": defaultdict(),
                "thread_safe": True,
            }
        )

        self.assertEqual(serialized, expected)

    def test_encode_categorical_column_profiler(self):
        profile = CategoricalColumn("0")

        serialized = json.dumps(profile, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "name": "0",
                "col_index": np.nan,
                "sample_size": 0,
                "metadata": dict(),
                "times": defaultdict(),
                "thread_safe": True,
                "_categories": defaultdict(int),
                "_CategoricalColumn__calculations": dict(),
                "_top_k_categories": None,
            }
        )

        self.assertEqual(serialized, expected)

    def test_encode_categorical_column_profiler_after_update(self):
        df_categorical = pd.Series(
            [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                "c",
                "c",
                "c",
                "c",
                "c",
            ]
        )
        profile = CategoricalColumn(df_categorical.name)

        with patch("time.time", side_effect=lambda: 0.0):
            profile.update(df_categorical)

        serialized = json.dumps(profile, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "name": None,
                "col_index": np.nan,
                "sample_size": 12,
                "metadata": {},
                "times": {"categories": 0.0},
                "thread_safe": True,
                "_categories": {"c": 5, "b": 4, "a": 3},
                "_CategoricalColumn__calculations": {},
                "_top_k_categories": None,
            },
        )

        self.assertEqual(serialized, expected)

    def test_encode_order_column_profiler(self):
        profile = OrderColumn("0")

        serialized = json.dumps(profile, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "order": None,
                "_last_value": None,
                "_first_value": None,
                "_piecewise": False,
                "_OrderColumn__calculations": dict(),
                "name": "0",
                "col_index": np.nan,
                "sample_size": 0,
                "metadata": dict(),
                "times": defaultdict(),
                "thread_safe": True,
            }
        )

        self.assertEqual(serialized, expected)

    def test_encode_order_column_profiler_after_update(self):
        profile = OrderColumn("0")

        df_order = pd.Series(["za", "z", "c", "a"])
        with patch("time.time", side_effect=lambda: 0.0):
            profile.update(df_order)

        serialized = json.dumps(profile, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "order": "descending",
                "_last_value": "a",
                "_first_value": "za",
                "_piecewise": False,
                "_OrderColumn__calculations": dict(),
                "name": "0",
                "col_index": np.nan,
                "sample_size": 4,
                "metadata": dict(),
                "times": {"order": 0.0},
                "thread_safe": True,
            }
        )

        self.assertEqual(serialized, expected)
