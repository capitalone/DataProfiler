import json
import unittest
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pandas as pd

from dataprofiler.profilers.base_column_profilers import BaseColumnProfiler
from dataprofiler.profilers.categorical_column_profile import CategoricalColumn
from dataprofiler.profilers.json_encoder import ProfileEncoder


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

        serialized = json.dumps(profile, cls=ProfileEncoder, sort_keys=True)

        expected = json.dumps(
            {
                "_CategoricalColumn__calculations": {},
                "_categories": {"a": 3, "b": 4, "c": 5},
                "_top_k_categories": None,
                "col_index": np.nan,
                "metadata": {},
                "name": None,
                "sample_size": 12,
                "thread_safe": True,
                "times": {"categories": 0.0},
            },
            sort_keys=True,
        )

        self.assertEqual(serialized, expected)
