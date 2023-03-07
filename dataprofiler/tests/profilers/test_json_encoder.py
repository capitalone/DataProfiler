import json
import unittest
from collections import defaultdict
from unittest.mock import patch

import numpy as np

from dataprofiler.profilers.base_column_profilers import BaseColumnProfiler
from dataprofiler.profilers.json_encoder import ProfileEncoder


class TestJsonEncoder(unittest.TestCase):
    def test_encode_base_column_profiler(self):
        with patch.multiple(BaseColumnProfiler, __abstractmethods__=set()):
            profile = BaseColumnProfiler(name="0")

        serialized = json.dumps(profile, cls=ProfileEncoder)
        exepcted = json.loads(
            json.dumps(
                {
                    "name": "0",
                    "col_index": np.nan,
                    "sample_size": 0,
                    "metadata": dict(),
                    "times": defaultdict(),
                    "thread_safe": True,
                }
            )
        )

        self.assertEqual(json.loads(serialized), exepcted)
