import os
import unittest

from dataprofiler.profilers.column_profile_compilers import ColumnStatsProfileCompiler
from dataprofiler.tests.profilers.test_base_column_profilers import (
    AbstractTestColumnProfiler,
)

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestColumnStatsProfiler(AbstractTestColumnProfiler, unittest.TestCase):

    column_profiler = ColumnStatsProfileCompiler
    profile_types = ["data type", "statistics"]

    def setUp(self):
        AbstractTestColumnProfiler.setUp(self)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()


if __name__ == "__main__":
    unittest.main()
