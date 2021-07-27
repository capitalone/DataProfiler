import unittest
from unittest import mock
import logging
from io import StringIO

from dataprofiler import DataLabeler, dp_logging
from dataprofiler.labelers.regex_model import RegexModel


class TestDPLogging(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        root_logger = logging.getLogger()
        root_logger.removeHandler(dp_logging.get_logger())
        dp_logging._logger = None

    def test_default_verbosity(self):
        # Ensure that default effective level is INFO
        self.assertEqual(logging.INFO,
                         logging.getLogger('DataProfiler').getEffectiveLevel())

    def test_set_verbosity(self):
        # Set verbosity to WARNING (won't print/log update messages)
        dp_logging.set_verbosity(logging.WARNING)
        self.assertEqual(logging.WARNING,
                         dp_logging.get_logger().getEffectiveLevel())

        # When self.assertLogs is set with a level where no logs were written
        # An AssertionError is thrown. This is what we expect in this case,
        # Since the only logs we want written are WARNING and up, but only INFO
        # Would have been written in the logic that creates output below
        msg = "no logs of level WARNING or higher triggered on " \
              "DataProfiler.labelers.character_level_cnn_model"
        with self.assertRaisesRegex(AssertionError, msg):
            with self.assertLogs(
                    'DataProfiler.labelers.character_level_cnn_model',
                    level=logging.WARNING):
                labeler = DataLabeler(labeler_type='structured', trainable=True)
                labeler.fit(['this', 'is', 'data'], ['UNKNOWN'] * 3, epochs=7)

        msg = "no logs of level WARNING or higher triggered on " \
              "DataProfiler.labelers.regex_model"
        with self.assertRaisesRegex(AssertionError, msg):
            with self.assertLogs('DataProfiler.labelers.regex_model',
                                 level=logging.WARNING):
                rm = RegexModel(label_mapping={'UNKNOWN': 0})
                rm.predict(data=['oh', 'boy', 'i', 'sure', 'love', 'regex'])

        # Set verbosity to INFO (will print/log update messages)
        dp_logging.set_verbosity(logging.INFO)
        self.assertEqual(logging.INFO,
                         dp_logging.get_logger().getEffectiveLevel())

        # Will write "EPOCH i" updates with statistics
        with self.assertLogs('DataProfiler.labelers.character_level_cnn_model',
                             level=logging.INFO) as logs:
            labeler = DataLabeler(labeler_type='structured', trainable=True)
            labeler.fit(['this', 'is', 'data'], ['UNKNOWN'] * 3, epochs=7)

        # Ensures it got logged
        self.assertTrue(len(logs.output) > 0)

        # Will write "Data Samples Processed: i" updates
        with self.assertLogs('DataProfiler.labelers.regex_model',
                             level=logging.INFO) as logs:
            rm = RegexModel(label_mapping={'UNKNOWN': 0})
            rm.predict(data=['oh', 'boy', 'i', 'sure', 'love', 'regex'])

        # Ensures it got logged
        self.assertTrue(len(logs.output) > 0)
