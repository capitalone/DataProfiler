import unittest
import logging

from dataprofiler import DataLabeler, dp_logging
from dataprofiler.labelers.regex_model import RegexModel


class TestDPLogging(unittest.TestCase):

    def test_set_verbosity(self):
        # Ensure that logs are written when verbose
        dp_logging.set_verbosity(logging.INFO)
        # Level should be set to INFO
        self.assertEqual(logging.INFO,
                         dp_logging.get_logger().getEffectiveLevel())

        # Will write "EPOCH i" updates with statistics
        with self.assertLogs('DataProfiler.labelers.character_level_cnn_model',
                             level=logging.INFO) as cm:
            labeler = DataLabeler(labeler_type='structured', trainable=True)
            labeler.fit(['this', 'is', 'data'], ['UNKNOWN'] * 3, epochs=7)

        # Check that 3 'EPOCH i' updates written for each epoch
        for i in range(7):
            # 3 entries indices corresponding to given epoch
            idxs = [i * 3, i * 3 + 1, i * 3 + 2]
            for idx in idxs:
                log = cm.output[idx]
                epoch_msg = (f"INFO:DataProfiler.labelers."
                             f"character_level_cnn_model:\rEPOCH {i}")
                self.assertEqual(epoch_msg, log[:len(epoch_msg)])

        # Will write "Data Samples Processed: i" updates
        with self.assertLogs('DataProfiler.labelers.regex_model',
                             level=logging.INFO) as cm:
            rm = RegexModel(label_mapping={'UNKNOWN': 0})
            rm.predict(data=['oh', 'boy', 'i', 'sure', 'love', 'regex'])

        # Check that 'Data Samples Process: i' written for each sample
        for i in range(5):
            log = cm.output[i]
            exp_log = (f"INFO:DataProfiler.labelers.regex_model:\rData Samples "
                       f"Processed: {i}   ")
            self.assertEqual(exp_log, log)

        # Ensure that no logs written when not verbose
        dp_logging.set_verbosity(logging.WARNING)
        # Level should be set to WARNING
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