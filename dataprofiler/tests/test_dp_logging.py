import unittest
from unittest import mock
import logging
from io import StringIO


@mock.patch('sys.stdout', new_callable=StringIO)
class TestDPLogging(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from dataprofiler import dp_logging
        root_logger = logging.getLogger()
        root_logger.removeHandler(dp_logging.get_logger())
        dp_logging._logger = None

    @classmethod
    def tearDown(cls):
        from dataprofiler import dp_logging
        root_logger = logging.getLogger()
        root_logger.removeHandler(dp_logging.get_logger())
        dp_logging._logger = None

    def test_default_verbosity(self, mock_stdout):
        # Ensure that default effective level is INFO
        self.assertEqual(logging.INFO,
                         logging.getLogger('DataProfiler').getEffectiveLevel())

    def test_set_verbosity(self, mock_stdout):
        from dataprofiler import dp_logging, DataLabeler
        from dataprofiler.labelers.regex_model import RegexModel
        # Set verbosity to WARNING (won't print/log update messages)
        dp_logging.set_verbosity(logging.WARNING)
        self.assertEqual(logging.WARNING,
                         logging.getLogger('DataProfiler').getEffectiveLevel())

        labeler = DataLabeler(labeler_type='structured', trainable=True)
        labeler.fit(['this', 'is', 'data'], ['UNKNOWN'] * 3, epochs=7)

        # Ensure it didn't get printed
        self.assertFalse(len(mock_stdout.getvalue()))

        # Reset StringIO stdout mock
        mock_stdout.truncate(0)
        mock_stdout.seek(0)

        rm = RegexModel(label_mapping={'UNKNOWN': 0})
        rm.predict(data=['oh', 'boy', 'i', 'sure', 'love', 'regex'])

        # Ensure it didn't get printed
        self.assertFalse(len(mock_stdout.getvalue()))

        # Reset StringIO stdout mock
        mock_stdout.truncate(0)
        mock_stdout.seek(0)

        # Set verbosity to INFO (will print/log update messages)
        dp_logging.set_verbosity(logging.INFO)
        self.assertEqual(logging.INFO,
                         logging.getLogger('DataProfiler').getEffectiveLevel())

        # Will write "EPOCH i" updates with statistics
        labeler = DataLabeler(labeler_type='structured', trainable=True)
        labeler.fit(['this', 'is', 'data'], ['UNKNOWN'] * 3, epochs=7)

        # Ensure it got printed
        self.assertTrue(len(mock_stdout.getvalue()))

        # Reset StringIO stdout mock
        mock_stdout.truncate(0)
        mock_stdout.seek(0)

        # Will write "Data Samples Processed: i" updates
        rm = RegexModel(label_mapping={'UNKNOWN': 0})
        rm.predict(data=['oh', 'boy', 'i', 'sure', 'love', 'regex'])

        # Ensure it got printed
        self.assertTrue(len(mock_stdout.getvalue()))
