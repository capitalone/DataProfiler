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
        dp_logging._dp_logger = None

    @classmethod
    def tearDownClass(cls):
        from dataprofiler import dp_logging
        root_logger = logging.getLogger()
        root_logger.removeHandler(dp_logging.get_logger())
        dp_logging._dp_logger = None

    def test_default_verbosity(self, mock_stdout):
        # Ensure that default effective level is INFO
        self.assertEqual(logging.INFO,
                         logging.getLogger('DataProfiler').getEffectiveLevel())

    def test_set_verbosity(self, mock_stdout):
        from dataprofiler import dp_logging

        # Initialize DataProfiler logger
        dp_logging.get_logger()

        # Make dummy logger that inherits from DataProfiler logger
        dummy_logger = logging.getLogger('DataProfiler.dummy.logger')

        # Set to INFO by default
        self.assertEqual(logging.INFO,
                         dummy_logger.getEffectiveLevel())

        # Info appears in stdout
        dummy_logger.info("this is info 1")
        self.assertIn("this is info 1", mock_stdout.getvalue())

        # Warnings appear in stdout
        dummy_logger.warning("this is warning 1")
        self.assertIn("this is warning 1", mock_stdout.getvalue())

        # Turn verbosity to WARNING, so that info doesn't appear in stdout
        dp_logging.set_verbosity(logging.WARNING)

        # Info will no longer appear in stdout
        dummy_logger.info("this is info 2")
        self.assertNotIn("this is info 2", mock_stdout.getvalue())

        # Warnings still appear in stdout
        dummy_logger.warning("this is warning 2")
        self.assertIn("this is warning 2", mock_stdout.getvalue())

        # Going back to INFO still works as expected
        dp_logging.set_verbosity(logging.INFO)

        dummy_logger.info("this is info 3")
        self.assertIn("this is info 3", mock_stdout.getvalue())

        dummy_logger.warning("this is warning 3")
        self.assertIn("this is warning 3", mock_stdout.getvalue())
