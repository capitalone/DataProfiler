from __future__ import print_function

import os
import logging
import unittest
from unittest import mock

import dataprofiler as dp

from . import test_utils

from dataprofiler import Data, Profiler, DataLabeler
from dataprofiler.labelers.regex_model import RegexModel


# This is taken from: https://github.com/rlworkgroup/dowel/pull/36/files
# undo when cpython#4800 is merged.
unittest.case._AssertWarnsContext.__enter__ = test_utils.patched_assert_warns


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestDataProfiler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(MODULE_PATH, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'csv/aws_honeypot_marx_geo.csv'),
                 type='csv'),
        ]

    def test_set_seed(self):
        import dataprofiler as dp
        self.assertEqual(dp.settings._seed, None)

        dp.set_seed(5)
        self.assertEqual(dp.settings._seed, 5)

        with self.assertRaisesRegex(ValueError,
                                    "Seed should be a non-negative integer."):
            dp.set_seed(-5)

        with self.assertRaisesRegex(ValueError,
                                    "Seed should be a non-negative integer."):
            dp.set_seed(5.2)

    def test_set_verbosity(self):
        # Ensure that logs are written when verbose
        dp.set_verbosity(True)
        # Level should be set to INFO
        self.assertEqual(logging.INFO, logging.getLogger().getEffectiveLevel())

        # Will write "EPOCH i" updates with statistics
        with self.assertLogs(level=logging.INFO) as cm:
            labeler = DataLabeler(labeler_type='structured', trainable=True)
            labeler.fit(['this', 'is', 'data'], ['UNKNOWN'] * 3, epochs=7)

        # Check that 3 'EPOCH i' updates written for each epoch
        for i in range(7):
            # 3 entries indices corresponding to given epoch
            idxs = [i * 3, i * 3 + 1, i * 3 + 2]
            for idx in idxs:
                log = cm.output[idx]
                epoch_msg = (f"INFO:DataProfiler.character_level_cnn_model:"
                             f"\rEPOCH {i}")
                self.assertEqual(epoch_msg, log[:len(epoch_msg)])

        # Will write "Data Samples Processed: i" updates
        with self.assertLogs(level=logging.INFO) as cm:
            rm = RegexModel(label_mapping={'UNKNOWN': 0})
            rm.predict(data=['oh', 'boy', 'i', 'sure', 'love', 'regex'])

        # Check that 'Data Samples Process: i' written for each sample
        for i in range(5):
            log = cm.output[i]
            exp_log = (f"INFO:DataProfiler.regex_model:\rData Samples "
                       f"Processed: {i}   ")
            self.assertEqual(exp_log, log)

        # Ensure that no logs written when not verbose
        dp.set_verbosity(False)
        # Level should be set to WARNING
        self.assertEqual(logging.WARNING, logging.getLogger().getEffectiveLevel())

        # When self.assertLogs is set with a level where no logs were written
        # An AssertionError is thrown. This is what we expect in this case,
        # Since the only logs we want written are WARNING and up, but only INFO
        # Would have been written in the logic that creates output below
        with self.assertRaisesRegex(AssertionError, "no logs of level WARNING "
                                                    "or higher triggered on "
                                                    "root"):
            with self.assertLogs(level=logging.WARNING) as cm:
                labeler = DataLabeler(labeler_type='structured', trainable=True)
                labeler.fit(['this', 'is', 'data'], ['UNKNOWN'] * 3, epochs=7)

        with self.assertRaisesRegex(AssertionError, "no logs of level WARNING "
                                                    "or higher triggered on "
                                                    "root"):
            with self.assertLogs(level=logging.WARNING) as cm:
                rm = RegexModel(label_mapping={'UNKNOWN': 0})
                rm.predict(data=['oh', 'boy', 'i', 'sure', 'love', 'regex'])

    def test_data_import(self):
        for file in self.input_file_names:
            data = Data(file['path'])
            self.assertEqual(data.data_type, file['type'])

    def test_data_profiling(self):
        for file in self.input_file_names:
            data = Data(file['path'])
            profile = Profiler(data)
            self.assertIsNotNone(profile.profile)
            self.assertIsNotNone(profile.report())

    def test_no_snappy(self):
        import sys
        import importlib
        import types
        orig_import = __import__
        # necessary for any wrapper around the library to test if snappy caught
        # as an issue

        def reload_data_profiler():
            """Recursively reload modules."""
            sys_modules = sys.modules.copy()
            for module_name, module in sys_modules.items():
                # Only reload top level of the dataprofiler
                if ('dataprofiler' in module_name and
                        len(module_name.split('.')) < 3):
                    if isinstance(module, types.ModuleType):
                        importlib.reload(module)

        def import_mock(name, *args):
            if name == 'snappy':
                raise ImportError('test')
            return orig_import(name, *args)

        with mock.patch('builtins.__import__', side_effect=import_mock):
            with self.assertWarns(ImportWarning) as w:
                import dataprofiler
                reload_data_profiler()

        self.assertEqual(
            str(w.warning),
            'Snappy must be installed to use parquet/avro datasets.'
            '\n\n'
            'For macOS use Homebrew:\n'
            '\t`brew install snappy`'
            '\n\n'
            'For linux use apt-get:\n`'
            '\tsudo apt-get -y install libsnappy-dev`\n',
        )

    def test_no_tensorflow(self):
        import sys
        import pandas
        orig_import = __import__
        # necessary for any wrapper around the library to test if snappy caught
        # as an issue

        def import_mock(name, *args):
            if name == 'tensorflow':
                raise ImportError('test')
            return orig_import(name, *args)

        with mock.patch('builtins.__import__', side_effect=import_mock):

            with self.assertWarnsRegex(RuntimeWarning,
                                       "Partial Profiler Failure"):
                modules_with_tf = [
                    'dataprofiler.labelers.character_level_cnn_model',
                ]
                for module in modules_with_tf:
                    if module in sys.modules:
                        del sys.modules[module]
                df = pandas.DataFrame([[1, 2.0], [1, 2.2], [-1, 3]])
                profile = Profiler(df)


if __name__ == '__main__':
    unittest.main()
