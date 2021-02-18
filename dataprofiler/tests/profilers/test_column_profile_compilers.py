from __future__ import print_function

import six
import unittest
from unittest import mock

from dataprofiler.profilers import column_profile_compilers as \
    col_pro_compilers


class TestBaseColumnProfileCompilerClass(unittest.TestCase):

    def test_cannot_instantiate(self):
        """showing we normally can't instantiate an abstract class"""
        with self.assertRaises(TypeError) as e:
            col_pro_compilers.BaseColumnProfileCompiler()
        self.assertEqual(
            "Can't instantiate abstract class BaseColumnProfileCompiler with "
            "abstract methods profile",
            str(e.exception)
        )

    @mock.patch.multiple(
        col_pro_compilers.BaseColumnProfileCompiler, __abstractmethods__=set(),
        _profilers=[mock.Mock()])
    @mock.patch.multiple(
        col_pro_compilers.ColumnStatsProfileCompiler, _profilers=[mock.Mock()])
    def test_add_profilers(self):
        compiler1 = col_pro_compilers.BaseColumnProfileCompiler(mock.Mock())
        compiler2 = col_pro_compilers.BaseColumnProfileCompiler(mock.Mock())

        # test incorrect type
        with self.assertRaisesRegex(TypeError,
                                    '`BaseColumnProfileCompiler` and `int` are '
                                    'not of the same profile compiler type.'):
            compiler1 + 3

        compiler3 = col_pro_compilers.ColumnStatsProfileCompiler(mock.Mock())
        compiler3._profiles = [mock.Mock()]
        with self.assertRaisesRegex(TypeError,
                                    '`BaseColumnProfileCompiler` and '
                                    '`ColumnStatsProfileCompiler` are '
                                    'not of the same profile compiler type.'):
            compiler1 + compiler3

        # test mismatched names
        compiler1.name = 'compiler1'
        compiler2.name = 'compiler2'
        with self.assertRaisesRegex(ValueError,
                                    'Column profile names are unmatched: '
                                    'compiler1 != compiler2'):
            compiler1 + compiler2

        # test mismatched profiles due to options
        compiler2.name = 'compiler1'
        compiler1._profiles = dict(test1=mock.Mock())
        compiler2._profiles = dict(test2=mock.Mock())
        with self.assertRaisesRegex(ValueError,
                                    'Column profilers were not setup with the '
                                    'same options, hence they do not calculate '
                                    'the same profiles and cannot be added '
                                    'together.'):
            compiler1 + compiler2

        # test success
        compiler1._profiles = dict(test=1)
        compiler2._profiles = dict(test=2)
        merged_compiler = compiler1 + compiler2
        self.assertEqual(3, merged_compiler._profiles['test'])
        self.assertEqual('compiler1', merged_compiler.name)

    @mock.patch.multiple(
        col_pro_compilers.BaseColumnProfileCompiler, __abstractmethods__=set())
    def test_no_profilers_error(self):
        with self.assertRaises(NotImplementedError) as e:
            col_pro_compilers.BaseColumnProfileCompiler(None)
        self.assertEqual("Must add profilers.", str(e.exception))

    def test_update_match_are_abstract(self):
        six.assertCountEqual(
            self,
            {'profile'},
            col_pro_compilers.BaseColumnProfileCompiler.__abstractmethods__
        )


if __name__ == '__main__':
    unittest.main()
