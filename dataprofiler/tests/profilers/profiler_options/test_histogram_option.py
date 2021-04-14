import re

from dataprofiler.profilers.profiler_options import HistogramOption

from .test_boolean_option import TestBooleanOption


class TestHistogramOption(TestBooleanOption):

    option_class = HistogramOption
    keys = []

    def test_init(self):
        option = self.get_options()
        self.assertTrue(option.is_enabled)
        self.assertEqual(option.method, ['auto'])

    def test_set_helper(self):
        option = self.get_options()

        # validate, variable path being passed
        expected_error = ("type object 'test.hist_bin_count' has no attribute "
                          "'is_enabled'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({'hist_bin_count.is_enabled': True}, 'test')

    def test_set(self):
        option = self.get_options()

        params_to_check = [
            dict(prop='is_enabled', value_list=[False, True]),
            dict(prop='method',
                 value_list=['auto', 'fd', 'doane', 'scott', 'rice', 'sturges',
                             'sqrt', ['sturges', 'doane']]),
            dict(prop='hist_bin_count',
                 value_list=['1', '10', '100', '1000', '99', '10000000'])
        ]

        # this code can be abstracted to limit code everywhere else
        # AKA, params_to_check would be the only needed code plus raise errors
        def _assert_set_helper(prop, value):
            option.set({prop: value})
            self.assertEqual(value, getattr(option, prop), msg=prop)

        for params in params_to_check:
            prop, value_list = params['prop'], params['value_list']
            for value in value_list:
                _assert_set_helper(prop, value)

        # Treat method as a BooleanOption
        expected_error = "type object 'method' has no attribute 'is_enabled'"
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'method.is_enabled': True})

        # Treat hist_bin_count as a BooleanOption
        expected_error = ("type object 'hist_bin_count' has no attribute "
                          "'is_enabled'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'hist_bin_count.is_enabled': True})
    
    def test_validate_helper(self):
        super(TestHistogramOption, self).test_validate_helper()

    def test_validate(self):

        super(TestHistogramOption, self).test_validate()

        params_to_check = [
            # non errors
            dict(prop='is_enabled', value_list=[False, True], errors=[]),
            dict(prop='method',
                 value_list=['auto', 'fd', 'doane', 'scott', 'rice', 'sturges',
                             'sqrt', ['sturges', 'doane']],
                 errors=[]),
            dict(prop='hist_bin_count',
                 value_list=[1, 10, 100, 1000, 99, 10000000],
                 errors=[]),

            # errors
            dict(prop='method', value_list=[1, [], False],
                 errors=[
                     "HistogramOption.method must be a string or list of "
                     "strings from the following: ['auto', 'fd', 'doane', "
                     "'scott', 'rice', 'sturges', 'sqrt']."
                 ]),
            dict(prop='method', value_list=['whoops', ["doane", "incorrect"]],
                 errors=[
                     "HistogramOption.method must be a subset or selection "
                     "from: ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges',"
                     " 'sqrt']."
                 ]),
            dict(prop='hist_bin_count', value_list=[-1, False, [], 1.2, 1.0],
                 errors=[
                     "HistogramOption.hist_bin_count must be an integer more "
                     "than 0."
                 ]),
        ]

        # # this code can be abstracted to limit code everywhere else
        # # AKA, params_to_check would be the only needed code
        # def _assert_validate_helper(prop, value):
        #

        # Default configuration is valid
        option = self.get_options()
        self.assertIsNone(option.validate(raise_error=False))

        for params in params_to_check:
            prop, value_list, expected_errors = (
                params['prop'], params['value_list'], params['errors']
            )
            option = self.get_options()
            for value in value_list:
                setattr(option, prop, value)
                validate_errors = option.validate(raise_error=False)
                if expected_errors:
                    self.assertListEqual(
                        expected_errors,
                        validate_errors,
                        msg='Errored for prop: {}, value: {}.'.format(prop,
                                                                      value))
                else:
                    self.assertIsNone(validate_errors)

        # test for invalidity between mismatched histogram method / bins
        option = self.get_options()
        with self.assertWarnsRegex(UserWarning,
                                   r'Setting HistogramOption.hist_bin_count '
                                   r'overrides HistogramOption.method. '
                                   r'HistogramOption.method will be ignored.'):
            option.hist_bin_count = 10
            option.method = 'auto'
            option.validate(raise_error=False)

        option = self.get_options()
        option.hist_bin_count = None
        option.method = None
        expected_errors = [r"Either HistogramOption.hist_bin_count or "
                           r"HistogramOption.method must be set to compute "
                           r"histograms."]
        errors = option.validate(raise_error=False)
        self.assertListEqual(expected_errors, errors)

        # this time testing raising an error
        with self.assertRaisesRegex(ValueError, expected_errors[0]):
            option.validate()
