import re

from dataprofiler.profilers.profiler_options import HistogramOption

from .test_boolean_option import TestBooleanOption


class TestHistogramOption(TestBooleanOption):

    option_class = HistogramOption
    keys = []

    def test_init(self):
        option = self.get_options()
        self.assertTrue(option.is_enabled)
        self.assertEqual(option.bin_count_or_method, 'auto')

    def test_set_helper(self):
        option = self.get_options()

        # validate, variable path being passed
        expected_error = ("type object 'test.bin_count_or_method' has no "
                          "attribute 'is_enabled'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({'bin_count_or_method.is_enabled': True}, 'test')

    def test_set(self):
        option = self.get_options()

        params_to_check = [
            dict(prop='is_enabled', value_list=[False, True]),
            dict(prop='bin_count_or_method',
                 value_list=[None, 'auto', 'fd', 'doane', 'scott', 'rice',
                             'sturges', 'sqrt', ['sturges', 'doane'], 1,
                             10, 100, 1000, 99, 10000000])
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

        # Treat bin_count_or_method as a BooleanOption
        expected_error = ("type object 'bin_count_or_method' has no attribute "
                          "'is_enabled'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'bin_count_or_method.is_enabled': True})

    def test_validate_helper(self):
        super(TestHistogramOption, self).test_validate_helper()

    def test_validate(self):

        super(TestHistogramOption, self).test_validate()

        params_to_check = [
            # non errors
            dict(prop='is_enabled', value_list=[False, True], errors=[]),
            dict(prop='bin_count_or_method',
                 value_list=['auto', 'fd', 'doane', 'scott', 'rice', 'sturges',
                             'sqrt', ['sturges', 'doane'], 1, 10, 100, 1000,
                             99, 10000000],
                 errors=[]),

            # errors
            dict(prop='bin_count_or_method',
                 value_list=[-1, 1.2, 1.0, [], False, 'whoops', ["doane", "incorrect"], '1'],
                 errors=[
                     "HistogramOption.bin_count_or_method must be an integer "
                     "more than 1, a string, or list of strings from the "
                     "following: ['auto', 'fd', 'doane', 'scott', 'rice', "
                     "'sturges', 'sqrt']."
                 ]),
        ]

        # # this code can be abstracted to limit code everywhere else
        # # AKA, for loop below could be abstracted to a utils func

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
                    self.assertIsNone(
                        validate_errors,
                        msg='Errored for prop: {}, value: {}.'.format(prop,
                                                                      value))

        # this time testing raising an error
        option.bin_count_or_method = 'fake method'
        expected_error = (
            r"HistogramOption.bin_count_or_method must be an integer more than "
            r"1, a string, or list of strings from the following: "
            r"\['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'].")
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()
