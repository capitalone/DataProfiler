from dataprofiler.profilers.profiler_options import TextProfilerOptions, \
    BooleanOption
from dataprofiler.tests.profilers.profiler_options.test_base_inspector_options \
     import TestBaseInspectorOptions


class TestTextProfilerOptions(TestBaseInspectorOptions):

    option_class = TextProfilerOptions
    keys = []

    def test_init(self):
        option = self.get_options()
        self.assertTrue(option.is_enabled)
        self.assertTrue(option.is_case_sensitive)
        self.assertIsNone(option.stop_words)
        self.assertTrue(option.words.is_enabled)
        self.assertTrue(option.vocab.is_enabled)

    def test_set_helper(self):
        option = self.get_options()

        # validate, variable path being passed
        expected_error = ("type object 'test.is_case_sensitive' has no "
                          "attribute 'is_enabled'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({'is_case_sensitive.is_enabled': True}, 'test')

        expected_error = ("type object 'test.stop_words' has no "
                          "attribute 'is_enabled'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({'stop_words.is_enabled': True}, 'test')

        expected_error = ("type object 'test.words.is_enabled' has no attribute "
                          "'other_props'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({'words.is_enabled.other_props': True}, 'test')

        expected_error = ("type object 'test.vocab.is_enabled' has no attribute "
                          "'other_props'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({'vocab.is_enabled.other_props': True}, 'test')

    def test_set(self):
        option = self.get_options()

        params_to_check = [
            dict(prop='is_enabled', value_list=[False, True]),
            dict(prop='is_case_sensitive', value_list=[False, True]),
            dict(prop='stop_words',
                 value_list=[None, ['word1', 'word2'], []]),
            dict(prop='words', value_list=[False, True]),
            dict(prop='vocab', value_list=[False, True]),
        ]

        # this code can be abstracted to limit code everywhere else
        # AKA, params_to_check would be the only needed code plus raise errors
        def _assert_set_helper(prop, value):
            if prop not in ['words', 'vocab']:
                option.set({prop: value})
                self.assertEqual(value, getattr(option, prop), msg=prop)
            else:
                prop_enable = '{}.is_enabled'.format(prop)
                option.set({prop_enable: value})
                self.assertEqual(value, option.properties[prop].is_enabled,
                                 msg=prop)

        for params in params_to_check:
            prop, value_list = params['prop'], params['value_list']
            for value in value_list:
                _assert_set_helper(prop, value)

        # Treat is_case_sensitive and stop_words as BooleanOption
        expected_error = ("type object 'is_case_sensitive' has no attribute "
                          "'is_enabled'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'is_case_sensitive.is_enabled': True})

        expected_error = ("type object 'stop_words' has no attribute "
                          "'is_enabled'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'stop_words.is_enabled': True})

        expected_error = ("type object 'words.is_enabled' has no attribute "
                          "'other_props'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'words.is_enabled.other_props': True})

        expected_error = ("type object 'vocab.is_enabled' has no attribute "
                          "'other_props'")
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'vocab.is_enabled.other_props': True})

    def test_validate_helper(self):
        super(TestTextProfilerOptions, self).test_validate_helper()

    def test_validate(self):

        super(TestTextProfilerOptions, self).test_validate()

        params_to_check = [
            # non errors
            dict(prop='is_enabled', value_list=[False, True], errors=[]),
            dict(prop='is_case_sensitive', value_list=[False, True], errors=[]),
            dict(prop='stop_words',
                 value_list=[None, ['word1', 'word2'], []],
                 errors=[]),
            dict(prop='words',
                 value_list=[BooleanOption(is_enabled=False),
                             BooleanOption(is_enabled=True)],
                 errors=[]),
            dict(prop='vocab',
                 value_list=[BooleanOption(is_enabled=False),
                             BooleanOption(is_enabled=True)],
                 errors=[]),

            # errors
            dict(prop='is_case_sensitive',
                 value_list=[2, 'string'],
                 errors=[
                     "TextProfilerOptions.is_case_sensitive must "
                     "be a Boolean."
                 ]),
            dict(prop='stop_words',
                 value_list=[2, 'a', [1, 2], ['a', 1, 'a']],
                 errors=[
                     "TextProfilerOptions.stop_words must be None "
                     "or list of strings."
                 ]),
            dict(prop='words',
                 value_list=[2, True],
                 errors=[
                     "TextProfilerOptions.words must be a "
                     "BooleanOption object."
                 ]),
            dict(prop='vocab',
                 value_list=[2, True],
                 errors=[
                     "TextProfilerOptions.vocab must be a "
                     "BooleanOption object."
                 ]),
        ]

        # Default configuration is valid
        option = self.get_options()
        self.assertIsNone(option.validate(raise_error=False))

        # # this code can be abstracted to limit code everywhere else
        # # AKA, for loop below could be abstracted to a utils func
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
        option = self.get_options()
        option.stop_words = 'fake word'
        expected_error = (
            "TextProfilerOptions.stop_words must be None "
            "or list of strings.")
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()
