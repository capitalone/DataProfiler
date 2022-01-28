import os
import unittest
from unittest import mock
import random
import pkg_resources
import json
from io import StringIO
import re

import numpy as np

from dataprofiler.labelers.data_processing import \
    BaseDataProcessor, CharPreprocessor, CharPostprocessor, \
    StructCharPreprocessor, StructCharPostprocessor, \
    DirectPassPreprocessor, RegexPostProcessor, StructRegexPostProcessor


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


test_parameters = {
    'test_a': 0,
    'test_b': ' ',
}

preprocessor_parameters = {
    'flatten_split': 0,
    'flatten_separator': ' ',
    'is_separate_at_max_len': True,

}

postprocessor_parameters = {
    'use_word_level_argmax': True,
    'output_format': 'character_argmax',
    'separators': (' ', ',', ';', "'", '"', ':', '\n', '\t', "."),
    'word_level_min_percent': 0.75,
}

struct_preprocessor_params = {
    'max_length': 4,
    'max_num_chars': 100,
    'oov_token':  'load_test',
    'sample_encoder': {
        "word_index": {"0": 2, "1": 3, "2": 4},
        "oov_token": "UNK",
        "word_counts": {"0": 260712, "1": 229990,"2": 185196},
        "num_words": 200,
        "index_word": {"1": "UNK", "2": "0", "3": "1", "4": "2"},
        "split": " ",
        "document_count": 283734,
        "lower": True,
        "index_docs": {"2": 111909},
        "filters": "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
        "char_level": True,
        "word_docs": {"0": 111909, "1": 120239, "2": 111142}
    }
}

struct_postprocessor_params = {}


def mock_open_func(filename, *args):
    if filename.find('preprocessor_parameters') >= 0:
        return StringIO(json.dumps(preprocessor_parameters))
    elif filename.find('postprocessor_parameters') >= 0:
        return StringIO(json.dumps(postprocessor_parameters))
    elif filename.find('test_parameters') >= 0:
        return StringIO(json.dumps(test_parameters))


def mock_open_struct_process(filename, *args):
    if filename.find('default_pre_fail') >= 0:
        return StringIO(json.dumps({}))
    elif filename.find('preprocessor_parameters') >= 0:
        return StringIO(json.dumps(struct_preprocessor_params))
    elif filename.find('postprocessor_parameters') >= 0:
        return StringIO(json.dumps(struct_postprocessor_params))


def setup_save_mock_open(mock_open):
    mock_file = StringIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file


@mock.patch('dataprofiler.labelers.data_processing.BaseDataProcessor.'
            '__abstractmethods__', set())
class TestBaseDataProcessor(unittest.TestCase):

    @staticmethod
    def mock_validate_parameters():
        # mock validate parameters, since none exist
        BaseDataProcessor._validate_parameters = mock.Mock()
        BaseDataProcessor._validate_parameters.return_value = None

    @mock.patch('dataprofiler.labelers.data_processing.BaseDataProcessor.'
                '_BaseDataProcessor__subclasses',
                new_callable=mock.PropertyMock)
    def test_register_subclass(self, mock_subclasses, *mocks):
        # remove not implemented func
        self.mock_validate_parameters()

        base_processor = BaseDataProcessor()
        base_processor._register_subclass()
        self.assertIn(
            mock.call().__setitem__('basedataprocessor', BaseDataProcessor),
            mock_subclasses.mock_calls)

    @mock.patch('dataprofiler.labelers.data_processing.BaseDataProcessor.'
                '_BaseDataProcessor__subclasses',
                new_callable=mock.PropertyMock)
    def test_get_class(self, mock_subclasses, *mocks):
        # remove not implemented func
        self.mock_validate_parameters()

        # setup mock return value
        mock_subclasses.return_value = dict(basedataprocessor=BaseDataProcessor)

        base_class = BaseDataProcessor.get_class('BaseDataProcessor')
        self.assertEqual(BaseDataProcessor, base_class)

    @mock.patch('dataprofiler.labelers.data_processing.BaseDataProcessor.'
                '__abstractmethods__', set())
    @mock.patch('dataprofiler.labelers.data_processing.BaseDataProcessor.'
                '_validate_parameters', return_value=None)
    def test_equality_checks(self, *mocks):

        FakeProcessor1 = type('FakeProcessor1', (BaseDataProcessor,), {})
        FakeProcessor2 = type('FakeProcessor2', (BaseDataProcessor,), {})

        fake_processor1 = FakeProcessor1(test=1)
        fake_processor1_1 = FakeProcessor1(test=1)
        fake_processor1_2 = FakeProcessor1(test=2)
        fake_processor2 = FakeProcessor2(test=1)

        # assert True if the same object
        self.assertEqual(fake_processor1, fake_processor1)

        # assert True if same class but same params
        self.assertEqual(fake_processor1, fake_processor1_1)

        # assert False if diff class even if same params
        self.assertNotEqual(fake_processor1, fake_processor2)

        # assert False if same class even diff params
        self.assertNotEqual(fake_processor1, fake_processor1_2)

    def test_set_parameters(self, *mocks):
        # patch validate
        self.mock_validate_parameters()

        # validate params set successfully
        params = {'test': 1}
        base_processor = BaseDataProcessor()
        base_processor.set_params(**params)

        self.assertDictEqual(params, base_processor._parameters)

        # test overwrite params
        params = {'test': 2}
        base_processor.set_params(**params)
        self.assertDictEqual(params, base_processor._parameters)

        # test invalid params
        base_processor._validate_parameters.side_effect = ValueError('test')
        with self.assertRaisesRegex(ValueError, 'test'):
            base_processor.set_params(**params)

    @mock.patch("builtins.open", side_effect=mock_open_func)
    def test_load_processor(self, mock_open, *property_mocks):
        # patch validate
        self.mock_validate_parameters()

        # call load processor func
        with mock.patch('dataprofiler.labelers.data_processing.'
                        'BaseDataProcessor.processor_type',
                        new_callable=mock.PropertyMock(return_value='test')):
            mocked_processor = BaseDataProcessor.load_from_disk('test/path')

        # assert internal functions called and validated with loaded parameters
        mocked_processor._validate_parameters.assert_called()
        mocked_processor._validate_parameters.assert_called_with(
            dict(test_a=0, test_b=' '))

        # assert attributes properly loaded/assigned in class.
        self.assertTrue(hasattr(mocked_processor, '_parameters'),
                        msg='Mocked object was not assigned attribute '
                            '`_parameters` in _load_processor.')
        self.assertEqual(dict(test_a=0, test_b=' '),
                         mocked_processor._parameters)

    @mock.patch('dataprofiler.labelers.data_processing.BaseDataProcessor.'
                'load_from_disk')
    def test_load_from_library(self, mocked_load, *mocks):
        # patch validate
        self.mock_validate_parameters()

        # call func
        BaseDataProcessor.load_from_library('default')

        # assert called with proper load_processor dirpath
        default_labeler_dir = pkg_resources.resource_filename(
            'resources', 'labelers'
        )
        mocked_load.assert_called_with(os.path.join(default_labeler_dir,
                                                    'default'))

    @mock.patch("builtins.open")
    def test_save_processor(self, mock_open, *mocks):
        # setup mocks
        mock_file = setup_save_mock_open(mock_open)

        # setup mocked class
        mocked_processor = mock.create_autospec(BaseDataProcessor)
        mocked_processor.processor_type = 'test'
        mocked_processor.get_parameters.return_value = {'test': 1}

        # call save processor func
        BaseDataProcessor._save_processor(mocked_processor, 'test')

        # assert parameters saved
        mock_open.assert_called_with('test/test_parameters.json', 'w')
        self.assertEqual('{"test": 1}', mock_file.getvalue())

        # close mocks
        StringIO.close(mock_file)

    @mock.patch('dataprofiler.labelers.data_processing.BaseDataProcessor.'
                '_save_processor')
    def test_save_to_disk(self, mocked_save, *mocks):
        # patch validate
        self.mock_validate_parameters()

        # call func
        base_processor = BaseDataProcessor()
        base_processor.save_to_disk('test/path')

        # assert _save_processor called with proper dirpath
        mocked_save.assert_called_with('test/path')


class TestDirectPassPreprocessor(unittest.TestCase):

    def test_registered_subclass(self):
        self.assertEqual(
            DirectPassPreprocessor,
            BaseDataProcessor.get_class(DirectPassPreprocessor.__name__))

    def test_validate_parameters(self):

        mock_processor = mock.Mock(spec=DirectPassPreprocessor)

        # test with no parameters, success
        DirectPassPreprocessor._validate_parameters(
            mock_processor, parameters={})

        # test with parameters, fail
        with self.assertRaisesRegex(
                ValueError, '`DirectPassPreprocessor` has no parameters.'):
            DirectPassPreprocessor._validate_parameters(
                mock_processor, parameters={'test': 'fail'})

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        DirectPassPreprocessor.help()
        self.assertIn("Parameters", mock_stdout.getvalue())
        self.assertIn("Input Format", mock_stdout.getvalue())

    def test_get_parameters(self):

        # test no params
        processor = DirectPassPreprocessor()
        self.assertDictEqual({}, processor.get_parameters())

    def test_process(self):
        preprocessor = DirectPassPreprocessor()

        data = np.array(['test'])
        labels = np.array([1, 1, 1, 1])

        # test w/o labels
        output = preprocessor.process(data)
        self.assertEqual(data, output)

        # test w/ labels
        output, output_labels = preprocessor.process(data, labels)
        self.assertEqual(data, output)
        self.assertTrue(np.array_equal(labels, output_labels))


class TestCharPreprocessor(unittest.TestCase):

    def test_registered_subclass(self):
        self.assertEqual(
            CharPreprocessor,
            BaseDataProcessor.get_class(CharPreprocessor.__name__))

    def test_validate_parameters(self):

        def test_raises(error_msg, flatten_split=0, flatten_separator='',
                        is_separate_at_max_len=True):
            with self.assertRaises(ValueError) as e:
                CharPreprocessor._validate_parameters(
                    mock_processor,
                    dict(flatten_split=flatten_split,
                         flatten_separator=flatten_separator,
                         is_separate_at_max_len=is_separate_at_max_len))
            self.assertEqual(error_msg, str(e.exception))

        def test_success(flatten_split=0, flatten_separator='',
                         is_separate_at_max_len=True):
            try:
                CharPreprocessor._validate_parameters(
                    mock_processor,
                    dict(flatten_split=flatten_split,
                         flatten_separator=flatten_separator,
                         is_separate_at_max_len=is_separate_at_max_len))
            except Exception as e:
                self.fail(str(e))

        mock_processor = mock.Mock(spec=CharPreprocessor)

        flatten_error_msg = '`flatten_split` must be a float or int >= 0 and ' \
                            '<= 1'
        separator_error_msg = '`flatten_separator` must be a str'
        at_max_len_error_msg = '`is_separate_at_max_len` must be a bool'
        test_cases = [
            # flatten_split test cases
            dict(params=dict(flatten_split=None), error_msg=flatten_error_msg),
            dict(params=dict(flatten_split=''), error_msg=flatten_error_msg),
            dict(params=dict(flatten_split=BaseDataProcessor),
                 error_msg=flatten_error_msg),
            dict(params=dict(flatten_split=-.1), error_msg=flatten_error_msg),
            dict(params=dict(flatten_split=1.1), error_msg=flatten_error_msg),
            dict(params=dict(flatten_split=float('nan')),
                 error_msg=flatten_error_msg),

            # no exception
            dict(params=dict(flatten_split=0), error_msg=None),
            dict(params=dict(flatten_split=1), error_msg=None),
            dict(params=dict(flatten_split=0.5), error_msg=None),
            dict(params=dict(flatten_split=1/3), error_msg=None),

            # flatten_separator test cases
            dict(params=dict(flatten_separator=None),
                 error_msg=separator_error_msg),
            dict(params=dict(flatten_separator=1),
                 error_msg=separator_error_msg),
            dict(params=dict(flatten_separator=BaseDataProcessor),
                 error_msg=separator_error_msg),

            # no exception
            dict(params=dict(flatten_separator=''), error_msg=None),
            dict(params=dict(flatten_separator=' '), error_msg=None),
            dict(params=dict(flatten_separator='abcdefghi'), error_msg=None),

            # is_separate_at_max_len test cases
            dict(params=dict(is_separate_at_max_len=None),
                 error_msg=at_max_len_error_msg),
            dict(params=dict(is_separate_at_max_len=1),
                 error_msg=at_max_len_error_msg),
            dict(params=dict(is_separate_at_max_len=BaseDataProcessor),
                 error_msg=at_max_len_error_msg),

            # no exception
            dict(params=dict(is_separate_at_max_len=False), error_msg=None),
            dict(params=dict(is_separate_at_max_len=True), error_msg=None),

            # combination error test cases
            dict(params=dict(flatten_split=None, flatten_separator=None),
                 error_msg='\n'.join([flatten_error_msg, separator_error_msg])),
            dict(params=dict(flatten_split=None, is_separate_at_max_len=None),
                 error_msg='\n'.join([flatten_error_msg,
                                      at_max_len_error_msg])),
            dict(params=dict(flatten_separator=None,
                             is_separate_at_max_len=None),
                 error_msg='\n'.join([separator_error_msg,
                                      at_max_len_error_msg])),
            dict(params=dict(flatten_split=None, flatten_separator=None,
                             is_separate_at_max_len=None),
                 error_msg='\n'.join([flatten_error_msg, separator_error_msg,
                                      at_max_len_error_msg])),
        ]

        for test_case in test_cases:
            if test_case['error_msg'] is None:
                test_success(**test_case['params'])
            else:
                test_raises(test_case['error_msg'], **test_case['params'])

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        CharPreprocessor.help()
        self.assertIn("Parameters", mock_stdout.getvalue())
        self.assertIn("Input Format", mock_stdout.getvalue())

    def test_get_parameters(self):

        # test default params
        processor = CharPreprocessor()
        self.assertDictEqual(dict(max_length=3400,
                                  default_label='UNKNOWN',
                                  pad_label='PAD',
                                  flatten_split=0,
                                  flatten_separator=' ',
                                  is_separate_at_max_len=False),
                             processor.get_parameters())

        # test set params
        params = dict(max_length=10,
                      default_label='test default',
                      pad_label='test pad',
                      flatten_split=1,
                      flatten_separator='test',
                      is_separate_at_max_len=True)
        processor = CharPreprocessor(**params)
        self.assertDictEqual(params, processor.get_parameters())

        # test subset set params
        params = dict(max_length=10,
                      default_label='test default',
                      pad_label='test pad',
                      flatten_split=1,
                      flatten_separator='test',
                      is_separate_at_max_len=True)
        processor = CharPreprocessor(**params)
        self.assertDictEqual(
            dict(max_length=10, default_label='test default'),
            processor.get_parameters(['max_length', 'default_label']))

    def test_find_nearest_sentence_break_before_ind(self):
        preprocessor = CharPreprocessor
        test_sentence = 'this is my test sentence. How nice.'

        start_ind = len(test_sentence) - 1
        break_ind = preprocessor._find_nearest_sentence_break_before_ind(
            test_sentence, start_ind=start_ind, min_ind=-1, separators=(' ',)
        )
        self.assertEqual(29, break_ind)

        start_ind = 28
        break_ind = preprocessor._find_nearest_sentence_break_before_ind(
            test_sentence, start_ind=start_ind, min_ind=-1, separators=(' ',)
        )
        self.assertEqual(25, break_ind)

        start_ind = len(test_sentence) - 1
        break_ind = preprocessor._find_nearest_sentence_break_before_ind(
            test_sentence, start_ind=start_ind, min_ind=-1,
            separators=('f', ' ')
        )
        self.assertEqual(29, break_ind)

        start_ind = len(test_sentence) - 1
        break_ind = preprocessor._find_nearest_sentence_break_before_ind(
            test_sentence, start_ind=start_ind, min_ind=-1,
            separators=('does not exits',)
        )
        self.assertEqual(-1, break_ind)

    def test_process_batch_helper(self):
        preprocessor = CharPreprocessor()

        label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2,
            "TEST2": 3,
            "TEST3": 4,
        }

        # test max_len < separator_len
        test_sentences = np.array(['test'])
        with self.assertRaisesRegex(ValueError,
                                    'The `flatten_separator` length cannot be '
                                    'more than or equal to the `max_length`'):
            process_generator = preprocessor._process_batch_helper(
                test_sentences, max_length=0, default_label='UNKNOWN',
                pad_label='PAD', label_mapping=label_mapping, batch_size=2)
            next(process_generator)

        # test a single sentence
        test_sentences = np.array(['this is my test sentence. How nice.'])
        expected_output = [
            dict(samples=['this', ' is']),
            dict(samples=[' my', ' test']),
            dict(samples=[' sent', 'ence.']),
            dict(samples=[' How', ' nice']),
            dict(samples=['.']),
        ]

        process_generator = preprocessor._process_batch_helper(
            test_sentences, max_length=5, default_label='UNKNOWN',
            pad_label='PAD', label_mapping=label_mapping, batch_size=2)

        process_output = [data for data in process_generator]
        self.assertListEqual(expected_output, process_output)

        # test multiple sentences, notice the difference in sample 4.
        # ' How'  -> 'How'
        test_sentences = np.array(['this is my test sentence.', 'How nice.'])
        expected_output = [
            dict(samples=['this', ' is']),
            dict(samples=[' my', ' test']),
            dict(samples=[' sent', 'ence.']),
            dict(samples=['How', ' nice']),
            dict(samples=['.']),
        ]

        process_generator = preprocessor._process_batch_helper(
            test_sentences, max_length=5, default_label='UNKNOWN',
            pad_label='PAD', label_mapping=label_mapping, batch_size=2)

        process_output = [data for data in process_generator]
        self.assertListEqual(expected_output, process_output)

        # test with label assignment
        # test a single sentence
        test_sentences = np.array(['this is my test sentence. How nice.'])
        labels = [
            [
                [5, 7, 'TEST1'],
                [11, 24, 'TEST2'],
                [26, 29, 'TEST1'],
                [30, 34, 'TEST2']
            ]
        ]
        expected_output = [
            dict(samples=['this', ' is'],
                 labels=[[1, 1, 1, 1, 0], [1, 2, 2, 0, 0]]),
            dict(samples=[' my', ' test'],
                 labels=[[1, 1, 1, 0, 0], [1, 3, 3, 3, 3]]),
            dict(samples=[' sent', 'ence.'],
                 labels=[[3, 3, 3, 3, 3], [3, 3, 3, 3, 1]]),
            dict(samples=[' How', ' nice'],
                 labels=[[1, 2, 2, 2, 0], [1, 3, 3, 3, 3]]),
            dict(samples=['.'], labels=[[1, 0, 0, 0, 0]]),
        ]

        process_generator = preprocessor._process_batch_helper(
            test_sentences, max_length=5, default_label='UNKNOWN',
            pad_label='PAD', labels=labels, label_mapping=label_mapping,
            batch_size=2)

        process_output = [data for data in process_generator]
        self.assertListEqual(expected_output, process_output)

        # test multiple sentences, notice the difference in sample 4.
        # ' How'  -> 'How'
        test_sentences = np.array(['this is my test sentence.', 'How nice.'])
        labels = [
            [[5, 7, 'TEST1'], [11, 24, 'TEST2']],
            [[0, 3, 'TEST1'], [4, 8, 'TEST2']]
        ]
        expected_output = [
            dict(samples=['this', ' is'],
                 labels=[[1, 1, 1, 1, 0], [1, 2, 2, 0, 0]]),
            dict(samples=[' my', ' test'],
                 labels=[[1, 1, 1, 0, 0], [1, 3, 3, 3, 3]]),
            dict(samples=[' sent', 'ence.'],
                 labels=[[3, 3, 3, 3, 3], [3, 3, 3, 3, 1]]),
            dict(samples=['How', ' nice'],
                 labels=[[2, 2, 2, 0, 0], [1, 3, 3, 3, 3]]),
            dict(samples=['.'], labels=[[1, 0, 0, 0, 0]]),
        ]

        process_generator = preprocessor._process_batch_helper(
            test_sentences, max_length=5, default_label='UNKNOWN',
            pad_label='PAD', labels=labels, label_mapping=label_mapping,
            batch_size=2)

        process_output = [data for data in process_generator]
        self.assertListEqual(expected_output, process_output)

    def test_process(self):
        preprocessor = CharPreprocessor(
            max_length=5, default_label='UNKNOWN', pad_label='PAD',)

        label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2,
            "TEST2": 3,
            "TEST3": 4,
        }

        # test max_len < separator_len
        test_sentences = np.array(['test'])
        preprocessor._parameters['max_length'] = 0
        with self.assertRaisesRegex(ValueError,
                                    'The `flatten_separator` length cannot be '
                                    'more than or equal to the `max_length`'):
            process_generator = preprocessor.process(
                test_sentences, label_mapping=label_mapping, batch_size=2)
            next(process_generator)

        # test labels, no label_mapping
        preprocessor._parameters['max_length'] = 10
        with self.assertRaisesRegex(ValueError,
                                    'If `labels` are specified, `label_mapping`'
                                    ' must also be specified'):
            process_generator = preprocessor.process(
                np.array(['test']), labels=['test'], label_mapping=None,
                batch_size=2)
            next(process_generator)

        # test a single sentence
        test_sentences = np.array(['this is my test sentence. How nice.'])
        expected_output = [
            np.array([['this'], [' is']]),
            np.array([[' my'], [' test']]),
            np.array([[' sent'], ['ence.']]),
            np.array([[' How'], [' nice']]),
            np.array([['.']]),
        ]

        # without labels process
        preprocessor._parameters['max_length'] = 5
        process_generator = preprocessor.process(
            test_sentences, label_mapping=label_mapping, batch_size=2)

        process_output = [data for data in process_generator]
        for expected, output in zip(expected_output, process_output):
            self.assertTrue((expected == output).all())

        # with labels process
        test_sentences = np.array(['this is my'])
        labels = [
            [
                [5, 7, 'TEST1'],
                [11, 24, 'TEST2'],
                [26, 29, 'TEST1'],
                [30, 34, 'TEST2']
            ]
        ]
        expected_sentence_output = [
            [['this'], [' is']],
            [[' my']],
        ]
        expected_labels_output = [
            np.array([
                   [[0., 1., 0., 0., 0.],  # this
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 0.]],

                   [[0., 1., 0., 0., 0.],  # is
                    [0., 0., 1., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.]]
            ]),

            np.array([[[0., 1., 0., 0., 0.],  # my
                       [0., 1., 0., 0., 0.],
                       [0., 1., 0., 0., 0.],
                       [1., 0., 0., 0., 0.],
                       [1., 0., 0., 0., 0.]]]),

        ]
        expected_output = tuple(zip(expected_sentence_output,
                                    expected_labels_output))
        process_generator = preprocessor.process(
            test_sentences, labels=labels, label_mapping=label_mapping,
            batch_size=2)

        process_output = [data for data in process_generator]
        for expected, output in zip(expected_output, process_output):
            self.assertIsInstance(output, tuple)
            self.assertTrue((expected[0] == output[0]).all())
            self.assertTrue((expected[1] == output[1]).all())

    def test_process_ending_null_string(self):

        preprocessor = CharPreprocessor(
            max_length=5, default_label='UNKNOWN', pad_label='PAD',
            flatten_split=1.0, is_separate_at_max_len=True,
        )

        label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2,
            "TEST2": 3,
            "TEST3": 4,
        }

        # test a single sentence
        test_sentences = np.array(['this\x00is\x00\x00\x00my test sentence. '
                                   'How nice.\x00\x00\x00'], dtype=object)
        expected_output = [
            np.array([['this\x00'], ['is\x00\x00\x00']], dtype=object),
            np.array([['my te'], ['st se']], dtype=object),
            np.array([['ntenc'], ['e. Ho']], dtype=object),
            np.array([['w nic'], ['e.\x00\x00\x00']], dtype=object),
        ]

        # without labels process
        process_generator = preprocessor.process(
            test_sentences, label_mapping=label_mapping, batch_size=2)

        # check to make sure string length is not stripped because ending in
        # \x00
        process_output = [data for data in process_generator]
        for expected, output_batch in zip(expected_output, process_output):
            for output in output_batch:
                print(output)
                self.assertEqual(5, len(output[0]))  # validates not trimmed
            np.testing.assert_equal(expected, output_batch)

    def test_process_input_checks(self):
        prep = CharPreprocessor()
        multi_dim_msg = re.escape("Multidimensional data given to "
                                  "CharPreprocessor. Consider using a different"
                                  " preprocessor or flattening data (and labels)")
        with self.assertRaisesRegex(ValueError, multi_dim_msg):
            next(prep.process(np.array([["this", "is"],
                                        ["two", "dimensions"]])))
        diff_length_msg = re.escape("Data and labels given to CharPreprocessor "
                                    "are different lengths, 2 != 1")
        with self.assertRaisesRegex(ValueError, diff_length_msg):
            next(prep.process(np.array(["two", "strings"]),
                              np.array([[(0, 1, "UNKNOWN")]],
                                       dtype="object"),
                              {"UNKNOWN": 1}))


class TestCharPostprocessor(unittest.TestCase):

    def test_registered_subclass(self):
        self.assertEqual(
            CharPostprocessor,
            BaseDataProcessor.get_class(CharPostprocessor.__name__))

    def test_validate_parameters(self):

        def test_raises(error_msg, default_label='UNKNOWN', pad_label='PAD',
                        flatten_separator=' ', use_word_level_argmax=True,
                        output_format='ner', separators=('',),
                        word_level_min_percent=0):
            with self.assertRaises(ValueError) as e:
                CharPostprocessor._validate_parameters(
                    mock_processor, dict(
                        default_label=default_label,
                        pad_label=pad_label,
                        flatten_separator=flatten_separator,
                        use_word_level_argmax=use_word_level_argmax,
                        output_format=output_format,
                        separators=separators,
                        word_level_min_percent=word_level_min_percent))
            self.assertEqual(error_msg, str(e.exception))

        def test_success(default_label='UNKNOWN', pad_label='PAD',
                         flatten_separator=' ', use_word_level_argmax=True,
                         output_format='ner', separators=('',),
                         word_level_min_percent=0):
            try:
                CharPostprocessor._validate_parameters(
                    mock_processor, dict(
                        default_label=default_label,
                        pad_label=pad_label,
                        flatten_separator=flatten_separator,
                        use_word_level_argmax=use_word_level_argmax,
                        output_format=output_format,
                        separators=separators,
                        word_level_min_percent=word_level_min_percent))
            except Exception as e:
                self.fail(str(e))

        mock_processor = mock.Mock(spec=CharPostprocessor)

        default_label_error_msg = "`default_label` must be a string."
        pad_label_error_msg = "`pad_label` must be a string."
        flatten_separator_error_msg = "`flatten_separator` must be a string."
        word_level_argmax_error_msg = '`use_word_level_argmax` must be a bool'
        output_format_error_msg = '`output_format` must be a str of value ' \
                                  '`character_argmax` or `ner`'
        separators_error_msg = '`separators` must be a list of str'
        word_min_percent_error_msg = '`word_level_min_percent` must be a float'\
                                     ' or int >= 0 and <= 1'
        test_cases = [
            # default_label test cases
            dict(params=dict(default_label=None),
                 error_msg=default_label_error_msg),
            dict(params=dict(default_label=1),
                 error_msg=default_label_error_msg),
            dict(params=dict(default_label=[]),
                 error_msg=default_label_error_msg),

            dict(params=dict(default_label='test'), error_msg=None),

            # pad_label test cases
            dict(params=dict(pad_label=None),
                 error_msg=pad_label_error_msg),
            dict(params=dict(pad_label=1),
                 error_msg=pad_label_error_msg),
            dict(params=dict(pad_label=[]),
                 error_msg=pad_label_error_msg),

            dict(params=dict(pad_label='test'), error_msg=None),

            # flatten_separator test cases
            dict(params=dict(flatten_separator=None),
                 error_msg=flatten_separator_error_msg),
            dict(params=dict(flatten_separator=1),
                 error_msg=flatten_separator_error_msg),
            dict(params=dict(flatten_separator=[]),
                 error_msg=flatten_separator_error_msg),

            dict(params=dict(flatten_separator='test'), error_msg=None),

            # use_word_level_argmax test cases
            dict(params=dict(use_word_level_argmax=None),
                 error_msg=word_level_argmax_error_msg),
            dict(params=dict(use_word_level_argmax=1),
                 error_msg=word_level_argmax_error_msg),
            dict(params=dict(use_word_level_argmax='error string'),
                 error_msg=word_level_argmax_error_msg),

            dict(params=dict(use_word_level_argmax=False), error_msg=None),
            dict(params=dict(use_word_level_argmax=True), error_msg=None),

            # output_format raise exception
            dict(params=dict(output_format=None),
                 error_msg=output_format_error_msg),
            dict(params=dict(output_format=''),
                 error_msg=output_format_error_msg),
            dict(params=dict(output_format=BaseDataProcessor),
                 error_msg=output_format_error_msg),
            dict(params=dict(output_format=-.1),
                 error_msg=output_format_error_msg),
            dict(params=dict(output_format=1.1),
                 error_msg=output_format_error_msg),
            dict(params=dict(output_format=float('nan')),
                 error_msg=output_format_error_msg),

            # no exception flatten_split
            dict(params=dict(output_format='character_argmax'), error_msg=None),
            dict(params=dict(output_format='NER'), error_msg=None),
            dict(params=dict(output_format='ner'), error_msg=None),

            # separators test cases
            dict(params=dict(separators=None), error_msg=separators_error_msg),
            dict(params=dict(separators=1), error_msg=separators_error_msg),
            dict(params=dict(separators=''),
                 error_msg=separators_error_msg),
            dict(params=dict(separators=BaseDataProcessor),
                 error_msg=separators_error_msg),
            dict(params=dict(separators=(' ', 1)),
                 error_msg=separators_error_msg),

            dict(params=dict(separators=('',)), error_msg=None),
            dict(params=dict(separators=(' ', 'f')), error_msg=None),

            # word_level_min_percent test cases
            dict(params=dict(word_level_min_percent=None),
                 error_msg=word_min_percent_error_msg),
            dict(params=dict(word_level_min_percent=''),
                 error_msg=word_min_percent_error_msg),
            dict(params=dict(word_level_min_percent=BaseDataProcessor),
                 error_msg=word_min_percent_error_msg),
            dict(params=dict(word_level_min_percent=-.1),
                 error_msg=word_min_percent_error_msg),
            dict(params=dict(word_level_min_percent=1.1),
                 error_msg=word_min_percent_error_msg),
            dict(params=dict(word_level_min_percent=float('nan')),
                 error_msg=word_min_percent_error_msg),

            # no exception
            dict(params=dict(word_level_min_percent=0), error_msg=None),
            dict(params=dict(word_level_min_percent=1), error_msg=None),
            dict(params=dict(word_level_min_percent=0.5), error_msg=None),
            dict(params=dict(word_level_min_percent=1 / 3), error_msg=None),

            # combination error test cases
            dict(params=dict(use_word_level_argmax=None, output_format=None,
                             separators=None, word_level_min_percent=None),
                 error_msg='\n'.join([word_level_argmax_error_msg,
                                      output_format_error_msg,
                                      separators_error_msg,
                                      word_min_percent_error_msg])),
        ]

        for test_case in test_cases:
            if test_case['error_msg'] is None:
                test_success(**test_case['params'])
            else:
                test_raises(test_case['error_msg'], **test_case['params'])

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        CharPostprocessor.help()
        self.assertIn("Parameters", mock_stdout.getvalue())
        self.assertIn("Output Format", mock_stdout.getvalue())

    def test_get_parameters(self):

        # test default params
        processor = CharPostprocessor()
        self.assertDictEqual(
            dict(default_label='UNKNOWN',
                 pad_label='PAD',
                 flatten_separator=" ",
                 use_word_level_argmax=False,
                 output_format='character_argmax',
                 separators=(' ', ',', ';', "'", '"', ':', '\n', '\t', "."),
                 word_level_min_percent=0.75),
            processor.get_parameters())

        # test set params
        params = dict(
            default_label='test_default',
            pad_label='test_pad',
            flatten_separator="test_flatten",
            use_word_level_argmax=True,
            output_format='ner',
            separators=(".",),
            word_level_min_percent=0)
        processor = CharPostprocessor(**params)
        self.assertDictEqual(params, processor.get_parameters())

        # test get subset params
        params = dict(
            default_label='test_default',
            pad_label='test_pad',
            flatten_separator="test_flatten",
            use_word_level_argmax=True,
            output_format='ner',
            separators=(".",),
            word_level_min_percent=0)
        processor = CharPostprocessor(**params)
        self.assertDictEqual(
            dict(default_label='test_default', pad_label='test_pad'),
            processor.get_parameters(['default_label', 'pad_label']))

    def test_word_level_argmax(self):

        # input data initialization
        data = np.array(['this is my test sentence.', 'How nice.', 'How nice'])
        predictions = [
            # this is my test sentence.
            [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
             3, 3, 1],
            # How nice.
            [2, 2, 1, 1, 3, 1, 3, 3, 1],
            # How nice
            [2, 2, 1, 1, 3, 1, 3, 3]
        ]
        label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2,
            "TEST2": 3,
            "TEST3": 4,
        }
        default_label = 'UNKNOWN'

        # separators = (), No change in predictions.
        processor = CharPostprocessor(separators=tuple())
        expected_output = [
            # this is my test sentence.
            [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
             3, 3, 1],
            # How nice.
            [2, 2, 1, 1, 3, 1, 3, 3, 1],
            # How nice
            [2, 2, 1, 1, 3, 1, 3, 3]
        ]
        output = processor._word_level_argmax(
            data, predictions, label_mapping, default_label)
        self.assertListEqual(expected_output, output)

        # word_level_min_argmax = 0.75
        processor = CharPostprocessor(word_level_min_percent=0.75)
        expected_output = [
            # this is my test sentence.
            [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 1],
            # How nice.
            [1, 1, 1, 1, 3, 3, 3, 3, 1],
            # How nice
            [1, 1, 1, 1, 3, 3, 3, 3]            
        ]
        output = processor._word_level_argmax(
            data, predictions, label_mapping, default_label)
        self.assertListEqual(expected_output, output)

        # word_level_min_argmax = 1.0
        processor = CharPostprocessor(word_level_min_percent=1.0)
        expected_output = [
            # this is my test sentence.
            [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1],
            # How nice.
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            # How nice
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        output = processor._word_level_argmax(
            data, predictions, label_mapping, default_label)
        self.assertListEqual(expected_output, output)

        # word_level_min_argmax = 0.0
        processor = CharPostprocessor(word_level_min_percent=0.0)
        expected_output = [
            # this is my test sentence.
            [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 1],
            # How nice.
            [2, 2, 2, 1, 3, 3, 3, 3, 1],
            # How nice
            [2, 2, 2, 1, 3, 3, 3, 3]
        ]
        output = processor._word_level_argmax(
            data, predictions, label_mapping, default_label)
        self.assertListEqual(expected_output, output)

    def test_convert_to_NER_format(self):
        # input data initialization
        data = np.array(['this is my test sentence.', 'How nice.', 'How nice'])
        predictions = [
            # this is my test sentence.
            [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
             3, 3, 1],
            # How nice.
            [2, 2, 1, 1, 3, 1, 3, 3, 1],
            # How nice
            [2, 2, 2, 1, 3, 1, 3, 3]
        ]
        label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2,
            "TEST2": 3,
            "TEST3": 4,
        }

        # pad, background default test
        processor = CharPostprocessor()
        expected_output = [
            [
                (5 ,  7, 'TEST1'),
                (11, 20, 'TEST2'),
                (20, 22, 'TEST1'),
                (22, 24, 'TEST2')],
            [
                (0, 2, 'TEST1'),
                (4, 5, 'TEST2'),
                (6, 8, 'TEST2')],
            [
                (0, 3, 'TEST1'),
                (4, 5, 'TEST2'),
                (6, 8, 'TEST2')]

        ]

        output = processor.convert_to_NER_format(
            predictions, label_mapping=label_mapping,
            default_label='UNKNOWN', pad_label='PAD')
        self.assertListEqual(expected_output, output)

        # pad, background default test
        processor = CharPostprocessor()
        expected_output = [
            [
                 ( 0,  5, 'UNKNOWN'),
                 ( 7, 11, 'UNKNOWN'),
                 (24, 25, 'UNKNOWN')],
            [
                 ( 2,  4, 'UNKNOWN'),
                 ( 5,  6, 'UNKNOWN'),
                 ( 8,  9, 'UNKNOWN')],
            [
                ( 3,  4, 'UNKNOWN'),
                ( 5,  6, 'UNKNOWN')]
        ]

        output = processor.convert_to_NER_format(
            predictions, label_mapping=label_mapping,
            default_label='TEST1', pad_label='TEST2')
        self.assertListEqual(expected_output, output)

    @mock.patch(
        'dataprofiler.labelers.data_processing.CharPostprocessor.'
        '_word_level_argmax'
    )
    @mock.patch(
        'dataprofiler.labelers.data_processing.CharPostprocessor.'
        'convert_to_NER_format'
    )
    def test_process_mocked(self, mock_convert_to_NER, mock_use_word_level):

        data = np.array([])
        predictions = dict(pred=[])
        label_mapping = dict(test=0)

        # test default
        processor = CharPostprocessor()
        output = processor.process(data, predictions, label_mapping)
        mock_convert_to_NER.assert_not_called()
        mock_use_word_level.assert_not_called()

        # test use_word_level_argmax=True, output_format='character_argmax'
        mock_convert_to_NER.reset_mock()
        mock_use_word_level.reset_mock()
        processor = CharPostprocessor(
            use_word_level_argmax=True, output_format='character_argmax')
        output = processor.process(data, predictions, label_mapping)
        mock_convert_to_NER.assert_not_called()
        mock_use_word_level.assert_called()

        # test use_word_level_argmax=False, output_format='NER'
        mock_convert_to_NER.reset_mock()
        mock_use_word_level.reset_mock()
        processor = CharPostprocessor(
            use_word_level_argmax=False, output_format='NER')
        output = processor.process(data, predictions, label_mapping)
        mock_convert_to_NER.assert_called()
        mock_use_word_level.assert_not_called()

        # test use_word_level_argmax=False, output_format='character_argmax'
        mock_convert_to_NER.reset_mock()
        mock_use_word_level.reset_mock()
        processor = CharPostprocessor(
            use_word_level_argmax=True, output_format='NER')
        output = processor.process(data, predictions, label_mapping)
        mock_convert_to_NER.assert_called()
        mock_use_word_level.assert_called()

    def test_process_integrated(self):
        data = np.array(['this is my test sentence.', 'How nice.'])
        predictions = dict(pred=[
            # this is my test sentence.
            [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
             3, 3, 1],
            # How nice.
            [2, 2, 1, 1, 3, 1, 3, 3, 1]
        ])
        label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2,
            "TEST2": 3,
            "TEST3": 4,
        }

        # pad, background default, output_format=char_argmax, word_level=False
        expected_output = dict(pred=[
            np.array([1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                      3, 2, 2, 3, 3, 1]),
            np.array([2, 2, 1, 1, 3, 1, 3, 3, 1])
        ])
        processor = CharPostprocessor()
        output_result = processor.process(
            data, predictions, label_mapping)

        for expected, output in zip(expected_output['pred'],
                                    output_result['pred']):
            self.assertTrue((expected == output).all())

        # pad, background default, output_format=char_argmax, word_level=True
        expected_output = dict(pred=[
            np.array([1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                      3, 3, 3, 3, 3, 1]),
            np.array([1, 1, 1, 1, 3, 3, 3, 3, 1])
        ])
        processor = CharPostprocessor(use_word_level_argmax=True)
        output_result = processor.process(
            data, predictions, label_mapping)

        for expected, output in zip(expected_output['pred'],
                                    output_result['pred']):
            self.assertTrue((expected == output).all())

        # pad, background default, output_format=NER, word_level=True
        expected_output = dict(pred=[
            [
                 ( 5,  7, 'TEST1'),
                 (11, 24, 'TEST2')],
            [
                 ( 4,  8, 'TEST2')]
        ])
        processor = CharPostprocessor(use_word_level_argmax=True,
                                                 output_format='NER')
        output = processor.process(data, predictions, label_mapping)

        self.assertDictEqual(expected_output, output)

    def test_match_sentence_lengths(self):
        processor = CharPostprocessor()

        # Original data
        data = np.array(['test', 'hellotomyfriends', 'lol'])

        # Prediction results with 5 separator length. Separator is labeled "1"
        # Words are labeled 2-4
        results = dict(pred=[
            #T  E  S  T                 H  E  L  L  O  T  O  M  Y  F
            [2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            #R  I  E  N  D  S                 L  O  L
            [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 1, 1],
        ])

        post_process_results = \
            processor.match_sentence_lengths(data,
                                             results,
                                             flatten_separator='\x01' * 5)
        expected_results = dict(pred=[
            np.array([2, 2, 2, 2]),
            np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            np.array([4, 4, 4])])
        for expected, output in zip(expected_results['pred'],
                                    post_process_results['pred']):
            self.assertTrue((expected == output).all())

        data = np.array(['Hello', 'Friends', 'Im', "Grant", "Sayonara"])
        # Prediction results with 3 separator length. Separator is labeled "1"
        # Words are labeled 2-6
        results = dict(pred=[
            # H  E  L  L  O           F  R  I  E  N  D  S
            [2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3],
            # I  M           G  R  A  N  T 
            [4, 4, 1, 1, 1, 5, 5, 5, 5, 5],
            # S  A  Y  O  N  A  R  A
            [6, 6, 6, 6, 6, 6, 6, 6]
        ])

        post_process_results = \
            processor.match_sentence_lengths(data,
                                             results,
                                             flatten_separator='\x01' * 3)

        expected_results = dict(pred=[
            np.array([2, 2, 2, 2, 2]),
            np.array([3, 3, 3, 3, 3, 3, 3]),
            np.array([4, 4]),
            np.array([5, 5, 5, 5, 5]),
            np.array([6, 6, 6, 6, 6, 6, 6, 6])])

        for expected, output in zip(expected_results['pred'],
                                    post_process_results['pred']):
            self.assertTrue((expected == output).all())

        # Test that results are not modified with inplace
        data = np.array(['test', 'hellotomyfriends', 'lol'])
        results = dict(pred=[
            #T  E  S  T                 H  E  L  L  O  T  O  M  Y  F
            [2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            #R  I  E  N  D  S                 L  O  L
            [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 1, 1],
        ])

        post_process_results = \
            processor.match_sentence_lengths(data,
                                             results,
                                             flatten_separator='\x01' * 5,
                                             inplace=False)
        self.assertNotEqual(results, post_process_results)

        post_process_results = \
            processor.match_sentence_lengths(data,
                                             results,
                                             flatten_separator='\x01' * 5,
                                             inplace=True)
        self.assertEqual(results, post_process_results)


class TestPreandPostCharacterProcessorConnection(unittest.TestCase):

    def test_flatten_convert(self):

        # initialize variables
        default_label = 'UNKNOWN'
        pad_label = 'PAD'
        test_sentences = np.array(['These', 'are', 'my test', 'sentences.'])
        batch_size = 2
        flatten_separator = ' '

        # no flattening, max length more than sentence lengths
        preprocessor = CharPreprocessor(
            max_length=200, default_label=default_label, pad_label=pad_label,
            flatten_split=0, flatten_separator=flatten_separator)
        postprocessor = CharPostprocessor()

        output_generator = preprocessor.process(
            test_sentences, batch_size=batch_size)

        # mimic model output as a sentence instead of prediction
        output = [list(data[0]) for batch_data in output_generator
                  for data in batch_data]
        output = postprocessor.process(
            test_sentences, dict(pred=output),
            label_mapping=dict(test=0)
        )
        reconstructed_test_sentences = [''.join(sentence) for sentence in
                                        output['pred']]
        self.assertListEqual(test_sentences.tolist(),
                             reconstructed_test_sentences)

        # no flattening, with overflow
        preprocessor = CharPreprocessor(
            max_length=4, default_label=default_label, pad_label=pad_label,
            flatten_split=0, flatten_separator=flatten_separator)
        postprocessor = CharPostprocessor(
            flatten_separator=flatten_separator)

        output_generator = preprocessor.process(
            test_sentences, batch_size=batch_size,
        )

        # mimic model output as a sentence instead of prediction
        output = [list(data[0]) for batch_data in output_generator
                  for data in batch_data]
        output = postprocessor.process(
            test_sentences, dict(pred=output), label_mapping=dict(test=0)
        )
        reconstructed_test_sentences = [''.join(sentence) for sentence in
                                        output['pred']]
        self.assertListEqual(test_sentences.tolist(),
                             reconstructed_test_sentences)

        # flattening, no overflow
        preprocessor = CharPreprocessor(
            max_length=200, default_label='UNKNOWN', pad_label='PAD',
            flatten_split=1, flatten_separator=flatten_separator)
        postprocessor = CharPostprocessor(
            default_label=default_label,
            pad_label=pad_label, flatten_separator=flatten_separator)

        output_generator = preprocessor.process(
            test_sentences, batch_size=batch_size)

        # mimic model output as a sentence instead of prediction
        output = [list(data[0]) for batch_data in output_generator
                  for data in batch_data]
        output = postprocessor.process(
            test_sentences, dict(pred=output), label_mapping=dict(test=0))
        reconstructed_test_sentences = [''.join(sentence) for sentence in
                                        output['pred']]
        self.assertListEqual(test_sentences.tolist(),
                             reconstructed_test_sentences)

        # flattening, with overflow
        preprocessor = CharPreprocessor(
            max_length=4, default_label=default_label, pad_label=pad_label,
            flatten_split=1, flatten_separator=flatten_separator)
        postprocessor = CharPostprocessor(
            default_label=default_label, pad_label=pad_label,
            flatten_separator=flatten_separator)

        output_generator = preprocessor.process(
            test_sentences, batch_size=batch_size)

        # mimic model output as a sentence instead of prediction
        output = [list(data[0]) for batch_data in output_generator
                  for data in batch_data]
        output = postprocessor.process(
            test_sentences, dict(pred=output), label_mapping=dict(test=0),
        )
        reconstructed_test_sentences = [''.join(sentence) for sentence in
                                        output['pred']]
        self.assertListEqual(test_sentences.tolist(),
                             reconstructed_test_sentences)

        # mixed flattening, with overflow
        test_sentences = np.array(['This', 'is', 'a significantly', 'more',
                                   'difficult sentence.'])
        preprocessor = CharPreprocessor(
            max_length=5, default_label=default_label, pad_label=pad_label,
            flatten_split=.75, flatten_separator=flatten_separator)
        postprocessor = CharPostprocessor(
            default_label=default_label, pad_label=pad_label,
            flatten_separator=flatten_separator)

        output_generator = preprocessor.process(
            test_sentences, batch_size=batch_size)

        # mimic model output as a sentence instead of prediction
        output = [list(data[0]) for batch_data in output_generator
                  for data in batch_data]
        output = postprocessor.process(
            test_sentences, dict(pred=output), label_mapping=dict(test=0)
        )
        reconstructed_test_sentences = [''.join(sentence) for sentence in
                                        output['pred']]
        self.assertListEqual(test_sentences.tolist(),
                             reconstructed_test_sentences)

        # flatten separator length more than 1
        flatten_separator = '\x01\x01\x01'
        test_sentences = np.array(['This', 'is', 'a significantly', 'more',
                                   'difficult sentence.'])
        preprocessor = CharPreprocessor(
            max_length=5, default_label=default_label, pad_label=pad_label,
            flatten_split=.75, flatten_separator=flatten_separator)
        postprocessor = CharPostprocessor(
            default_label=default_label, pad_label=pad_label,
            flatten_separator=flatten_separator)

        output_generator = preprocessor.process(
            test_sentences, batch_size=batch_size)

        # mimic model output as a sentence instead of prediction
        output = [list(data[0]) for batch_data in output_generator
                  for data in batch_data]
        output = postprocessor.process(
            test_sentences, dict(pred=output), label_mapping=dict(test=0))
        reconstructed_test_sentences = [''.join(sentence) for sentence in
                                        output['pred']]
        self.assertListEqual(test_sentences.tolist(),
                             reconstructed_test_sentences)


class TestStructCharPreprocessor(unittest.TestCase):

    def test_registered_subclass(self):
        self.assertEqual(
            StructCharPreprocessor,
            BaseDataProcessor.get_class(
                StructCharPreprocessor.__name__))

    def test_validate_parameters(self):

        def test_raises(error_msg, params):
            with self.assertRaises(ValueError) as e:
                processor._validate_parameters(params)
            self.assertEqual(error_msg, str(e.exception))

        def test_success(params):
            try:
                processor._validate_parameters(params)
            except Exception as e:
                self.fail(str(e))

        processor = StructCharPreprocessor()

        max_len_error_msg = '`max_length` must be an int > 0'
        default_label_error_msg = "`default_label` must be a string."
        pad_label_error_msg = "`pad_label` must be a string."
        separator_error_msg = '`flatten_separator` must be a str'
        at_max_len_error_msg = '`is_separate_at_max_len` must be a bool'
        test_cases = [
            # max_length test cases
            dict(param_list=
                 dict(max_length=
                      [None, '', BaseDataProcessor, -1, 1.1, float('nan')]),
                 error_msg=max_len_error_msg),

            # success case
            dict(param_list=dict(max_length=[1, 4, 100]),
                 error_msg=None),

            # default_label test cases
            dict(param_list=dict(default_label=[None, 1, []]),
                 error_msg=default_label_error_msg),

            # success case
            dict(params=dict(default_label='test'), error_msg=None),

            # pad_label test cases
            dict(param_list=dict(pad_label=[None, 1, []]),
                 error_msg=pad_label_error_msg),

            # success case
            dict(params=dict(pad_label='test'), error_msg=None),

            # flatten_separator test cases
            dict(param_list=
                 dict(flatten_separator=[None, 1, BaseDataProcessor]),
                 error_msg=separator_error_msg),

            # success case
            dict(param_list=dict(flatten_separator=['', ' ', 'abcdefghi']),
                 error_msg=None),

            # is_separate_at_max_len test cases
            dict(param_list=
                 dict(is_separate_at_max_len=[None, 1, BaseDataProcessor]),
                 error_msg=at_max_len_error_msg),

            # success case
            dict(param_list=dict(is_separate_at_max_len=[False, True]),
                 error_msg=None),

            # combination error test cases
            dict(params=dict(max_length=-1, default_label=None, pad_label=None,
                             flatten_separator=None,
                             is_separate_at_max_len=None),
                 error_msg='\n'.join([max_len_error_msg,
                                      default_label_error_msg,
                                      pad_label_error_msg,
                                      separator_error_msg,
                                      at_max_len_error_msg])),
        ]

        for test_case in test_cases:
            test_list = []
            if 'param_list' in test_case:
                # assume list only changes one value
                key = list(test_case['param_list'].keys())[0]
                for value in test_case['param_list'][key]:
                    test_list.append(dict([(key, value)]))
            elif 'params' in test_case:
                test_list = [test_case['params']]

            for params in test_list:
                # by default uses known working default values
                test_params = dict(max_length=3400,
                                   default_label='UNKNOWN',
                                   pad_label='PAD',
                                   flatten_separator="\x01" * 5,
                                   is_separate_at_max_len=False)

                # replaces default values with ones expected for test cases
                test_params.update(params)
                if test_case['error_msg'] is None:
                    test_success(test_params)
                else:
                    test_raises(test_case['error_msg'], test_params)

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        StructCharPreprocessor.help()
        self.assertIn("Parameters", mock_stdout.getvalue())
        self.assertIn("Input Format", mock_stdout.getvalue())

    def test_get_parameters(self):

        # test default params
        processor = StructCharPreprocessor()
        self.assertDictEqual(dict(max_length=3400,
                                  default_label='UNKNOWN',
                                  pad_label='PAD',
                                  flatten_separator='\x01'*5,
                                  is_separate_at_max_len=False),
                             processor.get_parameters())

        # test set params
        params = dict(max_length=10,
                      default_label='test default',
                      pad_label='test pad',
                      flatten_separator='test',
                      is_separate_at_max_len=True)
        processor = StructCharPreprocessor(**params)
        self.assertDictEqual(params, processor.get_parameters())

        # test subset set params
        params = dict(max_length=10,
                      default_label='test default',
                      pad_label='test pad',
                      flatten_separator='test',
                      is_separate_at_max_len=True)
        processor = StructCharPreprocessor(**params)
        self.assertDictEqual(
            dict(max_length=10, default_label='test default'),
            processor.get_parameters(['max_length', 'default_label']))

    def test_convert_to_unstructured_format(self):
        preprocessor = StructCharPreprocessor(
            max_length=10, default_label='UNKNOWN', pad_label='PAD', )

        # test a single sentence
        separator = preprocessor._parameters['flatten_separator']
        test_array = np.array(['this', ' is', 'my test sentence.', ' How ',
                               'nice.'])
        expected_text = 'this' + separator + ' is' + separator \
                        + 'my test sentence.' + separator + ' How ' \
                        + separator + 'nice.'

        # without labels process
        output_text, output_labels = \
            preprocessor.convert_to_unstructured_format(test_array, labels=None)

        self.assertEqual(expected_text, output_text)
        self.assertIsNone(output_labels)

        # with labels process
        test_array = np.array(['this', ' is', 'my test sentence.', ' How ',
                               'nice.'])
        labels = ['TEST1', 'TEST2', 'UNKNOWN', 'TEST2', 'TEST1']
        expected_labels = [
            (0, 4, 'TEST1'),
            (4, 9, 'PAD'),
            (9, 12, 'TEST2'),
            (12, 17, 'PAD'),
            (34, 39, 'PAD'),
            (39, 44, 'TEST2'),
            (44, 49, 'PAD'),
            (49, 54, 'TEST1'),
        ]

        output_text, output_labels = \
            preprocessor.convert_to_unstructured_format(
                test_array, labels=labels)

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_labels, output_labels)

    def test_process(self):
        preprocessor = StructCharPreprocessor(
            max_length=10, default_label='UNKNOWN', pad_label='PAD',)

        label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2,
            "TEST2": 3,
            "TEST3": 4,
        }

        # test max_len < separator_len
        test_array = np.array(['test'])
        preprocessor._parameters['max_length'] = 0
        with self.assertRaisesRegex(ValueError, 'The `flatten_separator` length '
                                                'cannot be more than or equal '
                                                 'to the `max_length`'):
            process_generator = preprocessor.process(
                test_array, label_mapping=label_mapping, batch_size=2)
            next(process_generator)

        # test labels, no label_mapping
        preprocessor._parameters['max_length'] = 10
        with self.assertRaisesRegex(ValueError,
                                    'If `labels` are specified, `label_mapping`'
                                    ' must also be specified'):
            process_generator = preprocessor.process(
                np.array(['test']), labels=np.array(['test']),
                label_mapping=None, batch_size=2)
            next(process_generator)

        # test a single sentence
        separator = preprocessor._parameters['flatten_separator']
        test_array = np.array(['this', ' is', 'my test sentence.', ' How ',
                               'nice.'])
        expected_output = [
            np.array([['this'], [' is' + separator + 'my']]),
            np.array([[' test'], [' sentence.']]),
            np.array([[' How '], ['nice.']]),
        ]

        # without labels process
        process_generator = preprocessor.process(
            test_array, label_mapping=label_mapping, batch_size=2)

        process_output = [data for data in process_generator]
        for expected, output in zip(expected_output, process_output):
            self.assertTrue((expected == output).all())

        # with labels process
        test_array = np.array(['this', ' is', 'my test.'])
        labels = np.array(['TEST1', 'TEST2', 'UNKNOWN'])
        expected_sentence_output = [
            np.array([['this'], [' is' + separator + 'my']]),
            np.array([[' test.']]),
        ]
        expected_labels_output = [
            np.array([
                   [[0., 0., 1., 0., 0.],  # 'this'
                    [0., 0., 1., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.]],

                   [[0., 0., 0., 1., 0.],  # ' is\x01\x01\x01\x01\x01my'
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.]]
            ]),

            np.array([
                    [[0., 1., 0., 0., 0.],  # ' test.'
                     [0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.]]
            ]),

        ]
        expected_output = tuple(zip(expected_sentence_output,
                                    expected_labels_output))
        process_generator = preprocessor.process(
            test_array, labels=labels, label_mapping=label_mapping,
            batch_size=2)

        process_output = [data for data in process_generator]
        for expected, output in zip(expected_output, process_output):
            self.assertIsInstance(output, tuple)
            self.assertTrue((expected[0] == output[0]).all())
            self.assertTrue((expected[1] == output[1]).all())

    def test_process_input_checks(self):
        prep = StructCharPreprocessor()
        diff_shape_msg = re.escape("Data and labels given to "
                                   "StructCharPreprocessor are of different "
                                   "shapes, (2, 1) != (1, 2)")
        with self.assertRaisesRegex(ValueError, diff_shape_msg):
            prep.process(np.array([["hello"], ["world"]]),
                         np.array([["UNKNOWN", "UNKNOWN"]]),
                         {"UNKNOWN": 1})
        multi_dim_msg = re.escape("Data given to StructCharPreprocessor was "
                                  "multidimensional, it will be flattened for "
                                  "model processing. Results may be inaccurate,"
                                  " consider reformatting data or changing "
                                  "preprocessor.")
        with self.assertWarnsRegex(Warning, multi_dim_msg):
            prep.process(np.array([["this", "is"], ["two", "dimensions"]]))


class TestStructCharPostprocessor(unittest.TestCase):

    def test_registered_subclass(self):
        self.assertEqual(
            StructCharPostprocessor,
            BaseDataProcessor.get_class(
                StructCharPostprocessor.__name__))
        
    def test_random_state_constructor(self):

        try:
            processor = StructCharPostprocessor(random_state=0)
            processor = StructCharPostprocessor(
                random_state=random.getstate())
        except Exception as e:
            self.fail(str(e))

        with self.assertRaisesRegex(ValueError,
                                    '`random_state` must be a random.Random.'):
            processor = StructCharPostprocessor(
                random_state=[None, None, None])

    def test_validate_parameters(self):

        def test_raises(error_msg, processor_params):
            with self.assertRaises(ValueError) as e:
                processor._validate_parameters(processor_params)
            self.assertEqual(error_msg, str(e.exception))

        def test_success(processor_params):
            try:
                processor._validate_parameters(processor_params)
            except Exception as e:
                self.fail(str(e))

        processor = StructCharPostprocessor()

        default_label_error_msg = "`default_label` must be a string."
        pad_label_error_msg = "`pad_label` must be a string."
        separator_error_msg = '`flatten_separator` must be a string.'
        random_state_error_msg = '`random_state` must be a random.Random.'
        test_cases = [
            # default_label test cases
            dict(param_list=dict(default_label=[None, 1, []]),
                 error_msg=default_label_error_msg),

            # success case
            dict(params=dict(default_label='test'), error_msg=None),

            # pad_label test cases
            dict(param_list=dict(pad_label=[None, 1, []]),
                 error_msg=pad_label_error_msg),

            # success case
            dict(params=dict(pad_label='test'), error_msg=None),

            # flatten_separator test cases
            dict(param_list=
                 dict(flatten_separator=[None, 1, BaseDataProcessor]),
                 error_msg=separator_error_msg),

            # success case
            dict(param_list=dict(flatten_separator=['', ' ', 'abcdefghi']),
                 error_msg=None),

            # random_state test cases
            dict(param_list=
                 dict(random_state=[[], 'string', 1.2, BaseDataProcessor]),
                 error_msg=random_state_error_msg),

            # success case
            dict(param_list=dict(random_state=[random.Random()]),
                 error_msg=None),

            # combination error test cases
            dict(params=dict(default_label=None, pad_label=None,
                             flatten_separator=None,
                             random_state=None),
                 error_msg='\n'.join([default_label_error_msg,
                                      pad_label_error_msg,
                                      separator_error_msg,
                                      random_state_error_msg])),
        ]

        for test_case in test_cases:
            test_list = []
            if 'param_list' in test_case:
                # assume list only changes one value
                key = list(test_case['param_list'].keys())[0]
                for value in test_case['param_list'][key]:
                    test_list.append(dict([(key, value)]))
            elif 'params' in test_case:
                test_list = [test_case['params']]

            for params in test_list:
                # by default uses known working default values
                test_params = dict(
                    default_label='UNKNOWN',
                    pad_label='PAD',
                    flatten_separator="\x01"*5,
                    random_state=random.Random())

                # replaces default values with ones expected for test cases
                test_params.update(params)
                if test_case['error_msg'] is None:
                    test_success(test_params)
                else:
                    test_raises(test_case['error_msg'], test_params)

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        StructCharPostprocessor.help()
        self.assertIn("Parameters", mock_stdout.getvalue())
        self.assertIn("Output Format", mock_stdout.getvalue())

    @mock.patch.object(random.Random, '__deepcopy__',
                       new=lambda self, obj: self, create=True)
    def test_get_parameters(self):

        # test default params
        random_state = random.Random(0)

        processor = StructCharPostprocessor(
            random_state=random_state)  # required because mock/isinstance fail
        self.assertDictEqual(dict(default_label='UNKNOWN',
                                  pad_label='PAD',
                                  flatten_separator='\x01'*5,
                                  is_pred_labels=True,
                                  random_state=random_state),
                             processor.get_parameters())

        # test set params
        params = dict(default_label='test default',
                      pad_label='test pad',
                      flatten_separator='test',
                      is_pred_labels=False,
                      random_state=random_state)
        processor = StructCharPostprocessor(**params)
        self.assertDictEqual(params, processor.get_parameters())

        # test subset set params
        params = dict(default_label='test default',
                      pad_label='test pad',
                      flatten_separator='test',
                      is_pred_labels=False,
                      random_state=random_state)
        processor = StructCharPostprocessor(**params)
        self.assertDictEqual(
            dict(default_label='test default', pad_label='test pad'),
            processor.get_parameters(['default_label', 'pad_label']))

    def test_process(self):

        # without confidence
        data = np.array(['this', ' is', 'my test sentence.', ' How ', 'nice.'])
        results = dict(pred=[
            # 'this'
            [2, 2, 2, 2],
            # ' is\x01\x01\x01\x01\x01my'
            [3, 3, 3, 0, 0, 0, 0, 0, 1, 1],
            # ' test'
            [1, 1, 1, 1, 1],
            # ' sentence.'
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # ' How '
            [3, 3, 3, 3, 3],
            # 'nice.'
            [2, 2, 2, 2, 2],
        ])
        label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2,
            "TEST2": 3,
            "TEST3": 4,
        }

        expected_output = dict(pred=np.array([
            'TEST1', 'TEST2', 'UNKNOWN', 'TEST2', 'TEST1'
        ]))
        processor = StructCharPostprocessor(
            default_label='UNKNOWN',
            pad_label='PAD',
            flatten_separator='\x01' * 5)
        output = processor.process(data, results, label_mapping)

        self.assertIn('pred', output)
        self.assertTrue((expected_output['pred'] == output['pred']).all())

        # test with is_pred_labels = False
        processor = StructCharPostprocessor(
            default_label='UNKNOWN',
            pad_label='PAD',
            is_pred_labels=False,
            flatten_separator='\x01' * 5)
        expected_output_ints = dict(pred=np.array([2, 3, 1, 3, 2]))
        output = processor.process(data, results, label_mapping)

        self.assertIn('pred', output)
        self.assertTrue((expected_output_ints['pred'] == output['pred']).all())

        # with confidences
        processor = StructCharPostprocessor(
            default_label='UNKNOWN',
            pad_label='PAD',
            is_pred_labels=True,
            flatten_separator='\x01' * 5)
        confidences = []
        for sample in results['pred']:
            confidences.append([])
            for label in sample:
                confidences[-1].append([0] * label + [1] + [0] * (4-label))
            confidences[-1] = confidences[-1]
        results['conf'] = confidences

        expected_confidence_output = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ])

        output = processor.process(data, results, label_mapping)
        self.assertIn('pred', output)
        self.assertIn('conf', output)
        self.assertTrue((expected_output['pred'] == output['pred']).all())
        self.assertTrue(np.array_equal(expected_confidence_output,
                                       output['conf']))

    def test_match_sentence_lengths(self):
        processor = StructCharPostprocessor()

        # Original data
        data = np.array(['test', 'hellotomyfriends', 'lol'])

        # Prediction results with 5 separator length. Separator is labeled "1"
        # Words are labeled 2-4
        results = dict(pred=[
            #T  E  S  T                 H  E  L  L  O  T  O  M  Y  F
            [2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            #R  I  E  N  D  S                 L  O  L
            [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 1, 1],
        ])

        post_process_results = \
            processor.match_sentence_lengths(data, 
                                             results,
                                             flatten_separator='\x01' * 5)
        expected_results = dict(pred=[
            np.array([2, 2, 2, 2]),
            np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            np.array([4, 4, 4])])
        for expected, output in zip(expected_results['pred'],
                                    post_process_results['pred']):
            self.assertTrue((expected == output).all())

        data = np.array(['Hello', 'Friends', 'Im', "Grant", "Sayonara"])
        # Prediction results with 3 separator length. Separator is labeled "1"
        # Words are labeled 2-6
        results = dict(pred=[
            #H  E  L  L  O           F  R  I  E  N  D  S
            [2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3],
            #I  M           G  R  A  N  T 
            [4, 4, 1, 1, 1, 5, 5, 5, 5, 5],
            #S  A  Y  O  N  A  R  A
            [6, 6, 6, 6, 6, 6, 6, 6]
        ])

        post_process_results = \
            processor.match_sentence_lengths(data,
                                             results,
                                             flatten_separator='\x01' * 3)

        expected_results = dict(pred=[
            np.array([2, 2, 2, 2, 2]),
            np.array([3, 3, 3, 3, 3, 3, 3]),
            np.array([4, 4]),
            np.array([5, 5, 5, 5, 5]),
            np.array([6, 6, 6, 6, 6, 6, 6, 6])])

        for expected, output in zip(expected_results['pred'],
                                    post_process_results['pred']):
            self.assertTrue((expected == output).all())

        # Test that results are not modified with inplace
        data = np.array(['test', 'hellotomyfriends', 'lol'])
        results = dict(pred=[
            #T  E  S  T                 H  E  L  L  O  T  O  M  Y  F
            [2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            #R  I  E  N  D  S                 L  O  L
            [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 1, 1],
        ])

        post_process_results = \
            processor.match_sentence_lengths(data, 
                                             results,
                                             flatten_separator='\x01' * 5,
                                             inplace=False)
        self.assertNotEqual(results, post_process_results)

        post_process_results = \
            processor.match_sentence_lengths(data,
                                             results,
                                             flatten_separator='\x01' * 5,
                                             inplace=True)
        self.assertEqual(results, post_process_results)


class TestRegexPostProcessor(unittest.TestCase):

    def test_registered_subclass(self):
        self.assertEqual(
            RegexPostProcessor,
            BaseDataProcessor.get_class(RegexPostProcessor.__name__))

    def test_validate_parameters(self):

        def test_raises(error_msg, processor_params):
            with self.assertRaises(ValueError) as e:
                processor._validate_parameters(processor_params)
            self.assertEqual(error_msg, str(e.exception))

        def test_success(processor_params):
            try:
                processor._validate_parameters(processor_params)
            except Exception as e:
                self.fail(str(e))

        processor = RegexPostProcessor()

        aggregation_func_error_msg1 = "`aggregation_func` must be a string."
        aggregation_func_error_msg2 = ("`aggregation_func` must be a one of "
                                       "['split', 'priority', 'random'].")
        priority_order_error_msg1 = ("`priority_order` cannot be None if " 
                                     "`aggregtation_func` == priority.")
        priority_order_error_msg2 = ("`priority_order` must be a list or " 
                                     "numpy.ndarray.")
        random_state_error_msg = '`random_state` must be a random.Random.'
        test_cases = [
            # aggregation_func test cases
            dict(param_list=dict(aggregation_func=[None, 1, []]),
                 error_msg=aggregation_func_error_msg1),
            dict(param_list=dict(aggregation_func=['a', 'test']),
                 error_msg=aggregation_func_error_msg2),

            # success case
            dict(params=dict(aggregation_func='split'), error_msg=None),
            dict(params=dict(aggregation_func='random'), error_msg=None),
            dict(params=dict(aggregation_func='priority'), error_msg=None),

            # priority_order test cases
            dict(param_list=dict(priority_order=[1, 'string']),
                 error_msg=priority_order_error_msg2),

            # priority_order error bc not set when priority is the agg func
            dict(params=dict(aggregation_func='priority',
                             priority_order=None),
                 error_msg='\n'.join([priority_order_error_msg1])),

            # success case
            dict(params=dict(priority_order=[4, 2, 1, 3]), error_msg=None),
            dict(params=dict(priority_order=np.array([4, 2, 1, 3])),
                 error_msg=None),

            # random_state test cases
            dict(param_list=
                 dict(random_state=[[], 'string', 1.2, BaseDataProcessor]),
                 error_msg=random_state_error_msg),

            # success case
            dict(param_list=dict(random_state=[random.Random()]),
                 error_msg=None),

            # combination error test cases
            dict(params=dict(aggregation_func=None,
                             priority_order='test',
                             random_state=None),
                 error_msg='\n'.join([aggregation_func_error_msg1,
                                      priority_order_error_msg2,
                                      random_state_error_msg])),

        ]

        for test_case in test_cases:
            test_list = []
            if 'param_list' in test_case:
                # assume list only changes one value
                key = list(test_case['param_list'].keys())[0]
                for value in test_case['param_list'][key]:
                    test_list.append(dict([(key, value)]))
            elif 'params' in test_case:
                test_list = [test_case['params']]

            for params in test_list:
                # by default uses known working default values
                test_params = dict()

                # replaces default values with ones expected for test cases
                test_params.update(params)
                if test_case['error_msg'] is None:
                    test_success(test_params)
                else:
                    test_raises(test_case['error_msg'], test_params)

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        RegexPostProcessor.help()
        self.assertIn("Parameters", mock_stdout.getvalue())
        self.assertIn("Output Format", mock_stdout.getvalue())

    @mock.patch.object(random.Random, '__deepcopy__',
                       new=lambda self, obj: self, create=True)
    def test_get_parameters(self):

        # test default params
        random_state = random.Random(0)

        processor = RegexPostProcessor(
            random_state=random_state)  # required because mock/isinstance fail
        self.assertDictEqual(dict(aggregation_func='split',
                                  priority_order=None,
                                  random_state=random_state),
                             processor.get_parameters())

        # test set params
        params = dict(aggregation_func='priority',
                      priority_order=[1, 2, 3],
                      random_state=random_state)
        processor = RegexPostProcessor(**params)
        self.assertDictEqual(params, processor.get_parameters())

        # test subset set params
        params = dict(aggregation_func='random',
                      priority_order=[1, 2, 3],
                      random_state=random_state)
        processor = RegexPostProcessor(**params)
        self.assertDictEqual(
            dict(aggregation_func='random', random_state=random_state),
            processor.get_parameters(['aggregation_func', 'random_state']))

    def test_process(self):

        label_mapping = label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2
        }
        data = None
        results = dict(pred=[
            np.array([[1, 1, 0],
                      [0, 0, 1],
                      [0, 0, 1],
                      [1, 1, 1]]),
            np.array([[0, 1, 0],
                      [1, 1, 1],
                      [1, 0, 1]])
        ])

        # aggregation_func = 'split'
        expected_output = dict(pred=[
            np.array([[0.5, 0.5, 0],
                      [0, 0, 1],
                      [0, 0, 1],
                      [1 / 3, 1 / 3, 1 / 3]]),
            np.array([[0, 1, 0],
                      [1 / 3, 1 / 3, 1 / 3],
                      [0.5, 0, 0.5]])
        ])
        processor = RegexPostProcessor(aggregation_func='split')
        process_output = processor.process(data, results, label_mapping)

        self.assertIn('pred', process_output)
        for expected, output in zip(expected_output['pred'],
                                    process_output['pred']):
            self.assertTrue(np.array_equal(expected, output))

        # aggregation_func = 'priority'
        priority_order = [1, 0, 2]
        expected_output = dict(pred=[
            np.array([1, 2, 2, 1]),
            np.array([1, 1, 0])
        ])
        processor = RegexPostProcessor(aggregation_func='priority',
                                       priority_order=priority_order)
        process_output = processor.process(data, results, label_mapping)

        self.assertIn('pred', process_output)
        for expected, output in zip(expected_output['pred'],
                                    process_output['pred']):
            self.assertTrue(np.array_equal(expected, output))

        # aggregation_func = 'random'

        # first random
        random_state = random.Random(0)
        expected_output = dict(pred=[
            np.array([0, 2, 2, 0]),
            np.array([1, 0, 0])
        ])
        processor = RegexPostProcessor(aggregation_func='random',
                                       random_state=random_state)
        process_output = processor.process(data, results, label_mapping)

        self.assertIn('pred', process_output)
        for expected, output in zip(expected_output['pred'],
                                    process_output['pred']):
            self.assertTrue(np.array_equal(expected, output))

        # second random
        random_state = random.Random(1)
        expected_output = dict(pred=np.array([
            np.array([1, 2, 2, 1]),
            np.array([1, 1, 2])
        ]))
        processor = RegexPostProcessor(aggregation_func='random',
                                       random_state=random_state)
        process_output = processor.process(data, results, label_mapping)

        self.assertIn('pred', process_output)
        for expected, output in zip(expected_output['pred'],
                                    process_output['pred']):
            self.assertTrue(np.array_equal(expected, output))

    def test_random_state_constructor(self):

        try:
            processor = RegexPostProcessor(random_state=0)
            processor = RegexPostProcessor(random_state=random.getstate())
        except Exception as e:
            self.fail(str(e))

        with self.assertRaisesRegex(ValueError,
                                    '`random_state` must be a random.Random.'):
            processor = RegexPostProcessor(
                random_state=[None, None, None])


class TestStructRegexPostProcessor(unittest.TestCase):

    def test_registered_subclass(self):
        self.assertEqual(
            StructRegexPostProcessor,
            BaseDataProcessor.get_class(StructRegexPostProcessor.__name__))

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        RegexPostProcessor.help()
        self.assertIn("Parameters", mock_stdout.getvalue())
        self.assertIn("Output Format", mock_stdout.getvalue())

    def test_set_parameters(self, *mocks):

        # validate params set successfully
        params = {'random_state': random.Random()}
        processor = StructRegexPostProcessor()
        processor.set_params(**params)

        # test invalid params
        with self.assertRaisesRegex(ValueError,
                                    "`random_state` must be a random.Random."):
            processor.set_params(random_state='bad')

        with self.assertRaisesRegex(ValueError,
                                    "aggregation_func is not an accepted "
                                    "parameter.\npriority_order is not an "
                                    "accepted parameter."):
            processor.set_params(aggregation_func='bad', priority_order='bad')

    def test_process(self):

        label_mapping = label_mapping = {
            'PAD': 0,
            'UNKNOWN': 1,
            "TEST1": 2
        }
        data = None
        results = dict(pred=[
            np.array([[1, 1, 0],
                      [0, 0, 1],
                      [0, 0, 1],
                      [1, 1, 1]]),
            np.array([[0, 1, 0],
                      [1, 1, 1],
                      [1, 0, 1]])],
            conf=None,  # this isn't used internally so can set to none
        )

        expected_output = dict(
            pred=np.array([2, 1]),
            conf=np.array([[5 / 24, 5 / 24, 14 / 24],
                           [5 / 18, 8 / 18, 5 / 18]])
        )
        processor = StructRegexPostProcessor()
        process_output = processor.process(data, results, label_mapping)

        self.assertIn('pred', process_output)
        np.testing.assert_almost_equal(expected_output['pred'],
                                       process_output['pred'])
        self.assertIn('conf', process_output)
        np.testing.assert_almost_equal(expected_output['conf'],
                                       process_output['conf'])

    @mock.patch("builtins.open")
    def test_save_processor(self, mock_open, *mocks):
        # setup mocks
        mock_file = setup_save_mock_open(mock_open)

        # setup mocked class
        mocked_processor = mock.create_autospec(BaseDataProcessor)
        mocked_processor.processor_type = 'test'
        regex_processor_mock = mock.Mock(spec=RegexPostProcessor)()
        random_mock = mock.Mock()
        random_mock.getstate.return_value = ['test']
        regex_processor_mock.get_parameters.return_value = \
            dict(random_state=random_mock)
        mocked_processor._parameters = \
            dict(regex_processor=regex_processor_mock)

        # call save processor func
        StructRegexPostProcessor._save_processor(mocked_processor, 'test')

        # assert parameters saved
        mock_open.assert_called_with('test/test_parameters.json', 'w')
        self.assertEqual('{"random_state": ["test"]}', mock_file.getvalue())

        # close mocks
        StringIO.close(mock_file)

    def test_random_state_constructor(self):

        try:
            processor = StructRegexPostProcessor(random_state=0)
            processor = StructRegexPostProcessor(random_state=random.getstate())
        except Exception as e:
            self.fail(str(e))

        with self.assertRaisesRegex(ValueError,
                                    '`random_state` must be a random.Random.'):
            processor = StructRegexPostProcessor(
                random_state=[None, None, None])
