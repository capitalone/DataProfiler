import unittest
from unittest import mock

from dataprofiler.labelers import base_model


class TestBaseModel(unittest.TestCase):

    @mock.patch('dataprofiler.labelers.base_model.BaseModel.'
                '_BaseModel__subclasses',
                new_callable=mock.PropertyMock)
    def test_register_subclass(self, mock_subclasses):
        # remove not implemented func

        fake_class = type('FakeModel', (base_model.BaseModel,), {})
        fake_class.__abstractmethods__ = []
        fake_class._register_subclass()
        # base_model.BaseModel._register_subclass()
        self.assertIn(
            mock.call().__setitem__('fakemodel', fake_class),
            mock_subclasses.mock_calls)

    @mock.patch('dataprofiler.labelers.base_model.BaseModel.'
                '__abstractmethods__', set())
    @mock.patch('dataprofiler.labelers.base_model.BaseModel.'
                '_validate_parameters', return_value=None)
    def test_equality_checks(self, *mocks):

        FakeModel1 = type('FakeModel1', (base_model.BaseModel,), {})
        FakeModel2 = type('FakeModel2', (base_model.BaseModel,), {})

        fake_model1 = FakeModel1(label_mapping={'a': 1, 'b': 2},
                                 parameters={'test': 1})
        fake_model1_1 = FakeModel1(label_mapping={'a': 1, 'b': 2},
                                   parameters={'test': 1})
        fake_model1_2 = FakeModel1(label_mapping={'c': 2},
                                   parameters={'test': 1})
        fake_model1_3 = FakeModel1(label_mapping={'a': 1, 'b': 2},
                                   parameters={'Different': 1})
        fake_model2 = FakeModel2(label_mapping={'a': 1, 'b': 2},
                                 parameters={'a': 1, 'b': 2})

        # assert True if the same object
        self.assertEqual(fake_model1, fake_model1)

        # assert True if same class but same params / label_mapping
        self.assertEqual(fake_model1, fake_model1_1)

        # assert False if diff class even if same params / label_mapping
        self.assertNotEqual(fake_model1, fake_model2)

        # assert False if same class even diff params / label_mapping
        self.assertNotEqual(fake_model1, fake_model1_2)
        self.assertNotEqual(fake_model1, fake_model1_3)

    # @mock.patch('data_profiler.labelers.base_model.BaseModel._validate_parameters')
    def test_get_parameters(self):

        mock_model = mock.Mock(spec=base_model.BaseModel)
        mock_model._label_mapping = {'a': 1, 'c': '2'}
        mock_model._parameters = {'test1': 1, 'test2': '2'}

        # base case
        params = base_model.BaseModel.get_parameters(mock_model)
        self.assertDictEqual(
            {'label_mapping': {'a': 1, 'c': '2'}, 'test1': 1, 'test2': '2'},
            params)

        # param list w/o label_mapping
        param_list = ['test1']
        params = base_model.BaseModel.get_parameters(mock_model, param_list)
        self.assertDictEqual({'test1': 1}, params)

        # param list w/ label_mapping
        param_list = ['test2', 'label_mapping']
        params = base_model.BaseModel.get_parameters(mock_model, param_list)
        self.assertDictEqual(
            {'label_mapping': {'a': 1, 'c': '2'}, 'test2': '2'}, params)

    def test_set_parameters(self):

        # validate params set successfully
        mock_model = mock.Mock(spec=base_model.BaseModel)
        mock_model._parameters = dict()
        params = {'test': 1}
        base_model.BaseModel.set_params(mock_model, **params)
        self.assertDictEqual(params, mock_model._parameters)

        # test overwrite params
        params = {'test': 2}
        base_model.BaseModel.set_params(mock_model, **params)
        self.assertDictEqual(params, mock_model._parameters)

        # test invalid params
        mock_model._validate_parameters.side_effect = ValueError('test')
        with self.assertRaisesRegex(ValueError, 'test'):
            base_model.BaseModel.set_params(mock_model, **params)

    @mock.patch.multiple('dataprofiler.labelers.base_model.BaseModel',
                         __abstractmethods__=set(),
                         _validate_parameters=mock.MagicMock(return_value=None))
    def test_add_labels(self, *args):

        # setup model with mocked abstract methods
        mock_model = base_model.BaseModel(
            label_mapping={'NEW_LABEL': 1}, parameters={})

        # assert bad label inputs
        with self.assertRaisesRegex(TypeError, '`label` must be a str.'):
            mock_model.add_label(label=None)

        with self.assertRaisesRegex(TypeError, '`label` must be a str.'):
            mock_model.add_label(label=1)

        # assert existing label
        label = 'NEW_LABEL'
        with self.assertWarnsRegex(UserWarning,
                                   'The label, `{}`, already exists in the '
                                   'label mapping.'.format(label)):
            mock_model.add_label(label)

        # assert bad same_as input
        with self.assertRaisesRegex(TypeError, '`same_as` must be a str.'):
            mock_model.add_label(label='test', same_as=1)

        label = 'NEW_LABEL_2'
        same_as = 'DOES_NOT_EXIST'
        with self.assertRaisesRegex(ValueError,
                                    '`same_as` value: {}, did not exist in the '
                                    'label_mapping.'.format(same_as)):
            mock_model.add_label(label, same_as)

        # assert successful add
        label = 'NEW_LABEL_2'
        mock_model.add_label(label)
        self.assertDictEqual(
            {'NEW_LABEL': 1, 'NEW_LABEL_2': 2}, mock_model._label_mapping)

        # assert successful add w/ same_as
        label = 'NEW_LABEL_3'
        mock_model.add_label(label, same_as='NEW_LABEL')
        self.assertDictEqual(
            {'NEW_LABEL': 1, 'NEW_LABEL_2': 2, 'NEW_LABEL_3': 1},
            mock_model._label_mapping)

    def test_set_label_mapping_parameters(self):

        # setup mock
        mock_model = mock.Mock(spec=base_model.BaseModel)
        mock_model._convert_labels_to_label_mapping.side_effect = \
            base_model.BaseModel._convert_labels_to_label_mapping

        # assert non value is not accepted.
        with self.assertRaisesRegex(TypeError,
                                    "Labels must either be a non-empty encoding"
                                    " dict which maps labels to index encodings"
                                    " or a list."):
            base_model.BaseModel.set_label_mapping(
                mock_model, label_mapping=None)

        # non-acceptable value case
        with self.assertRaisesRegex(TypeError,
                                    "Labels must either be a non-empty encoding"
                                    " dict which maps labels to index encodings"
                                    " or a list."):
            base_model.BaseModel.set_label_mapping(
                mock_model, label_mapping=1)

        # assert error for empty label_mapping dict
        with self.assertRaisesRegex(TypeError,
                                    "Labels must either be a non-empty encoding"
                                    " dict which maps labels to index encodings"
                                    " or a list."):
            base_model.BaseModel.set_label_mapping(mock_model, label_mapping={})

        # assert label_map set
        base_model.BaseModel.set_label_mapping(
            mock_model, label_mapping={'test': 'test'})
        self.assertDictEqual({'test': 'test'}, mock_model._label_mapping)

    def test_convert_labels_to_encodings(self, *mocks):

        # test label list to label_mapping
        labels = ['a', 'b', 'd', 'c']
        label_mapping = base_model.BaseModel._convert_labels_to_label_mapping(
            labels, requires_zero_mapping=True)
        self.assertDictEqual(dict(a=0, b=1, d=2, c=3), label_mapping)

        # test label dict to label_mapping
        labels = dict(a=1, b=2, d=3, c=4)
        label_mapping = base_model.BaseModel._convert_labels_to_label_mapping(
            labels, requires_zero_mapping=True)
        self.assertDictEqual(dict(a=1, b=2, d=3, c=4), label_mapping)
