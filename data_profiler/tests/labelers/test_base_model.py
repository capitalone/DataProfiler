import unittest
from unittest import mock

from data_profiler.labelers import base_model


class TestBaseModel(unittest.TestCase):

    @mock.patch('data_profiler.labelers.base_model.BaseModel.'
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

    @mock.patch('data_profiler.labelers.base_model.BaseModel.'
                '__abstractmethods__', set())
    @mock.patch('data_profiler.labelers.base_model.BaseModel.'
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

    def test_set_label_mapping_parameters(self):

        # setup mock
        mock_model = mock.Mock(spec=base_model.BaseModel)

        # base case
        with self.assertRaisesRegex(ValueError,
                                    "`label_mapping` must be a dict which maps "
                                    "labels to index encodings."):
            base_model.BaseModel.set_label_mapping(
                mock_model, label_mapping=None)

        # assert error for empty label_mapping dict
        with self.assertRaisesRegex(ValueError,
                                    "`label_mapping` must be a dict which maps "
                                    "labels to index encodings."):
            base_model.BaseModel.set_label_mapping(mock_model, label_mapping={})

        # assert label_map set
        base_model.BaseModel.set_label_mapping(
            mock_model, label_mapping={'test': 'test'})
        self.assertDictEqual({'test': 'test'}, mock_model._label_mapping)
