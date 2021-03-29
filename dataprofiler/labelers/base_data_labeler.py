import sys
import os
import warnings
import json
import pkg_resources

import numpy as np
import pandas as pd

from .. import data_readers
from . import data_processing
from .base_model import BaseModel

default_labeler_dir = pkg_resources.resource_filename(
    'resources', 'labelers'
)


class BaseDataLabeler(object):

    _default_model_loc = None

    def __init__(self, dirpath=None, load_options=None):
        """
        Initialize DataLabeler class.

        :param dirpath: path to data labeler
        :param load_options: optional arguments to include for load i.e. class
                             for model or processors
        """
        if dirpath is not None and not isinstance(dirpath, str):
            raise ValueError('`dirpath` must be a file directory where a '
                             'DataLabeler exists.')
        # Example: self._model is an instance of BaseModel
        self._model = None

        # Example: self._preprocessor and self._postprocessor are instances of
        # DataProcessing
        self._preprocessor = None
        self._postprocessor = None

        # load default model
        if dirpath or self._default_model_loc:
            if dirpath is None:
                dirpath = os.path.join(default_labeler_dir,
                                       self._default_model_loc)
            self._load_data_labeler(dirpath, load_options)

    def __eq__(self, other):
        """
        Checks if two data labelers are equal with one another, only checks
        important variables

        :param self: a data labeler
        :param other: a data labeler
        :type self: BaseDataLabeler
        :type other: BaseDataLabeler
        :return: Whether or not self and other are equal
        :rtype: Bool
        """
        if self._preprocessor != other.preprocessor \
                or self._model != other.model \
                or self._postprocessor != other.postprocessor:
            return False
        return True

    def help(self):
        """
        Help function describing alterable parameters, input data formats
        for preprocessors, and output data formats for postprocessors.

        :return: None
        """

        print("DataLabeler Information:")
        print("=" * 80)
        sys.stdout.write("Preprocessor: ")
        self._preprocessor.help()

        print("\n" + "=" * 80)
        sys.stdout.write("Model: ")
        self._model.help()

        print("\n" + "=" * 80)
        sys.stdout.write("Postprocessor: ")
        self._postprocessor.help()

    @property
    def label_mapping(self):
        """
        Retrieves the label encodings

        :return: dictionary for associating labels to indexes
        """
        return self._model.label_mapping

    @property
    def reverse_label_mapping(self):
        """
        Retrieves the index to label encoding

        :return: dictionary for associating indexes to labels
        """
        return self._model.reverse_label_mapping

    @property
    def labels(self):
        """
        Retrieves the label

        :return: list of labels
        """
        return self._model.labels

    @property
    def preprocessor(self):
        """
        Retrieves the data preprocessor

        :return: returns the preprocessor instance
        """
        return self._preprocessor

    @property
    def model(self):
        """
        Retrieves the data labeler model

        :return: returns the model instance
        """
        return self._model

    @property
    def postprocessor(self):
        """
        Retrieves the data postprocessor

        :return: returns the postprocessor instance
        """
        return self._postprocessor

    @staticmethod
    def _check_and_return_valid_data_format(data, fit_or_predict='fit'):
        """
        Checks incoming data to match the specified fit or predict format.

        :param data: data to check
        :type data: Union[pandas.DataFrame, pandas.Series, numpy.array, list]
        :param fit_or_predict: if the data needs to be in fit or predict format
        :type fit_or_predict: str
        :return: validated and formatted data
        """
        if fit_or_predict not in ['fit', 'predict']:
            raise ValueError('`fit_or_predict` must equal `fit` or `predict`')

        # Pull dataframe out of data reader object
        if isinstance(data, data_readers.base_data.BaseData):
            data = data.data

        if isinstance(data, list):
            data = np.array(data, dtype="object")
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            data = data.values
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "Data must be imported using the data_readers, "
                "pd.DataFrames, np.ndarrays, or lists."
            )

        if fit_or_predict == "fit":
            return data
        else:
            return np.reshape(data, -1)

    def set_params(self, params):
        """
        Allows user to set parameters of pipeline components in the following
        format:
            params = dict(
                preprocessor=dict(...),
                model=dict(...),
                postprocessor=dict(...)
            )
        where the key,values pairs for each pipeline component must match
        parameters that exist in their components.

        :param params: dictionary containing a key for a given pipeline
            component and its associated value of parameters as such:
                dict(preprocessor=dict(...), model=dict(...),
                postprocessor=dict(...))
        :type params: dict
        :return: None
        """
        is_params_error = True
        if params and isinstance(params, dict):
            unknown_keys = set(params.keys()) - \
                           {'preprocessor', 'model', 'postprocessor'}
            if not unknown_keys:
                is_params_error = False

        if is_params_error:
            raise ValueError(
                'The params dict must have the following format:\n'
                'params=dict(preprocessor=dict(...), model=dict(...), '
                'postprocessor=dict(...)), where each sub-dict contains '
                'parameters of the specified data_labeler pipeline components.')
        elif not self._preprocessor and 'preprocessor' in params \
                or not self._model and 'model' in params \
                or not self._postprocessor and 'postprocessor' in params:
            raise ValueError(
                'Parameters for the preprocessor, model, or postprocessor were '
                'specified when one or more of these were not set in the '
                'DataLabeler.'
            )

        if self._preprocessor and 'preprocessor' in params:
            self._preprocessor.set_params(**params['preprocessor'])
        if self._model and 'model' in params:
            self._model.set_params(**params['model'])
        if self._postprocessor and 'postprocessor' in params:
            self._postprocessor.set_params(**params['postprocessor'])

        self.check_pipeline(skip_postprocessor=self._postprocessor is None,
                            error_on_mismatch=False)

    def add_label(self, label, same_as=None):
        """
        Adds a label to the data labeler.

        :param label: new label being added to the data labeler
        :type label: str
        :param same_as: label to have the same encoding index as for multi-label
            to single encoding index.
        :type same_as: str
        :return: None
        """
        self._model.add_label(label, same_as)

    def set_labels(self, labels):
        """
        Sets the labels for the data labeler.

        :param labels: new labels in either encoding list or dict
        :type labels: list or dict
        :return: None
        """
        # convert to valid format
        self._model.set_label_mapping(label_mapping=labels)

    def predict(self, data, batch_size=32, predict_options=None,
                error_on_mismatch=False, verbose=1):
        """
        Predicts labels of input data based with the data labeler model.

        :param data: data to be predicted upon
        :param batch_size: batch size of prediction
        :param predict_options: optional parameters to allow for predict as a
            dict, i.e.  dict(show_confidences=True)
        :param error_on_mismatch: if true, errors instead of warns on parameter
            mismatches in pipeline
        :param verbose: Flag to determine whether to print status or not
        :return: predictions
        """

        if predict_options is None:
            predict_options = {}
        data = self._check_and_return_valid_data_format(
            data, fit_or_predict='predict'
        )

        # check for valid pipeline
        self.check_pipeline(
            skip_postprocessor=False, error_on_mismatch=error_on_mismatch)

        # preprocess
        samples = self._preprocessor.process(data, batch_size=batch_size)

        # predicting:
        results = self._model.predict(
            samples, batch_size,
            show_confidences=predict_options.get('show_confidences', False),
            verbose=verbose
        )

        # postprocessing:
        results = self._postprocessor.process(data, results, self.label_mapping)

        return results

    def set_preprocessor(self, data_processor):
        """
        Set the data preprocessor for the data labeler

        :param data_processor: processor to set as the preprocessor
        :type data_processor: data_processing.BaseDataPreprocessor
        :return: None
        """
        if not isinstance(data_processor, data_processing.BaseDataPreprocessor):
            raise TypeError('The specified preprocessor was not of the correct'
                            ' type, `DataProcessing`.')
        self._preprocessor = data_processor

    def set_model(self, model):
        """
        Set the model for the data labeler

        :param model: model to use within the data labeler
        :type model: base_model.BaseModel
        :return: None
        """
        if not isinstance(model, BaseModel):
            raise TypeError('The specified model was not of the correct'
                            ' type, `BaseModel`.')
        self._model = model

    def set_postprocessor(self, data_processor):
        """
        Set the data postprocessor for the data labeler

        :param data_processor: processor to set as the postprocessor
        :type data_processor: data_processing.BaseDataPostprocessor
        :return: None
        """
        if not isinstance(data_processor, data_processing.BaseDataPostprocessor):
            raise TypeError('The specified postprocessor was not of the '
                            'correct type, `DataProcessing`.')
        self._postprocessor = data_processor

    def check_pipeline(self, skip_postprocessor=False, error_on_mismatch=False):
        """
        Checks whether the processors and models connect together without error.

        :param skip_postprocessor: skip checking postprocessor is valid in
            pipeline
        :type skip_postprocessor: bool
        :param error_on_mismatch: if true, errors instead of warns on parameter
            mismatches in pipeline
        :type error_on_mismatch: bool
        :return: bool indicating valid pipeline
        """
        messages = []

        def get_parameter_overlap_mismatches(param_dict1, param_dict2):
            """
            Get mismatching parameters in dictionary if same key exists.

            :param param_dict1: 1st set of dictionary of parameters
            :type param_dict1: dict
            :param param_dict2: 2nd set of dictionary of parameters
            :type param_dict2: dict
            :return: None
            """
            param_mismatch_overlaps = []
            for key in param_dict1:
                if key in param_dict2 and param_dict1[key] != param_dict2[key]:
                    param_mismatch_overlaps.append(key)
            return param_mismatch_overlaps

        model_params = self._model.get_parameters()
        preprocessor_params = self._preprocessor.get_parameters()

        mismatch_overlaps = get_parameter_overlap_mismatches(
            model_params, preprocessor_params
        )
        for param in mismatch_overlaps:
            messages.append(
                'Model and preprocessor value for `{}` do not match. {} != '
                '{}'.format(param, model_params[param],
                            preprocessor_params[param])
            )

        if not skip_postprocessor:
            postprocessor_params = self._postprocessor.get_parameters()
            mismatch_overlaps = get_parameter_overlap_mismatches(
                model_params, postprocessor_params
            )
            for param in mismatch_overlaps:
                messages.append(
                    'Model and postprocessor value for `{}` do not match. '
                    '{} != {}'.format(param, model_params[param],
                                      postprocessor_params[param])
                )
            mismatch_overlaps = get_parameter_overlap_mismatches(
                preprocessor_params, postprocessor_params
            )
            for param in mismatch_overlaps:
                messages.append(
                    'Preprocessor and postprocessor value for `{}` do not '
                    'match. {} != {}'.format(param,
                                             preprocessor_params[param],
                                             postprocessor_params[param])
                )
        if messages:
            if error_on_mismatch:
                raise RuntimeError('\n'.join(messages))
            warnings.warn('\n'.join(messages), category=RuntimeWarning)

    @staticmethod
    def _load_parameters(dirpath, load_options=None):
        """
        Loads the data labeler parameters

        :param dirpath: directory where the saved datalabeler exists.
        :type dirpath: str
        :param load_options: optional arguments to include for load i.e. class
                             for model or processors
        :type load_options: dict
        :return: data labeler parameter dict
        """
        if not load_options:
            load_options = {}

        with open(os.path.join(dirpath, 'data_labeler_parameters.json')) as fp:
            params = json.load(fp)
            
        if 'model_class' in load_options:
            model_class = load_options.get('model_class')
            if not isinstance(model_class, BaseModel):
                raise TypeError('`model_class` must be a BaseModel')
            param_model_class = params.get('model', {}).get('class', None)
            if param_model_class != model_class.__name__:
                raise ValueError('The load_options model class does not match '
                                 'the required DataLabeler model.\n {} != {}'.
                                 format(model_class.__name__,
                                        param_model_class))
            params['model']['class'] = model_class
        if 'preprocessor_class' in load_options:
            processor_class = load_options.get('preprocessor_class')
            if not isinstance(processor_class,
                              data_processing.BaseDataPreprocessor):
                raise TypeError('`preprocessor_class` must be a '
                                'BaseDataPreprocessor')
            param_processor_class = params.get('preprocessor', {}).\
                get('class', None)
            if param_processor_class != processor_class:
                raise ValueError('The load_options preprocessor class does not '
                                 'match the required DataLabeler preprocessor.'
                                 '\n {} != {}'.
                                 format(processor_class, param_processor_class))
            params['preprocessor']['class'] = load_options.get(
                'preprocessor_class')
        if 'postprocessor_class' in load_options:
            processor_class = load_options.get('postprocessor_class')
            if not isinstance(processor_class,
                              data_processing.BaseDataPostprocessor):
                raise TypeError('`postprocessor_class` must be a '
                                'BaseDataPostprocessor')
            param_processor_class = params.get('postprocessor', {}).\
                get('class', None)
            if param_processor_class != processor_class.__name__:
                raise ValueError(
                    'The load_options postprocessor class does not match '
                    'the required DataLabeler postprocessor.\n {} != {}'.
                    format(processor_class.__name__, param_processor_class))
            params['postprocessor']['class'] = load_options.get(
                'postprocessor_class')
        return params

    def _load_model(self, model_class, dirpath):
        """
        Load the data labeler model either by using a provided model class or
        retrieving a registered data labeler model.

        :param model_class: class of model being loaded
        :type model_class: Union[BaseModel, str]
        :param dirpath: directory where the saved DataLabeler model exists.
        :type dirpath: str
        :return: None
        """
        if isinstance(model_class, str):
            model_class = BaseModel.get_class(model_class)
        if not model_class:
            raise ValueError('`model_class`, {}, was not set in load_options '
                             'and could not be found as a registered model '
                             'class in BaseModel.'.format(str(model_class)))
        self.set_model(model_class.load_from_disk(dirpath))

    def _load_preprocessor(self, processor_class, dirpath):
        """
        Loads the preprocessor for the data labeler.

        :param processor_class: class of model being loaded
        :type processor_class: Union[data_processing.BaseDataProcessor, str]
        :param dirpath: directory where the saved DataLabeler model exists.
        :type dirpath: str
        :return: None
        """
        if isinstance(processor_class, str):
            processor_class = data_processing.BaseDataProcessor.get_class(
                processor_class)
        if not processor_class:
            raise ValueError('`processor_class`, {}, was not set in load_options '
                             'and could not be found as a registered model '
                             'class in BaseDataProcessor.'.format(
                str(processor_class)))
        self.set_preprocessor(processor_class.load_from_disk(dirpath))

    def _load_postprocessor(self, processor_class, dirpath):
        """
        Loads the postprocessor for the data labeler.

        :param processor_class: class of model being loaded
        :type processor_class: Union[data_processing.BaseDataProcessor, str]
        :param dirpath: directory where the saved DataLabeler model exists.
        :type dirpath: str
        :return: None
        """
        if isinstance(processor_class, str):
            processor_class = data_processing.BaseDataProcessor.get_class(
                processor_class)
        if not processor_class:
            raise ValueError('`processor_class`, {}, was not set in '
                             'load_options and could not be found as a '
                             'registered model class in BaseDataProcessor.'.
                             format(str(processor_class)))
        self.set_postprocessor(processor_class.load_from_disk(dirpath))

    def _load_data_labeler(self, dirpath, load_options=None):
        """
        Loads and initializes the data data labeler in the given path.

        :param dirpath: location of data labeler info files.
        :type dirpath: str
        :param load_options: optional arguments to include for load i.e. class
                             for model or processors
        :type load_options: dict
        :return: DataLabeler class
        """

        # get loaded parameters
        params = self._load_parameters(dirpath, load_options)
        model_params = params.get('model')
        preprocessor_params = params.get('preprocessor')
        postprocessor_params = params.get('postprocessor')

        # setup data labeler based on parameters
        self._load_model(model_params.get('class'), dirpath)
        self._load_preprocessor(preprocessor_params.get('class'), dirpath)
        self._load_postprocessor(postprocessor_params.get('class'), dirpath)

    @classmethod
    def load_from_library(cls, name):
        """
        Loads the data labeler from the data labeler zoo in the library.

        :param name: name of the data labeler.
        :type name: str
        :return: DataLabeler class
        """
        return cls(os.path.join(default_labeler_dir, name))

    @classmethod
    def load_from_disk(cls, dirpath, load_options=None):
        """
        Loads the data labeler from a saved location on disk.

        :param dirpath: path to data labeler files.
        :type dirpath: str
        :param load_options: optional arguments to include for load i.e. class
                             for model or processors
        :type load_options: dict
        :return: DataLabeler class
        """
        return cls(dirpath, load_options)

    @classmethod
    def load_with_components(cls, preprocessor, model, postprocessor):
        """
        Loads the data labeler from a its set  of components.

        :param preprocessor: processor to set as the preprocessor
        :type preprocessor: data_processing.BaseDataPreprocessor
        :param model: model to use within the data labeler
        :type model: base_model.BaseModel
        :param postprocessor: processor to set as the postprocessor
        :type postprocessor: data_processing.BaseDataPostprocessor
        :return:
        """
        data_labeler = type("CustomDataLabeler", (BaseDataLabeler,), {})()
        data_labeler.set_preprocessor(preprocessor)
        data_labeler.set_model(model)
        data_labeler.set_postprocessor(postprocessor)
        return data_labeler

    def _save_model(self, dirpath):
        """
        Saves the data labeler model.

        :param dirpath: path to save the data labeler
        :type dirpath: str
        :return: None
        """
        self._model.save_to_disk(dirpath)

    def _save_preprocessor(self, dirpath):
        """
        Saves the preprocessor for the data labeler.

        :param dirpath: path to save the data processor
        :type dirpath: str
        :return: None
        """
        self._preprocessor.save_to_disk(dirpath)

    def _save_postprocessor(self, dirpath):
        """
        Saves the postprocessor for the data labeler.

        :param dirpath: path to save the data processor
        :type dirpath: str
        :return: None
        """
        self._postprocessor.save_to_disk(dirpath)

    def _save_parameters(self, dirpath):
        """
        Data labeler specific parameters to save.

        :param dirpath: location to save the parameters
        :type dirpath: str
        :return: None
        """
        parameters = {
            'model': {
                'class': self._model.__class__.__name__
            },
            'preprocessor': {
                'class': self._preprocessor.__class__.__name__,
            },
            'postprocessor': {
                'class': self._postprocessor.__class__.__name__,

            },
        }
        with open(os.path.join(dirpath,
                               'data_labeler_parameters.json'), 'w') as fp:
            json.dump(parameters, fp)

    def _save_data_labeler(self, dirpath):
        """
        Saves each component of the data labeler to the specified location.

        :param dirpath: path to where to save the data labeler.
        :type dirpath: str
        :return: None
        """
        self._save_parameters(dirpath)
        self._save_model(dirpath)
        self._save_preprocessor(dirpath)
        self._save_postprocessor(dirpath)

    def save_to_disk(self, dirpath):
        """
        Saves the data labeler to the specified location

        :param dirpath: location to save the data labeler.
        :type dirpath: str
        :return: None
        """
        # note diff from saving to cloud which would create a temp path then
        # delete
        self._save_data_labeler(dirpath)


class TrainableDataLabeler(BaseDataLabeler):

    def fit(self, x, y, validation_split=0.2, labels=None, reset_weights=False,
            batch_size=32, epochs=1, error_on_mismatch=False):
        """
        Fits the data labeler model for the dataset.

        :param x: samples to fit model
        :type x: Union[pd.DataFrame, pd.Series, np.ndarray]
        :param y: labels associated with the samples to fit model
        :type y: Union[pd.DataFrame, pd.Series, np.ndarray]
        :param validation_split: split of the data to have as cross-validation
            data
        :type validation_split: float
        :param labels: Encoding or number of labels if refit is needed to new
            labels
        :type labels: Union[list, dict]
        :param reset_weights:  Flag to determine whether or not to reset the
            weights
        :type reset_weights: bool
        :param batch_size: Size of each batch sent to data labeler model
        :type batch_size: int
        :param epochs: number of epochs to iterate over the dataset and send to
            the model
        :type epochs: int
        :param error_on_mismatch: if true, errors instead of warns on parameter
            mismatches in pipeline
        :type error_on_mismatch: bool
        :return: model output
        """

        # input validation checks
        x = self._check_and_return_valid_data_format(x, fit_or_predict='fit')
        y = self._check_and_return_valid_data_format(y, fit_or_predict='fit')

        num_samples = len(x)
        if num_samples == 0 or len(y) == 0:
            raise ValueError("No data or labels to fit.")
        elif num_samples != len(y):
            raise ValueError("Data and labels must be the same length.")
        elif validation_split < 0. or validation_split >= 1.0:
            raise ValueError(
                "`validation_split` must be >= 0 and less than 1.0")

        # check pipeline
        self.check_pipeline(
            skip_postprocessor=False, error_on_mismatch=error_on_mismatch)

        # fit to model
        if labels is not None:
            self.set_labels(labels)
        if reset_weights:
            self._model.reset_weights()

        # shuffle input data
        shuffle_inds = np.random.permutation(num_samples)
        x = x[shuffle_inds]
        y = y[shuffle_inds]

        # free memory
        del shuffle_inds

        # preprocess data
        cv_split_index = max(1, int(num_samples * (1 - validation_split)))
        train_data = self._preprocessor.process(
            x[:cv_split_index], labels=y[:cv_split_index],
            label_mapping=self.label_mapping, batch_size=batch_size)
        cv_data = None if not validation_split or cv_split_index < 2 else \
            self._preprocessor.process(
                x[cv_split_index:], labels=y[cv_split_index:],
                label_mapping=self.label_mapping, batch_size=batch_size)

        results = []
        for i in range(epochs):
            results.append(self._model.fit(train_data, cv_data))
            if i < epochs - 1:
                # shuffle input data
                shuffle_inds = np.random.permutation(cv_split_index)
                train_data_x = x[shuffle_inds]
                train_data_y = y[shuffle_inds]

                # free memory
                del shuffle_inds

                train_data = self._preprocessor.process(
                    train_data_x, labels=train_data_y,
                    label_mapping=self.label_mapping, batch_size=batch_size)
                cv_data = None if not validation_split or cv_split_index < 2 \
                    else self._preprocessor.process(
                        x[cv_split_index:], labels=y[cv_split_index:],
                        label_mapping=self.label_mapping, batch_size=batch_size)
        return results

    def set_model(self, model):
        """
        Set the model for a trainable data labeler. Model must have a train
        function to be able to be set.

        :param model: model to use within the data labeler
        :type model: base_model.BaseModel
        :return: None
        """
        if not hasattr(model, 'fit'):
            raise ValueError('`model` must have a fit function to be '
                             'trainable.')
        BaseDataLabeler.set_model(self, model)

    @classmethod
    def load_with_components(cls, preprocessor, model, postprocessor):
        """
        Loads the data labeler from a its set  of components.

        :param preprocessor: processor to set as the preprocessor
        :type preprocessor: data_processing.BaseDataPreprocessor
        :param model: model to use within the data labeler
        :type model: base_model.BaseModel
        :param postprocessor: processor to set as the postprocessor
        :type postprocessor: data_processing.BaseDataPostprocessor
        :return:
        """
        data_labeler = type(
            "CustomTrainableDataLabeler", (TrainableDataLabeler,), {})()
        data_labeler.set_preprocessor(preprocessor)
        data_labeler.set_model(model)
        data_labeler.set_postprocessor(postprocessor)
        return data_labeler

