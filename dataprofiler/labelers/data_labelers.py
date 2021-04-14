import os
import pkg_resources

import pandas as pd

from .. import data_readers
from .base_data_labeler import BaseDataLabeler, TrainableDataLabeler

default_labeler_dir = pkg_resources.resource_filename(
    'resources', 'labelers'
)


def train_structured_labeler(data, default_label=None, save_dirpath=None, epochs=2):
    """
    Uses provided data to create and save a structured data labeler

    :param data: data to be trained upon
    :type data: Union[None, pd.DataFrame]
    :param save_dirpath: path to save data labeler
    :type save_dirpath: Union[None, str]
    :param epochs: number of epochs to loop training the data
    :type epochs: int
    :return:
    """
    if isinstance(data, data_readers.base_data.BaseData) \
            and not isinstance(data, data_readers.text_data.TextData):
        data = data.data
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        raise TypeError(
            "Input data must be either a `pd.DataFrame` or a "
            "`data_profiler.Data` and not of type `TextData`."
        )
    if save_dirpath is not None:
        if not isinstance(save_dirpath, str):
            raise TypeError("The output dirpath must be a string.")
        elif not os.access(save_dirpath, os.W_OK):
            raise ValueError(
                "The `save_dirpath` is not valid or not accessible."
            )

    # prep data
    value_label_df = data.reset_index(drop=True).melt()
    value_label_df.columns = [1, 0]  # labels=1, values=0 in that order
    value_label_df = value_label_df.astype(str)

    data_labeler = DataLabeler(labeler_type='structured', trainable=True)
    labels = value_label_df[1].unique().tolist()

    # set default label to the data labeler pipeline
    if default_label:
        params = {'default_label': default_label}
        data_labeler.set_params(
            {'preprocessor': params, 'model': params, 'postprocessor': params})

    data_labeler.fit(
        x=value_label_df[0], y=value_label_df[1], labels=labels, epochs=epochs)
    if save_dirpath:
        data_labeler.save_to_disk(save_dirpath)
    return data_labeler


class UnstructuredDataLabeler(BaseDataLabeler):
    _default_model_loc = 'unstructured_model'


class StructuredDataLabeler(BaseDataLabeler):
    _default_model_loc = 'structured_model'


class DataLabeler(object):
    
    labeler_classes = dict(
        structured=StructuredDataLabeler,
        unstructured=UnstructuredDataLabeler,
    )

    def __new__(cls, labeler_type, dirpath=None, load_options=None,
                trainable=False):

        data_labeler = cls.labeler_classes.get(labeler_type, None)
        if data_labeler is None:
            raise ValueError(
                'No DataLabeler class types matched the input, `{}`. '
                'Allowed types {}.'.format(
                    labeler_type, list(cls.labeler_classes.keys())))
        if trainable:
            if dirpath is None:
                dirpath = os.path.join(default_labeler_dir,
                                       data_labeler._default_model_loc)
            return TrainableDataLabeler(dirpath, load_options)
        return data_labeler(dirpath, load_options)
