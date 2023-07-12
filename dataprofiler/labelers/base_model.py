"""Contains abstract classes for labeling data."""
from __future__ import annotations

import abc
import copy
import inspect
import warnings
from typing import Any, Callable, Type, TypeVar, cast

from dataprofiler._typing import DataArray

T = TypeVar("T", bound="BaseModel")


class AutoSubRegistrationMeta(abc.ABCMeta):
    """For registering subclasses."""

    _register_subclass: Callable[[type[AutoSubRegistrationMeta]], None]

    def __new__(
        cls, clsname: str, bases: tuple[type, ...], attrs: dict[str, object]
    ) -> type[T]:
        """Create auto registration object and return new class."""
        new_class = cast(
            Type[T],
            super().__new__(cls, clsname, bases, attrs),
        )
        new_class._register_subclass()
        return new_class


class BaseModel(metaclass=abc.ABCMeta):
    """For labeling data."""

    __subclasses: dict[str, type[BaseModel]] = {}
    __metaclass__ = abc.ABCMeta

    # boolean if the label mapping requires the mapping for index 0 reserved
    requires_zero_mapping: bool = False

    def __init__(self, label_mapping: list | dict, parameters: dict) -> None:
        """
        Initialize Base Model.

        Only model and model parameters are stored here.
        :param label_mapping: label mapping of the model or list of labels to be
            converted into the label mapping
        :type label_mapping: Union[list, dict]
        :param parameters: Contains all the appropriate parameters for the model.
                           Must contain num_labels.
        :type parameters: dict
        :return: None
        """
        # initialize class
        self._model: Any = None
        self._validate_parameters(parameters)
        self._parameters: dict = parameters
        self._label_mapping: dict[str, int] = None  # type: ignore

        self.set_label_mapping(label_mapping)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize and register subclass."""
        super().__init_subclass__(**kwargs)
        cls._register_subclass()

    def __eq__(self, other: object) -> bool:
        """
        Check if two models are equal with one another.

        May only check important variables, i.e. may not check model itself.

        :param self: a model
        :param other: a model
        :type self: BaseModel
        :type other: BaseModel
        :return: Whether or not self and other are equal
        :rtype: bool
        """
        if (
            type(self) != type(other)
            or not isinstance(other, BaseModel)
            or self._parameters != other._parameters
            or self._label_mapping != other._label_mapping
        ):
            return False
        return True

    @classmethod
    def _register_subclass(cls) -> None:
        """Register a subclass for the class factory."""
        if not inspect.isabstract(cls):
            cls.__subclasses[cls.__name__.lower()] = cls

    @property
    def label_mapping(self) -> dict[str, int]:
        """Return mapping of labels to their encoded values."""
        if self._label_mapping:
            return copy.deepcopy(self._label_mapping)
        return {}

    @property
    def reverse_label_mapping(self) -> dict[int, str]:
        """
        Return reversed order of current labels.

        Useful for when needed to extract Labels via indices.
        """
        return {v: k for k, v in self.label_mapping.items()}

    @property
    def labels(self) -> list[str]:
        """
        Retrieve the label.

        :return: list of labels
        """
        return [
            v
            for k, v in sorted(
                self.reverse_label_mapping.items(), key=lambda item: item[0]
            )
        ]

    @staticmethod
    def _convert_labels_to_label_mapping(
        labels: list[str] | dict[str, int], requires_zero_mapping: bool
    ) -> dict:
        """
        Convert the new labels set to be in an encoding dict if not already.

        :param labels: Labels to convert to an encoding dict
        :type labels: Union[list, dict]
        :param requires_zero_mapping: boolean if the label mapping requires the
            mapping for index 0 reserved.
        :type requires_zero_mapping: bool
        :return: label encoding dict
        """
        if isinstance(labels, dict):
            return labels

        # if list
        start_index = 0 if requires_zero_mapping else 1
        return dict(zip(labels, list(range(start_index, start_index + len(labels)))))

    @property
    def num_labels(self) -> int:
        """Return max label mapping."""
        return max(self.label_mapping.values()) + 1

    @classmethod
    def get_class(cls, class_name: str) -> type[BaseModel] | None:
        """Get subclasses."""
        # Import possible internal models
        from .character_level_cnn_model import CharacterLevelCnnModel  # NOQA
        from .column_name_model import ColumnNameModel  # NOQA
        from .regex_model import RegexModel  # NOQA

        return cls.__subclasses.get(class_name.lower(), None)

    def get_parameters(self, param_list: list[str] | None = None) -> dict:
        """
        Return a dict of parameters from the model given a list.

        :param param_list: list of parameters to retrieve from the model.
        :type param_list: List[str]
        :return: dict of parameters
        """
        if param_list is None:
            parameters = copy.deepcopy(self._parameters)
            parameters["label_mapping"] = copy.deepcopy(self._label_mapping)
            return parameters

        param_dict = {}
        for param in param_list:
            if param in self._parameters:
                param_dict[param] = self._parameters.get(param)
            elif param == "label_mapping":
                param_dict["label_mapping"] = self._label_mapping
            else:
                raise ValueError(
                    "`{}` does not exist as a parameter in {}.".format(
                        param, self.__class__.__name__
                    )
                )
        return copy.deepcopy(param_dict)

    def set_params(self, **kwargs: Any) -> None:
        """Set the parameters if they exist given kwargs."""
        # first check if any parameters are invalid
        self._validate_parameters(kwargs)

        for param in kwargs:
            self._parameters[param] = kwargs[param]

    def add_label(self, label: str, same_as: str | None = None) -> None:
        """
        Add a label to the data labeler.

        :param label: new label being added to the data labeler
        :type label: str
        :param same_as: label to have the same encoding index as for multi-label
            to single encoding index.
        :type same_as: str
        :return: None
        """
        # validate label
        if not label or not isinstance(label, str):
            raise TypeError("`label` must be a str.")
        elif label in self._label_mapping:
            warnings.warn(
                "The label, `{}`, already exists in the label " "mapping.".format(label)
            )
            return

        # validate same_as
        if same_as and not isinstance(same_as, str):
            raise TypeError("`same_as` must be a str.")
        elif same_as and same_as not in self._label_mapping:
            raise ValueError(
                "`same_as` value: {}, did not exist in the "
                "label_mapping.".format(same_as)
            )

        # add label to label_mapping
        max_label_ind = max(self._label_mapping.values())
        self._label_mapping[label] = self._label_mapping.get(
            same_as, max_label_ind + 1  # type: ignore
        )

    def set_label_mapping(self, label_mapping: list[str] | dict[str, int]) -> None:
        """
        Set the labels for the model.

        :param label_mapping: label mapping of the model or list of labels to be
            converted into the label mapping
        :type label_mapping: Union[list, dict]
        :return: None
        """
        if not isinstance(label_mapping, (list, dict)) or not label_mapping:
            raise TypeError(
                "Labels must either be a non-empty encoding dict "
                "which maps labels to index encodings or a list."
            )
        label_mapping = self._convert_labels_to_label_mapping(
            label_mapping, self.requires_zero_mapping
        )
        self._label_mapping = copy.deepcopy(label_mapping)

    @abc.abstractmethod
    def _need_to_reconstruct_model(self) -> bool:
        """
        Determinine whether or not to reconstruct the model.

        :return: bool of whether to reconstruct model
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_parameters(self, parameters: dict) -> None:
        """
        Validate the parameters sent in.

        Raise error if parameters are bogus.
        :param parameters:
            Contains all the appropriate parameters for the model.
        :type parameters: dict
        :return: None
        """
        raise NotImplementedError()

    @classmethod
    def help(cls) -> None:
        """
        Help describe alterable parameters.

        :return: None
        """
        param_docs = cast(str, inspect.getdoc(cls._validate_parameters))
        param_start_ind = param_docs.find("parameters:\n") + 12
        param_end_ind = param_docs.find(":type parameters:") - 1

        help_str = (
            cls.__name__
            + "\n\n"
            + "Parameters:\n"
            + param_docs[param_start_ind:param_end_ind]
        )
        print(help_str)

    @abc.abstractmethod
    def _construct_model(self) -> None:
        """
        Construct model for the data labeler.

        This also serves as a weight reset.

        :return: None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _reconstruct_model(self) -> None:
        """
        Reconstruct appropriate layers if number of number of labels is altered.

        :return: None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_weights(self) -> None:
        """
        Reset the weights of the model.

        :return: None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(
        self,
        data: DataArray,
        batch_size: int,
        show_confidences: bool,
        verbose: bool,
    ) -> dict:
        """
        Predict the data with the current model.

        :param data: model input data to predict on
        :type data: iterator of data to process
        :param batch_size: number of samples in the batch of data
        :type batch_size: int
        :param show_confidences: whether user wants prediction confidences
        :type show_confidences: bool
        :param verbose: Flag to determine whether to print status or not
        :type verbose: bool
        :return: char level predictions and confidences
        :rtype: dict
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def load_from_disk(cls, dirpath: str) -> BaseModel:
        """
        Load whole model from disk with weights.

        :param dirpath: directory path where you want to load the model from
        :type dirpath: str
        :return: loaded model
        :rtype: BaseModel
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save_to_disk(self, dirpath: str) -> None:
        """
        Save whole model to disk with weights.

        :param dirpath: directory path where you want to save the model to
        :type dirpath: str
        :return: None
        """
        raise NotImplementedError()


class BaseTrainableModel(BaseModel, metaclass=abc.ABCMeta):
    """Contains abstract method for training models."""

    @abc.abstractmethod
    def fit(
        self,
        train_data: DataArray,
        val_data: DataArray,
        batch_size: int | None = None,
        epochs: int | None = None,
        label_mapping: dict[str, int] | None = None,
        reset_weights: bool = False,
        verbose: bool = True,
    ) -> tuple[dict, float | None, dict]:
        """
        Train the current model with the training data and validation data.

        :param train_data: Training data used to train model
        :type train_data: Union[pd.DataFrame, pd.Series, np.ndarray]
        :param val_data: Validation data used to validate the training
        :type val_data: Union[pd.DataFrame, pd.Series, np.ndarray]
        :param batch_size: Used to determine number of samples in each batch
        :type batch_size: int
        :param epochs: Used to determine how many epochs to run
        :type epochs: int
        :param label_mapping: Mapping of the labels
        :type label_mapping: dict
        :param reset_weights: Flag to determine whether or not to reset the
            model's weights
        :type reset_weights: bool
        :return: history, f1, f1_report
        :rtype: Tuple[dict, float, dict]
        """
        raise NotImplementedError()
