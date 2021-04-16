#!/usr/bin/env python
"""
coding=utf-8
Specify the options when running the data profiler.
"""
import warnings
import abc
import copy
from ..labelers.base_data_labeler import BaseDataLabeler


class BaseOption(object):

    @property
    def properties(self):
        """
        Returns a copy of the option properties.

        :return: dictionary of the option's properties attr: value
        :rtype: dict
        """
        return copy.deepcopy(self.__dict__)

    def _set_helper(self, options, variable_path):
        """
        Set all the options. Send in a dict that contains all of or a subset of
        the appropriate options. Set the values of the options. Will raise error
        if the formatting is improper.

        :param options: dict containing the options you want to set.
        :type options: dict
        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: None
        """
        if not isinstance(options, dict):
            raise ValueError("The options must be a dictionary.")

        if not isinstance(variable_path, str):
            raise ValueError("The variable path must be a string.")

        for option in options:
            option_list = option.split(".", 1)
            option_name = option_list[0]
            option_variable_path = variable_path + '.' + option_name \
                if variable_path else option_name
            if option_name in self.properties:
                option_prop = getattr(self, option_name)
                if isinstance(option_prop, BaseOption):
                    option_key = option_list[1]
                    option_prop._set_helper(
                        {option_key: options[option]},
                        variable_path=option_variable_path
                    )
                elif len(option_list) > 1:
                    raise AttributeError(
                        "type object '{}' has no attribute '{}'".format(
                            option_variable_path, option_list[1]))
                else:
                    setattr(self, option_name, options[option])

        for option_name in self.properties:
            option = getattr(self, option_name)
            if isinstance(option, BaseOption):
                option_variable_path = variable_path + '.' + option_name \
                    if variable_path else option_name
                option._set_helper(options, variable_path=option_variable_path)

    def set(self, options):
        """
        Set all the options. Send in a dict that contains all of or a subset of 
        the appropriate options. Set the values of the options. Will raise error
        if the formatting is improper.

        :param options: dict containing the options you want to set.
        :type options: dict
        :return: None
        """
        if not isinstance(options, dict):
            raise ValueError("The options must be a dictionary.")
        self._set_helper(options, variable_path='')

    @abc.abstractmethod
    def _validate_helper(self, variable_path=''):
        """
        Validates the options do not conflict and cause errors and returns
        possible errors

        :param variable_path: Current path to variable set.
        :type variable_path: str
        :return: List of errors (if raise_error is false)
        :rtype: list(str)
        """
        raise NotImplementedError()

    def validate(self, raise_error=True):
        """
        Validates the options do not conflict and cause errors. Raises
        error/warning if so.

        :param raise_error: Flag that raises errors if true. Returns errors if
            false.
        :type raise_error: bool
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        errors = self._validate_helper()
        if raise_error and errors:
            raise ValueError('\n'.join(errors))
        elif errors: 
            return errors


class BooleanOption(BaseOption):

    def __init__(self, is_enabled=True):
        """
        Boolean option

        :ivar is_enabled: boolean option to enable/disable the option.
        :vartype is_enabled: bool
        """
        self.is_enabled = is_enabled

    def _validate_helper(self, variable_path='BooleanOption'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        if not isinstance(variable_path, str):
            raise ValueError("The variable path must be a string.")

        errors = []
        if not isinstance(self.is_enabled, bool):
            errors = ["{}.is_enabled must be a Boolean.".format(variable_path)]
        return errors


class HistogramOption(BooleanOption):

    def __init__(self, is_enabled=True, bin_count_or_method='auto'):
        """Options for histograms

        :ivar is_enabled: boolean option to enable/disable the option.
        :vartype is_enabled: bool
        :ivar bin_count_or_method: bin count or the method with which to
            calculate histograms
        :vartype bin_count_or_method: Union[str, int, list(str)]
        """
        self.bin_count_or_method = bin_count_or_method
        super().__init__(is_enabled=is_enabled)

    def _validate_helper(self, variable_path='HistogramOption'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        errors = super()._validate_helper(variable_path=variable_path)

        if self.bin_count_or_method is not None:
            valid_methods = ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges',
                             'sqrt']
            
            value = self.bin_count_or_method
            if isinstance(value, str):
                value = [value]
            if isinstance(value, int) and value >= 1:
                pass  # use errors below if not a passing int
            elif (not isinstance(value, list) or len(value) < 1
                  or not all([isinstance(item, str) for item in value]) or
                  not set(value).issubset(set(valid_methods))):
                errors.append("{}.bin_count_or_method must be an integer more "
                              "than 1, a string, or list of strings from the "
                              "following: {}.".format(variable_path,
                                                      valid_methods))
        return errors


class BaseColumnOptions(BooleanOption):

    def __init__(self):
        """
        Base options for all the columns.

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        """
        super().__init__(is_enabled=True)

    def _validate_helper(self, variable_path='BaseColumnOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        return super()._validate_helper(variable_path) 

    def is_prop_enabled(self, prop):
        """
        Checks to see if a property is enabled or not and returns boolean.

        :param prop: The option to check if it is enabled
        :type prop: String
        :return: Whether or not the property is enabled
        :rtype: Boolean
        """
        is_enabled = True
        if prop not in self.properties:
            raise AttributeError("Property \"{}\" does not exist in {}."
                                 .format(prop, self.__class__.__name__))
        option_prop = getattr(self, prop)
        if isinstance(option_prop, bool):
            is_enabled = option_prop
        elif isinstance(option_prop, BooleanOption):
            is_enabled = option_prop.is_enabled
        return is_enabled


class NumericalOptions(BaseColumnOptions):

    def __init__(self):
        """
        Options for the Numerical Stats Mixin

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar min: boolean option to enable/disable min
        :vartype min: BooleanOption
        :ivar max: boolean option to enable/disable max
        :vartype max: BooleanOption
        :ivar sum: boolean option to enable/disable sum
        :vartype sum: BooleanOption
        :ivar variance: boolean option to enable/disable variance
        :vartype variance: BooleanOption
        :ivar histogram_and_quantiles: boolean option to enable/disable
            histogram_and_quantiles
        :vartype histogram_and_quantiles: BooleanOption
        :ivar is_numeric_stats_enabled: boolean to enable/disable all numeric
            stats
        :vartype is_numeric_stats_enabled: bool
        """
        self.min = BooleanOption(is_enabled=True)
        self.max = BooleanOption(is_enabled=True)
        self.sum = BooleanOption(is_enabled=True)
        self.variance = BooleanOption(is_enabled=True)
        self.histogram_and_quantiles = HistogramOption()
        BaseColumnOptions.__init__(self)

    @property
    def is_numeric_stats_enabled(self):
        """
        Returns the state of numeric stats being enabled / disabled. If any
        numeric stats property is enabled it will return True, otherwise it
        will return False.

        :return: true if any numeric stats property is enabled, otherwise false
        :rtype bool:
        """
        if self.min.is_enabled or self.max.is_enabled or self.sum.is_enabled \
                or self.variance.is_enabled \
                or self.histogram_and_quantiles.is_enabled:
            return True
        return False

    @is_numeric_stats_enabled.setter
    def is_numeric_stats_enabled(self, value):
        """
        This property will enable or disable all numeric stats properties:
        min, max, sum, variance, histogram_and_quantiles

        :param value: boolean to enable/disable all numeric stats properties
        :type value: bool
        :return: None
        """
        self.min.is_enabled = value
        self.max.is_enabled = value
        self.sum.is_enabled = value
        self.variance.is_enabled = value
        self.histogram_and_quantiles.is_enabled = value

    @property
    def properties(self):
        """
        Includes at least:
            is_enabled: Turns on or off the column.
        """
        props = super().properties
        props['is_numeric_stats_enabled'] = self.is_numeric_stats_enabled
        return props

    def _validate_helper(self, variable_path='NumericalOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        if not variable_path:
            variable_path = self.__class__.__name__

        errors = super()._validate_helper(variable_path=variable_path)
        for item in ["histogram_and_quantiles", "min", "max", "sum",
                     "variance"]:
            if not isinstance(self.properties[item], BooleanOption):
                errors.append("{}.{} must be a BooleanOption."
                              .format(variable_path, item))
            else:
                errors += self.properties[item]._validate_helper(
                    variable_path=variable_path + '.' + item)

        if not self.properties["sum"].is_enabled and \
                self.properties["variance"].is_enabled:
            errors.append("{}: The numeric stats must toggle on the sum "
                          "if the variance is toggled on."
                          .format(variable_path))

        # warn user if all stats are disabled
        if not errors:
            if not self.is_numeric_stats_enabled:
                variable_path = variable_path + '.numeric_stats' \
                    if variable_path else self.__class__.__name__
                warnings.warn("{}: The numeric stats are completely disabled."
                              .format(variable_path))
        return errors


class IntOptions(NumericalOptions):

    def __init__(self):
        """
        Options for the Int Column

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar min: boolean option to enable/disable min
        :vartype min: BooleanOption
        :ivar max: boolean option to enable/disable max
        :vartype max: BooleanOption
        :ivar sum: boolean option to enable/disable sum
        :vartype sum: BooleanOption
        :ivar variance: boolean option to enable/disable variance
        :vartype variance: BooleanOption
        :ivar histogram_and_quantiles: boolean option to enable/disable
            histogram_and_quantiles
        :vartype histogram_and_quantiles: BooleanOption
        :ivar is_numeric_stats_enabled: boolean to enable/disable all numeric
            stats
        :vartype is_numeric_stats_enabled: bool
        """
        NumericalOptions.__init__(self)

    def _validate_helper(self, variable_path='IntOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        return super()._validate_helper(variable_path) 


class PrecisionOptions(BooleanOption):
    
    def __init__(self, is_enabled=True, sample_ratio=None):
        """
        Options for precision

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar sample_ratio: float option to determine ratio of valid
                            float samples in determining percision.
                            This ratio will override any defaults.
        :vartype sample_ratio: float
        """
        self.sample_ratio = sample_ratio
        super().__init__(is_enabled=is_enabled)

    def _validate_helper(self, variable_path='PrecisionOptions'):
        """
        Validates the options do not conflict and cause errors.
        
        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: List of strings
        """
        errors = super()._validate_helper(variable_path=variable_path)    
        if self.sample_ratio is not None:
            if not isinstance(self.sample_ratio, float) \
               and not isinstance(self.sample_ratio, int):
                errors.append("{}.sample_ratio must be a float."
                              .format(variable_path))                
            if (isinstance(self.sample_ratio, float) \
               or isinstance(self.sample_ratio, int)) \
               and (self.sample_ratio < 0 or self.sample_ratio > 1.0):
                errors.append("{}.sample_ratio must be a float between 0 and 1."
                              .format(variable_path))                
        
        return errors
    
class FloatOptions(NumericalOptions):

    def __init__(self):
        """
        Options for the Float Column.

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar min: boolean option to enable/disable min
        :vartype min: BooleanOption
        :ivar max: boolean option to enable/disable max
        :vartype max: BooleanOption
        :ivar sum: boolean option to enable/disable sum
        :vartype sum: BooleanOption
        :ivar variance: boolean option to enable/disable variance
        :vartype variance: BooleanOption
        :ivar histogram_and_quantiles: boolean option to enable/disable
            histogram_and_quantiles
        :vartype histogram_and_quantiles: BooleanOption
        :ivar is_numeric_stats_enabled: boolean to enable/disable all numeric
            stats
        :vartype is_numeric_stats_enabled: bool
        """
        NumericalOptions.__init__(self)
        self.precision = PrecisionOptions(is_enabled=True)

    def _validate_helper(self, variable_path='FloatOptions'):
        """
        Validates the options do not conflict and cause errors.
        
        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        errors = super()._validate_helper(variable_path=variable_path)
        if not isinstance(self.precision, PrecisionOptions):
            errors.append("{}.precision must be a PrecisionOptions."
                          .format(variable_path))
        errors += self.precision._validate_helper(variable_path+'.precision')
        return errors


class TextOptions(NumericalOptions):

    def __init__(self):
        """
        Options for the Text Column:

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar vocab: boolean option to enable/disable vocab
        :vartype vocab: BooleanOption
        :ivar min: boolean option to enable/disable min
        :vartype min: BooleanOption
        :ivar max: boolean option to enable/disable max
        :vartype max: BooleanOption
        :ivar sum: boolean option to enable/disable sum
        :vartype sum: BooleanOption
        :ivar variance: boolean option to enable/disable variance
        :vartype variance: BooleanOption
        :ivar histogram_and_quantiles: boolean option to enable/disable
            histogram_and_quantiles
        :vartype histogram_and_quantiles: BooleanOption
        :ivar is_numeric_stats_enabled: boolean to enable/disable all numeric
            stats
        :vartype is_numeric_stats_enabled: bool
        """
        NumericalOptions.__init__(self)
        self.vocab = BooleanOption(is_enabled=True)

    def _validate_helper(self, variable_path='TextOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        errors = super()._validate_helper(variable_path=variable_path)
        if not isinstance(self.vocab, BooleanOption):
            errors.append("{}.vocab must be a BooleanOption."
                          .format(variable_path))
        errors += self.vocab._validate_helper(variable_path + '.vocab')
        return errors


class DateTimeOptions(BaseColumnOptions):

    def __init__(self):
        """
        Options for the Datetime Column

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        """
        BaseColumnOptions.__init__(self)

    def _validate_helper(self, variable_path='DateTimeOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        return super()._validate_helper(variable_path) 


class OrderOptions(BaseColumnOptions):

    def __init__(self):
        """
        Options for the Order Column

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        """
        BaseColumnOptions.__init__(self)

    def _validate_helper(self, variable_path='OrderOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        return super()._validate_helper(variable_path) 


class CategoricalOptions(BaseColumnOptions):

    def __init__(self):
        """
        Options for the Categorical Column

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        """
        BaseColumnOptions.__init__(self)

    def _validate_helper(self, variable_path='CategoricalOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        return super()._validate_helper(variable_path) 


class DataLabelerOptions(BaseColumnOptions):

    def __init__(self):
        """
        Options for the Data Labeler Column.

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar data_labeler_dirpath: String to load data labeler from
        :vartype data_labeler_dirpath: str
        :ivar max_sample_size: Int to decide sample size
        :vartype max_sample_size: int
        """
        BaseColumnOptions.__init__(self)
        self.data_labeler_dirpath = None
        self.max_sample_size = None
        self.data_labeler_object = None

    def __deepcopy__(self, memo):
        """
        Override deepcopy for data labeler object
        Adapted from https://stackoverflow.com/questions/1500718/
        how-to-override-the-copy-deepcopy-operations-for-a-python-object/40484215
        :param memo: data object needed to copy
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'data_labeler_object':
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def properties(self):
        """
        Returns a copy of the option properties.

        :return: dictionary of the option's properties attr: value
        :rtype: dict
        """
        props = {k: copy.deepcopy(v)
                 for k,v in self.__dict__.items() if k != 'data_labeler_object'}
        props['data_labeler_object'] = self.data_labeler_object
        return props

    def _validate_helper(self, variable_path='DataLabelerOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        errors = super()._validate_helper(variable_path=variable_path)
        
        if self.data_labeler_dirpath is not None and \
                not isinstance(self.data_labeler_dirpath, str):
            errors.append("{}.data_labeler_dirpath must be a string."
                          .format(variable_path))
            
        if self.data_labeler_object is not None and \
                not isinstance(self.data_labeler_object, BaseDataLabeler):
            errors.append("{}.data_labeler_object must be a BaseDataLabeler "
                          "object."
                          .format(variable_path))
        if self.data_labeler_object is not None and \
                self.data_labeler_dirpath is not None:
            warnings.warn("The data labeler passed in will be used,"
                          " not through the directory of the default model")
            
        if self.max_sample_size is not None and \
                not isinstance(self.max_sample_size, int):
            errors.append("{}.max_sample_size must be an integer."
                          .format(variable_path))
        elif self.max_sample_size is not None and self.max_sample_size <= 0:
            errors.append("{}.max_sample_size must be greater than 0."
                          .format(variable_path))
        return errors


class StructuredOptions(BaseOption):

    def __init__(self):
        """
        Constructs the StructuredOptions object with default values.

        :ivar int: option set for int profiling.
        :vartype int: IntOptions
        :ivar float: option set for float profiling.
        :vartype float: FloatOptions
        :ivar datetime: option set for datetime profiling.
        :vartype datetime: DateTimeOptions
        :ivar text: option set for text profiling.
        :vartype text: TextOptions
        :ivar order: option set for order profiling.
        :vartype order: OrderOptions
        :ivar category: option set for category profiling.
        :vartype category: CategoricalOptions
        :ivar data_labeler: option set for data_labeler profiling.
        :vartype data_labeler: DataLabelerOptions
        """
        self.multiprocess = BooleanOption()
        self.int = IntOptions()
        self.float = FloatOptions()
        self.datetime = DateTimeOptions()
        self.text = TextOptions()
        self.order = OrderOptions()
        self.category = CategoricalOptions()
        self.data_labeler = DataLabelerOptions()

    @property
    def enabled_columns(self):
        """Returns a list of the enabled profiler columns."""
        enabled_columns = list()
        for key, value in self.properties.items():
            if value.is_enabled:
                enabled_columns.append(key)
        return enabled_columns

    def _validate_helper(self, variable_path='StructuredOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        if not isinstance(variable_path, str):
            raise ValueError("The variable path must be a string.")

        errors = []

        prop_check = dict([
            ('multiprocess', BooleanOption),
            ('int', IntOptions),
            ('float', FloatOptions),
            ('datetime', DateTimeOptions),
            ('text', TextOptions),
            ('order', OrderOptions),
            ('category', CategoricalOptions),
            ('data_labeler', DataLabelerOptions)
        ])

        for column in self.properties:
            if not isinstance(self.properties[column], prop_check[column]):
                errors.append("{}.{} must be a(n) {}.".format(
                    variable_path, column, prop_check[column].__name__))
            errors += self.properties[column]._validate_helper(
                variable_path=(variable_path + '.' + column
                               if variable_path else column))
        return errors


class ProfilerOptions(BaseOption):

    def __init__(self):
        """
        Initializes the ProfilerOptions object.

        :ivar structured_options: option set for structured dataset profiling.
        :vartype structured_options: StructuredOptions
        """
        self.structured_options = StructuredOptions()

    def _validate_helper(self, variable_path='ProfilerOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        if not isinstance(variable_path, str):
            raise ValueError("The variable path must be a string.")

        errors = []
        if not isinstance(self.structured_options, StructuredOptions):
            errors.append("{}.structured_options must be a StructuredOptions."
                          .format(variable_path))

        errors += self.structured_options._validate_helper(
            variable_path=variable_path + '.structured_options')

        return errors
