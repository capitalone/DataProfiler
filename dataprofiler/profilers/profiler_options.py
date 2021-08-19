#!/usr/bin/env python
"""
coding=utf-8
Specify the options when running the data profiler.
"""
import re
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

            is_check_all = False
            if option_name == '*':
                option_list = option_list[1].split(".", 1)
                option_name = option_list[0]
                is_check_all = True

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
            elif len(option_list) > 1 or is_check_all:
                for class_option_name in self.properties:
                    class_option = getattr(self, class_option_name)
                    if isinstance(class_option, BaseOption):
                        option_variable_path = (
                            variable_path + '.' + class_option_name
                            if variable_path else class_option_name)
                        class_option._set_helper(
                            {option: options[option]},
                            variable_path=option_variable_path)
            else:
                error_path = (variable_path if variable_path
                              else self.__class__.__name__)
                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(
                        error_path, option))

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

    def __eq__(self, other):
        """
        Determines equality by ensuring equality of all attributes, some of
        which may be Options objects themselves.
        """
        if not isinstance(other, self.__class__):
            return False

        return self.__dict__ == other.__dict__


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

class ModeOption(BooleanOption):

    def __init__(self, is_enabled=True, max_k_modes=5):
        """Options for mode estimation

        :ivar is_enabled: boolean option to enable/disable the option.
        :vartype is_enabled: bool
        :ivar top_k_modes: the max number of modes to return, if applicable
        :vartype top_k_modes: int
        """
        self.top_k_modes = max_k_modes
        super().__init__(is_enabled=is_enabled)

    def _validate_helper(self, variable_path='ModeOption'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        errors = super()._validate_helper(variable_path=variable_path)

        if self.top_k_modes is not None and (
                not isinstance(self.top_k_modes, int)
                or self.top_k_modes < 1):
            errors.append("{}.top_k_modes must be either None"
                          " or a positive integer".format(variable_path))
        return errors

class BaseInspectorOptions(BooleanOption):

    def __init__(self, is_enabled=True):
        """
        Base options for all the columns.

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        """
        super().__init__(is_enabled=is_enabled)

    def _validate_helper(self, variable_path='BaseInspectorOptions'):
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


class NumericalOptions(BaseInspectorOptions):

    def __init__(self):
        """
        Options for the Numerical Stats Mixin

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar min: boolean option to enable/disable min
        :vartype min: BooleanOption
        :ivar max: boolean option to enable/disable max
        :vartype max: BooleanOption
        :ivar mode: option to enable/disable mode and set return count
        :vartype mode: ModeOption
        :ivar median: option to enable/disable median
        :vartype median: BooleanOption
        :ivar sum: boolean option to enable/disable sum
        :vartype sum: BooleanOption
        :ivar variance: boolean option to enable/disable variance
        :vartype variance: BooleanOption
        :ivar skewness: boolean option to enable/disable skewness
        :vartype skewness: BooleanOption
        :ivar kurtosis: boolean option to enable/disable kurtosis
        :vartype kurtosis: BooleanOption
        :ivar histogram_and_quantiles: boolean option to enable/disable
            histogram_and_quantiles
        :vartype histogram_and_quantiles: BooleanOption
        :ivar bias_correction : boolean option to enable/disable existence of bias
        :vartype bias: BooleanOption
        :ivar num_zeros: boolean option to enable/disable num_zeros
        :vartype num_zeros: BooleanOption
        :ivar num_negatives: boolean option to enable/disable num_negatives
        :vartype num_negatives: BooleanOption
        :ivar is_numeric_stats_enabled: boolean to enable/disable all numeric
            stats
        :vartype is_numeric_stats_enabled: bool
        """
        self.min = BooleanOption(is_enabled=True)
        self.max = BooleanOption(is_enabled=True)
        self.mode = ModeOption(is_enabled=True)
        self.median = BooleanOption(is_enabled=True)
        self.sum = BooleanOption(is_enabled=True)
        self.variance = BooleanOption(is_enabled=True)
        self.skewness = BooleanOption(is_enabled=True)
        self.kurtosis = BooleanOption(is_enabled=True)
        self.median_abs_deviation = BooleanOption(is_enabled=True)
        self.num_zeros = BooleanOption(is_enabled=True)
        self.num_negatives = BooleanOption(is_enabled=True)
        self.histogram_and_quantiles = HistogramOption()
        # By default, we correct for bias
        self.bias_correction = BooleanOption(is_enabled=True)
        BaseInspectorOptions.__init__(self)

    @property
    def is_numeric_stats_enabled(self):
        """
        Returns the state of numeric stats being enabled / disabled. If any
        numeric stats property is enabled it will return True, otherwise it
        will return False.

        :return: true if any numeric stats property is enabled, otherwise false
        :rtype bool:
        """
        if self.min.is_enabled or self.max.is_enabled \
                or self.mode.is_enabled or self.sum.is_enabled \
                or self.variance.is_enabled or self.skewness.is_enabled \
                or self.kurtosis.is_enabled or self.median.is_enabled \
                or self.median_abs_deviation.is_enabled \
                or self.histogram_and_quantiles.is_enabled \
                or self.num_zeros.is_enabled or self.num_negatives.is_enabled:
            return True
        return False

    @is_numeric_stats_enabled.setter
    def is_numeric_stats_enabled(self, value):
        """
        This property will enable or disable all numeric stats properties:
        min, max, sum, variance, skewness, kurtosis, histogram_and_quantiles,
        num_zeros, num_negatives

        :param value: boolean to enable/disable all numeric stats properties
        :type value: bool
        :return: None
        """
        self.min.is_enabled = value
        self.max.is_enabled = value
        self.mode.is_enabled = value
        self.median.is_enabled = value
        self.sum.is_enabled = value
        self.variance.is_enabled = value
        self.skewness.is_enabled = value
        self.kurtosis.is_enabled = value
        self.median_abs_deviation.is_enabled = value
        self.num_zeros.is_enabled = value
        self.num_negatives.is_enabled = value
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
        for item in ["histogram_and_quantiles", "min", "max", "sum", "mode",
                     "median", "variance", "skewness", "kurtosis",
                     "median_abs_deviation", "bias_correction",
                     "num_zeros", "num_negatives"]:
            if not isinstance(self.properties[item], BooleanOption):
                errors.append("{}.{} must be a BooleanOption."
                              .format(variable_path, item))
            else:
                errors += self.properties[item]._validate_helper(
                    variable_path=variable_path + '.' + item)

        # Error checks for dependent calculations
        sum_disabled = not self.properties["sum"].is_enabled
        var_disabled = not self.properties["variance"].is_enabled
        skew_disabled = not self.properties["skewness"].is_enabled
        kurt_disabled = not self.properties["kurtosis"].is_enabled
        mad_disabled = not self.properties["median_abs_deviation"].is_enabled
        histogram_disabled = \
            not self.properties["histogram_and_quantiles"].is_enabled
        if sum_disabled and not var_disabled:
            errors.append(
                "{}: The numeric stats must toggle on the sum "
                "if the variance is toggled on."
                .format(variable_path))
        if (sum_disabled or var_disabled) and not skew_disabled:
            errors.append("{}: The numeric stats must toggle on the "
                          "sum and variance if skewness is toggled on."
                          .format(variable_path))
        if (sum_disabled or var_disabled or skew_disabled) \
                and not kurt_disabled:
            errors.append("{}: The numeric stats must toggle on sum,"
                          " variance, and skewness if kurtosis is "
                          "toggled on.".format(variable_path))
        if histogram_disabled and not mad_disabled:
            errors.append("{}: The numeric stats must toggle on histogram "
                          "and quantiles if median absolute deviation is "
                          "toggled on.".format(variable_path))

        mode_disabled = not self.properties["mode"].is_enabled
        median_disabled = not self.properties["median"].is_enabled
        histogram_disabled = \
            not self.properties["histogram_and_quantiles"].is_enabled
        if histogram_disabled:
            if not mode_disabled:
                errors.append("{}: The numeric stats must toggle on histogram "
                              "and quantiles if mode is "
                              "toggled on.".format(variable_path))
            if not median_disabled:
                errors.append("{}: The numeric stats must toggle on histogram "
                              "and quantiles if median is "
                              "toggled on.".format(variable_path))

        # warn user if all stats are disabled
        if not errors:
            if not self.is_numeric_stats_enabled:
                variable_path = variable_path + '.numeric_stats' \
                    if variable_path else self.__class__.__name__
                warnings.warn(
                    "{}: The numeric stats are completely disabled."
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
        :ivar mode: option to enable/disable mode and set return count
        :vartype mode: ModeOption
        :ivar median: option to enable/disable median
        :vartype median: BooleanOption
        :ivar sum: boolean option to enable/disable sum
        :vartype sum: BooleanOption
        :ivar variance: boolean option to enable/disable variance
        :vartype variance: BooleanOption
        :ivar skewness: boolean option to enable/disable skewness
        :vartype skewness: BooleanOption
        :ivar kurtosis: boolean option to enable/disable kurtosis
        :vartype kurtosis: BooleanOption
        :ivar histogram_and_quantiles: boolean option to enable/disable
            histogram_and_quantiles
        :vartype histogram_and_quantiles: BooleanOption
        :ivar bias_correction : boolean option to enable/disable existence of bias
        :vartype bias: BooleanOption
        :ivar num_zeros: boolean option to enable/disable num_zeros
        :vartype num_zeros: BooleanOption
        :ivar num_negatives: boolean option to enable/disable num_negatives
        :vartype num_negatives: BooleanOption
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
        :ivar mode: option to enable/disable mode and set return count
        :vartype mode: ModeOption
        :ivar median: option to enable/disable median
        :vartype median: BooleanOption
        :ivar sum: boolean option to enable/disable sum
        :vartype sum: BooleanOption
        :ivar variance: boolean option to enable/disable variance
        :vartype variance: BooleanOption
        :ivar skewness: boolean option to enable/disable skewness
        :vartype skewness: BooleanOption
        :ivar kurtosis: boolean option to enable/disable kurtosis
        :vartype kurtosis: BooleanOption
        :ivar histogram_and_quantiles: boolean option to enable/disable
            histogram_and_quantiles
        :vartype histogram_and_quantiles: BooleanOption
        :ivar bias_correction : boolean option to enable/disable existence of bias
        :vartype bias: BooleanOption
        :ivar num_zeros: boolean option to enable/disable num_zeros
        :vartype num_zeros: BooleanOption
        :ivar num_negatives: boolean option to enable/disable num_negatives
        :vartype num_negatives: BooleanOption
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
        errors += self.precision._validate_helper(variable_path + '.precision')
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
        :ivar mode: option to enable/disable mode and set return count
        :vartype mode: ModeOption
        :ivar median: option to enable/disable median
        :vartype median: BooleanOption
        :ivar sum: boolean option to enable/disable sum
        :vartype sum: BooleanOption
        :ivar variance: boolean option to enable/disable variance
        :vartype variance: BooleanOption
        :ivar skewness: boolean option to enable/disable skewness
        :vartype skewness: BooleanOption
        :ivar kurtosis: boolean option to enable/disable kurtosis
        :vartype kurtosis: BooleanOption
        :ivar bias_correction : boolean option to enable/disable existence of bias
        :vartype bias: BooleanOption
        :ivar histogram_and_quantiles: boolean option to enable/disable
            histogram_and_quantiles
        :vartype histogram_and_quantiles: BooleanOption
        :ivar num_zeros: boolean option to enable/disable num_zeros
        :vartype num_zeros: BooleanOption
        :ivar num_negatives: boolean option to enable/disable num_negatives
        :vartype num_negatives: BooleanOption
        :ivar is_numeric_stats_enabled: boolean to enable/disable all numeric
            stats
        :vartype is_numeric_stats_enabled: bool
        """
        NumericalOptions.__init__(self)
        self.vocab = BooleanOption(is_enabled=True)
        self.num_zeros = BooleanOption(is_enabled=False)
        self.num_negatives = BooleanOption(is_enabled=False)

    def _validate_helper(self, variable_path='TextOptions'):
        """
        Validates the options do not conflict and cause errors. Also validates
        that some options (num_zeros and num_negatives) are set to be disabled
        by default

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

        if self.properties["num_zeros"].is_enabled:
            errors.append("{}.num_zeros should always be disabled,"
                          " num_zeros.is_enabled = False"
                          .format(variable_path))

        if self.properties["num_negatives"].is_enabled:
            errors.append("{}.num_negatives should always be disabled,"
                          " num_negatives.is_enabled = False"
                          .format(variable_path))
        return errors

    @property
    def is_numeric_stats_enabled(self):
        """
        Returns the state of numeric stats being enabled / disabled. If any
        numeric stats property is enabled it will return True, otherwise it
        will return False. Although it seems redundant, this method is needed
        in order for the function below, the setter function
        also called is_numeric_stats_enabled, to properly work.

        :return: true if any numeric stats property is enabled, otherwise false
        :rtype bool:
        """
        if self.min.is_enabled or self.max.is_enabled \
                or self.sum.is_enabled or self.mode.is_enabled \
                or self.variance.is_enabled or self.skewness.is_enabled \
                or self.kurtosis.is_enabled or self.median.is_enabled \
                or self.median_abs_deviation.is_enabled \
                or self.histogram_and_quantiles.is_enabled:
            return True
        return False

    @is_numeric_stats_enabled.setter
    def is_numeric_stats_enabled(self, value):
        """
        This property will enable or disable all numeric stats properties:
        min, max, sum, variance, skewness, kurtosis, histogram_and_quantiles,

        :param value: boolean to enable/disable all numeric stats properties
        :type value: bool
        :return: None
        """
        self.min.is_enabled = value
        self.max.is_enabled = value
        self.mode.is_enabled = value
        self.median.is_enabled = value
        self.sum.is_enabled = value
        self.variance.is_enabled = value
        self.skewness.is_enabled = value
        self.kurtosis.is_enabled = value
        self.median_abs_deviation.is_enabled = value
        self.histogram_and_quantiles.is_enabled = value

class DateTimeOptions(BaseInspectorOptions):

    def __init__(self):
        """
        Options for the Datetime Column

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        """
        BaseInspectorOptions.__init__(self)

    def _validate_helper(self, variable_path='DateTimeOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        return super()._validate_helper(variable_path)


class OrderOptions(BaseInspectorOptions):

    def __init__(self):
        """
        Options for the Order Column

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        """
        BaseInspectorOptions.__init__(self)

    def _validate_helper(self, variable_path='OrderOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        return super()._validate_helper(variable_path)


class CategoricalOptions(BaseInspectorOptions):

    def __init__(self, is_enabled=True, top_k_categories=None):
        """
        Options for the Categorical Column

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar top_k_categories: number of categories to be displayed when called
        :vartype top_k_categories: [None, int]
        """
        BaseInspectorOptions.__init__(self, is_enabled=is_enabled)
        self.top_k_categories = top_k_categories

    def _validate_helper(self, variable_path='CategoricalOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        errors = super()._validate_helper(variable_path)
        if self.top_k_categories is not None and (
                not isinstance(self.top_k_categories, int)
                or self.top_k_categories < 1):
            errors.append("{}.top_k_categories must be either None"
                          " or a positive integer".format(variable_path))
        return errors


class CorrelationOptions(BaseInspectorOptions):

    def __init__(self, is_enabled=False, columns=None):
        """
        Options for the Correlation between Columns

        :ivar is_enabled: boolean option to enable/disable.
        :vartype is_enabled: bool
        :ivar columns: Columns considered to calculate correlation
        :vartype columns: list()
        """
        BaseInspectorOptions.__init__(self, is_enabled=is_enabled)
        self.columns = columns

    def _validate_helper(self, variable_path='CorrelationOptions'):
        """
        Validates the options do not conflict and cause errors.

        :param variable_path: current path to variable set.
        :type variable_path: str
        :return: list of errors (if raise_error is false)
        :rtype: list(str)
        """
        errors = super()._validate_helper(variable_path=variable_path)

        if (self.columns is not None and
                (not isinstance(self.columns, list)
                 or len(self.columns) <= 1
                 or not all(isinstance(item, str)
                            for item in self.columns))):
            errors.append("{}.columns must be None "
                          "or list of strings "
                          "with at least two elements.".format(variable_path))
        return errors


class DataLabelerOptions(BaseInspectorOptions):

    def __init__(self):
        """
        Options for the Data Labeler Column.

        :ivar is_enabled: boolean option to enable/disable the column.
        :vartype is_enabled: bool
        :ivar data_labeler_dirpath: String to load data labeler from
        :vartype data_labeler_dirpath: str
        :ivar max_sample_size: Int to decide sample size
        :vartype max_sample_size: int
        :ivar data_labeler_object: DataLabeler object used in profiler
        :vartype max_sample_size: BaseDataLabeler
        """
        BaseInspectorOptions.__init__(self)
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
                 for k, v in self.__dict__.items() if k != 'data_labeler_object'}
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

class TextProfilerOptions(BaseInspectorOptions):

    def __init__(self, is_enabled=True, is_case_sensitive=True,
                 stop_words=None, top_k_chars=None,
                 top_k_words=None):
        """
        Constructs the TextProfilerOption object with default values.

        :ivar is_enabled: boolean option to enable/disable the option.
        :vartype is_enabled: bool
        :ivar is_case_sensitive: option set for case sensitivity.
        :vartype is_case_sensitive: bool
        :ivar stop_words: option set for stop words.
        :vartype stop_words: Union[None, list(str)]
        :ivar top_k_chars: option set for number of top common characters.
        :vartype top_k_chars: Union[None, int]
        :ivar top_k_words: option set for number of top common words.
        :vartype top_k_words: Union[None, int]
        :ivar words: option set for word update.
        :vartype words: BooleanOption
        :ivar vocab: option set for vocab update.
        :vartype vocab: BooleanOption
        """
        super().__init__(is_enabled=is_enabled)
        self.is_case_sensitive = is_case_sensitive
        self.stop_words = stop_words
        self.top_k_chars = top_k_chars
        self.top_k_words = top_k_words
        self.vocab = BooleanOption(is_enabled=True)
        self.words = BooleanOption(is_enabled=True)

    def _validate_helper(self, variable_path='TextProfilerOptions'):
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

        if not isinstance(self.is_case_sensitive, bool):
            errors.append("{}.is_case_sensitive must be a Boolean."
                          .format(variable_path))

        if (self.stop_words is not None and
                (not isinstance(self.stop_words, list)
                 or not all(isinstance(item, str)
                            for item in self.stop_words))):
            errors.append("{}.stop_words must be None "
                          "or list of strings.".format(variable_path))

        if (self.top_k_chars is not None and
                (not isinstance(self.top_k_chars, int) or self.top_k_chars <= 0)):
            errors.append("{}.top_k_chars must be None "
                          "or positive integer.".format(variable_path))

        if (self.top_k_words is not None and
                (not isinstance(self.top_k_words, int) or self.top_k_words <= 0)):
            errors.append("{}.top_k_words must be None "
                          "or positive integer.".format(variable_path))

        if not isinstance(self.vocab, BooleanOption):
            errors.append("{}.vocab must be a BooleanOption "
                          "object."
                          .format(variable_path))

        if not isinstance(self.words, BooleanOption):
            errors.append("{}.words must be a BooleanOption "
                          "object."
                          .format(variable_path))

        return errors


class StructuredOptions(BaseOption):

    def __init__(self, null_values=None):
        """
        Constructs the StructuredOptions object with default values.

        :param null_values: null values we input.
        :vartype null_values: Union[None, dict]
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
        :ivar correlation: option set for correlation profiling.
        :vartype correlation: CorrelationOptions
        :ivar chi2_homogeneity: option set for chi2_homogeneity matrix
        :vartype chi2_homogeneity: BooleanOption()
        :ivar null_values: option set for defined null values
        :vartype null_values: Union[None, dict]
        """
        # Option variables
        self.multiprocess = BooleanOption()
        self.int = IntOptions()
        self.float = FloatOptions()
        self.datetime = DateTimeOptions()
        self.text = TextOptions()
        self.order = OrderOptions()
        self.category = CategoricalOptions()
        self.data_labeler = DataLabelerOptions()
        self.correlation = CorrelationOptions()
        self.chi2_homogeneity = BooleanOption(is_enabled=True)
        # Non-Option variables
        self.null_values = null_values

    @property
    def enabled_profiles(self):
        """Returns a list of the enabled profilers for columns."""
        enabled_profiles = list()
        # null_values does not have is_enabled
        properties = self.properties
        properties.pop('null_values')
        for key, value in properties.items():
            if value.is_enabled:
                enabled_profiles.append(key)
        return enabled_profiles

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
            ('data_labeler', DataLabelerOptions),
            ('correlation', CorrelationOptions),
            ('chi2_homogeneity', BooleanOption)
        ])
        properties = self.properties
        properties.pop('null_values')
        for column in (properties):
            if not isinstance(self.properties[column], prop_check[column]):
                errors.append("{}.{} must be a(n) {}.".format(
                    variable_path, column, prop_check[column].__name__))
            else:
                errors += self.properties[column]._validate_helper(
                    variable_path=(variable_path + '.' + column
                                   if variable_path else column))

        if self.null_values is not None and not (
                isinstance(self.null_values, dict) and
                all(isinstance(key, str) and
                    (isinstance(value, re.RegexFlag) or value == 0)
                    for key, value in self.null_values.items())):
            errors.append("{}.null_values must be either None or "
                          "a dictionary that contains keys of type str "
                          "and values == 0 or are instances of "
                          "a re.RegexFlag".format(variable_path))

        if isinstance(self.category, CategoricalOptions) and \
                isinstance(self.chi2_homogeneity, BooleanOption) and not \
                self.category.is_enabled and self.chi2_homogeneity.is_enabled:
            errors.append("Categorical statistics must be enabled if "
                          "Chi-squared test in enabled.".format(variable_path))

        return errors


class UnstructuredOptions(BaseOption):

    def __init__(self):
        """
        Constructs the UnstructuredOptions object with default values.

        :ivar text: option set for text profiling.
        :vartype text: TextProfilerOptions
        :ivar data_labeler: option set for data_labeler profiling.
        :vartype data_labeler: DataLabelerOptions
        """
        self.text = TextProfilerOptions()
        self.data_labeler = DataLabelerOptions()

    @property
    def enabled_profiles(self):
        """Returns a list of the enabled profilers."""
        enabled_profiles = list()
        for key, value in self.properties.items():
            if value.is_enabled:
                enabled_profiles.append(key)
        return enabled_profiles

    def _validate_helper(self, variable_path='UnstructuredOptions'):
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
            ('text', TextProfilerOptions),
            ('data_labeler', DataLabelerOptions)
        ])

        for prop in self.properties:
            if not isinstance(self.properties[prop], prop_check[prop]):
                errors.append("{}.{} must be a(n) {}.".format(
                    variable_path, prop, prop_check[prop].__name__))
            else:
                errors += self.properties[prop]._validate_helper(
                    variable_path=(variable_path + '.' + prop
                                   if variable_path else prop))
        return errors


class ProfilerOptions(BaseOption):

    def __init__(self):
        """
        Initializes the ProfilerOptions object.

        :ivar structured_options: option set for structured dataset profiling.
        :vartype structured_options: StructuredOptions
        :ivar unstructured_options: option set for unstructured dataset profiling.
        :vartype unstructured_options: UnstructuredOptions
        """
        self.structured_options = StructuredOptions()
        self.unstructured_options = UnstructuredOptions()

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

        if not isinstance(self.unstructured_options, UnstructuredOptions):
            errors.append("{}.unstructured_options must be an UnstructuredOptions."
                          .format(variable_path))

        errors += self.unstructured_options._validate_helper(
            variable_path=variable_path + '.unstructured_options')

        return errors

    def set(self, options):
        """
        Overwrites BaseOption.set since the type (unstructured/structured) may
        need to be specified if the same options exist within both
        self.structured_options and self.unstructured_options

        :param options: Dictionary of options to set
        :type options: dict
        :Return: None
        """
        if not isinstance(options, dict):
            raise ValueError("The options must be a dictionary.")

        # Options that need further specification
        overlap_options = {"data_labeler_object", "data_labeler_dirpath",
                           "text.is_enabled", "text.vocab"}

        # Specification needed for overlap_options above
        option_specifications = {
            "*", "structured_options", "unstructured_options"}

        # Function to see if any overlap options present in option being set
        def overlap_opt_set(opt):
            for overlap_opt in overlap_options:
                if overlap_opt in opt:
                    return True
            return False

        overlap_dict = dict()
        for option in options:
            # Tried to set an overlap option without specifying struct/unstruct
            if (option.split(".")[0] not in option_specifications
                    and overlap_opt_set(option)):
                overlap_dict[option] = options[option]

        if overlap_dict:
            raise ValueError(f"Attempted to set options {overlap_dict} in "
                             f"ProfilerOptions without specifying whether "
                             f"to set them for StructuredOptions or "
                             f"UnstructuredOptions.")

        # Set options as normal if none were overlapping
        self._set_helper(options, variable_path='')
