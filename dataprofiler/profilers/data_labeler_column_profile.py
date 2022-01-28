import operator

import numpy as np

from . import BaseColumnProfiler
from . import utils
from ..labelers.data_labelers import DataLabeler
from .profiler_options import DataLabelerOptions


class DataLabelerColumn(BaseColumnProfiler):
    
    type = "data_labeler"
    
    def __init__(self, name, options=None):
        """
        Initialization of Data Label profiling for structured datasets.

        :param name: name of column being profiled
        :type name: String
        :param options: Options for the data labeler column
        :type options: DataLabelerOptions
        """
        BaseColumnProfiler.__init__(self, name)

        self._max_sample_size = 1000
        if options:
            if not isinstance(options, DataLabelerOptions):
                raise ValueError("DataLabelerColumn parameter 'options' must be"
                                 " of type DataLabelerOptions.")
            if options.max_sample_size:
                self._max_sample_size = options.max_sample_size

        self.data_labeler = None
        if options and options.data_labeler_object:
            self.data_labeler = options.data_labeler_object
        if self.data_labeler is None:
            data_labeler_dirpath = None
            if options:
                data_labeler_dirpath = options.data_labeler_dirpath

            self.data_labeler = DataLabeler(
                labeler_type='structured',
                dirpath=data_labeler_dirpath,
                load_options=None)

        self.reverse_label_mapping = self.data_labeler.reverse_label_mapping
        num_labels = self.data_labeler.model.num_labels

        # remove PAD from output (reserved zero index)
        if self.data_labeler.model.requires_zero_mapping:
            self.reverse_label_mapping.pop(0, None)
            num_labels -= 1

        self._possible_data_labels = list(self.reverse_label_mapping.values())
        self._possible_data_labels = [  # sort the data_labels based on index
            x for _, x in sorted(zip(
                self.reverse_label_mapping.keys(), self._possible_data_labels)
            )
        ]
        self.rank_distribution = dict(
            [(key, 0) for key in self._possible_data_labels])
        self._sum_predictions = np.zeros(num_labels)

        # rank distribution variables
        self._top_k_voting = 1
        self._min_voting_prob = 0.20

        # data label prediction variables
        self._min_prob_differential = 0.20
        self._top_k_labels = 3
        self._min_top_label_prob = 0.35

        self.__calculations = {}
        self._filter_properties_w_options(self.__calculations, options)

        self.thread_safe = False

    @staticmethod
    def assert_equal_conditions(data_labeler, data_labeler2):
        """
        Ensures data labelers have the same values. Raises error otherwise.
        
        :param data_labeler: first data_labeler
        :param data_labeler2: second data_labeler
        :type data_labeler: DataLabelerColumn
        :type data_labeler2: DataLabelerColumn
        :return: None
        """
        if data_labeler._top_k_voting != data_labeler2._top_k_voting:
            raise ValueError("Sorry, can't merge profiles: {} are not the same "
                             "in both DataLabeler Profilers being merged, "
                             "as required".format('_top_k_voting'))
        if data_labeler._min_voting_prob != data_labeler2._min_voting_prob:
            raise ValueError("Sorry, can't merge profiles: {} are not the same "
                             "in both DataLabeler Profilers being merged, "
                             "as required".format('_min_voting_prob'))
        if data_labeler._min_prob_differential != data_labeler2._min_prob_differential:
            raise ValueError("Sorry, can't merge profiles: {} are not the same "
                             "in both DataLabeler Profilers being merged, "
                             "as required".format('_min_voting_prob'))
        if data_labeler._top_k_labels != data_labeler2._top_k_labels:
            raise ValueError("Sorry, can't merge profiles: {} are not the same "
                             "in both DataLabeler Profilers being merged, "
                             "as required".format('_top_k_labels'))
        if data_labeler._min_top_label_prob != data_labeler2._min_top_label_prob:
            raise ValueError("Sorry, can't merge profiles: {} are not the same "
                             "in both DataLabeler Profilers being merged, "
                             "as required".format('_min_top_label_prob'))
        if data_labeler._possible_data_labels != data_labeler2._possible_data_labels:
            raise ValueError("Sorry, can't merge profiles: {} are not the same "
                             "in both DataLabeler Profilers being merged, "
                             "as required".format('_possible_data_labels'))
        if data_labeler.data_labeler != data_labeler2.data_labeler:
            raise ValueError("Sorry, can't merge profiles: DataLabeler1 and "
                             "DataLabeler2 have different Models for labeling.")

    def __add__(self, other):
        """
        Merges the properties of two DataLabelerColumn profiles
        
        :param self: first profile
        :param other: second profile
        :type self: DataLabelerColumn
        :type other: DataLabelerColumn
        :return: New DataLabelerColumn merged profile
        """
        if not isinstance(other, DataLabelerColumn):
            raise TypeError("Unsupported operand type(s) for +: "
                            "'DataLabelerColumn' and '{}'".format(
                                other.__class__.__name__))
        
        if self.data_labeler != other.data_labeler \
                or self._max_sample_size != other._max_sample_size:
            raise AttributeError("Cannot merge. The data labeler and/or the max "
                                 "sample size are not the same for both column "
                                 "profiles.")
        
        self.assert_equal_conditions(self, other)

        # recreate options so the DataLabeler is transferred and not duplicated
        options = DataLabelerOptions()
        options.max_sample_size = self._max_sample_size
        options.data_labeler_object = self.data_labeler

        merged_profile = DataLabelerColumn(self.name, options)
        BaseColumnProfiler._add_helper(merged_profile, self, other)

        #Set all common variables
        merged_profile.data_labeler = self.data_labeler
        merged_profile._possible_data_labels = self._possible_data_labels
        merged_profile._top_k_voting = self._top_k_voting
        merged_profile._min_voting_prob = self._min_voting_prob
        merged_profile._min_prob_differential = self._min_prob_differential
        merged_profile._top_k_labels = self._top_k_labels
        merged_profile._min_top_label_prob = self._min_top_label_prob
        merged_profile._max_sample_size = self._max_sample_size
        merged_profile._top_k_voting = self._top_k_voting

        #Combine rank distribution
        merged_profile.rank_distribution = {key: self.rank_distribution.get(key, 0) +
                                                 other.rank_distribution.get(key, 0) for key in
                                            set(self.rank_distribution) | set(other.rank_distribution)}

        #Combine Sum Predictions
        merged_profile._sum_predictions = self._sum_predictions + other._sum_predictions
        
        self._merge_calculations(merged_profile.__calculations,
                                 self.__calculations,
                                 other.__calculations)
        return merged_profile

    @property
    def data_label(self):
        """
        Returns the data labels which best fit the data it has seen based on
        the DataLabeler used. Data labels must be within the minimum probability
        differential of the top predicted value. If nothing is more than
        minimum top label value, it says it could not determine the data label.
        """
        if not self.sample_size:
            return None

        ranks_items = self.rank_distribution.items()
        ordered_top_k_rank = np.array(sorted(
            ranks_items, key=operator.itemgetter(1), reverse=True
        ))[:self._top_k_labels]
        top_k_probabilities = np.fromiter(
            map(operator.itemgetter(1), ordered_top_k_rank), dtype=float
        ) / sum(self.rank_distribution.values())
        is_value_close = top_k_probabilities - top_k_probabilities[0] >= \
                                                    -self._min_prob_differential

        data_label = '|'.join(map(
            operator.itemgetter(0), ordered_top_k_rank[is_value_close]
        ))
        top_label = ordered_top_k_rank[0][0]
        if self.label_representation[top_label] < self._min_top_label_prob:
            return "could not determine"
        return data_label

    @property
    def avg_predictions(self):
        """
        Averages all sample predictions for each data label.
        """
        if not self.sample_size:
            return None

        avg_predictions = self._sum_predictions / self.sample_size
        return dict(zip(self._possible_data_labels, avg_predictions))

    @property
    def label_representation(self):
        """
        Representation of label found within the dataset based on ranked voting.
        When top_k=1, this is simply the distribution of data labels found
        within the dataset.
        """
        if not self.sample_size:
            return None

        label_representation = dict([(key, 0) for key in self._possible_data_labels])
        total_votes = max(1, sum(list(self.rank_distribution.values())))
        for key in label_representation:
            label_representation[key] = \
                self.rank_distribution[key] / total_votes
        return label_representation

    @property
    def profile(self):
        """
        Property for profile. Returns the profile of the column.
        """
        profile = {
            "data_label": self.data_label,
            "avg_predictions": self.avg_predictions,
            "data_label_representation": self.label_representation,
            "times": self.times
        }
        return profile

    def diff(self, other_profile, options=None):
        """
        Generates the differences between the orders of two DataLabeler columns

        :return: Dict containing the differences between orders in their
        appropriate output formats
        :rtype: dict
        """
        differences = super().diff(other_profile, options)

        labels = self.data_label.split('|')
        avg_preds = self.avg_predictions
        label_rep = self.label_representation
        other_labels = other_profile.data_label.split('|')
        other_avg_preds = other_profile.avg_predictions
        other_label_rep = other_profile.label_representation

        differences = {
            "data_label": utils.find_diff_of_lists_and_sets(labels, other_labels),
            "avg_predictions": utils.find_diff_of_dicts(avg_preds, other_avg_preds),
            "label_representation": utils.find_diff_of_dicts(label_rep, other_label_rep)
        }
        return differences

    @BaseColumnProfiler._timeit(name='data_labeler_predict')
    def _update_predictions(self, df_series, prev_dependent_properties=None,
                            subset_properties=None):
        """
        Method for updating the column profile properties with a cleaned
        dataset and the known profile of the dataset.

        :param prev_dependent_properties: Contains all the previous properties
        that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: dict
        :param df_series: Data to be profiled
        :type df_series: pandas.DataFrame
        :return: None
        """
        predictions = self.data_labeler.predict(
            df_series, predict_options=dict(show_confidences=True))
        # remove PAD from output (reserved zero index)
        if self.data_labeler.model.requires_zero_mapping:
            ignore_value = 0  # PAD index
            predictions['conf'] = np.delete(
                predictions['conf'], ignore_value, axis=1)
        sum_predictions = np.sum(predictions['conf'], axis=0)
        self._sum_predictions += sum_predictions

        rank_predictions = np.argpartition(
            predictions['conf'], axis=1, kth=-self._top_k_voting
        )
        start_index = 0
        if self.data_labeler.model.requires_zero_mapping:
            start_index = 1
        for i in range(rank_predictions.shape[0]):
            sorted_rank = rank_predictions[i][-self._top_k_voting:]
            sorted_rank = sorted_rank[np.argsort(predictions['conf'][i][sorted_rank])]
            for rank_position, value in enumerate(sorted_rank):
                if predictions['conf'][i][value] > self._min_voting_prob:
                    self.rank_distribution[
                        self.reverse_label_mapping[value + start_index]
                    ] += rank_position + 1

    def _update_helper(self, df_series_clean, profile):
        """
        Updating the column profile properties
        
        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: float profile dictionary
        :type profile: dict
        :return: None
        """
        self._update_column_base_properties(profile)

    def update(self, df_series):
        """
        Updates the column profile.
        
        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: None
        """
        if len(df_series) == 0:
            return self
        
        sample_size = min(len(df_series), self._max_sample_size)
        df_series = df_series.sample(sample_size)

        profile = dict(sample_size=sample_size)
        self._update_predictions(df_series=df_series,
                                 prev_dependent_properties={},
                                 subset_properties=profile)
        BaseColumnProfiler._perform_property_calcs(
            self, self.__calculations, df_series=df_series,
            prev_dependent_properties={}, subset_properties=profile)
        self._update_helper(df_series, profile)

        return self
