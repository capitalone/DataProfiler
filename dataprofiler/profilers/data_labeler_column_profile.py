"""Contains class for for profiling data labeler col."""
from __future__ import annotations

import operator
from typing import Dict, List, Optional, cast

import numpy as np
from pandas import DataFrame, Series

from ..labelers.base_data_labeler import BaseDataLabeler
from ..labelers.data_labelers import DataLabeler
from . import BaseColumnProfiler, utils
from .profiler_options import DataLabelerOptions


class DataLabelerColumn(BaseColumnProfiler):
    """Sublass of BaseColumnProfiler for profiling data labeler col."""

    type = "data_labeler"

    def __init__(self, name: Optional[str], options: DataLabelerOptions = None) -> None:
        """
        Initialize Data Label profiling for structured datasets.

        :param name: name of column being profiled
        :type name: String
        :param options: Options for the data labeler column
        :type options: DataLabelerOptions
        """
        BaseColumnProfiler.__init__(self, name)

        self._max_sample_size = 1000
        if options:
            if not isinstance(options, DataLabelerOptions):
                raise ValueError(
                    "DataLabelerColumn parameter 'options' must be"
                    " of type DataLabelerOptions."
                )
            if options.max_sample_size:
                self._max_sample_size = options.max_sample_size

        self.data_labeler: BaseDataLabeler = None  # type: ignore[assignment]
        if options and options.data_labeler_object:
            self.data_labeler = options.data_labeler_object
        if self.data_labeler is None:
            data_labeler_dirpath = None
            if options:
                data_labeler_dirpath = options.data_labeler_dirpath

            self.data_labeler = DataLabeler(
                labeler_type="structured",
                dirpath=data_labeler_dirpath,
                load_options=None,
            )

        self._reverse_label_mapping: Optional[Dict] = None
        self._possible_data_labels: Optional[List[str]] = None
        self._rank_distribution: Dict[str, int] = None  # type: ignore[assignment]
        self._sum_predictions: np.ndarray = None  # type: ignore[assignment]

        # rank distribution variables
        self._top_k_voting: int = 1
        self._min_voting_prob: float = 0.20

        # data label prediction variables
        self._min_prob_differential: float = 0.20
        self._top_k_labels: int = 3
        self._min_top_label_prob: float = 0.35

        self.__calculations: Dict = {}
        self._filter_properties_w_options(self.__calculations, options)

        self.thread_safe = False

    @staticmethod
    def assert_equal_conditions(
        data_labeler: DataLabelerColumn, data_labeler2: DataLabelerColumn
    ) -> None:
        """
        Ensure data labelers have the same values. Raise error otherwise.

        :param data_labeler: first data_labeler
        :param data_labeler2: second data_labeler
        :type data_labeler: DataLabelerColumn
        :type data_labeler2: DataLabelerColumn
        :return: None
        """
        if data_labeler._top_k_voting != data_labeler2._top_k_voting:
            raise ValueError(
                "Sorry, can't merge profiles: {} are not the same "
                "in both DataLabeler Profilers being merged, "
                "as required".format("_top_k_voting")
            )
        if data_labeler._min_voting_prob != data_labeler2._min_voting_prob:
            raise ValueError(
                "Sorry, can't merge profiles: {} are not the same "
                "in both DataLabeler Profilers being merged, "
                "as required".format("_min_voting_prob")
            )
        if data_labeler._min_prob_differential != data_labeler2._min_prob_differential:
            raise ValueError(
                "Sorry, can't merge profiles: {} are not the same "
                "in both DataLabeler Profilers being merged, "
                "as required".format("_min_voting_prob")
            )
        if data_labeler._top_k_labels != data_labeler2._top_k_labels:
            raise ValueError(
                "Sorry, can't merge profiles: {} are not the same "
                "in both DataLabeler Profilers being merged, "
                "as required".format("_top_k_labels")
            )
        if data_labeler._min_top_label_prob != data_labeler2._min_top_label_prob:
            raise ValueError(
                "Sorry, can't merge profiles: {} are not the same "
                "in both DataLabeler Profilers being merged, "
                "as required".format("_min_top_label_prob")
            )
        if data_labeler.possible_data_labels != data_labeler2.possible_data_labels:
            raise ValueError(
                "Sorry, can't merge profiles: {} are not the same "
                "in both DataLabeler Profilers being merged, "
                "as required".format("possible_data_labels")
            )
        if data_labeler.data_labeler != data_labeler2.data_labeler:
            raise ValueError(
                "Sorry, can't merge profiles: DataLabeler1 and "
                "DataLabeler2 have different Models for labeling."
            )

    def __add__(self, other: DataLabelerColumn) -> DataLabelerColumn:
        """
        Merge the properties of two DataLabelerColumn profiles.

        :param self: first profile
        :param other: second profile
        :type self: DataLabelerColumn
        :type other: DataLabelerColumn
        :return: New DataLabelerColumn merged profile
        """
        if not isinstance(other, DataLabelerColumn):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'DataLabelerColumn' and '{}'".format(other.__class__.__name__)
            )

        if (
            self.data_labeler != other.data_labeler
            or self._max_sample_size != other._max_sample_size
        ):
            raise AttributeError(
                "Cannot merge. The data labeler and/or the max "
                "sample size are not the same for both column "
                "profiles."
            )

        self.assert_equal_conditions(self, other)

        # recreate options so the DataLabeler is transferred and not duplicated
        options = DataLabelerOptions()
        options.max_sample_size = self._max_sample_size
        options.data_labeler_object = self.data_labeler

        merged_profile = DataLabelerColumn(self.name, options)
        BaseColumnProfiler._add_helper(merged_profile, self, other)

        # Set all common variables
        merged_profile.data_labeler = self.data_labeler
        merged_profile._possible_data_labels = self._possible_data_labels
        merged_profile._top_k_voting = self._top_k_voting
        merged_profile._min_voting_prob = self._min_voting_prob
        merged_profile._min_prob_differential = self._min_prob_differential
        merged_profile._top_k_labels = self._top_k_labels
        merged_profile._min_top_label_prob = self._min_top_label_prob
        merged_profile._max_sample_size = self._max_sample_size
        merged_profile._top_k_voting = self._top_k_voting

        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )

        # Combine rank distribution
        if self.sample_size or other.sample_size:
            merged_profile._rank_distribution = {
                key: self._rank_distribution.get(key, 0)
                + other._rank_distribution.get(key, 0)
                for key in set(self._rank_distribution) | set(other._rank_distribution)
            }

            # Combine Sum Predictions
            merged_profile._sum_predictions = (
                self._sum_predictions + other._sum_predictions
            )
        return merged_profile

    @property
    def reverse_label_mapping(self) -> Dict:
        """Return reverse label mapping."""
        if self._reverse_label_mapping is None:
            self._reverse_label_mapping = self.data_labeler.reverse_label_mapping
            if self.data_labeler.model.requires_zero_mapping:
                self._reverse_label_mapping.pop(0, None)
        return self._reverse_label_mapping

    @property
    def possible_data_labels(self) -> List[str]:
        """Return possible data labels."""
        if self._possible_data_labels is None:
            self._possible_data_labels = list(self.reverse_label_mapping.values())
            self._possible_data_labels = [  # sort the data_labels based on index
                x
                for _, x in sorted(
                    zip(self.reverse_label_mapping.keys(), self._possible_data_labels)
                )
            ]
        return self._possible_data_labels

    @property
    def rank_distribution(self) -> Dict[str, int]:
        """Return rank distribution."""
        if self._rank_distribution is None:
            self._rank_distribution = {key: 0 for key in self.possible_data_labels}
        return self._rank_distribution

    @property
    def sum_predictions(self) -> np.ndarray:
        """Sum predictions."""
        if self._sum_predictions is None:
            num_labels = self.data_labeler.model.num_labels
            if self.data_labeler.model.requires_zero_mapping:
                num_labels -= 1
            self._sum_predictions = np.zeros(num_labels)
        return self._sum_predictions

    @sum_predictions.setter
    def sum_predictions(self, value: np.ndarray) -> None:
        """Update sum predictions."""
        self._sum_predictions = value

    @property
    def data_label(self) -> Optional[str]:
        """
        Return data labels which best fit data it has seen based on DataLabeler used.

        Data labels must be within the minimum probability
        differential of the top predicted value. If nothing is more than
        minimum top label value, it says it could not determine the data label.
        """
        if not self.sample_size:
            return None

        ranks_items = self.rank_distribution.items()
        ordered_top_k_rank = np.array(
            sorted(ranks_items, key=operator.itemgetter(1), reverse=True)
        )[: self._top_k_labels]
        top_k_probabilities = np.fromiter(
            map(operator.itemgetter(1), ordered_top_k_rank), dtype=float
        ) / sum(self.rank_distribution.values())
        is_value_close = (
            top_k_probabilities - top_k_probabilities[0] >= -self._min_prob_differential
        )

        data_label = "|".join(
            map(operator.itemgetter(0), ordered_top_k_rank[is_value_close])
        )
        top_label = ordered_top_k_rank[0][0]
        if cast(Dict, self.label_representation)[top_label] < self._min_top_label_prob:
            return "could not determine"
        return data_label

    @property
    def avg_predictions(self) -> Optional[Dict[str, float]]:
        """Average all sample predictions for each data label."""
        if not self.sample_size:
            return None

        avg_predictions = self.sum_predictions / self.sample_size
        return dict(zip(self.possible_data_labels, avg_predictions))

    @property
    def label_representation(self) -> Optional[Dict[str, float]]:
        """
        Represent label found within the dataset based on ranked voting.

        When top_k=1, this is simply the distribution of data labels found
        within the dataset.
        """
        if not self.sample_size:
            return None

        label_representation: Dict[str, float] = {
            key: 0 for key in self.possible_data_labels
        }
        total_votes = max(1, sum(list(self.rank_distribution.values())))
        for key in label_representation:
            label_representation[key] = self.rank_distribution[key] / total_votes
        return label_representation

    @property
    def profile(self) -> Dict:
        """Return the profile of the column."""
        profile = {
            "data_label": self.data_label,
            "avg_predictions": self.avg_predictions,
            "data_label_representation": self.label_representation,
            "times": self.times,
        }
        return profile

    def report(self, remove_disabled_flag: bool = False) -> Dict:
        """
        Return report.

        Private abstract method.

        :param remove_disabled_flag: flag to determine if disabled
            options should be excluded in the report.
        :type remove_disabled_flag: boolean
        """
        return self.profile

    def diff(self, other_profile: DataLabelerColumn, options: Dict = None) -> Dict:
        """
        Generate differences between the orders of two DataLabeler columns.

        :return: Dict containing the differences between orders in their
        appropriate output formats
        :rtype: dict
        """
        differences = super().diff(other_profile, options)

        self_labels = None
        if self.sample_size:
            self_labels = cast(str, self.data_label).split("|")
        other_labels = None
        if other_profile.sample_size:
            other_labels = cast(str, other_profile.data_label).split("|")
        avg_preds = self.avg_predictions
        label_rep = self.label_representation
        other_avg_preds = other_profile.avg_predictions
        other_label_rep = other_profile.label_representation

        differences = {
            "data_label": utils.find_diff_of_lists_and_sets(self_labels, other_labels),
            "avg_predictions": utils.find_diff_of_dicts(avg_preds, other_avg_preds),
            "label_representation": utils.find_diff_of_dicts(
                label_rep, other_label_rep
            ),
        }
        return differences

    @BaseColumnProfiler._timeit(name="data_labeler_predict")
    def _update_predictions(
        self,
        df_series: DataFrame,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """
        Update col profile properties with clean dataset and its known profile.

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
            df_series, predict_options=dict(show_confidences=True)
        )
        # remove PAD from output (reserved zero index)
        if self.data_labeler.model.requires_zero_mapping:
            ignore_value = 0  # PAD index
            predictions["conf"] = np.delete(predictions["conf"], ignore_value, axis=1)
        sum_predictions = np.sum(predictions["conf"], axis=0)
        self.sum_predictions += sum_predictions

        rank_predictions = np.argpartition(
            predictions["conf"], axis=1, kth=-self._top_k_voting
        )
        start_index = 0
        if self.data_labeler.model.requires_zero_mapping:
            start_index = 1
        for i in range(rank_predictions.shape[0]):
            sorted_rank = rank_predictions[i][-self._top_k_voting :]
            sorted_rank = sorted_rank[np.argsort(predictions["conf"][i][sorted_rank])]
            for rank_position, value in enumerate(sorted_rank):
                if predictions["conf"][i][value] > self._min_voting_prob:
                    self.rank_distribution[
                        self.reverse_label_mapping[value + start_index]
                    ] += (rank_position + 1)

    def _update_helper(self, df_series_clean: Series, profile: Dict) -> None:
        """
        Update the column profile properties.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: float profile dictionary
        :type profile: dict
        :return: None
        """
        self._update_column_base_properties(profile)

    def update(self, df_series: Series) -> DataLabelerColumn:
        """
        Update the column profile.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: updated DataLabelerColumn
        :rtype: DataLabelerColumn
        """
        if len(df_series) == 0:
            return self

        sample_size = min(len(df_series), self._max_sample_size)
        df_series = df_series.sample(sample_size)

        profile = dict(sample_size=sample_size)
        self._update_predictions(
            df_series=df_series, prev_dependent_properties={}, subset_properties=profile
        )
        BaseColumnProfiler._perform_property_calcs(
            self,
            self.__calculations,
            df_series=df_series,
            prev_dependent_properties={},
            subset_properties=profile,
        )
        self._update_helper(df_series, profile)

        return self
