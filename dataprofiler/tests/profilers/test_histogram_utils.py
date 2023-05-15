import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

import dataprofiler
from dataprofiler.profilers import NumericStatsMixin, histogram_utils


# Mocks for property functions
def mock_stddev():
    return 1.0


def mock_invalid_stddev():
    return -1.0


def mock_get_percentile(percentile):
    if percentile == [75]:
        return 2.0
    if percentile == [25]:
        return 1.0


def mock_sqrt_return_none(profile):
    return None


def mock_sqrt_return_nan(profile):
    return float("nan")


class TestColumn(NumericStatsMixin):
    def __init__(self):
        NumericStatsMixin.__init__(self)
        self.times = defaultdict(float)
        self.match_count = 5
        self.min = 1
        self.max = 5
        self._biased_skewness = 1.0
        self._stored_histogram["histogram"]["bin_counts"] = [1, 1, 1, 1, 1, 1]
        self._stored_histogram["histogram"]["bin_edges"] = [0, 1, 2, 3, 4, 5, 6]

    def update(self, df_series):
        pass

    def _filter_properties_w_options(self, calculations, options):
        pass


class TestHistogramUtils(unittest.TestCase):
    def test_ptp(self):
        # Maximum greater than minimum
        maximum = 15
        minimum = 12
        expected = 3
        actual = histogram_utils._ptp(maximum, minimum)
        self.assertEqual(expected, actual)

        # Minimum greater than maximum
        maximum = 12
        minimum = 15
        expected = -3
        actual = histogram_utils._ptp(maximum, minimum)
        self.assertEqual(expected, actual)

        # Minimum equal to maximum
        maximum = 15
        minimum = 15
        expected = 0
        actual = histogram_utils._ptp(maximum, minimum)
        self.assertEqual(expected, actual)

    def test_calc_doane_bin_width_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        with mock.patch(
            "dataprofiler.profilers.NumericStatsMixin.stddev", new_callable=mock_stddev
        ):
            # Case 1: min, max, match_count, biased_skewness, and stddev are set
            expected_dataset_size = profile.match_count
            expected_minimum = profile.min
            expected_maximum = profile.max
            sg1 = np.sqrt(
                6.0
                * (expected_dataset_size - 2)
                / ((expected_dataset_size + 1.0) * (expected_dataset_size + 3))
            )
            expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
                1.0
                + np.log2(expected_dataset_size)
                + np.log2(1.0 + np.absolute(1.0) / sg1)
            )

            actual = histogram_utils._calc_doane_bin_width_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 2: min, max, biased_skewness, and stddev are set.
            # match_count doesn't exist
            delattr(profile, "match_count")
            expected_dataset_size = sum(
                profile._stored_histogram["histogram"]["bin_counts"]
            )
            expected_minimum = profile.min
            expected_maximum = profile.max
            sg1 = np.sqrt(
                6.0
                * (expected_dataset_size - 2)
                / ((expected_dataset_size + 1.0) * (expected_dataset_size + 3))
            )
            expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
                1.0
                + np.log2(expected_dataset_size)
                + np.log2(1.0 + np.absolute(1.0) / sg1)
            )
            actual = histogram_utils._calc_doane_bin_width_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 3 max, biased_skewness, and stddev are set.
            # match_count doesn't exist and min is None
            profile.min = None
            expected_dataset_size = sum(
                profile._stored_histogram["histogram"]["bin_counts"]
            )
            expected_minimum = profile._stored_histogram["histogram"]["bin_edges"][0]
            expected_maximum = profile.max
            sg1 = np.sqrt(
                6.0
                * (expected_dataset_size - 2)
                / ((expected_dataset_size + 1.0) * (expected_dataset_size + 3))
            )
            expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
                1.0
                + np.log2(expected_dataset_size)
                + np.log2(1.0 + np.absolute(1.0) / sg1)
            )
            actual = histogram_utils._calc_doane_bin_width_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 4 biased_skewness, and stddev are set.
            # match_count doesn't exist and both min and max are None
            profile.max = None
            expected_dataset_size = sum(
                profile._stored_histogram["histogram"]["bin_counts"]
            )
            expected_minimum = profile._stored_histogram["histogram"]["bin_edges"][0]
            expected_maximum = profile._stored_histogram["histogram"]["bin_edges"][-1]
            sg1 = np.sqrt(
                6.0
                * (expected_dataset_size - 2)
                / ((expected_dataset_size + 1.0) * (expected_dataset_size + 3))
            )
            expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
                1.0
                + np.log2(expected_dataset_size)
                + np.log2(1.0 + np.absolute(1.0) / sg1)
            )
            actual = histogram_utils._calc_doane_bin_width_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 5 match_count (dataset_size) is < 2
            profile.match_count = 0
            actual = histogram_utils._calc_doane_bin_width_from_profile(profile)
            self.assertEqual(0.0, actual)

        with mock.patch(
            "dataprofiler.profilers.NumericStatsMixin.stddev",
            new_callable=mock_invalid_stddev,
        ):
            # Case 6 match_count (dataset_size) is > 2 and profile.stddev < 0
            profile.match_count = 5
            actual = histogram_utils._calc_doane_bin_width_from_profile(profile)
            self.assertEqual(0.0, actual)

    def test_calc_rice_bin_width_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        # Case 1: min, max, and match_count are set
        expected_dataset_size = profile.match_count
        expected_minimum = profile.min
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
            2.0 * expected_dataset_size ** (1.0 / 3)
        )
        actual = histogram_utils._calc_rice_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 2: min and max set. match_count doesn't exist
        delattr(profile, "match_count")
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile.min
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
            2.0 * expected_dataset_size ** (1.0 / 3)
        )
        actual = histogram_utils._calc_rice_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 3 max is set. match_count doesn't exist and min is None
        profile.min = None
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile._stored_histogram["histogram"]["bin_edges"][0]
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
            2.0 * expected_dataset_size ** (1.0 / 3)
        )
        actual = histogram_utils._calc_rice_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 4 match_count doesn't exist and both min and max are None
        profile.max = None
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile._stored_histogram["histogram"]["bin_edges"][0]
        expected_maximum = profile._stored_histogram["histogram"]["bin_edges"][-1]
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
            2.0 * expected_dataset_size ** (1.0 / 3)
        )
        actual = histogram_utils._calc_rice_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

    def test_calc_sturges_bin_width_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        # Case 1: min, max, and match_count are set
        expected_dataset_size = profile.match_count
        expected_minimum = profile.min
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
            np.log2(expected_dataset_size) + 1.0
        )
        actual = histogram_utils._calc_sturges_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 2: min and max set. match_count doesn't exist
        delattr(profile, "match_count")
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile.min
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
            np.log2(expected_dataset_size) + 1.0
        )
        actual = histogram_utils._calc_sturges_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 3 max is set. match_count doesn't exist and min is None
        profile.min = None
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile._stored_histogram["histogram"]["bin_edges"][0]
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
            np.log2(expected_dataset_size) + 1.0
        )
        actual = histogram_utils._calc_sturges_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 4 match_count doesn't exist and both min and max are None
        profile.max = None
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile._stored_histogram["histogram"]["bin_edges"][0]
        expected_maximum = profile._stored_histogram["histogram"]["bin_edges"][-1]
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / (
            np.log2(expected_dataset_size) + 1.0
        )
        actual = histogram_utils._calc_sturges_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

    def test_calc_sqrt_bin_width_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        # Case 1: min, max, and match_count are set
        expected_dataset_size = profile.match_count
        expected_minimum = profile.min
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / np.sqrt(
            expected_dataset_size
        )
        actual = histogram_utils._calc_sqrt_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 2: min and max set. match_count doesn't exist
        delattr(profile, "match_count")
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile.min
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / np.sqrt(
            expected_dataset_size
        )
        actual = histogram_utils._calc_sqrt_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 3 max is set. match_count doesn't exist and min is None
        profile.min = None
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile._stored_histogram["histogram"]["bin_edges"][0]
        expected_maximum = profile.max
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / np.sqrt(
            expected_dataset_size
        )
        actual = histogram_utils._calc_sqrt_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 4 match_count doesn't exist and both min and max are None
        profile.max = None
        expected_dataset_size = sum(
            profile._stored_histogram["histogram"]["bin_counts"]
        )
        expected_minimum = profile._stored_histogram["histogram"]["bin_edges"][0]
        expected_maximum = profile._stored_histogram["histogram"]["bin_edges"][-1]
        expected = histogram_utils._ptp(expected_maximum, expected_minimum) / np.sqrt(
            expected_dataset_size
        )
        actual = histogram_utils._calc_sqrt_bin_width_from_profile(profile)
        self.assertEqual(expected, actual)

    def test_calc_fd_bin_width_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        with mock.patch(
            "dataprofiler.profilers.NumericStatsMixin._get_percentile",
            side_effect=mock_get_percentile,
        ):
            # Case 1: match_count is set
            expected_dataset_size = profile.match_count
            expected = 2.0 * 1.0 * expected_dataset_size ** (-1.0 / 3.0)
            actual = histogram_utils._calc_fd_bin_width_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 2: match_count doesn't exist
            delattr(profile, "match_count")
            expected_dataset_size = sum(
                profile._stored_histogram["histogram"]["bin_counts"]
            )
            expected = 2.0 * 1.0 * expected_dataset_size ** (-1.0 / 3.0)
            actual = histogram_utils._calc_fd_bin_width_from_profile(profile)
            self.assertEqual(expected, actual)

    def test_calc_auto_bin_width_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        with mock.patch(
            "dataprofiler.profilers.histogram_utils._calc_fd_bin_width_from_profile"
        ) as fd_mock:
            with mock.patch(
                "dataprofiler.profilers.histogram_utils._calc_sturges_bin_width_from_profile"
            ) as sturges_mock:
                # Case 1: Freedman-Diaconis calc > Sturges calc
                fd_mock.return_value = 1
                sturges_mock.return_value = 0
                expected = 0
                actual = histogram_utils._calc_auto_bin_width_from_profile(profile)
                self.assertEqual(expected, actual)

                # Case 2: Freedman-Diaconis calc < Sturges calc
                fd_mock.return_value = 1
                sturges_mock.return_value = 2
                expected = 1
                actual = histogram_utils._calc_auto_bin_width_from_profile(profile)
                self.assertEqual(expected, actual)

                # Case 3: Freedman-Diaconis calc returns None
                fd_mock.return_value = None
                sturges_mock.return_value = 0
                expected = 0
                actual = histogram_utils._calc_auto_bin_width_from_profile(profile)
                self.assertEqual(expected, actual)

                # Case 4: Both calcs return None
                fd_mock.return_value = None
                sturges_mock.return_value = None
                actual = histogram_utils._calc_auto_bin_width_from_profile(profile)
                self.assertIsNone(actual)

    def test_calc_scott_bin_width_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        with mock.patch(
            "dataprofiler.profilers.NumericStatsMixin.stddev", new_callable=mock_stddev
        ):
            # Case 1: match_count and stddev are set
            expected_dataset_size = profile.match_count
            expected = (24.0 * np.pi**0.5 / expected_dataset_size) ** (1.0 / 3.0) * 1
            actual = histogram_utils._calc_scott_bin_width_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 2: match_count doesn't exist and stddev is set
            delattr(profile, "match_count")
            expected_dataset_size = sum(
                profile._stored_histogram["histogram"]["bin_counts"]
            )
            expected = (24.0 * np.pi**0.5 / expected_dataset_size) ** (1.0 / 3.0) * 1
            actual = histogram_utils._calc_scott_bin_width_from_profile(profile)
            self.assertEqual(expected, actual)

    def test_calculate_bins_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        # Case 1: bin method not in set of valid bin methods
        with self.assertRaises(ValueError):
            histogram_utils._calculate_bins_from_profile(profile, "test_not_in_method")

        # Case 2: min, max, and match_count are set and bin_method is valid
        expected_buckets = 3
        actual = histogram_utils._calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(expected_buckets, actual)

        # Case 3: min and max set. match_count doesn't exist
        delattr(profile, "match_count")
        expected_buckets = 3
        actual = histogram_utils._calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(expected_buckets, actual)

        # Case 4 max is set. match_count doesn't exist and min is None
        profile.min = None
        expected_buckets = 3
        actual = histogram_utils._calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(expected_buckets, actual)

        # Case 5 match_count doesn't exist and both min and max are None
        profile.max = None
        expected_buckets = 3
        actual = histogram_utils._calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(expected_buckets, actual)

        # Case 6 dataset_size is zero
        profile.match_count = 0
        actual = histogram_utils._calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(1, actual)

        # Case 7 calculated width is None
        with mock.patch.dict(
            dataprofiler.profilers.histogram_utils._hist_bin_width_selectors_for_profile,
            {"sqrt": mock_sqrt_return_none},
        ):
            profile = TestColumn()
            actual = histogram_utils._calculate_bins_from_profile(profile, "sqrt")
            self.assertEqual(1, actual)

        # Case 8 calculated width is float NaN
        with mock.patch.dict(
            dataprofiler.profilers.histogram_utils._hist_bin_width_selectors_for_profile,
            {"sqrt": mock_sqrt_return_nan},
        ):
            profile = TestColumn()
            actual = histogram_utils._calculate_bins_from_profile(profile, "sqrt")
            self.assertEqual(1, actual)
