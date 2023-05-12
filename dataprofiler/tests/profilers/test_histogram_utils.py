import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

from dataprofiler.profilers import NumericStatsMixin, histogram_utils


# Mocks for property functions
def mock_stddev():
    return 1.0


def mock__invalid_stddev():
    return -1.0


def mock_get_percentile(percentile):
    if percentile == [75]:
        return 2.0
    if percentile == [25]:
        return 1.0


class TestColumn(NumericStatsMixin):
    def __init__(self):
        NumericStatsMixin.__init__(self)
        self.match_count = 0
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

    def test_hist_bin_doane_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        with mock.patch(
            "dataprofiler.profilers.NumericStatsMixin.stddev", new_callable=mock_stddev
        ):
            # Case 1: min, max, match_count, biased_skewness, and stddev are set
            expected = 4 / (1 + 2.321928094887362 + 1.3967037745666426)
            actual = histogram_utils._hist_bin_doane_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 2: min, max, biased_skewness, and stddev are set.
            # match_count doesn't exist
            delattr(profile, "match_count")
            expected = 4 / (1 + 2.584962500721156 + 1.3896687739916025)
            actual = histogram_utils._hist_bin_doane_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 3 max, biased_skewness, and stddev are set.
            # match_count doesn't exist and min is None
            profile.min = None
            expected = 5 / (1 + 2.584962500721156 + 1.3896687739916025)
            actual = histogram_utils._hist_bin_doane_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 4 biased_skewness, and stddev are set.
            # match_count doesn't exist and both min and max are None
            profile.max = None
            expected = 6 / (1 + 2.584962500721156 + 1.3896687739916025)
            actual = histogram_utils._hist_bin_doane_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 5 match_count (dataset_size) is < 2
            profile.match_count = 0
            actual = histogram_utils._hist_bin_doane_from_profile(profile)
            self.assertEqual(0.0, actual)

        with mock.patch(
            "dataprofiler.profilers.NumericStatsMixin.stddev",
            new_callable=mock__invalid_stddev,
        ):
            # Case 6 match_count (dataset_size) is > 2 and profile.stddev < 0
            profile.match_count = 5
            actual = histogram_utils._hist_bin_doane_from_profile(profile)
            self.assertEqual(0.0, actual)

    def test_hist_bin_rice_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        # Case 1: min, max, and match_count are set
        expected = 4 / (2 * 5 ** (1 / 3))
        actual = histogram_utils._hist_bin_rice_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 2: min and max set. match_count doesn't exist
        delattr(profile, "match_count")
        expected = 4 / (2 * 6 ** (1 / 3))
        actual = histogram_utils._hist_bin_rice_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 3 max is set. match_count doesn't exist and min is None
        profile.min = None
        expected = 5 / (2 * 6 ** (1 / 3))
        actual = histogram_utils._hist_bin_rice_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 4 match_count doesn't exist and both min and max are None
        profile.max = None
        expected = 6 / (2 * 6 ** (1 / 3))
        actual = histogram_utils._hist_bin_rice_from_profile(profile)
        self.assertEqual(expected, actual)

    def test_hist_bin_sturges_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        # Case 1: min, max, and match_count are set
        expected = 4 / (2.321928094887362 + 1)
        actual = histogram_utils._hist_bin_sturges_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 2: min and max set. match_count doesn't exist
        delattr(profile, "match_count")
        expected = 4 / (2.584962500721156 + 1)
        actual = histogram_utils._hist_bin_sturges_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 3 max is set. match_count doesn't exist and min is None
        profile.min = None
        expected = 5 / (2.584962500721156 + 1)
        actual = histogram_utils._hist_bin_sturges_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 4 match_count doesn't exist and both min and max are None
        profile.max = None
        expected = 6 / (2.584962500721156 + 1)
        actual = histogram_utils._hist_bin_sturges_from_profile(profile)
        self.assertEqual(expected, actual)

    def test_hist_bin_sqrt_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        # Case 1: min, max, and match_count are set
        expected = 4 / 2.23606797749979
        actual = histogram_utils._hist_bin_sqrt_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 2: min and max set. match_count doesn't exist
        delattr(profile, "match_count")
        expected = 4 / 2.449489742783178
        actual = histogram_utils._hist_bin_sqrt_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 3 max is set. match_count doesn't exist and min is None
        profile.min = None
        expected = 5 / 2.449489742783178
        actual = histogram_utils._hist_bin_sqrt_from_profile(profile)
        self.assertEqual(expected, actual)

        # Case 4 match_count doesn't exist and both min and max are None
        profile.max = None
        expected = 6 / 2.449489742783178
        actual = histogram_utils._hist_bin_sqrt_from_profile(profile)
        self.assertEqual(expected, actual)

    def test_hist_bin_fd_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        with mock.patch(
            "dataprofiler.profilers.NumericStatsMixin._get_percentile",
            side_effect=mock_get_percentile,
        ):
            # Case 1: match_count is set
            expected = 2.0 * 1.0 * 5 ** (-1 / 3)
            actual = histogram_utils._hist_bin_fd_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 2: match_count doesn't exist
            delattr(profile, "match_count")
            expected = 2.0 * 1.0 * 6 ** (-1 / 3)
            actual = histogram_utils._hist_bin_fd_from_profile(profile)
            self.assertEqual(expected, actual)

    def test_hist_bin_auto_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        with mock.patch(
            "dataprofiler.profilers.histogram_utils._hist_bin_fd_from_profile"
        ) as fd_mock:
            with mock.patch(
                "dataprofiler.profilers.histogram_utils._hist_bin_sturges_from_profile"
            ) as sturges_mock:
                # Case 1: Freedman-Diaconis calc > Sturges calc
                fd_mock.return_value = 1
                sturges_mock.return_value = 0
                expected = 0
                actual = histogram_utils._hist_bin_auto_from_profile(profile)
                self.assertEqual(expected, actual)

                # Case 2: Freedman-Diaconis calc < Sturges calc
                fd_mock.return_value = 1
                sturges_mock.return_value = 2
                expected = 1
                actual = histogram_utils._hist_bin_auto_from_profile(profile)
                self.assertEqual(expected, actual)

                # Case 3: Freedman-Diaconis calc returns None
                fd_mock.return_value = None
                sturges_mock.return_value = 0
                expected = 0
                actual = histogram_utils._hist_bin_auto_from_profile(profile)
                self.assertEqual(expected, actual)

                # Case 4: Both calcs return None
                fd_mock.return_value = None
                sturges_mock.return_value = None
                actual = histogram_utils._hist_bin_auto_from_profile(profile)
                self.assertIsNone(actual)

    def test_hist_bin_scott_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        with mock.patch(
            "dataprofiler.profilers.NumericStatsMixin.stddev", new_callable=mock_stddev
        ):
            # Case 1: match_count and stddev are set
            expected = (24.0 * np.pi**0.5 / 5) ** (1.0 / 3.0) * 1
            actual = histogram_utils._hist_bin_scott_from_profile(profile)
            self.assertEqual(expected, actual)

            # Case 2: match_count doesn't exist and stddev is set
            delattr(profile, "match_count")
            expected = (24.0 * np.pi**0.5 / 6) ** (1.0 / 3.0) * 1
            actual = histogram_utils._hist_bin_scott_from_profile(profile)
            self.assertEqual(expected, actual)

    def test_calculate_bins_from_profile(self):
        # Initial setup of profile
        profile = TestColumn()

        # Case 1: bin method not in set of valid bin methods
        with self.assertRaises(ValueError):
            histogram_utils.calculate_bins_from_profile(profile, "test_not_in_method")

        # Case 2: min, max, and match_count are set and bin_method is valid
        expected_sqrt_answer = 4 / 2.23606797749979
        expected_buckets = int(np.ceil(4 / expected_sqrt_answer))
        actual = histogram_utils.calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(expected_buckets, actual)

        # Case 3: min and max set. match_count doesn't exist
        delattr(profile, "match_count")
        expected_sqrt_answer = 4 / 2.449489742783178
        expected_buckets = int(np.ceil(4 / expected_sqrt_answer))
        actual = histogram_utils.calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(expected_buckets, actual)

        # Case 4 max is set. match_count doesn't exist and min is None
        profile.min = None
        expected_sqrt_answer = 5 / 2.449489742783178
        expected_buckets = int(np.ceil(5 / expected_sqrt_answer))
        actual = histogram_utils.calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(expected_buckets, actual)

        # Case 5 match_count doesn't exist and both min and max are None
        profile.max = None
        expected_sqrt_answer = 6 / 2.449489742783178
        expected_buckets = int(np.ceil(6 / expected_sqrt_answer))
        actual = histogram_utils.calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(expected_buckets, actual)

        # Case 6 dataset_size is zero
        profile.match_count = 0
        actual = histogram_utils.calculate_bins_from_profile(profile, "sqrt")
        self.assertEqual(1, actual)

        # Case 7 calculated width is None
        profile.match_count = 5
        with mock.patch(
            "dataprofiler.profilers.histogram_utils._hist_bin_sqrt_from_profile"
        ) as sqrt_mock:
            sqrt_mock.return_value = None
            actual = histogram_utils.calculate_bins_from_profile(profile, "sqrt")
            self.assertEqual(1, actual)

            # Case 8 calculated width is float NaN
            sqrt_mock.return_value = float("nan")
            actual = histogram_utils.calculate_bins_from_profile(profile, "sqrt")
            self.assertEqual(1, actual)
