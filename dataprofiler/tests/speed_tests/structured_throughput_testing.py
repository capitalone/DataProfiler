import json
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import sys

    sys.path.insert(0, "../../..")
    import dataprofiler as dp
except ImportError:
    import dataprofiler as dp


# suppress TF warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


################################################################################
######################## set any optional changes here #########################
################################################################################
options = dp.ProfilerOptions()

# these two options default to True if commented out
options.structured_options.multiprocess.is_enabled = False
# options.structured_options.data_labeler.is_enabled = False

# parameter alteration
ALLOW_SUBSAMPLING = True  # profiler to subsample the dataset if large
PERCENT_TO_NAN = 0.0  # Value must be between 0 and 100


sample_sizes = [100, 1000, 5000, 7500, int(1e5)]
################################################################################

if __name__ == "__main__":

    # set seed
    random.seed(0)
    np.random.seed(0)
    dp.set_seed(0)

    # load data
    data = dp.Data("data/time_structured_profiler.csv")

    # [0] allows model to be initialzied and added to labeler
    sample_sizes = [0] + sample_sizes
    profile_times = []
    for sample_size in sample_sizes:
        # setup time dict

        print(f"Evaluating sample size: {sample_size}")
        df = data.data.sample(sample_size, replace=True).reset_index(drop=True)

        if PERCENT_TO_NAN:
            samples_to_nan = int(len(df) * PERCENT_TO_NAN / 100)
            for col_name in df:
                ind_to_nan = random.sample(list(df.index), samples_to_nan)
                df[col_name][ind_to_nan] = "None"

        # time profiling
        start_time = time.time()
        if ALLOW_SUBSAMPLING:
            profiler = dp.Profiler(df, options=options)
        else:
            profiler = dp.Profiler(df, samples_per_update=len(df), options=options)
        total_time = time.time() - start_time

        # get overall time for merging profiles
        start_time = time.time()
        try:
            merged_profile = profiler + profiler
        except ValueError:
            pass  # empty profile merge if 0 data
        merge_time = time.time() - start_time

        # get times for each profile in the columns
        for profile in profiler.profile:
            compiler_times = defaultdict(list)

            for compiler_name in profile.profiles:
                compiler = profile.profiles[compiler_name]
                inspector_times = dict()
                for inspector_name in compiler._profiles:
                    inspector = compiler._profiles[inspector_name]
                    inspector_times[inspector_name] = inspector.times
                compiler_times[compiler_name] = inspector_times
            column_profile_time = {
                "name": profile.name,
                "sample_size": sample_size,
                "total_time": total_time,
                "column": compiler_times,
                "merge": merge_time,
                "percent_to_nan": PERCENT_TO_NAN,
                "allow_subsampling": ALLOW_SUBSAMPLING,
                "is_data_labeler": options.structured_options.data_labeler.is_enabled,
                "is_multiprocessing": options.structured_options.multiprocess.is_enabled,
            }
            profile_times += [column_profile_time]

        # add time for for Top-level
        if sample_size:
            profile_times += [
                {
                    "name": "StructuredProfiler",
                    "sample_size": sample_size,
                    "total_time": total_time,
                    "column": profiler.times,
                    "merge": merge_time,
                    "percent_to_nan": PERCENT_TO_NAN,
                    "allow_subsampling": ALLOW_SUBSAMPLING,
                    "is_data_labeler": options.structured_options.data_labeler.is_enabled,
                    "is_multiprocessing": options.structured_options.multiprocess.is_enabled,
                }
            ]

        print(f"COMPLETE sample size: {sample_size}")
        print(f"Profiled in {total_time} seconds")
        print(f"Merge in {merge_time} seconds")
        print()

    # Print dictionary with profile times
    print("Results Saved")
    # print(json.dumps(profile_times, indent=4))

    # only works if columns all have unique names
    times_table = (
        pd.json_normalize(profile_times).set_index(["name", "sample_size"]).sort_index()
    )

    # save json and times table
    with open("structured_profiler_times.json", "w") as fp:
        json.dump(profile_times, fp, indent=4)
    times_table.to_csv("structured_profiler_times.csv")
