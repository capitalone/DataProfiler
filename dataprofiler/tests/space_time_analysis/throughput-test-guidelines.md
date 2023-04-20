# Throughput Evaluation Guidelines

Standardizing the tests and datasets for throughput comparison is necessary for
accurate version / commit comparisons. This guide seeks to standardize the
testing mechanism for both structured and unstructured datasets.


# Structured Dataset Throughput Evaluation

The test script `structured_throughput_testing.py` has been provided to simplify
the throughput testing procedure. Simply running the script will provide a
printed output as well as four files and saved to the working directory of where
the script was ran.

  * `structured_profile_times.json`: dict of total time, time to merge, and
      runtimes for each of the profiled functions within the library
  * `structured_profile_times.csv`: a flattened table of the above json
  * `profile_space_analysis.bin`: a bin file that contains information on the
      spatial analysis of running the dp.Profiler function
  * `merge_space_analysis.bin`: a bin file that contains information on the
      spatial analysis of merging two profiles together

Total time and merge time can be used for comparing the overall runtime changes,
whereas the individual function times can detail bottlenecks or speed changes as
a result of alterations to a property's calculation.

The spatial analysis `bin` files can be viewed in different report formats with memray.
For example running:
```console
python3 -m memray flamegraph profile_space_analysis.bin -o profile_space_analysis.html
```
Gives a html formatted flamegraph that displays the distribution of space allocated by
function calls involved in the dp.Profiler

The script can be run as follows:
```console
python structured_throughput_testing.py
```

### Tunable parameters

The script has a set of parameters which can be tuned to evaluate how specific
conditions affect profiling speed. The parameters are as follows:

  * TIME_ANALYSIS:             turns on or off the time analysis functionality
  * SPACE_ANALYSIS:            turns on or off the space analysis functionality
  * multiprocess:              turns on or off multiprocessing (True / False)
  * data_labeler               turns on or off the data labeler (True / False)
  * ALLOW_SUBSAMPLING:         turns on or off subsampling for large data (True / False)
  * PERCENT_TO_NAN:            percentage of data in each column to set as NaN (0 - 100)
  * sample_sizes:              list of dataset sizes to evaluate
  * DATASET_PATH:              path to a pre-existing dataset for analysis


### Generated Dataset Details

The dataset generated in this script will be, by default a combination of the following data classes:
  * text - string with length >= 256 but < 1000
  * string - string with length >= 1 but < 256
  * categorical - entry consisting of chars ranged from A-E
  * integer - python primitive integer ranged from -1e6 to 1e6
  * float - python3 primitive float ranged from -1e6 to 1e6 rounded to 3 significant figures
  * ordered - Ordered column of integers
  * datetime - entry following any valid format from the Datapofiler's datetime format list
by default the dataset will consist of 100000 rows of each of the above data classes.


### Provided Dataset Details

The dataset evaluated in this script utilizes a subset of data from the CSV
test dataset `aws_honeypot_marx_geo.csv`. This dataset contains 4 columns such
that it would equally test the 4 primitive data types evaluated by the
Profilers (datetime, int, float, and text). Additionally, the dataset had all
null values removed so that NaNs were only provided if desired by the user
options.

The following code was used to create this dataset:
```python
# executed within: "dataprofiler/tests/speed_tests"
import random

import numpy as np
import pandas as pd

# set seed for repeatable
random.seed(0)
np.random.seed(0)


# dataprofiler/tests/data/csv/aws_honeypot_marx_geo.csv
# read data wihtout altering it
data = pd.read_csv('../data/csv/aws_honeypot_marx_geo.csv', dtype=str)

# fill most of comments so not completely NA
data['comment'][data['comment'].isna()] = data['host'][data['comment'].isna()]

# select subset of data
data = data[['datetime', 'srcport', 'latitude', 'comment']]

# remove any columns that have NA
data = data[~data.isna().any(axis=1)].reset_index(drop=True)

# save the dataset
data.to_csv('data/time_structured_profiler.csv', index=False)
```

# Reporting findings to github


### Obtaining space flamegraphs
After running `structured_space_time_analysis.py` script there will be `.bin` files created in the `./space_analysis`
directory. To generate flamegraphs from all the generated `.bin` files:
```console
./create_flamegraphs.sh
```


### Attaching report to PR
When running these space/time analyses, you can include the following format in your Pull Request description for
validation of improvement:
```
- <1st dataset size>
  - Profile
    - Pre change
        - Total space allocated: <Total space taken in flame graph>
        - Line specific space allocated (Line number/s): <Space taken by soon-to-be-changed line in flame graph>
        - Profile runtime: <from time report text file>
    - Post change
        - Total space allocated: <Total space taken in flame graph>
        - Line specific space allocated (Line number/s): <Space taken by soon-to-be-changed line in flame graph>
        - Profile runtime: <from time report text file>
  - Merge
    - Pre change
        - Total space allocated: <Total space taken in flame graph>
        - Line specific space allocated (Line number/s): <Space taken by soon-to-be-changed line in flame graph>
        - Merge runtime: <from time report text file>
    - Post change
        - Total space allocated: <Total space taken in flame graph>
        - Line specific space allocated (Line number/s): <Space taken by changed line in flame graph>
        - Merge runtime: <from time report text file>
- <2nd dataset size>
...
```

# Unstructured Dataset Throughput Evaluation

TBD
