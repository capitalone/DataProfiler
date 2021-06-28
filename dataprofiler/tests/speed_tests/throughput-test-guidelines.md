# Throughput Evaluation Guidelines

Standardizing the tests and datasets for throughput comparison is necessary for
accurate version / commit comparisons. This guide seeks to standardize the
testing mechanism for both structured and unstructured datasets.


# Structured Dataset Throughput Evaluation

The test script `structured_throughput_testing.py` has been provided to simplify
the throughput testing procedure. Simply running the script will provide a
printed output as well as two files and saved to the working directory of where
the script was ran.

  * `structured_profile_times.json`: dict of total time, time to merge, and
      runtimes for each of the profiled functions within the library
  * `structured_profile_times.csv`: flattened table of the the above json

Total time and merge time can be used for comparing the overall runtime changes,
whereas the individual function times can detail bottlenecks or speed changes as
a result of alterations to a property's calculation.

The script can be ran as follows

```console
python structured_throughput_testing.py
```


# Unstructured Dataset Throughput Evaluation

TBD
