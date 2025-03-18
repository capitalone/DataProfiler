.. _roadmap:

Roadmap
*******

For more detailed tasks, checkout the repo's github issues page here: 
`Github Issues <https://github.com/capitalone/DataProfiler/issues>`_.


Data Reader Updates
===================
- Read data from S3 bucket
    - All in the current `dp.Data()` API paradigm, we want to enable passing an S3 bucket file path to read in data from AWS s3.
- Pass list of data file paths to data reader
- Pass in linst of data frames to data reader 

New Model
=========
- Transformer model from sensitive data detection

Historical Profiles 
===================
- Some questions about Historical Profiles / need to step back and rething design to start:
    - Meta profile on top?
    - Stored windowed info inside? Etc...
- Branch with current state of Historical Profiles
- Two example notebooks of current state: 
    - Notebook example `one <https://github.com/capitalone/DataProfiler/blob/feature/historical_profiler/examples/historical_profiler.ipynb>`_.
    - Notebook example `two <https://github.com/capitalone/DataProfiler/blob/feature/historical_profiler/examples/WIP_historical_profiler_advanced.ipynb>`_.


Conditional Report Metric
=========================
- Based on what is populated on other metrics in the report, have "secondary" / "derivatives" of that number (or that number in conjunction with another number) populate in thie report as well.
- For example, if null_count is not None, then populate a null_percent key with a value of the dividence of (null_count / sample_count).

Space / Time Testing
====================
- Automatic comparison testing for space and time analysis on PR’s
    - Standardize a report for space time analysis for future comparisons (create baseline numbers)
    - Include those in integration tests that will automatically run on code when it is changed in PRs
- Could be an optional test, if the user thinks there is concern around the change driving an issue in the library performance 

Testing Suite Upgrades
======================
- Add mocking to unit tests where mocking is not utilized
- Integration testing separated out from the unit testing suite. Determine how to only run remotely during PRs
- Backward compatibility testing along with informative warnings and errors when a user is utilizing incompatible versions of the library and saved profile object

Historical Versions
===================
- Legacy version upgrades to enable patches to prior versions of the Data Profiler 

Miscellaneous
==============
- Refact/or Pandas to Polars DataFrames
- Spearman correlation calculation
- Workflow Profiles
