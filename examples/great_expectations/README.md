## Data Profiler + Great Expectation
[Great Expectations](https://greatexpectations.io/) is an open source tool built around the idea that should always know what to expect from your data. It helps data teams eliminate pipeline debt, through data testing, documentation, and profiling.

### What is an Expectation
An Expectation is a declarative statement that a computer can evaluate, and that is semantically meaningful to humans, like expect_column_values_to_be_unique or expect_column_mean_to_be_between.
Expectations are implemented as classes that provide a rich interface to the rest of the library to support validation, profiling, and translation.
Some Expectations are implemented in the core library.
Many others are contributed by the community of data practitioners that bring their domain knowledge and share it as Expectations.

See a full list of Data Profiler Expectations available through Great Expectations on the [package page](https://greatexpectations.io/packages/capitalone_dataprofiler_expectations).
If you are interested in learning more about expectations, here is a link to the [docs](https://greatexpectations.io/expectations).

### About these Notebooks
These notebook examples will utilize a dataset containing individual salary information across the world from the year 2020 until 2022.
Each example will run through a different scenario that uses the data and a custom Data Profiler expectation.
These scenarios provide a practical example of how expectations can be used in real world contexts for data quality checks.

### Local Setup
```shell
cd examples/great_expectations/ # cd to great expectations examples
python3 -m venv venv # create venv
source venv/bin/activate # activate venv
pip install git+https://github.com/great-expectations/great_expectations.git@develop#egg=capitalone-dataprofiler-expectations\&subdirectory=contrib/capitalone_dataprofiler_expectations/ # install data profiler expectation package
great_expectations init # init great expectations
python -m ipykernel install --name=venv # create a new kernel to use in jupyter notebook
```

In order to run the notebooks run `jupyter notebook`.
Once you have a notebook open, select **Kernel** &rarr; **Change kernel** &rarr; **venv** at the top.
