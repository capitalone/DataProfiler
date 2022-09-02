## Data Profiler + Great Expectation
[Great Expectations](https://greatexpectations.io/) is an open source tool built around the idea that should always know what to expect from your data. It helps data teams eliminate pipeline debt, through data testing, documentation, and profiling.

### What is an Expectation
An Expectation is a declarative statement that a computer can evaluate, and that is semantically meaningful to humans, like expect_column_values_to_be_unique or expect_column_mean_to_be_between. 
Expectations are implemented as classes that provide a rich interface to the rest of the library to support validation, profiling, and translation. 
Some Expectations are implemented in the core library. 
Many others are contributed by the community of data practitioners that bring their domain knowledge and share it as Expectations.

See a full list of Data Profiler Expectations available through Great Expectations on the [package page](https://greatexpectations.io/packages/capitalone_dataprofiler_expectations). 
If you are interested in learning more about expectations, here is a link to the [docs](https://greatexpectations.io/expectations).

### Local Setup
```shell
cd examples/great_expectations/ # cd to great expectations examples
python3 -m venv venv # create venv
source venv/bin/activate # activate venv
pip install git+https://github.com/great-expectations/great_expectations.git@develop#subdirectory=contrib/capitalone_dataprofiler_expectations/ # install data profiler expectation package
great_expectations init # init great expectations
```
