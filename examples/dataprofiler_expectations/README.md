## Data Profiler + Great Expectation
[Great Expectations](https://greatexpectations.io/) is an open source tool built around the idea that should always know what to expect from your data. It helps data teams eliminate pipeline debt, through data testing, documentation, and profiling.

### What is an Expectation
An expectation is a functional data quality assertions. 
They can be very simple rules, such as expecting a numerical datapoint to be less than a specified value. 
Alternatively, Expectations can be quite complex, such as expecting that data metrics on a more recent set of data doesn't deviate more than a specified percent from the original dataset.
With expectations, you have the freedom to set the ranges or thresholds for your assertions.

See a full list of Data Profiler Expectations available through Great Expectations on the [package page](https://greatexpectations.io/packages/capitalone_dataprofiler_expectations). 
### Local Setup
1. Navigate to the GE examples directory: `cd examples/dataprofiler_expectations/`
2. Initialize your virtual environment: `python3 -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install Capital One's DataProfiler Expectations: `pip install git+https://github.com/great-expectations/great_expectations.git@develop#subdirectory=contrib/capitalone_dataprofiler_expectations/`
5. Run `great_expectations init`
