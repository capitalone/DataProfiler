## Data Profiler + Great Expectation
[Great Expectations](https://greatexpectations.io/) is an open source tool built around the idea that should always know what to expect from your data. It helps data teams eliminate pipeline debt, through data testing, documentation, and profiling.

See a full list of Data Profiler Expectations available through Great Expectations on the [package page](https://greatexpectations.io/packages/capitalone_dataprofiler_expectations). 
### Local Setup
1. Navigate to the GE examples directory: `cd examples/great_expectations/`
2. Initialize your virtual environment: `python3 -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Download and install Capital One's DataProfiler Expectations
   1. Clone the great expectations: `git clone https://github.com/great-expectations/great_expectations.git`
   2. Add `py_modules=[]` to `great_expectations/contrib/capitalone_dataprofiler_expectations/setup.py` under `setuptools.setup`
   3. Install the package to virtual environment: `pip install -e  great_expectations/contrib/capitalone_dataprofiler_expectations`
5. Run `great_expectations init`
