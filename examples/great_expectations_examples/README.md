## Data Profiler + Great Expectation
[Great Expectations](https://greatexpectations.io/) is an open source tool built around the idea that should always know what to expect from your data. It helps data teams eliminate pipeline debt, through data testing, documentation, and profiling.

See a full list of Data Profiler Expectations available through Great Expectations on the [package page](https://greatexpectations.io/packages/capitalone_dataprofiler_expectations). 
### Local Setup
1. Initialize your virtual environment: `python3 -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate`
3. In the `great_expectations_examples` directory run `pip install -r requirements.txt`
4. Download and install Capital One's DataProfiler Expectations
   1. Clone the repo ([link to repo](https://github.com/great-expectations/great_expectations))
   2. Add `py_modules=[]` to `great_expectations/contrib/capitalone_dataprofiler_expectations/setup.py` under `setuptools.setup`
   3. Install the package to virtual environment: `pip install -e <path_to_downloaded_package>`

Initialize Great Expectations:
- Run `great_expectations init` in the terminal at root of this repo.
    - NOTE: This step is crucial in order generate a `DataContext` that we will obtain later