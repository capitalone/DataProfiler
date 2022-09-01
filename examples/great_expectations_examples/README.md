## Instructions

### Offline work
Begin by creating a new directory for your project. We advise using a virtual environment which you can make by executing the following commands:
 - Initialize your virtual environment: `python3 -m venv venv`
 - Activate the virtual environment: `source venv/bin/activate`

Now install the following packages:
- In the `great_expectations_examples` directory run `pip install -r requirements.txt`
- Capital One's DataProfiler Expectations: `pip install capitalone_dataprofiler_expectations`
    - NOTE: this package is currently not published. You can download the package [here](https://github.com/great-expectations/great_expectations/tree/develop/contrib/capitalone_dataprofiler_expectations), and install it using: `pip install -e <path_to_downloaded_package>`
    - Once the package is downloaded, the following line might need to be added to `great_expectations/contrib/capitalone_dataprofiler_expectations/setup.py` if the `pip install` is failing
        - `py_modules=[]`

Initialize Great Expectations:
- Run the following command to initialize Great Expectations: `great_expectations init`
    - NOTE: This step is crucial in order generate a `DataContext` that we will obtain later