.. _install:

Install
*******


Snappy Installation
===================

This is required to profile parquet/avro datasets

MacOS with homebrew:

.. code-block:: console

    brew install snappy


Linux install:

.. code-block:: console

    sudo apt-get -y install libsnappy-dev


Data Profiler Installation
==========================

NOTE: Installation for python3

virtualenv install:

.. code-block:: console
    
    python3 -m pip install virtualenv


Setup virtual env:

.. code-block:: console

    python3 -m virtualenv --python=python3 venv3
    source venv3/bin/activate


Install requirements:

.. code-block:: console

    pip3 install -r requirements.txt

Install labeler dependencies:

.. code-block:: console

    pip3 install -r requirements-ml.txt


Install via the repo -- Build setup.py and install locally:

.. code-block:: console

    python3 setup.py sdist bdist bdist_wheel
    pip3 install dist/DataProfiler*-py3-none-any.whl


If you see:

.. code-block:: console

    ERROR: Double requirement given:dataprofiler==X.Y.Z from dataprofiler/dist/DataProfiler-X.Y.Z-py3-none-any.whl (already in dataprofiler==X2.Y2.Z2 from dataprofiler/dist/DataProfiler-X2.Y2.Z2-py3-none-any.whl, name='dataprofiler')

This means that you have multiple versions of the DataProfiler distribution 
in the dist folder.
To resolve, either remove the older one or delete the folder and rerun the steps
above.

Install via github:

.. code-block:: console

    pip3 install git+https://github.com/capitalone/dataprofiler.git#egg=dataprofiler



Testing
=======

For testing, install test requirements:

.. code-block:: console

    pip3 install -r requirements-test.txt


To run all unit tests, use:

.. code-block:: console

    DATAPROFILER_SEED=0 python3 -m unittest discover -p "test*.py"


To run file of unit tests, use form:

.. code-block:: console

    DATAPROFILER_SEED=0 python3 -m unittest discover -p test_profile_builder.py


To run a file with Pytest use:

.. code-block:: console

    DATAPROFILER_SEED=0 pytest dataprofiler/tests/data_readers/test_csv_data.py -v


To run individual of unit test, use form:

.. code-block:: console
    
    DATAPROFILER_SEED=0 python3 -m unittest dataprofiler.tests.profilers.test_profile_builder.TestProfiler


