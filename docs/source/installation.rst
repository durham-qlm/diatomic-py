Installation
============

PyPi Installation
-----------------

To install from the PyPi package repository, do

.. code-block:: shell

    python -m pip install diatomic-py

Conda Installation
------------------

To install into the active conda environment

.. code-block:: shell

    conda install diatomic-py


Installation from source
------------------------

.. code-block:: shell

    # Installs essentials + matplotlib
    python -m pip install ".[plotting]"

    # OR if you also want to run test suite:
    python -m pip install ".[test,plotting]"
    pytest

    # OR if you want to develop code for the package
    python -m pip install -e ".[dev,plotting]"
    pre-commit install
