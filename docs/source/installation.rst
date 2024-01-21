Installation
============

PyPi Installation
-----------------

To install from the PyPi package repository, do

.. code-block:: shell

    python -m pip install diatomic-py

Installation from source
------------------------

.. code-block:: shell

    git clone https://github.com/durham-qlm/diatomic-py.git
    cd diatomic-py

    # Installs essentials + matplotlib
    python -m pip install ".[plotting]"

    # OR if you also want to run test suite:
    python -m pip install ".[test,plotting]"
    pytest

    # OR if you want to develop code for the package
    python -m pip install -e ".[dev,plotting]"
    pre-commit install
