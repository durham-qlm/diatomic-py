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

    # Installs essentials only
    python -m pip install .

    # Installs essentials + plotting support
    python -m pip install ".[plotting]"

    # Installs plotting support and optional progress bars
    python -m pip install ".[plotting,progress]"

Development dependencies are managed with dependency groups. With uv:

.. code-block:: shell

    uv sync --group dev
    uv run pre-commit install
    uv run pytest
