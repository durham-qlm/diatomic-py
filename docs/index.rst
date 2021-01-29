.. Diatomic-py documentation master file, created by
   sphinx-quickstart on Wed Jan 27 17:58:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to Diatomic-py's documentation!
=======================================
Diatomic-py is a python module for computing the interaction between the hyperfine structure of a diatomic molecule and external fields.

Features
--------

Diatomic-py is a very flexible program, and can currently perform the following calculations for singlet-sigma ground states:

 * AC and DC Stark maps
 * Breit-Rabi diagrams
 * Hyperfine state compositions

Through fundamental representations of the hyperfine structure many more effects can be observed by combining diatomic-py with numpy and scipy.

Coming Soon
-----------

 * Plotting functionality
 * Fitting of experimental data
 * Doublet-sigma molecules

Installation
------------
Install diatomic-py by downloading the .whl file from GitHub and running:
    ``pip install diatomic-py.whl``

on windows use:
    ``python -m pip install diatomic-py.whl``

Usage
-----

Typical usage of this module requires the user-facing Calculate module.
    ``from diatom import Calculate``

An example calculation is shown in the ipython notebook on `GitHub`_.

.. _GitHub: https://github.com/JakeBlackmore/Diatomic-Py/blob/master/Example%20Scripts/Example%20Calculations.ipynb

Access to all of the individual terms is available through the Hamiltonian module
    ``from diatom import Hamiltonian``

Full descriptions of all functions can be found using the links below:

.. toctree::
   :maxdepth: 2

   diatom


License
-------

Diatomic-py is licensed under the 3-Clause BSD License

Credits
-------
  Author: Jacob A Blackmore

Contribute
----------
If you want to contribute please contact either j.a.blackmore@durham.ac.uk via email or use the GitHub page at https://github.com/JakeBlackmore/Diatomic-Py
