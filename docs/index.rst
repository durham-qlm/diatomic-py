.. Diatomic-py documentation master file, created by
   sphinx-quickstart on Wed Jan 27 17:58:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Diatomic-py's documentation!
=======================================
Diatomic-py is a python module for computing the interaction between the hyperfine structure of a diatomic molecule and external fields.

Features
--------

Diatomic-py is a very flexible program, and can currently perform the following calculations for singlet-sigma ground states
*AC and DC Stark maps
*Breit-Rabi diagrams
*Hyperfine state compositions


Installation
------------
Install diatomic-py by downloading the .whl file from GitHub and running:
  pip install diatomic-py.whl

on windows use:
  python -m pip install diatomic-py.whl

Usage
-----

Typical usage of this module requires the user-facing Calculate module.
  from diatom import Calculate

An example calculation is shown in the ipython notebook.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
-------

Diatomic-py is licensed under the 3-Clause BSD License

Credits
-------
  Author: Jacob A Blackmore

Contribute
----------
If you want to contribute please contact either j.a.blackmore@durham.ac.uk via email or use the GitHub page at https://github.com/JakeBlackmore/Diatomic-Py
