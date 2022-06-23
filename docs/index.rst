.. Diatomic-py documentation master file, created by
   sphinx-quickstart on Wed Jan 27 17:58:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to Diatomic-py's documentation!
=======================================
Diatomic-py is a python module for computing the interaction between the hyperfine structure of a simple diatomic molecule and external fields.

Features
--------

Diatomic-py is a very flexible program, and can currently perform the following calculations for <sup>1</sup>Σ ground states:

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

Typical usage of this module requires the user-facing calculate module.
    ``from diatom import calculate``
or
    ``import diatomic.calculate as calculate``


An example calculations are shown in the python scripts on `GitHub`_.

.. _GitHub: https://github.com/JakeBlackmore/Diatomic-Py/blob/master/example scripts

Access to all of the individual terms is available through the hamiltonian module
    ``from diatomic import hamiltonian``

Full descriptions of all functions can be found using the links below:

.. toctree::
   :maxdepth: 2

   diatomic

Paper
-----
If you use our work for academic purposes you can cite us using:

J.A.Blackmore *et al.* Diatomic-py: A python module for calculating the rotational and hyperfine structure of <sup>1</sup>Σ molecules, [Arxiv *e-prints* 2205.05686](https://arxiv.org/abs/2205.05686) (2022).

License
-------

Diatomic-py is licensed under the 3-Clause BSD License

Credits
-------
  Author: Jacob A Blackmore - jacob.blackmore@physics.ox.ac.uk

Contribute
----------
If you want to contribute please contact either jacob.blackmore@physics.ox.ac.uk via email or use the GitHub page at https://github.com/JakeBlackmore/Diatomic-Py
