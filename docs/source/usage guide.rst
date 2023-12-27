Usage Guide
===========

.. role:: python(code)
   :language: python

Usage of the package can be broken down into 3 steps:

#. Defining the molecular system.
#. Creating the Hamiltonian for the system.
#. Diagonalising the Hamiltonian for the system.
#. Calculating quantities from the result of the diagonalisation.

Defining the molecular system
-----------------------------

.. code-block:: python

    from diatomic.systems import SingletSigmaMolecule

    mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
    mol.Nmax = 2

Will define a Singlet Sigma molecular system from a preset stored in the dictionary :python:`SingletSigmaMolecule.presets`.
We also specify the maximum rotational manifold to consider in our calculations.
This object contains all the molecular constants needed to generate Hamiltonians and derive results, and will be passed into functions for context.


Creating the Hamiltonian for the system
---------------------------------------

We often wish to produce a plot of energy levels against a varied parameter. To do this, we must produce a Hamiltonian for each sample of the varied parameter.
For example, this could be the Zeeman splitting of the energy levels due to the presence of magnetic field in the Z direction.

.. code-block:: python

    import diatomic.operators as operators

    # Generate Hamiltonians
    H0 = operators.hyperfine_ham(mol)
    Hz = operators.zeeman_ham(mol)

    # Parameter Space
    GAUSS = 1e-4  # T
    B_MIN_GAUSS = 0.001
    B_MAX_GAUSS = 300
    B = np.linspace(B_MIN_GAUSS, B_MAX_GAUSS, 300) * GAUSS

    # Overall Hamiltonian
    Htot = H0 + Hz * B[:, None, None]

Here we generate the necessary Hamiltonians initially, however :python:`Hz` is defined for a unit magnetic field.
Therefore the total Hamiltonian is the bare hyperfine Hamiltonian :python:`H0` plus the Zeeman hamiltonian scaled by the varied magnetic field.
The :python:`None` s in :python:`B[:, None, None]` simply scale the :python:`B` array to have the right number of array dimensions to broadcast.


Diagonalising the Hamiltonian for the system
--------------------------------------------

The package comes with a convenience function to perform this step.

.. code-block:: python

    import diatomic.calculate as calculate

    eigenenergies, eigenstates = calculate.solve_system(Htot)

This is almost equivalent to :python:`numpy.linalg.eigh`, however that function sorts eigenstates for all Hamiltonians by increasing eigenenergy.
The problem with this is, there will be points where eigenenergies of the system cross while varying the varied parameter, but we want these to be continuous.
In other words, we want the eigenstates :python:`eigenstates[:,:,i]` (where the first free dimension selects the varied parameter value, and the second selects the corresponding basis state component)
to vary smoothly over the varied parameter. The convenience function :python:`calculate.solve_system` attempts to solve this issue by maximising eigenstate overlap between variations of the parameter.

By default, when comparing eigenstates step to step, it will match all states with all other states to maximise overlap. This is the safest option, but not necessarily the fastest.
If you have small changes to eigenenergies between steps in the varied parameters, it is likely the crossings will be local, and we only need to consider matching between the nearby eigenenergies.
This is an option that can be passed into the function, :python:`calculate.solve_system(Htot, num_diagonals=4)` for example. This would only consider matching eigenstates to ones within +-4 indices from the previous
step, speeding up the computation. This however, cannot guarantee continuity, only have it be very likely.

This step will likely be the performance bottleneck for your code. Diagonalisation is an expensive operation, taking :math:`\mathcal{O}(N^3)`, where :math:`N`` is the number of basis elements.


Calculating quantities from the result of the diagonalisation
-------------------------------------------------------------

Now we have performed the computationally expensive diagonalisation, we can derive results from it.
For example, we may wish to label the states in a more useful way with their appropriate quantum numbers.
In the magnetic field range considered here, the only good quantum numbers throughout the range are N and MF.
They will not, however, uniquely label a state, so we can have an additional index counting up in energy for states with common labels.

.. code-block:: python

    eigenlabels = calculate.label_states(mol, eigenstates[-1], ["N", "MF"], index_repeats=True)

Here, :python:`mol` passes context, :python:`eigenstates[-1]` is the eigenstates at a specific value of the varied parameter, in this case the last value, 300G.
:python:`["N", "MF"]` are the labels we try to fit. Finally, :python:`index_repeats=True` adds an additional index for repeats as described above.

Another useful quantity is the transition dipole moment from one state to another, which tells us how well we can couple two states with incident EM radiation in order to drive a transition.

.. code-block:: python

    groundstate = 0

    transition_sigma_plus = calculate.transition_electric_moments(
         mol, eigenstates, h=1, from_states=groundstate
    )
    transition_pi = calculate.transition_electric_moments(
         mol, eigenstates, h=0, from_states=groundstate
    )
    transition_sigma_minus = calculate.transition_electric_moments(
         mol, eigenstates, h=-1, from_states=groundstate
    )

Here we calculate the transition dipole moments for the three different helicities of incoming radiation :math:`\sigma^+, \pi, \sigma^-`.
By default this function would calculate transition dipole moments from all states to all states, however here we consider only from the state with index 0, the ground state at high field, which our above labeling gave :math:`(0,5)_0`.
