import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

from diatomic.systems import SingletSigmaMolecule
import diatomic.operators as operators
import diatomic.calculate as solver

GAUSS = 1e-4  # T
MHz = scipy.constants.h * 1e6

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 2

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)

# Parameter Space
B = np.linspace(0.001, 300, 50) * GAUSS

# Overall Hamiltonian
Htot = H0 + Hz * B[:, None, None]

# Solve (diagonalise) Hamiltonians
eigenenergies, eigenstates = solver.solve_system(Htot)

# Plot results
fig, (ax_up, ax_down) = plt.subplots(2, 1, sharex=True)

ax_down.plot(B / GAUSS, eigenenergies[:, 0:32] / MHz, c="k", lw=0.5, alpha=0.3)
ax_up.plot(B / GAUSS, eigenenergies[:, 32:128] / MHz, c="k", lw=0.5, alpha=0.3)

ax_down.set_xlabel("Magnetic Field (G)")
fig.supylabel("Energy / h (MHz)")

plt.show()
