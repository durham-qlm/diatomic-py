import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

from diatomic.systems import SingletSigmaMolecule
import diatomic.operators as operators
import diatomic.calculate as calculate
import diatomic.plotting as plotting

GAUSS = 1e-4  # T
MHz = scipy.constants.h * 1e6
kWpercm2 = 1e7

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 2

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)
Hac = operators.ac_ham(mol, a02=mol.a02[817], beta=0)

# Parameter Space
B = 181.6 * GAUSS
Iten_max = 20
STEPS = 400
Inten = np.linspace(0.001, Iten_max, STEPS) * kWpercm2

# Overall Hamiltonian
Htot = H0 + Hz * B + Hac * Inten[:, None, None]

# Solve (diagonalise) Hamiltonians
eigenenergies, eigenstates = calculate.solve_system(Htot)

# Apply labels (in some way arbitrary) warn if duplicate
eigenlabels = calculate.label_states(mol, eigenstates[0], ["N", "MF"])

# Plot results
fig, ax = plt.subplots(1, 1, sharex=True)

transition_energies = eigenenergies[:, :]  # - np.min(eigenenergies, axis=1)[:, None]

for i in range(32, 128):
    print(eigenlabels[i])
    ax.plot(Inten / kWpercm2, transition_energies[:, i] / MHz, c="k", lw=0.5, alpha=0.3)

    colours = np.zeros((STEPS, 4))
    colours[:, 2] = np.abs(eigenstates[:, 64, i])
    colours[:, 3] = np.abs(eigenstates[:, 64, i])

    cl = plotting.colorline(
        ax,
        Inten / kWpercm2,
        transition_energies[:, i] / MHz,
        colors=colours,
        linewidth=2.0,
    )

ax.set_xlim(0, Iten_max)

ax.set_xlabel("Laser intensity (kW cm$^{-2}$)")
ax.set_ylabel("Transition energy from $N=0, M_F=5$,  $E$ / $h$ (MHz)")

fig.show()

input("enter to continue")
