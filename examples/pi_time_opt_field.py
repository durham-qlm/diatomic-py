import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

import diatomic
from diatomic.systems import SingletSigmaMolecule
import diatomic.plotting as plotting
import diatomic.operators as operators
import diatomic.calculate as calculate

pi = scipy.constants.pi
bohr = scipy.constants.physical_constants["Bohr radius"][0]
eps0 = scipy.constants.epsilon_0

GAUSS = 1e-4  # T
MHz = scipy.constants.h * 1e6
muN = scipy.constants.physical_constants["nuclear magneton"][0]
H_BAR = scipy.constants.hbar
kWpercm2 = 1e7
START_FIELD = 181.699 * GAUSS

# Set logging
diatomic.configure_logging()

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 4
mol.a0 = 1800 * 4 * pi * eps0 * bohr**3
mol.a2 = 1997 * 4 * pi * eps0 * bohr**3

# Parameter Space
B_STEPS = 200
B_MIN_GAUSS = 0.001
B_MAX_GAUSS = 400
B = np.linspace(B_MIN_GAUSS, B_MAX_GAUSS, B_STEPS) * GAUSS

Inten = 3.07 * kWpercm2
DESIRED_ETA = 3

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)
Hac = operators.ac_ham(mol, mol.a02[1064], beta=0)

# Overall Hamiltonian
Htot = H0 + Hz * B[:, None, None] + Hac * Inten

# Solve (diagonalise) Hamiltonians
eigenergies, eigstates = calculate.solve_system(Htot)

# Label states
eiglabels = calculate.label_states(mol, eigstates[-1], ["N", "MF"])

# Calculate transition strength
sigma_plus_coupling = calculate.transition_electric_moments(mol, eigstates, +1)
sigma_minus_coupling = calculate.transition_electric_moments(mol, eigstates, -1)


def label_to_indices(labels, N, MF):
    labels = np.asarray(labels)
    indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
    return indices


T_G = np.zeros((mol.Nmax, B_STEPS), dtype=np.double)

# Create figure
fig, (axs, axs2) = plt.subplots(2, mol.Nmax, sharex=True)

for N in range(1, mol.Nmax + 1):
    from_label = (N - 1, N - 1 + 5)
    to_label = (N, N + 5)

    to_neighbours_label = (N, N - 2 + 5)

    from_index = label_to_indices(eiglabels, *from_label)[0]
    to_index = label_to_indices(eiglabels, *to_label)[0]

    to_neighbours_indices = label_to_indices(eiglabels, *to_neighbours_label)

    deltas = (
        np.abs(eigenergies[:, to_neighbours_indices].T - eigenergies[:, to_index])
        / H_BAR
    )

    specific_coupling = sigma_plus_coupling[:, from_index, to_index]

    gammas = np.abs(
        sigma_minus_coupling[:, from_index, to_neighbours_indices].T / specific_coupling
    )

    r = (4 * gammas**2 + gammas**4) / (deltas**2)

    er = np.sqrt(np.sum(r, axis=0))

    T_G[N - 1] = np.pi * er * 10 ** (DESIRED_ETA / 2) / 4

    # Plotting
    axup = axs[N - 1]
    axdown = axs2[N - 1]
    axup.plot(
        B / GAUSS,
        (
            eigenergies[:, 32 * (N) ** 2 : 32 * (N + 1) ** 2].T
            - eigenergies[:, to_index]
        ).T
        / MHz,
        c="k",
        lw=0.5,
        alpha=0.1,
    )

    axup.plot(B / GAUSS, np.zeros((B_STEPS)), c="green", lw=1, alpha=0.8)
    axup.set_ylim(-0.5, 1)

    for off_res_index in to_neighbours_indices:
        alpha_values = (
            sigma_minus_coupling[:, from_index, off_res_index] / (mol.d0)
        ) ** 0.5
        colors = np.zeros((B_STEPS, 4))
        colors[:, 0] = 1  # red
        colors[:, 3] = alpha_values
        plotting.colorline(
            axup,
            B / GAUSS,
            (eigenergies[:, off_res_index] - eigenergies[:, to_index]) / MHz,
            colors,
            linewidth=1.5,
        )

    axdown.plot(B / GAUSS, T_G[N - 1] / 1e-6, c="green")
    axdown.set_ylim(1e0, 1e3)
    axdown.set_yscale("log")
    axup.set_title(rf"$|{N-1},{N-1}\rangle \rightarrow |{N},{N}\rangle$")
    axup.axvline(START_FIELD / GAUSS, dashes=(3, 2), color="k", linewidth=1)
    axdown.axvline(START_FIELD / GAUSS, dashes=(3, 2), color="k", linewidth=1)


axs[0].set_xlim(0, B_MAX_GAUSS)
axs[0].set_ylabel("Detuning (MHz)")
axs2[0].set_ylabel(r"$t_\pi$ / $\mu s$")
fig.supxlabel("Magnetic Field (G)")
fig.suptitle(r"$(\sigma_+ + \sigma_-)/\sqrt{2}$, 3.07 $kW cm^{-2}$ trap depth")

fig.show()

input("enter to continue")
