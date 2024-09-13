import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

import diatomic
from diatomic.systems import SingletSigmaMolecule
import diatomic.plotting as plotting
import diatomic.operators as operators
import diatomic.calculate as calculate

# Define helpful constants
pi = scipy.constants.pi
bohr = scipy.constants.physical_constants["Bohr radius"][0]
eps0 = scipy.constants.epsilon_0

GAUSS = 1e-4  # T
MHz = scipy.constants.h * 1e6
muN = scipy.constants.physical_constants["nuclear magneton"][0]
H_BAR = scipy.constants.hbar
kWpercm2 = 1e7


# Set logging
diatomic.configure_logging()

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 2

# 0.2V
INTEN1065 = 2.88 * kWpercm2
B = 181.699 * GAUSS

# INTEN_STEPS = 200
# INTEN_MIN = 0.001
# INTEN_MAX = 20
# INTEN1065 = np.linspace(INTEN_MIN, INTEN_MAX, INTEN_STEPS) * kWpercm2

INTEN_STEPS = 100
INTEN_MIN = 0.001
INTEN_MAX = 6
INTEN817 = np.linspace(INTEN_MIN, INTEN_MAX, INTEN_STEPS) * kWpercm2

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)

fig, axs = plt.subplots(5, 5, constrained_layout=True)
print(axs.shape)

for k, axh in enumerate(axs):
    for j, ax in enumerate(axh):
        print(k, j)
        pol_1065 = (j - 2) * (5 / 360) * 2 * np.pi
        pol_817 = (k - 2) * (6 / 360) * 2 * np.pi

        Hac1065 = operators.ac_ham(mol, a02=mol.a02[1065], beta=pol_1065)
        Hac817 = operators.ac_ham(mol, a02=mol.a02[817], beta=pol_817)

        # Overall Hamiltonian
        Htot = H0 + Hz * B + Hac1065 * INTEN1065 + Hac817 * INTEN817[:, None, None]

        # Solve (diagonalise) Hamiltonians
        eigenergies, eigstates = calculate.solve_system(Htot)

        eiglabels = calculate.label_states(
            mol, eigstates[0], ["N", "MF"], index_repeats=True
        )

        def label_to_indices(labels, N, MF):
            labels = np.asarray(labels)
            indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
            return indices

        rovibgroundstate = label_to_indices(eiglabels, 0, 5)[0]
        oneone = label_to_indices(eiglabels, 1, 6)[0]

        pi_coupling = calculate.transition_electric_moments(
            mol, eigstates, 0, from_states=[rovibgroundstate]
        )
        sigma_plus_coupling = calculate.transition_electric_moments(
            mol, eigstates, +1, from_states=[rovibgroundstate]
        )
        sigma_minus_coupling = calculate.transition_electric_moments(
            mol, eigstates, -1, from_states=[rovibgroundstate]
        )

        c = ("blue", "red", "green")
        coupling_matrices = (pi_coupling, sigma_plus_coupling, sigma_minus_coupling)
        for i in range(32, 128):
            eigenergy = eigenergies[:, i] - eigenergies[:, rovibgroundstate]
            eiglabel = eiglabels[i]

            ax.plot(INTEN817 / kWpercm2, eigenergy / MHz, c="k", alpha=0.1)
            if eiglabel[1] == 4 or eiglabel[1] == 5 or eiglabel[1] == 6:
                pol_index = eiglabel[1] - 5
                c_plot = c[pol_index]
                coupling_matrix = coupling_matrices[pol_index]

                alpha_values = (coupling_matrix[:, 0, i] / (mol.d0)) ** 0.5
                colors = np.zeros((INTEN_STEPS, 4))
                if pol_index == 0:
                    c_on = 2
                elif pol_index == 1:
                    c_on = 1
                else:
                    c_on = 0
                colors[:, c_on] = 1  # red
                colors[:, 3] = alpha_values
                plotting.colorline(
                    ax,
                    INTEN817 / kWpercm2,
                    eigenergy / MHz,
                    colors,
                    linewidth=1.5,
                )

                # ax.plot(INTEN817/ kWpercm2, eigenergy / MHz, c=c_plot, alpha=1)

        # ax.set_ylabel("$E/h$ (MHz)")
        # ax.set_xlabel("817nm Intensity ($kW/cm^2$)")

        ax.set_xlim(0, INTEN_MAX)
        ax.set_ylim(980.12, 980.5)
        if k != 4:
            ax.set_xticks([])
        if j != 0:
            ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(f"{(k-2)*6} degrees 817 pol")
        if k == 0:
            ax.set_title(f"{(j-2)*5} degrees 1065 pol")

        ax.axvline(3.634, c="k", lw=1, linestyle="--")
        ax.axvspan(3.634 * 0.4, 3.634 * 1.1, alpha=0.05, color="k")


fig.supylabel("Transition from $(0,5)_0$ (MHz)")
fig.supxlabel("817nm Intensity ($kW/cm^2$)")

plt.show()
input("Enter to close")
