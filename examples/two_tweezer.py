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

fig, ((axul,axur),(axdl,axdr)) = plt.subplots(2,2, constrained_layout=True)

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 3

# 0.2V
B = 181.699 * GAUSS

INTEN_STEPS_1065 = 50
INTEN_MIN_1065 = 0.001
INTEN_MAX_1065 = 50
INTEN1065 = np.linspace(INTEN_MIN_1065, INTEN_MAX_1065, INTEN_STEPS_1065) * kWpercm2

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)
Hac1065 = operators.ac_ham(mol, a02=mol.a02[1065], beta=np.pi/40)
Hac817 = operators.ac_ham(mol, a02=mol.a02[817], beta=np.pi/50)

# Overall Hamiltonian
Htot = H0 + Hz * B + Hac1065 * INTEN1065[:, None, None]

# Solve (diagonalise) Hamiltonians
eigenergies, eigstates = calculate.solve_system(Htot)

eiglabels = calculate.label_states(mol, eigstates[0], ["N", "MF"], index_repeats=True)


def label_to_indices(labels, N, MF):
    labels = np.asarray(labels)
    indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
    return indices


rovibgroundstate = label_to_indices(eiglabels, 0, 5)[0]

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
    eigenergy = eigenergies[:, i]
    eiglabel = eiglabels[i]

    axul.plot(INTEN1065 / kWpercm2, eigenergy / MHz, c="k", alpha=0.1)
    if eiglabel[1] == 4 or eiglabel[1] == 5 or eiglabel[1] == 6:
        pol_index = eiglabel[1] - 5
        c_plot = c[pol_index]
        coupling_matrix = coupling_matrices[pol_index]

        alpha_values = (coupling_matrix[:, 0, i] / (mol.d0)) ** 0.5
        colors = np.zeros((INTEN_STEPS_1065, 4))
        if pol_index == 0:
            c_on = 2
        elif pol_index == 1:
            c_on = 1
        else:
            c_on = 0
        colors[:, c_on] = 1  # red
        colors[:, 3] = alpha_values
        plotting.colorline(
            axul,
            INTEN1065 / kWpercm2,
            eigenergy / MHz,
            colors,
            linewidth=1.5,
        )

        # ax.plot(INTEN817/ kWpercm2, eigenergy / MHz, c=c_plot, alpha=1)

for i in range(0,32):
    eigenergy = eigenergies[:, i]
    eiglabel = eiglabels[i]

    axdl.plot(INTEN1065 / kWpercm2, eigenergy / MHz, c="k", alpha=0.1)

    if i == rovibgroundstate:
        axdl.plot(INTEN1065 / kWpercm2, eigenergy / MHz, c="red", alpha=0.8)


INTEN_STEPS_817 = 50
INTEN_MIN_817 = 0.001
INTEN_MAX_817 = 50
INTEN817 = np.linspace(INTEN_MIN_817, INTEN_MAX_817, INTEN_STEPS_817) * kWpercm2

Htot = H0 + Hz * B + Hac1065 * INTEN_MAX_1065*kWpercm2 + Hac817 * INTEN817[:, None, None]

# Solve (diagonalise) Hamiltonians
eigenergies, eigstates = calculate.solve_system(Htot)

eiglabels = calculate.label_states(mol, eigstates[0], ["N", "MF"], index_repeats=True)

def label_to_indices(labels, N, MF):
    labels = np.asarray(labels)
    indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
    return indices


rovibgroundstate = label_to_indices(eiglabels, 0, 5)[0]

pi_coupling = calculate.transition_electric_moments(
    mol, eigstates, 0, from_states=[rovibgroundstate]
)
sigma_plus_coupling = calculate.transition_electric_moments(
    mol, eigstates, +1, from_states=[rovibgroundstate]
)
sigma_minus_coupling = calculate.transition_electric_moments(
    mol, eigstates, -1, from_states=[rovibgroundstate]
)

bicromatic_817 = -mol.a02[1065][1]/mol.a02[817][1]*INTEN_MAX_1065
axins = axur.inset_axes(
    [0.05, 0.6, 0.4, 0.38],
    xlim=(bicromatic_817-3, bicromatic_817+3), ylim=(974.3, 975.4), xticklabels=[], yticklabels=[])

axinsbicromatic = axur.inset_axes(
    [0.55, 0.6, 0.4, 0.38],
    xlim=(bicromatic_817-3, bicromatic_817+3), ylim=(974.3, 975.4), xticklabels=[], yticklabels=[])

axinsbicromatic.set_xlabel(r"$I_{817} = -\frac{\alpha^{1065}_2}{\alpha^{817}_2} I_{1065}$")

for ax in (axur, axins):
    c = ("blue", "red", "green")
    coupling_matrices = (pi_coupling, sigma_plus_coupling, sigma_minus_coupling)
    for i in range(32, 128):
        eigenergy = eigenergies[:, i]
        eiglabel = eiglabels[i]

        ax.plot(INTEN817 / kWpercm2, eigenergy / MHz, c="k", alpha=0.1)
        if eiglabel[1] == 4 or eiglabel[1] == 5 or eiglabel[1] == 6:
            pol_index = eiglabel[1] - 5
            c_plot = c[pol_index]
            coupling_matrix = coupling_matrices[pol_index]

            alpha_values = (coupling_matrix[:, 0, i] / (mol.d0)) ** 0.5
            colors = np.zeros((INTEN_STEPS_817, 4))
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

for i in range(0,32):
    eigenergy = eigenergies[:, i]
    eiglabel = eiglabels[i]

    axdr.plot(INTEN817 / kWpercm2, eigenergy / MHz, c="k", alpha=0.1)

    if i == rovibgroundstate:
        axdr.plot(INTEN817 / kWpercm2, eigenergy / MHz, c="red", alpha=0.8)


############
        
Htot = H0 + Hz * B + Hac817 * INTEN817[:, None, None] + Hac1065 * -mol.a02[817][1]/mol.a02[1065][1] * INTEN817[:,None,None]

# Solve (diagonalise) Hamiltonians
eigenergies, eigstates = calculate.solve_system(Htot)

eiglabels = calculate.label_states(mol, eigstates[0], ["N", "MF"], index_repeats=True)

def label_to_indices(labels, N, MF):
    labels = np.asarray(labels)
    indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
    return indices


rovibgroundstate = label_to_indices(eiglabels, 0, 5)[0]

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
    eigenergy = eigenergies[:, i]
    eiglabel = eiglabels[i]

    axinsbicromatic.plot(INTEN817 / kWpercm2, eigenergy / MHz, c="k", alpha=0.1)
    if eiglabel[1] == 4 or eiglabel[1] == 5 or eiglabel[1] == 6:
        pol_index = eiglabel[1] - 5
        c_plot = c[pol_index]
        coupling_matrix = coupling_matrices[pol_index]

        alpha_values = (coupling_matrix[:, 0, i] / (mol.d0)) ** 0.5
        colors = np.zeros((INTEN_STEPS_817, 4))
        if pol_index == 0:
            c_on = 2
        elif pol_index == 1:
            c_on = 1
        else:
            c_on = 0
        colors[:, c_on] = 1  # red
        colors[:, 3] = alpha_values
        plotting.colorline(
            axinsbicromatic,
            INTEN817 / kWpercm2,
            eigenergy / MHz,
            colors,
            linewidth=1.5,
        )



axul.set_ylabel("$E/h$ (MHz)")
axdl.set_xlabel("1065nm Intensity ($kW/cm^2$)")
axdr.set_xlabel("817nm Intensity ($kW/cm^2$)")

axul.set_ylim(973,981.5)
axur.set_ylim(973,981.5)

axdl.set_ylim(-6.5,1)
axdr.set_ylim(-6.5,1)

axul.set_xlim(0,50)
axur.set_xlim(0,50)

axdl.set_xlim(0,50)
axdr.set_xlim(0,50)

axur.axvline(bicromatic_817, c='k', linestyle='--')
axdr.axvline(bicromatic_817, c='k', linestyle='--')
axins.axvline(bicromatic_817, c='k', linestyle='--')
axinsbicromatic.axvline(bicromatic_817, c='k', linestyle='--')

axur.set_yticks([])
axdr.set_yticks([])

axur.indicate_inset_zoom(axins, edgecolor="black")

plt.show()
input("Enter to close")
