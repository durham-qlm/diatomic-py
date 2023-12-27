import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.optimize import leastsq

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


def label_to_indices(labels, N, MF):
    labels = np.asarray(labels)
    indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
    return indices


# Set logging
diatomic.configure_logging()

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 2

# Load CSV real data
IntPi, IntPiErrUp, IntPiErrDwn, TransPiCent, TransPiCentErr = np.genfromtxt(
    "./examples/pi_transitions.csv",
    delimiter=",",
    skip_header=1,
    dtype=None,
    encoding=None,
    unpack=True,
)
(
    IntSig,
    IntSigErrUp,
    IntSigErrDwn,
    TransCent1,
    TransCent1Err,
    TransCent2,
    TransCent2Err,
) = np.genfromtxt(
    "./examples/sigma_transitions.csv",
    delimiter=",",
    skip_header=1,
    dtype=None,
    encoding=None,
    unpack=True,
)

INTEN1065 = 3.07 * kWpercm2
B = 181.699 * GAUSS

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)

Hac_unit_aniso = operators.unit_ac_aniso(mol.Nmax, mol.Ii[0], mol.Ii[1], Beta=0)

# Seed Hamiltonian
Hseed = H0 + Hz * B


# Define residuals function
# outer_i=0
def residuals(params, sigma_int, pi_int, sigma_minus_data, sigma_plus_data, pi_data):
    a2_1065, a2_817 = params

    # Sigma data
    Htot = (
        Hseed
        + (sigma_int[:, None, None] * kWpercm2 * a2_817 + INTEN1065 * a2_1065)
        * Hac_unit_aniso
    )

    eigenergies, eigstates = calculate.solve_system(Htot)
    eiglabels = calculate.label_states(mol, eigstates[0, :, 0:128], ["N", "MF"])
    sigma_minus_index = label_to_indices(eiglabels, 1, 4)[1]
    sigma_plus_index = label_to_indices(eiglabels, 1, 6)[0]
    start_index = label_to_indices(eiglabels, 0, 5)[0]

    res_sigma_minus = (
        sigma_minus_data
        - (eigenergies[:, sigma_minus_index] - eigenergies[:, start_index]) / MHz
    ) / TransCent1Err
    res_sigma_plus = (
        sigma_plus_data
        - (eigenergies[:, sigma_plus_index] - eigenergies[:, start_index]) / MHz
    ) / TransCent2Err

    # Pi data
    Htot = (
        Hseed
        + (pi_int[:, None, None] * kWpercm2 * a2_817 + INTEN1065 * a2_1065)
        * Hac_unit_aniso
    )

    eigenergies, eigstates = calculate.solve_system(Htot)
    eiglabels = calculate.label_states(mol, eigstates[0, :, 0:128], ["N", "MF"])
    pi_index = label_to_indices(eiglabels, 1, 5)[2]
    start_index = label_to_indices(eiglabels, 0, 5)[0]

    res_pi = (
        pi_data - (eigenergies[:, pi_index] - eigenergies[:, start_index]) / MHz
    ) / TransPiCentErr

    return np.concatenate((res_sigma_minus, res_sigma_plus, res_pi))


p0 = [mol.a02[1065][1], mol.a02[817][1]]  # Initial guess for the parameters

arg_list = (IntSig, IntPi, TransCent1, TransCent2, TransPiCent)

popt, pcov, _, _, ier = leastsq(residuals, p0, args=arg_list, full_output=True)

perr = np.sqrt(np.diag(pcov))

to_cgs = 4 * pi * eps0 * bohr**3
print("1065nm, 817nm")
print("Optimal:  ", popt / to_cgs)
print("Error:    ", perr / to_cgs)

# Plotting...
pplot = popt  # [-2700 * to_cgs]
Hac1065 = operators.ac_ham(mol, (0, pplot[0]), beta=0)
Hac817 = operators.ac_ham(mol, (0, pplot[1]), beta=0)

# Parameter Space
INTEN_STEPS = 200
INTEN_MIN = 0.001
INTEN_MAX = 52
INTEN817 = np.linspace(INTEN_MIN, INTEN_MAX, INTEN_STEPS) * kWpercm2

# Overall Hamiltonian
Htot = H0 + Hz * B + Hac1065 * INTEN1065 + Hac817 * INTEN817[:, None, None]

# Solve (diagonalise) Hamiltonians
eigenergies, eigstates = calculate.solve_system(Htot)

eiglabels = calculate.label_states(mol, eigstates[-1], ["N", "MF"], index_repeats=True)

# Calculate transition strength
pi_coupling = calculate.transition_electric_moments(mol, eigstates, 0)
sigma_plus_coupling = calculate.transition_electric_moments(mol, eigstates, +1)
sigma_minus_coupling = calculate.transition_electric_moments(mol, eigstates, -1)

start_state_index = label_to_indices(eiglabels, 0, 5)[0]

fig, (ax1, axd) = plt.subplots(
    2, 1, sharex=True, figsize=(6, 6), constrained_layout=True, height_ratios=[3, 1]
)
for ax in [ax1]:
    for off_res_index in range(32, 128):
        ax.plot(
            INTEN817 / kWpercm2,
            (eigenergies[:, off_res_index] - eigenergies[:, start_state_index]) / MHz,
            c="k",
            alpha=0.04,
        )

        match eiglabels[off_res_index, 1]:
            case 4:
                c_on = 0
                coupling_matrix = sigma_minus_coupling
            case 5:
                c_on = 2
                coupling_matrix = pi_coupling
            case 6:
                c_on = 1
                coupling_matrix = sigma_plus_coupling

        alpha_values = (
            coupling_matrix[:, start_state_index, off_res_index] / (mol.d0)
        ) ** 0.5
        colors = np.zeros((INTEN_STEPS, 4))
        colors[:, c_on] = 1  # red
        colors[:, 3] = alpha_values
        plotting.colorline(
            ax,
            INTEN817 / kWpercm2,
            (eigenergies[:, off_res_index] - eigenergies[:, start_state_index]) / MHz,
            colors,
            linewidth=1.5,
        )

    ax.errorbar(
        IntPi,
        TransPiCent,
        yerr=TransPiCentErr,
        xerr=(IntPiErrDwn, IntPiErrUp),
        ls="none",
        fmt="o",
        color="darkblue",
        ecolor="darkblue",
        elinewidth=3,
        capsize=0,
    )
    ax.errorbar(
        IntSig,
        TransCent1,
        yerr=TransCent1Err,
        xerr=(IntSigErrDwn, IntSigErrUp),
        ls="none",
        fmt="o",
        color="darkred",
        ecolor="darkred",
        elinewidth=3,
        capsize=0,
    )
    ax.errorbar(
        IntSig,
        TransCent2,
        yerr=TransCent2Err,
        xerr=(IntSigErrDwn, IntSigErrUp),
        ls="none",
        fmt="o",
        color="darkgreen",
        ecolor="darkgreen",
        elinewidth=3,
        capsize=0,
    )


arg_list = (IntSig, IntPi, TransCent1, TransCent2, TransPiCent)
resid = residuals(popt, *arg_list)

i = 0
jump = len(IntSig)
axd.scatter(IntSig, resid[i : i + jump], c="darkred")
i += jump
axd.scatter(IntSig, resid[i : i + jump], c="darkgreen")
i += jump
axd.scatter(IntPi, resid[i:], c="darkblue")
axd.axhline(0, dashes=(3, 2), color="k", linewidth=1)
axd.set_ylim(-3, 3)
axd.set_ylabel("norm. resid.")

ax1.set_ylim(979.5, 983)

# ax1.set_xlim(0.3, 52)
# ax1.set_xscale("log")

ax1.set_xlim(0, 52)
ax1.set_xscale("linear")

ax1.set_ylabel("Transition Frequency (MHz)")
axd.set_xlabel("817nm Intensity ($kW/cm^2$)")

ax1.set_title(
    rf"$(\alpha_2(1065nm) = ({popt[0]/to_cgs:.0f} \pm {perr[0]/to_cgs:.0f})"
    " \cdot 4 \pi \epsilon_0 a_0) @ 3.07 kW/cm^2$\n"
    rf"$\alpha_2(817nm) = ({popt[1]/to_cgs:.0f} \pm {perr[1]/to_cgs:.0f})"
    " \cdot 4 \pi \epsilon_0 a_0$"
)

# fig.savefig("alpha2_regression.jpg", dpi=300)

fig.show()
input("Enter to close")
