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


# Example usage (angles in radians):
# omega, gamma, delta=0.0,-np.pi/4,np.pi/2
# omega, gamma, delta=0.0,0,0
omega, gamma, delta = 0.2, -0.7, 1.4
plotting.plot_polarization_ellipse(omega=omega, gamma=gamma, delta=delta)


def label_to_indices(labels, N, MF):
    labels = np.asarray(labels)
    indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
    return indices


# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 2

# Tests:
# Hac = operators.ac_ham(mol, a02=mol.a02[817], beta=0)
# print(Hac)
# should be the same as

# ratio=Hac_prime/Hac
# plt.matshow(np.real(ratio))
# plt.show()

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)

n_per_seg = [70, 250]
cum_n_per_seg = np.cumsum(n_per_seg)

inten_1064 = np.linspace(0, 5 * kWpercm2, n_per_seg[0])
inten_1013 = np.linspace(0.4 * kWpercm2, 120 * kWpercm2, n_per_seg[1])

steps = np.arange(0, cum_n_per_seg[-1], 1)
inten_1065_all = np.concatenate((inten_1064, [inten_1064[-1]] * n_per_seg[1]))
inten_1013_all = np.concatenate(([0.0] * n_per_seg[0], inten_1013))

Hac_1065 = operators.ac_ham_ellip(mol, mol.a02[1065], np.pi / 2, 0, 0)
Hac_1013 = operators.ac_ham_ellip(mol, mol.a02[1013], omega, gamma, delta)

# Parameter Space
B = 181.6 * GAUSS

# Overall Hamiltonian
Htot = (
    H0
    + Hz * B
    + Hac_1065 * inten_1065_all[:, None, None]
    + Hac_1013 * inten_1013_all[:, None, None]
)

# Solve (diagonalise) Hamiltonians
eigenenergies, eigenstates = calculate.solve_system(Htot)

# Apply labels (in some way arbitrary) warn if duplicate
eigenlabels = calculate.label_states(mol, eigenstates[0], ["N", "MF"])

ground_state_index = label_to_indices(eigenlabels, 0, 5)[0]

excited_state_index = label_to_indices(eigenlabels, 1, 6)[0]


transition_sigma_plus = calculate.transition_electric_moments(
    mol, eigenstates[:, :, :], h=1, from_states=[ground_state_index]
)
transition_pi = calculate.transition_electric_moments(
    mol, eigenstates[:, :, :], h=0, from_states=[ground_state_index]
)
transition_sigma_minus = calculate.transition_electric_moments(
    mol, eigenstates[:, :, :], h=-1, from_states=[ground_state_index]
)

# %% Plot results
fig, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)

transition_energies = eigenenergies - (eigenenergies[:, ground_state_index])[:, None]

cs = ["green", "blue", "red"]
rgbis = [1, 2, 0]
h = [transition_sigma_plus, transition_pi, transition_sigma_minus]

for ax, ir, xaxis in zip(
    axs,
    [(0, cum_n_per_seg[0]), (cum_n_per_seg[0], cum_n_per_seg[1])],
    [inten_1064, inten_1013],
):
    for i in range(32, 128):
        ax.plot(
            xaxis / kWpercm2,
            transition_energies[ir[0] : ir[1], i] / MHz,
            c="k",
            lw=0.5,
            alpha=0.3,
        )

    for c, rgbi, transition_elements in zip(cs, rgbis, h):
        for eigindex in range(32, 128):
            colours = np.zeros((ir[1] - ir[0], 4))
            colours[:, rgbi] = 1
            colours[:, 3] = np.minimum(
                (
                    transition_elements[ir[0] : ir[1], ground_state_index, eigindex]
                    / mol.d0
                )
                ** 0.5,
                1,
            )
            plotting.colorline(
                ax,
                xaxis / kWpercm2,
                transition_energies[ir[0] : ir[1], eigindex] / MHz,
                colors=colours,
                linewidth=1.5,
            )
    ax.set_xlim(xaxis[0] / kWpercm2, xaxis[-1] / kWpercm2)


# ax.set_xlim(0, Iten_max)

axs[0].set_ylabel("Transition energy from $N=0, M_F=5$,  $E$ / $h$ (MHz)")
axs[0].set_xlabel("Laser intensity (kW cm$^{-2}$)")

left_frac_keep = 0.0
fig.get_layout_engine().set(rect=(left_frac_keep, 0, 1 - left_frac_keep, 1))
fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0, wspace=0.00)


fig.show()
# %%
