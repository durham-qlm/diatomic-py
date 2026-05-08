import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

import diatomic
from diatomic.systems import SingletSigmaMolecule
import diatomic.operators as operators
import diatomic.calculate as calculate
import diatomic.plotting as plotting

plt.close("all")

# Use logging.DEBUG here to include nested timings such as diagonalisation chunks.
diatomic.configure_logging(level=logging.INFO)

GAUSS = 1e-4  # T
MHz = scipy.constants.h * 1e6
kWpercm2 = 1e7


# Example usage (angles in radians):

phi_rad = 28.31 * np.pi / 180
chi_rad = -38.73 * np.pi / 180
omega = 7 * np.pi / 180
gamma = np.arccos(np.cos(2 * chi_rad) * np.cos(2 * phi_rad)) / 2
delta = np.arctan2(np.sin(2 * chi_rad), np.cos(2 * chi_rad) * np.sin(2 * phi_rad))

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

n_per_seg = [30, 50]
cum_n_per_seg = np.cumsum(n_per_seg)

inten_1064 = np.linspace(0, 3.5 * kWpercm2, n_per_seg[0])
inten_1013 = np.linspace(0.4 * kWpercm2, 30 * kWpercm2, n_per_seg[1])

steps = np.arange(0, cum_n_per_seg[-1], 1)
inten_1065_all = np.concatenate((inten_1064, [inten_1064[-1]] * n_per_seg[1]))
inten_1013_all = np.concatenate(([0.0] * n_per_seg[0], inten_1013))

pi = scipy.constants.pi
bohr = scipy.constants.physical_constants["Bohr radius"][0]
eps0 = scipy.constants.epsilon_0
mol.a02[1013] = (
    2000 * 4 * pi * eps0 * bohr**3,
    300 * 4 * pi * eps0 * bohr**3,
    # This is an effective polarisability that is fit, assuming we're
    # in the centre of a gaussian beam with
)

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
eigenenergies, eigenstates = calculate.solve_system(Htot, progress=True, chunk_size=10)

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
fig, axs = plt.subplots(
    1, 2, sharey=True, constrained_layout=True, width_ratios=[1, 1.7], figsize=(5, 3)
)

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
    ax.set_xlim(0, xaxis[-1] / kWpercm2)


# ax.set_xlim(0, Iten_max)

axs[0].set_ylabel("Transition energy from $N=0, M_F=5$,  $E$ / $h$ (MHz)")
axs[0].set_xlabel("Laser intensity (kW cm$^{-2}$)")

ax = axs[1]
# Transitions 1
x = np.array(
    [
        1.5356828,
        15.35682798,
        23.03524197,
        7.67841399,
        3.0713656,
        6.14273119,
        18.42819357,
        9.21409679,
        21.49955917,
        12.28546238,
    ]
)
y = (
    np.array(
        [
            np.float64(331.6106864481848),
            np.float64(320.89166453475735),
            np.float64(289.7531361859122),
            np.float64(330.5508966782318),
            np.float64(336.4885058538581),
            np.float64(330.083396556891),
            np.float64(321.4298551735948),
            np.float64(324.52180289709264),
            np.float64(305.16994128711707),
            np.float64(324.3887610551789),
        ]
    )
    * 1e-3
    + 980
)
y_err = (
    np.array(
        [
            np.float64(2.1463699698868024),
            np.float64(2.7700190634181783),
            np.float64(6.562231056433836),
            np.float64(2.4678978089987265),
            np.float64(2.9049032158257564),
            np.float64(3.764564783148028),
            np.float64(4.449869217617698),
            np.float64(3.6191614084959127),
            np.float64(6.6809861388009555),
            np.float64(3.987889400755413),
        ]
    )
    * 1e-3
)

ax.errorbar(x, y, yerr=y_err, label=r"$\sigma^-$", color="C1", ls="", marker="o")

# Transitions2:
x = np.array(
    [
        1.5356828,
        15.35682798,
        23.03524197,
        7.67841399,
        3.0713656,
        6.14273119,
        18.42819357,
        9.21409679,
        21.49955917,
        12.28546238,
    ]
)

y = (
    np.array(
        [
            np.float64(425.49742477776186),
            np.float64(406.2795382772449),
            np.float64(388.90481947892937),
            np.float64(416.50177925912845),
            np.float64(426.14822943673744),
            np.float64(419.37164619504875),
            np.float64(403.901668838014),
            np.float64(417.0304986487666),
            np.float64(398.32200437744837),
            np.float64(414.2388687171375),
        ]
    )
    * 1e-3
    + 980
)

y_err = (
    np.array(
        [
            np.float64(2.04981983432611),
            np.float64(2.027069157894005),
            np.float64(4.256746368501268),
            np.float64(1.5867481189039707),
            np.float64(1.9845456462277902),
            np.float64(1.8762370061072797),
            np.float64(2.4265769416635914),
            np.float64(2.4671794821000037),
            np.float64(3.3996318771016223),
            np.float64(2.3930829280340262),
        ]
    )
    * 1e-3
)

ax.errorbar(x, y, yerr=y_err, label=r"$\sigma^+$", color="C2", ls="", marker="o")

## Transitions3:
x = np.array(
    [
        7.67841399,
        3.0713656,
        0.0,
        15.35682798,
        23.03524197,
        12.28546238,
        18.42819357,
        9.21409679,
    ]
)
y = (
    np.array(
        [
            np.float64(125.1988701960018),
            np.float64(118.09189109566739),
            np.float64(103.42209953488081),
            np.float64(141.00574344370693),
            np.float64(173.1775987081843),
            np.float64(135.19581968253385),
            np.float64(157.39057481624636),
            np.float64(124.07610656619853),
        ]
    )
    * 1e-3
    + 980
)

y_err = (
    np.array(
        [
            np.float64(2.0994381095300803),
            np.float64(3.6139219593759506),
            np.float64(3.358416847428735),
            np.float64(2.993453583114279),
            np.float64(3.2105469503306976),
            np.float64(2.379663626218664),
            np.float64(4.601307136404175),
            np.float64(2.7836196409112777),
        ]
    )
    * 1e-3
)


ax.errorbar(x, y, yerr=y_err, label=r"$\pi$", color="C0", ls="", marker="o")

ax.legend()

ax.set_ylim(980, 980.8)
left_frac_keep = 0.0
fig.get_layout_engine().set(rect=(left_frac_keep, 0, 1 - left_frac_keep, 1))
fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0, wspace=0.00)


plt.show()
