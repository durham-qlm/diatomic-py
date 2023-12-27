import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import diatomic
from diatomic.systems import SingletSigmaMolecule
import diatomic.operators as operators
import diatomic.calculate as calculate
from diatomic.plotting import plot_rotational_2d

# Set logging
diatomic.configure_logging()

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("RigidRotor")
mol.Nmax = 6

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hdc = operators.dc_ham(mol)
Hz = operators.zeeman_ham(mol)

# Parameter Space
E_STEPS = 200
E = np.linspace(0.001, 25, E_STEPS) * (mol.Brot / mol.d0)  # V/m

# Overall Hamiltonian
Htot = H0 + Hdc * E[:, None, None] + Hz * 1e-8

# Solve (diagonalise) Hamiltonians
eigenenergies, eigenstates = calculate.solve_system(Htot)

# Apply labels (in some way arbitrary) warn if duplicate
eigenlabels = calculate.label_states(mol, eigenstates[0], ["N", "MN"])

# Sort by label lexicographically
eigenlabels, eigenenergies, eigenstates = calculate.sort_by_labels(
    eigenlabels, eigenenergies, eigenstates
)

# Calculate derived values
electric_moments = calculate.electric_moment(mol, eigenstates)

# Plot results
fig, (axl, axr) = plt.subplots(1, 2, constrained_layout=True)

cmaps = ["Purples", "Blues", "Greens", "Reds", "YlOrBr", "autumn", "autumn", "autumn"]
positions = []

n_states = len(eigenenergies[0])

for eigenenergy, eigenlabel in zip(eigenenergies.T, eigenlabels):
    N = eigenlabel[0]
    M = eigenlabel[1]
    if N > 4:
        break
    cmap = mpl.colormaps[cmaps[N]]
    col = cmap((N + M + 1) / (2 * N + 2))
    if M >= 0:  # \pm M_N are degenerate
        axl.plot(E / (mol.Brot / mol.d0), eigenenergy / mol.Brot, c=col)
        pm_string = r"\pm" if M != 0 else ""

        axl.text(
            E[-1] / (mol.Brot / mol.d0),
            eigenenergy[-1] / mol.Brot,
            rf"$|{int(N)}, {pm_string}{int(M)}\rangle$",
        )

axl.set_xlim(0, 25)
axl.set_ylim(-20, 30)
axl.set_xlabel("Electric Field $(B_0/d_0)$")
axl.set_ylabel("Energy $(B_0)$")

#####

state = 2
probs = np.abs(eigenstates[:, :, state]) ** 2

important_count = 7
sorti = np.argsort(-np.abs(eigenstates[1, :, state]) ** 2)

importantprobs = np.abs(eigenstates[:, sorti[:important_count], state]) ** 2

colors = []
plot_probs = np.zeros((important_count, E_STEPS))
ii = 0
# for i in sorti[:important_count]:
for j, (N, M) in enumerate(operators.sph_iter(7)):
    for i in sorti[:important_count]:
        if j == i:
            cmap = mpl.colormaps[cmaps[N]]
            colors.append(cmap((N + M + 1) / (2 * N + 2)))
            plot_probs[ii, :] = np.abs(eigenstates[:, i, state]) ** 2
            ii += 1

axr.stackplot(
    E / (mol.Brot / mol.d0), plot_probs, colors=colors, lw=0.5, ec="face", aa=True
)
axr.set_ylabel(r"$|\langle N,M|\tilde{N}=1,M=0\rangle|^2$")
axr.set_xlabel("Electric Field $(B_0/d_0)$")
axr.set_ylim(0, 1.0)
axr.set_xlim(0, 25)

axr.text(3, 0.5, r"$|1,0\rangle$", ha="center", va="center")
axr.text(9, 0.2, r"$|0,0\rangle$", ha="center", va="center")
axr.text(15, 0.5, r"$|2,0\rangle$", ha="center", va="center")
axr.text(19, 0.75, r"$|3,0\rangle$", ha="center", va="center")
axr.text(22.5, 0.94, r"$|4,0\rangle$", ha="center", va="center")


# This works actually
rect = [0.1, 0.1, 0.2, 0.2]
ax_inset = fig.add_axes(rect)
E_where_index = int(0.7 * E_STEPS)
eigstate_index = 2

plot_rotational_2d(ax_inset, mol, eigenstates[E_where_index, :, eigstate_index])
con1 = mpl.patches.ConnectionPatch(
    (0, 1),
    (
        E[E_where_index] / (mol.Brot / mol.d0),
        eigenenergies[E_where_index, eigstate_index] / mol.Brot,
    ),
    coordsA="axes fraction",
    coordsB="data",
    axesA=ax_inset,
    axesB=axl,
)
fig.add_artist(con1)
con2 = mpl.patches.ConnectionPatch(
    (1, 0),
    (
        E[E_where_index] / (mol.Brot / mol.d0),
        eigenenergies[E_where_index, eigstate_index] / mol.Brot,
    ),
    coordsA="axes fraction",
    coordsB="data",
    axesA=ax_inset,
    axesB=axl,
)
fig.add_artist(con2)

rect = [0.15, 0.83, 0.15, 0.15]
ax_inset = fig.add_axes(rect)
E_where_index = int(0.7 * E_STEPS)
eigstate_index = 19

plot_rotational_2d(ax_inset, mol, eigenstates[E_where_index, :, eigstate_index])
con1 = mpl.patches.ConnectionPatch(
    (1, 1),
    (
        E[E_where_index] / (mol.Brot / mol.d0),
        eigenenergies[E_where_index, eigstate_index] / mol.Brot,
    ),
    coordsA="axes fraction",
    coordsB="data",
    axesA=ax_inset,
    axesB=axl,
)
fig.add_artist(con1)
con2 = mpl.patches.ConnectionPatch(
    (1, 0),
    (
        E[E_where_index] / (mol.Brot / mol.d0),
        eigenenergies[E_where_index, eigstate_index] / mol.Brot,
    ),
    coordsA="axes fraction",
    coordsB="data",
    axesA=ax_inset,
    axesB=axl,
)
fig.add_artist(con2)

plt.show()
