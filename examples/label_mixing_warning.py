import warnings

import numpy as np

from diatomic.systems import SingletSigmaMolecule
import diatomic.operators as operators
import diatomic.calculate as calculate

GAUSS = 1e-4  # T
KW_PER_CM2 = 1e7

INTENSITY_KW_PER_CM2 = 4.0
B_FIELDS_GAUSS = [0.001, 181.6]
BETAS = [0, np.sqrt(1 / 3)]
STATE_INDEX = 149
LABELS = ["N", "MF"]


def format_label(label):
    label_names = LABELS + ["k"]
    return (
        "("
        + ", ".join(f"{name}={value}" for name, value in zip(label_names, label))
        + ")"
    )


def label_case(mol, h0, hz, beta, b_gauss):
    hac = operators.ac_ham(mol, mol.a02[1064], beta=beta)
    htot = h0 + hz * b_gauss * GAUSS + hac * INTENSITY_KW_PER_CM2 * KW_PER_CM2

    _, eigstates = calculate.solve_system(htot)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        eiglabels = calculate.label_states(
            mol,
            eigstates,
            LABELS,
            index_repeats=True,
            warn_mixed=True,
            min_weight=0.99,
        )

    return eiglabels, caught_warnings


def main():
    mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
    mol.Nmax = 3

    h0 = operators.hyperfine_ham(mol)
    hz = operators.zeeman_ham(mol)

    print(f"RbCs, Nmax={mol.Nmax}")
    print(f"1064 nm tweezer intensity = {INTENSITY_KW_PER_CM2} kW/cm^2")
    print(f"Inspecting Python state index {STATE_INDEX}")
    print()

    for b_gauss in B_FIELDS_GAUSS:
        for beta in BETAS:
            eiglabels, caught_warnings = label_case(mol, h0, hz, beta, b_gauss)

            print(f"B = {b_gauss:g} G, beta = {beta:.6g}")
            print(f"  label[{STATE_INDEX}] = {format_label(eiglabels[STATE_INDEX])}")

            if caught_warnings:
                for warning in caught_warnings:
                    print(f"  label_states warning: {warning.message}")
            else:
                print("  no mixed-label warning")

            print()


if __name__ == "__main__":
    main()
