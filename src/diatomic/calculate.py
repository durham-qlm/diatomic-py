from diatomic import log_time
import diatomic.operators as operators

import numpy as np
import warnings


@log_time
def solve_system(hamiltonians, num_diagonals=None):
    """
    Solves for the eigenenergies and eigenstates of given Hamiltonian(s over 1
    parameter).

    This function diagonalizes the input Hamiltonian(s) to find the corresponding
    eigenenergies and eigenstates. It supports both single Hamiltonian and a set
    of Hamiltonians (e.g., varying with time or another parameter).

    It takes care to make sure the eigenenergies and eigenstates vary smoothly with
    respect to the varying parameter, unlike `np.eigh`.

    Args:
        hamiltonians (np.ndarray): A single Hamiltonian matrix or an array of
            Hamiltonian matrices.
        num_diagonals (int | None, optional): The number of local swaps above and below
            to consider when smoothing eigenfunctions. Changing this to a number can
            speed up smoothing calculations for large numbers of eigenstates. If None,
            compare all eigenstates to all other eigenstates when smoothing (safest).

    Returns:
        tuple: (eigenenergies, eigenstates).

    Raises:
        ValueError: If the input Hamiltonian has more than three dimensions.
    """
    eigenenergies_raw, eigenstates_raw = log_time(np.linalg.eigh)(hamiltonians)
    if hamiltonians.ndim == 2:
        return eigenenergies_raw, eigenstates_raw
    elif hamiltonians.ndim == 3:
        eigenenergies, eigenstates = sort_smooth(
            eigenenergies_raw, eigenstates_raw, num_diagonals=num_diagonals
        )
        return eigenenergies, eigenstates
    else:
        raise ValueError(
            "Too many dimensions, solve_system doesn't support smoothing"
            " eigenvalues over >1 parameters"
        )


def _matrix_prod_diagonal(A, B, d=0):
    """
    Computes the dth diagonal of the product of two matrices A and B,
    without computing the entire matrix product.

    Args:
        A (np.ndarray): The first matrix operand.
        B (np.ndarray): The second matrix operand.
        d (int, optional): The diagonal offset. Defaults to 0, which gives the
            main diagonal.

    Returns:
        np.ndarray: The computed diagonal elements as a one-dimensional array.
    """
    A_slice = slice(-d if d < 0 else None, -d if d > 0 else None)
    B_slice = slice(d if d > 0 else None, d if d < 0 else None)

    diag = np.einsum("...ab,...ba->...a", A[..., A_slice, :], B[..., B_slice])

    return diag


@log_time
def sort_smooth(eigvals, eigvecs, num_diagonals=None):
    """
    Smooths the eigenvalue trajectories as a function of an external parameter, rather
    than by energy order, as is the case with `np.eigh`.

    This function sorts the eigenvalues and eigenvectors to maintain continuity as the
    external parameter changes, preventing sudden jumps in the eigenvalue ordering.

    Args:
        eigvals (np.ndarray): Eigenvalues array with shape (steps, eigstate).
        eigvecs (np.ndarray): Eigenvectors array with shape (steps, basis, eigstate).
        num_diagonals (int | None, optional): The number of local swaps above and below
            to consider when smoothing eigenfunctions. Changing this to a number can
            speed up smoothing calculations for large numbers of eigenstates. If None,
            compare all eigenstates to all other eigenstates when smoothing (safest).

    Returns:
        tuple: The smoothed eigenvalues and eigenvectors.

    Raises:
        ValueError: If eigvecs does not have three dimensions.
    """
    if eigvecs.ndim != 3:
        raise ValueError("eigvecs is not a set of eigenvectors over one parameter.")

    param_step_count = eigvecs.shape[0]
    basis_size = eigvecs.shape[1]
    eigstate_count = eigvecs.shape[2]

    # Compute overlap matrix between (k)th and (k-1)th
    if num_diagonals is not None:
        k = min(num_diagonals, eigstate_count - 1)
        diagonals = np.zeros((param_step_count - 1, 2 * k + 1, eigstate_count))
        for diag_num in range(-k, k + 1):
            my_prod_diag = np.abs(
                _matrix_prod_diagonal(
                    eigvecs[:-1].swapaxes(-1, -2), eigvecs[1:].conj(), d=diag_num
                )
            )
            if diag_num > 0:
                diagonals[
                    :, k - diag_num, diag_num : diag_num + my_prod_diag.shape[1]
                ] = my_prod_diag
            else:
                diagonals[:, k - diag_num, 0 : my_prod_diag.shape[1]] = my_prod_diag

        best_overlaps = np.argmax(diagonals, axis=-2) - np.arange(
            k, k - eigstate_count, -1
        )
    else:
        overlap_matrices = np.abs(eigvecs[:-1].swapaxes(-1, -2) @ eigvecs[1:].conj())
        best_overlaps = np.argmax(overlap_matrices, axis=1)

    # Cumulative permutations
    integrated_permutations = np.empty((param_step_count, eigstate_count), dtype=int)
    integrated_permutations[-1] = np.arange(eigstate_count)

    for i in range(param_step_count - 2, -1, -1):
        integrated_permutations[i] = best_overlaps[i][integrated_permutations[i + 1]]

    # Rearrange to maintain continuity
    sorted_eigvals = eigvals[
        np.arange(param_step_count)[:, None], integrated_permutations
    ]
    sorted_eigvecs = eigvecs[
        np.arange(param_step_count)[:, None, None],
        np.arange(basis_size)[None, :, None],
        integrated_permutations[:, None, :],
    ]

    return sorted_eigvals, sorted_eigvecs


def _get_prior_repeat_indices(labels):
    """
    Given a numpy array, returns a list of 'prior repeat indices'. Each entry
    of the array that repeats gets an incrementing index on each subsequent occurrence.

    For example, given labels = [[1,1],[2,3],[1,1]], we see [1,1] repeats, and its
    second occurrence gets a 'prior repeat index' of 1. The output is [0,0,1].

    Args:
        labels (np.array): 2D array with labels.

    Returns:
        np.array: 1D array with 'prior repeat indices'.
    """

    double_labels = (2 * labels).astype(int)

    # Identify unique labels, get inverse indices and counts of labels
    _, inverse_indices, counts = np.unique(
        double_labels, return_inverse=True, return_counts=True, axis=0
    )

    # Initialize array for repeat indices
    repeat_indices = np.zeros(double_labels.shape[0], dtype=int)

    # For each unique label
    for idx, count in enumerate(counts):
        # Find all instances of the unique label
        instances = np.nonzero(inverse_indices == idx)[0]

        # Assign repeat indices in order of occurrence
        repeat_indices[instances] = np.arange(count)

    return repeat_indices


def _solve_quadratic(a, b, c):
    """Solve a quadratic equation a*x^2+b*x+c=0 .

    This is a simple function to solve the quadratic formula for x.
    Returns the most positive value of x supported.

    Args:
        a,b,c (floats) - coefficients in quadratic

    Returns:
        x (float) - maximum value of x supported by equation

    """
    x1 = (-b + np.sqrt((b**2) - (4 * (a * c)))) / (2 * a)
    x2 = (-b - np.sqrt((b**2) - (4 * (a * c)))) / (2 * a)
    return np.maximum(x1, x2)


@log_time
def label_states(mol, eigstates, labels, index_repeats=False):
    """
    Labels the eigenstates of a molecule with quantum numbers corresponding to
    specified labels. The returned numbers will only be good if the state is
    well-represented in the desired basis.

    This function processes a list of labels that specify which quantum numbers to
    calculate for the given eigenstates of a molecule. It uses the angular momentum
    operators for the molecule and the eigenstates to calculate these quantum numbers.

    The function supports labels for the total angular momentum vectors
    (N, F, I, I1, I2) for the z-component of them respectively (M<>).
    It also has an option to return the indices of repeated labels.

    Args:
        mol: An object representing the molecule, which contains the maximum angular
            momentum (Nmax) and intrinsic angular momenta (Ii).
        eigstates (ndarray): An array of eigenstates for which the labels are to be
            calculated.
        labels (list of str): A list of strings indicating which quantum numbers to
            calculate. Valid labels are "MN", "MF", "MI", "MI1", "MI2" for M-components
            and "N", "F", "I", "I1", "I2" for squared total angular momentum vectors.
        index_repeats (bool, optional): If True, includes the label indices of repeated
            labels in the output. Defaults to False.

    Returns:
        ndarray: A two-dimensional array where each row corresponds to the desired
            quantum numbers for each eigenstate, and each column corresponds to a label.
    """

    N_op, I1_op, I2_op = operators.generate_vecs(mol.Nmax, mol.Ii[0], mol.Ii[1])

    out_labels = []

    for label in labels:
        if label[0] == "M":
            match label:
                case "MN":
                    Tz = N_op[2]
                case "MF":
                    F = N_op + I1_op + I2_op
                    Tz = F[2]
                case "MI":
                    Itot = I1_op + I2_op
                    Tz = Itot[2]
                case "MI1":
                    Tz = I1_op[2]
                case "MI2":
                    Tz = I2_op[2]
                case _:
                    warnings.warn("Invalid label, ignoring")
                    continue
            double_M_labels = np.rint(
                2 * np.einsum("ik,ij,jk->k", np.conj(eigstates), Tz, eigstates)
            ).real.astype(int)
            M_labels = np.array([operators.HalfInt(of=dl) for dl in double_M_labels])
            out_labels.append(M_labels)
        else:
            match label:
                case "N":
                    T2 = operators.vector_dot(N_op, N_op)
                case "F":
                    F = N_op + I1_op + I2_op
                    T2 = operators.vector_dot(F, F)
                case "I":
                    Itot = I1_op + I2_op
                    T2 = operators.vector_dot(Itot, Itot)
                case "I1":
                    T2 = operators.vector_dot(I1_op, I1_op)
                case "I2":
                    T2 = operators.vector_dot(I2_op, I2_op)
                case _:
                    warnings.warn("Invalid label, ignoring")
                    continue
            T2eigval = np.einsum("ik,ij,jk->k", np.conj(eigstates), T2, eigstates)
            double_T_labels = np.rint(
                2 * _solve_quadratic(1, 1, -1 * T2eigval)
            ).real.astype(int)
            T_labels = np.array([operators.HalfInt(of=dl) for dl in double_T_labels])
            out_labels.append(T_labels)

    if index_repeats:
        out_labels.append(_get_prior_repeat_indices(np.array(out_labels).T))

    return np.array(out_labels).T


def sort_by_labels(eigenlabels, eigenenergies, eigenstates):
    """
    Sorts eigenstates and eigenenergies according to the provided labels
    lexicographically.

    This function sorts the eigenstates and eigenenergies in ascending order based
    on the eigenlabels, which can be quantum numbers or any other identifiers for
    the states. They are sorted with the first label value taking preference, and then
    the second if the first drew, as with a language dictionary.

    Args:
        eigenlabels (np.ndarray): Array of labels corresponding to eigenstates.
        eigenenergies (np.ndarray): Array of eigenenergies to be sorted.
        eigenstates (np.ndarray): Array of eigenstates to be sorted.

    Returns:
        tuple: The sorted eigenlabels, eigenenergies, and eigenstates.
    """
    indices = np.lexsort(eigenlabels[:, ::-1].T)
    return eigenlabels[indices], eigenenergies[:, indices], eigenstates[:, :, indices]


@log_time
def transition_electric_moments(mol, eigenstates, h=0, from_states=slice(None)):
    """
    Calculates the electric dipole transition coupling strengths between molecular
    eigenstates.

    Args:
        eigenstates (ndarray): An array containing the eigenstates of the molecule.
        mol: An object representing the molecule.
        h (int, optional): The helicity of the transition, +1 for sigma+, 0 for pi,
            -1 for sigma-.
        from_states (int, slice, or list, optional): The indices of the initial states
            from which transitions are considered. If an integer is passed, it is
            internallyconverted to a list. By default, all states are
            considered (slice(None)).

    Returns:
        ndarray: An array containing the absolute values of the transition electric
            moments for the specified transitions between eigenstates.
    """

    if isinstance(from_states, int):
        from_states = [from_states]

    dipole_op = mol.d0 * operators.expanded_unit_dipole_operator(mol, h)

    dipole_matrix_elements = (
        eigenstates[..., from_states].conj().swapaxes(-1, -2) @ dipole_op @ eigenstates
    )

    if h != 0:
        """
        After rotating wave approximation for sigma +/-, interaction matrix elements
        have d_pm matrix elements appearing above and below the diagonal respectively.
        """
        dipole_op_other = -dipole_op.conj().T

        opposite_dipole_matrix_elements = (
            eigenstates[..., from_states].conj().swapaxes(-1, -2)
            @ dipole_op_other
            @ eigenstates
        )

        all_mask = np.ones(dipole_matrix_elements.shape)
        mask = np.tril(all_mask, k=-1)
        mask_other = np.triu(all_mask, k=1)

        transition_electric_moments = (
            dipole_matrix_elements * mask + opposite_dipole_matrix_elements * mask_other
        )
    else:
        transition_electric_moments = dipole_matrix_elements

    return np.abs(transition_electric_moments)


def magnetic_moment(mol, eigstates):
    """
    Calculates the magnetic moments of each eigenstate.

    Args:
        eigstates (np.ndarray): array of eigenstates from diagonalisation
        mol: The molecule used to generate the eigenstates

    Returns:
        ndarray: An array corresponding to the magnetic moments of the eigenstates.
    """

    muz = -1 * operators.zeeman_ham(mol)

    mu = np.einsum(
        "vae,ab,vbe->ve", np.conjugate(eigstates), muz, eigstates, optimize="optimal"
    )
    return mu.real


def electric_moment(mol, eigstates):
    """
    Calculates the electric moments of each eigenstate.

    Args:
        eigstates (np.ndarray): array of eigenstates from diagonalisation
        mol: The molecule used to generate the eigenstates

    Returns:
        ndarray: An array corresponding to the electric moments of the eigenstates.
    """

    dz = -1 * operators.dc_ham(mol)

    d = np.einsum(
        "vae,ab,vbe->ve", np.conjugate(eigstates), dz, eigstates, optimize="optimal"
    )
    return d.real
