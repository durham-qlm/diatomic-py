from diatomic import log_time
import diatomic.operators as operators

import numpy as np
from scipy.optimize import linear_sum_assignment
import warnings


@log_time
def solve_system(hamiltonians, num_diagonals=None, progress=False, chunk_size=None):
    """
    Solves for the eigenenergies and eigenstates of given Hamiltonian(s) over one
    parameter.

    This function diagonalizes the input Hamiltonian(s) to find the corresponding
    eigenenergies and eigenstates. It supports both single Hamiltonian and a set
    of Hamiltonians (e.g., varying with time or another parameter).

    It takes care to make sure the eigenenergies and eigenstates vary smoothly with
    respect to the varying parameter, unlike `np.eigh`.

    Args:
        hamiltonians (np.ndarray): A single Hamiltonian matrix or an array of
            Hamiltonian matrices.
        num_diagonals (int | None, optional): The number of local swaps above and below
            to consider when smoothing eigenfunctions. If None, compare all
            eigenstates to all other eigenstates when smoothing (safest).
        progress (bool, optional): If True, show a progress bar while diagonalising
            scan-shaped Hamiltonians. Uses tqdm if available. Defaults to False.
        chunk_size (int | None, optional): Number of Hamiltonians to diagonalise in
            each batch when using chunked diagonalisation. If None, the original
            fully batched diagonalisation is used unless progress=True, in which case
            chunk_size defaults to 1 so the progress bar can update after each step.

    Returns:
        tuple: (eigenenergies, eigenstates).

    Raises:
        ValueError: If the input Hamiltonian has more than three dimensions.
    """
    if hamiltonians.ndim == 2:
        eigenenergies_raw, eigenstates_raw = log_time(np.linalg.eigh)(hamiltonians)
        return eigenenergies_raw, eigenstates_raw
    elif hamiltonians.ndim == 3:
        eigenenergies_raw, eigenstates_raw = _diagonalize_scan(
            hamiltonians, progress=progress, chunk_size=chunk_size
        )
        eigenenergies, eigenstates = sort_smooth(
            eigenenergies_raw, eigenstates_raw, num_diagonals=num_diagonals
        )
        return eigenenergies, eigenstates
    else:
        raise ValueError(
            "Too many dimensions, solve_system doesn't support smoothing"
            " eigenvalues over >1 parameters"
        )


def _progress_iterator(iterable, progress=False, total=None, desc=None):
    if not progress:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:
        warnings.warn(
            "progress=True was requested, but tqdm is not installed. "
            "Continuing without a progress bar.",
            stacklevel=3,
        )
        return iterable

    return tqdm(iterable, total=total, desc=desc)


@log_time
def _diagonalize_scan(hamiltonians, progress=False, chunk_size=None):
    """
    Diagonalise a scan of Hamiltonians, optionally in chunks for progress reporting.
    """
    if chunk_size is not None:
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer or None.")
        chunk_size = int(chunk_size)

    if chunk_size is None and not progress:
        return np.linalg.eigh(hamiltonians)

    step_count = hamiltonians.shape[0]
    if step_count == 0:
        return np.linalg.eigh(hamiltonians)

    if chunk_size is None:
        chunk_size = 1

    chunk_starts = range(0, step_count, chunk_size)
    chunk_count = (step_count + chunk_size - 1) // chunk_size

    eigenenergies = None
    eigenstates = None
    for start in _progress_iterator(
        chunk_starts, progress=progress, total=chunk_count, desc="Diagonalising"
    ):
        stop = min(start + chunk_size, step_count)
        chunk_energies, chunk_states = np.linalg.eigh(hamiltonians[start:stop])

        if eigenenergies is None:
            eigenenergies = np.empty(
                (step_count, chunk_energies.shape[-1]), dtype=chunk_energies.dtype
            )
            eigenstates = np.empty(
                (step_count, *chunk_states.shape[-2:]), dtype=chunk_states.dtype
            )

        eigenenergies[start:stop] = chunk_energies
        eigenstates[start:stop] = chunk_states

    return eigenenergies, eigenstates


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


def _overlap_permutation(overlap_matrix):
    """
    Return the previous-step index assigned to each current-step eigenstate.

    The assignment is one-to-one, unlike taking an independent argmax for each
    state. This prevents duplicated trajectories when overlap maxima are tied or
    nearly degenerate.
    """
    # Maximise total overlap by minimising the negative overlap. Entries outside
    # a restricted band are -inf, so give them a large finite cost for scipy.
    cost_matrix = np.where(np.isfinite(overlap_matrix), -overlap_matrix, 1e12)
    row_ind, col_ind = linear_sum_assignment(
        cost_matrix
    )  # Minimises cost of bipartite matching
    permutation = np.empty(overlap_matrix.shape[1], dtype=int)
    permutation[col_ind] = row_ind
    return permutation


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
            to consider when smoothing eigenfunctions. Restricting this can reduce
            work if eigenstates only swap locally in energy order. If None, compare
            all eigenstates to all other eigenstates when smoothing (safest).

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

    # Compute overlap_matrices[step, i, j] is |<old_i | new_j>| between neighbouring
    # parameter values. A large value means eigenstate j at step+1 is most like
    # eigenstate i at step.
    if num_diagonals is not None:
        k = min(num_diagonals, eigstate_count - 1)
        overlap_matrices = np.full(
            (param_step_count - 1, eigstate_count, eigstate_count), -np.inf
        )
        # Fill only the requested band around the energy-ordered diagonal. This
        # assumes crossings are local in energy index, but still enforces a
        # one-to-one assignment inside that band below.
        for diag_num in range(-k, k + 1):
            my_prod_diag = np.abs(
                _matrix_prod_diagonal(
                    eigvecs[:-1].swapaxes(-1, -2), eigvecs[1:].conj(), d=diag_num
                )
            )
            if diag_num > 0:
                rows = np.arange(my_prod_diag.shape[1])
                cols = rows + diag_num
            else:
                cols = np.arange(my_prod_diag.shape[1])
                rows = cols - diag_num

            overlap_matrices[:, rows, cols] = my_prod_diag
    else:
        overlap_matrices = np.abs(eigvecs[:-1].swapaxes(-1, -2) @ eigvecs[1:].conj())

    best_overlaps = np.empty((param_step_count - 1, eigstate_count), dtype=int)
    for i, overlap_matrix in enumerate(overlap_matrices):
        # For each current state, choose one distinct previous state. A simple
        # argmax can assign the same previous state to several current states,
        # duplicating eigenvectors and losing others.
        best_overlaps[i] = _overlap_permutation(overlap_matrix)

    # Calculate cumulative permutations based on best overlaps.
    # Work backwards so the final parameter value keeps the usual energy order,
    # while earlier steps are relabelled to connect smoothly to it.
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


def _as_halfint_array(double_values):
    return np.array([operators.HalfInt(of=int(dl)) for dl in double_values])


def _operator_expectation(operator, eigstates):
    return np.einsum(
        "ik,ij,jk->k", np.conj(eigstates), operator, eigstates, optimize="optimal"
    )


def _is_diagonal(operator):
    return np.allclose(operator, np.diag(np.diag(operator)))


def _weights_for_state_labels(
    component_double_labels, component_probabilities, double_labels
):
    weights = np.zeros(double_labels.shape[0])
    for double_label in np.unique(double_labels):
        state_mask = double_labels == double_label
        component_mask = component_double_labels == double_label
        if np.any(component_mask):
            weights[state_mask] = component_probabilities[component_mask][
                :, state_mask
            ].sum(axis=0)
    return weights


def _weights_for_operator_labels(operator, eigstates, double_labels, angular=False):
    if _is_diagonal(operator):
        operator_eigvals = np.diag(operator).real
        operator_eigvecs = None
    else:
        operator_eigvals, operator_eigvecs = np.linalg.eigh(operator)

    if angular:
        component_double_labels = np.rint(
            2 * _solve_quadratic(1, 1, -1 * operator_eigvals)
        ).astype(int)
    else:
        component_double_labels = np.rint(2 * operator_eigvals).astype(int)

    if operator_eigvecs is None:
        component_probabilities = np.abs(eigstates) ** 2
    else:
        component_probabilities = np.abs(operator_eigvecs.conj().T @ eigstates) ** 2

    return _weights_for_state_labels(
        component_double_labels, component_probabilities, double_labels
    )


def _joint_weights_for_diagonal_labels(
    component_label_columns, state_label_columns, eigstates
):
    component_labels = np.column_stack(component_label_columns)
    state_labels = np.column_stack(state_label_columns)
    probabilities = np.abs(eigstates) ** 2

    weights = np.zeros(eigstates.shape[1])
    for state_idx, state_label in enumerate(state_labels):
        component_mask = np.all(component_labels == state_label, axis=1)
        weights[state_idx] = probabilities[component_mask, state_idx].sum()
    return weights


def _warn_mixed_states(label_names, labels, weights, min_weight):
    mixed_state_indices = np.flatnonzero(weights < min_weight)
    if mixed_state_indices.size == 0:
        return

    worst_state_idx = mixed_state_indices[np.argmin(weights[mixed_state_indices])]
    label_text = ", ".join(
        f"{label_name}={labels[worst_state_idx, i]}"
        for i, label_name in enumerate(label_names)
    )
    warnings.warn(
        f"{mixed_state_indices.size} state(s) have less than {min_weight:.0%} "
        f"weight in their assigned ({', '.join(label_names)}) label sector; "
        "labels may be unreliable. "
        f"Worst case: state {worst_state_idx} labelled ({label_text}) has "
        f"weight {weights[worst_state_idx]:.3f}.",
        stacklevel=3,
    )


@log_time
def label_states(
    mol,
    eigstates,
    labels,
    index_repeats=False,
    basis_idx=None,
    warn_mixed=True,
    min_weight=0.9,
):
    """
    Labels the eigenstates of a molecule with quantum numbers corresponding to
    specified labels. The returned numbers will only be good if the state is
    well-represented in the desired basis.

    This function processes a list of labels that specify which quantum numbers to
    calculate for the given eigenstates of a molecule. It uses the angular momentum
    operators for the molecule and the eigenstates to calculate these quantum numbers.

    The function supports labels for the total angular momentum vectors
    (N, F, I, I1, I2) and for their z-components respectively (M<>).
    Labels are inferred from operator expectation values. If warn_mixed is True,
    the function warns when an assigned label sector carries less than min_weight
    of an eigenstate's probability weight.

    Args:
        mol: An object representing the molecule, which contains the maximum angular
            momentum (Nmax), minimum angular momentum (Nmin), and intrinsic angular
            momenta (Ii).
        eigstates (ndarray): Eigenstates for which the labels are to be calculated.
            The expected shape is (basis, states). A single eigenvector with shape
            (basis,) is also accepted.
        labels (list of str): A list of strings indicating which quantum numbers to
            calculate. Valid labels are "MN", "MF", "MI", "MI1", "MI2" for M-components
            and "N", "F", "I", "I1", "I2" for squared total angular momentum vectors.
        index_repeats (bool, optional): If True, includes the label indices of repeated
            labels in the output. Defaults to False.
        basis_idx (ndarray | None, optional): Indices mapping a cropped eigstates
            basis back into the full uncoupled basis. Required when eigstates were
            diagonalised in a cropped basis.
        warn_mixed (bool, optional): If True, warns when labels are assigned to
            states with weak weight in the labelled sector. Defaults to True.
        min_weight (float, optional): Minimum sector weight before warning that a
            label may be unreliable. Defaults to 0.9.

    Returns:
        ndarray: A two-dimensional array where each row corresponds to the desired
            quantum numbers for each eigenstate, and each column corresponds to a label.
    """

    eigstates = np.asarray(eigstates)
    if eigstates.ndim == 1:
        eigstates = eigstates[:, None]  # treat as one eigenvector (column)

    N_op, I1_op, I2_op = operators.generate_vecs(
        mol.Nmax, mol.Ii[0], mol.Ii[1], Nmin=mol.Nmin
    )
    full_dim = N_op.shape[-1]
    vec_dim = eigstates.shape[0]

    if basis_idx is None:
        if vec_dim == full_dim:
            basis_idx = None
        else:
            raise ValueError(
                f"eigstates are dimension {vec_dim}, but full basis is {full_dim}. "
                "Provide basis_idx so label_states can crop operators consistently."
            )

    if basis_idx is not None:
        basis_idx = np.asarray(basis_idx, dtype=int)
        if basis_idx.size != vec_dim:
            raise ValueError(
                (
                    f"basis_idx length {basis_idx.size} does not match"
                    f"eigstates dimension {vec_dim}."
                )
            )

    def crop_operator(operator):
        if basis_idx is None:
            return operator
        return operator[basis_idx, :][:, basis_idx]

    out_labels = []
    diagonal_label_names = []
    diagonal_component_label_columns = []
    diagonal_state_label_columns = []
    diagonal_output_label_columns = []

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
            Tz = crop_operator(Tz)
            double_M_labels = np.rint(
                2 * _operator_expectation(Tz, eigstates)
            ).real.astype(int)
            M_labels = _as_halfint_array(double_M_labels)
            out_labels.append(M_labels)

            if warn_mixed:
                diagonal_label_names.append(label)
                diagonal_component_label_columns.append(
                    np.rint(2 * np.diag(Tz).real).astype(int)
                )
                diagonal_state_label_columns.append(double_M_labels)
                diagonal_output_label_columns.append(M_labels)
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
            T2 = crop_operator(T2)
            T2eigval = _operator_expectation(T2, eigstates)
            double_T_labels = np.rint(
                2 * _solve_quadratic(1, 1, -1 * T2eigval)
            ).real.astype(int)
            T_labels = _as_halfint_array(double_T_labels)
            out_labels.append(T_labels)

            if warn_mixed:
                if _is_diagonal(T2):
                    diagonal_label_names.append(label)
                    diagonal_component_label_columns.append(
                        np.rint(
                            2 * _solve_quadratic(1, 1, -1 * np.diag(T2).real)
                        ).astype(int)
                    )
                    diagonal_state_label_columns.append(double_T_labels)
                    diagonal_output_label_columns.append(T_labels)
                else:
                    label_weights = _weights_for_operator_labels(
                        T2, eigstates, double_T_labels, angular=True
                    )
                    _warn_mixed_states(
                        [label], np.array([T_labels]).T, label_weights, min_weight
                    )

    if warn_mixed and diagonal_label_names:
        joint_weights = _joint_weights_for_diagonal_labels(
            diagonal_component_label_columns, diagonal_state_label_columns, eigstates
        )
        _warn_mixed_states(
            diagonal_label_names,
            np.array(diagonal_output_label_columns).T,
            joint_weights,
            min_weight,
        )

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
        eigenenergies (np.ndarray): Array of eigenenergies to be sorted along the
            last axis. Single-Hamiltonian and scan-shaped outputs are supported.
        eigenstates (np.ndarray): Array of eigenstates to be sorted along the last
            axis. Single-Hamiltonian and scan-shaped outputs are supported.

    Returns:
        tuple: The sorted eigenlabels, eigenenergies, and eigenstates.
    """
    indices = np.lexsort(eigenlabels[:, ::-1].T)
    return (
        eigenlabels[indices],
        np.take(eigenenergies, indices, axis=-1),
        np.take(eigenstates, indices, axis=-1),
    )


@log_time
def transition_electric_moments(
    mol, eigenstates, h=0, from_states=slice(None), to_states=slice(None)
):
    """
    Calculates electric dipole transition matrix elements between molecular
    eigenstates.

    These are the absolute values of <from|d_h|to>, where d_h is the spherical
    component of the molecule-fixed electric dipole operator transformed into the
    lab basis and expanded in the eigenstate basis. Multiplying by an electric
    field amplitude gives the corresponding dipole coupling energy scale.

    Args:
        mol: An object representing the molecule.
        eigenstates (ndarray): An array containing the eigenstates of the molecule,
            with shape (..., basis, states).
        h (int, optional): The helicity of the transition, +1 for sigma+, 0 for pi,
            -1 for sigma-.
        from_states (int, slice, or list, optional): The indices of the initial states
            from which transitions are considered. If an integer is passed, it is
            internally converted to a list. By default, all states are
            considered (slice(None)). These indices refer to the last axis of the
            supplied eigenstates array, so they are relative to any cropped
            eigenstate array.
        to_states (int, slice, or list, optional): The indices of the final states
            to which transitions are considered. If an integer is passed, it is
            internally converted to a list. By default, all states are
            considered (slice(None)).
            These indices also refer to the last axis of the supplied eigenstates
            array, so they are relative to any cropped eigenstate array.

    Returns:
        ndarray: Absolute transition electric moments with shape
            eigenstates.shape[:-2] + (n_from, n_to). The second-to-last axis
            enumerates the selected initial/from-state columns, in from_states order.
            The last axis enumerates the selected final/to-state columns, in
            to_states order. Output indices are positions inside those supplied
            from_states and to_states selections, not necessarily the original
            eigenstate indices.
    """

    state_count = eigenstates.shape[-1]
    all_state_indices = np.arange(state_count)
    if isinstance(from_states, (int, np.integer)):
        from_states = [int(from_states)]
    if isinstance(to_states, (int, np.integer)):
        to_states = [int(to_states)]

    from_state_indices = np.atleast_1d(all_state_indices[from_states])
    to_state_indices = np.atleast_1d(all_state_indices[to_states])

    dipole_op = mol.d0 * operators.expanded_unit_dipole_operator(mol, h)

    dipole_matrix_elements = (
        eigenstates[..., from_states].conj().swapaxes(-1, -2)
        @ dipole_op
        @ eigenstates[..., to_states]
    )

    if h != 0:
        # For circularly polarised light the electric field has a rotating
        # positive-frequency component and its complex conjugate. After the
        # rotating-wave approximation, opposite transition directions use
        # Hermitian-conjugate spherical dipole components. With the convention used
        # in the dipole operators, (d_h)^\dagger = -d_-h for h = +/-1, hence the
        # second operator below is -d_h^\dagger.
        #
        # This function uses the supplied eigenstate index order to decide which
        # side of the transition matrix diagonal an element lies on. Entries above
        # and below that diagonal are treated as opposite transition directions, so
        # they are filled with the two conjugate rotating components. If eigenstates
        # is cropped, the indices are relative to that cropped array.
        #
        # Because we return absolute values, swapping from_states and to_states
        # should give the transpose of the same coupling strengths.
        dipole_op_other = -dipole_op.conj().T

        opposite_dipole_matrix_elements = (
            eigenstates[..., from_states].conj().swapaxes(-1, -2)
            @ dipole_op_other
            @ eigenstates[..., to_states]
        )

        mask = from_state_indices[:, None] > to_state_indices[None, :]
        mask_other = from_state_indices[:, None] < to_state_indices[None, :]

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
        mol: The molecule used to generate the eigenstates
        eigstates (np.ndarray): Eigenstates from diagonalisation with shape
            (basis, states) or (steps, basis, states).

    Returns:
        ndarray: Magnetic moments with shape eigstates.shape[:-2] + (states,).
    """

    muz = -1 * operators.zeeman_ham(mol)

    mu = np.einsum(
        "...ae,ab,...be->...e",
        np.conjugate(eigstates),
        muz,
        eigstates,
        optimize="optimal",
    )
    return mu.real


def electric_moment(mol, eigstates):
    """
    Calculates the electric moments of each eigenstate.

    Args:
        mol: The molecule used to generate the eigenstates
        eigstates (np.ndarray): Eigenstates from diagonalisation with shape
            (basis, states) or (steps, basis, states).

    Returns:
        ndarray: Electric moments with shape eigstates.shape[:-2] + (states,).
    """

    dz = -1 * operators.dc_ham(mol)

    d = np.einsum(
        "...ae,ab,...be->...e",
        np.conjugate(eigstates),
        dz,
        eigstates,
        optimize="optimal",
    )
    return d.real
