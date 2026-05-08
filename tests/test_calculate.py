import numpy as np
import pytest
from diatomic.systems import SingletSigmaMolecule
import diatomic.operators as operators

from diatomic.calculate import (
    _get_prior_repeat_indices,
    electric_moment,
    label_states,
    magnetic_moment,
    solve_system,
    sort_smooth,
    sort_by_labels,
    transition_electric_moments,
    _matrix_prod_diagonal,
    _solve_quadratic,
)


def test_solve_quadratic():
    a, b, c = 1, -3, 2  # = (x-1)(x-2)
    assert _solve_quadratic(a, b, c) == 2.0

    # = (x-[1,2,3])(x-2)
    a, b, c = 1, np.array([-3, -4, -5]), np.array([2, 4, 6])
    assert (_solve_quadratic(a, b, c) == np.array([2.0, 2.0, 3.0])).all()


def test_get_prior_repeat_indices():
    test_labels = np.array(
        [[1, 1], [2, 2], [3, 3], [3, 3], [3, 2], [2, 2], [5, 5], [3, 3], [4, 4]]
    )
    expected_output = np.array([0, 0, 0, 1, 0, 1, 0, 2, 0])

    assert (_get_prior_repeat_indices(test_labels) == expected_output).all()


def test_label_states():
    mol = SingletSigmaMolecule.from_preset("Testing")
    mol.Nmax = 2
    mol.Ii = (operators.HalfInt(of=3), 1)

    basis_size = (
        operators.num_proj_with_below(mol.Nmax, Nmin=mol.Nmin)
        * operators.num_proj(mol.Ii[0])
        * operators.num_proj(mol.Ii[1])
    )

    states = np.zeros((basis_size, 3))

    N_state = np.zeros((operators.num_proj_with_below(mol.Nmax, Nmin=mol.Nmin)))
    N_state[0] = 1  # N=0,MN=0

    i1_state = np.zeros((operators.num_proj(mol.Ii[0])))
    i1_state[1] = 1  # MI1=1/2

    i2_state = np.zeros((operators.num_proj(mol.Ii[1])))
    i2_state[1] = 1  # MI2 = 0

    state = np.kron(N_state, np.kron(i1_state, i2_state))

    states[:, 0] = state

    ##############

    N_state = np.zeros((operators.num_proj_with_below(mol.Nmax, Nmin=mol.Nmin)))
    N_state[1] = 1  # N=1,MN=1

    i1_state = np.zeros((operators.num_proj(mol.Ii[0])))
    i1_state[0] = 1  # MI1=3/2

    i2_state = np.zeros((operators.num_proj(mol.Ii[1])))
    i2_state[2] = 1  # MI2 = -1

    state = np.kron(N_state, np.kron(i1_state, i2_state))

    states[:, 1] = state

    ################

    N_state = np.zeros((operators.num_proj_with_below(mol.Nmax, Nmin=mol.Nmin)))
    N_state[0] = 1  # N=0,MN=0

    i1_state = np.zeros((operators.num_proj(mol.Ii[0])))
    i1_state[1] = 1  # MI1=1/2

    i2_state = np.zeros((operators.num_proj(mol.Ii[1])))
    i2_state[1] = 1  # MI2 = 0

    state = np.kron(N_state, np.kron(i1_state, i2_state))

    states[:, 2] = state

    labels = label_states(mol, states, ["N", "MN", "MI1", "MI2"])

    expected = np.array(
        [
            [0, 0, operators.HalfInt(of=1), 0],
            [1, 1, operators.HalfInt(of=3), -1],
            [0, 0, operators.HalfInt(of=1), 0],
        ]
    )

    assert (labels == expected).all()

    expected_count = np.array(
        [
            [0, 0, operators.HalfInt(of=1), 0, 0],
            [1, 1, operators.HalfInt(of=3), -1, 0],
            [0, 0, operators.HalfInt(of=1), 0, 1],
        ]
    )

    labels_count = label_states(
        mol, states, ["N", "MN", "MI1", "MI2"], index_repeats=True
    )
    assert (labels_count == expected_count).all()


def test_label_states_warns_when_projection_label_is_mixed():
    mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
    mol.Nmax = 0

    basis_size = (
        operators.num_proj_with_below(mol.Nmax, Nmin=mol.Nmin)
        * operators.num_proj(mol.Ii[0])
        * operators.num_proj(mol.Ii[1])
    )
    states = np.zeros((basis_size, 1))

    mf_zero_idx = operators.uncoupled_basis_pos(
        0,
        0,
        operators.HalfInt(of=3),
        operators.HalfInt(of=-3),
        mol.Ii[0],
        mol.Ii[1],
        Nmin=mol.Nmin,
    )
    mf_one_idx = operators.uncoupled_basis_pos(
        0,
        0,
        operators.HalfInt(of=3),
        operators.HalfInt(of=-1),
        mol.Ii[0],
        mol.Ii[1],
        Nmin=mol.Nmin,
    )

    states[mf_zero_idx, 0] = np.sqrt(0.5)
    states[mf_one_idx, 0] = np.sqrt(0.5)

    with pytest.warns(UserWarning, match="less than 90%"):
        labels = label_states(mol, states, ["MF"])

    assert labels[0, 0] == operators.HalfInt(of=1)


def test_label_states_warns_when_angular_momentum_label_is_mixed():
    mol = SingletSigmaMolecule.from_preset("RigidRotor")
    mol.Nmax = 1

    basis_size = operators.num_proj_with_below(mol.Nmax, Nmin=mol.Nmin)
    states = np.zeros((basis_size, 1))

    n_zero_idx = operators.uncoupled_basis_pos(
        0, 0, 0, 0, mol.Ii[0], mol.Ii[1], Nmin=mol.Nmin
    )
    n_one_idx = operators.uncoupled_basis_pos(
        1, 0, 0, 0, mol.Ii[0], mol.Ii[1], Nmin=mol.Nmin
    )

    states[n_zero_idx, 0] = np.sqrt(0.5)
    states[n_one_idx, 0] = np.sqrt(0.5)

    with pytest.warns(UserWarning, match="less than 90%"):
        labels = label_states(mol, states, ["N"])

    assert labels[0, 0] == operators.HalfInt(of=1)


def test_sort_smooth_keeps_one_to_one_assignments_for_tied_overlaps():
    eigvals = np.array([[0.0, 1.0], [2.0, 3.0]])
    eigvecs = np.empty((2, 2, 2))
    eigvecs[0] = np.eye(2)
    eigvecs[1] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    _, sorted_eigvecs = sort_smooth(eigvals, eigvecs)

    np.testing.assert_allclose(sorted_eigvecs[0].T @ sorted_eigvecs[0], np.eye(2))


def test_solve_system_chunked_scan_matches_batched_scan():
    hamiltonians = np.array(
        [
            [[0.0, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 2.0]],
            [[0.2, 0.1, 0.0], [0.1, 1.1, 0.2], [0.0, 0.2, 2.1]],
            [[0.4, 0.1, 0.0], [0.1, 1.2, 0.2], [0.0, 0.2, 2.2]],
            [[0.6, 0.1, 0.0], [0.1, 1.3, 0.2], [0.0, 0.2, 2.3]],
        ]
    )

    batched_energies, batched_states = solve_system(hamiltonians)
    chunked_energies, chunked_states = solve_system(hamiltonians, chunk_size=2)

    np.testing.assert_allclose(chunked_energies, batched_energies)
    state_overlaps = np.abs(
        np.einsum("sbe,sbe->se", np.conj(batched_states), chunked_states)
    )
    np.testing.assert_allclose(state_overlaps, np.ones_like(state_overlaps))


def test_solve_system_rejects_invalid_chunk_size():
    hamiltonians = np.stack([np.eye(2), 2 * np.eye(2)])

    with pytest.raises(ValueError, match="chunk_size"):
        solve_system(hamiltonians, chunk_size=0)


def test_sort_by_labels_accepts_single_and_scan_outputs():
    labels = np.array([[1, 0], [0, 0], [1, -1]])
    eigenenergies = np.array([10.0, 20.0, 30.0])
    eigenstates = np.eye(3)

    sorted_labels, sorted_energies, sorted_states = sort_by_labels(
        labels, eigenenergies, eigenstates
    )

    np.testing.assert_array_equal(sorted_labels, np.array([[0, 0], [1, -1], [1, 0]]))
    np.testing.assert_array_equal(sorted_energies, np.array([20.0, 30.0, 10.0]))
    np.testing.assert_array_equal(sorted_states, np.eye(3)[:, [1, 2, 0]])

    scan_energies = np.vstack([eigenenergies, eigenenergies + 1])
    scan_states = np.stack([eigenstates, 2 * eigenstates])

    _, sorted_scan_energies, sorted_scan_states = sort_by_labels(
        labels, scan_energies, scan_states
    )

    np.testing.assert_array_equal(sorted_scan_energies, scan_energies[:, [1, 2, 0]])
    np.testing.assert_array_equal(sorted_scan_states, scan_states[:, :, [1, 2, 0]])


def test_moments_accept_single_and_scan_outputs():
    mol = SingletSigmaMolecule.from_preset("RigidRotor")
    mol.Nmax = 1

    eigenstates = np.eye(4)
    magnetic_single = magnetic_moment(mol, eigenstates)
    electric_single = electric_moment(mol, eigenstates)

    scan_states = np.stack([eigenstates, eigenstates])
    magnetic_scan = magnetic_moment(mol, scan_states)
    electric_scan = electric_moment(mol, scan_states)

    np.testing.assert_allclose(magnetic_scan[0], magnetic_single)
    np.testing.assert_allclose(magnetic_scan[1], magnetic_single)
    np.testing.assert_allclose(electric_scan[0], electric_single)
    np.testing.assert_allclose(electric_scan[1], electric_single)


def test_transition_electric_moments():
    mol = SingletSigmaMolecule.from_preset("Testing")
    mol.Nmax = 1
    mol.Ii = (0, 0)

    d0_matrix = transition_electric_moments(mol, np.identity(4), h=0)
    expected = np.sqrt(1 / 3) * np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    assert np.allclose(d0_matrix, expected)

    d0_from_0 = transition_electric_moments(mol, np.identity(4), h=0, from_states=[0])
    expected = np.array([0, 0, np.sqrt(1 / 3), 0])
    assert np.allclose(d0_from_0, expected)

    d0_from_0 = transition_electric_moments(mol, np.identity(4), h=0, from_states=0)
    expected = np.array([0, 0, np.sqrt(1 / 3), 0])
    assert np.allclose(d0_from_0, expected)

    d0_from_1 = transition_electric_moments(mol, np.identity(4), h=0, from_states=1)
    expected = np.array([0, 0, 0, 0])
    assert np.allclose(d0_from_1, expected)

    d0_from_01 = transition_electric_moments(
        mol, np.identity(4), h=0, from_states=[0, 2]
    )
    expected = np.sqrt(1 / 3) * np.array([[0, 0, 1, 0], [1, 0, 0, 0]])
    assert np.allclose(d0_from_01, expected)

    d0_from_011 = transition_electric_moments(mol, np.identity(4), h=1)
    expected = np.sqrt(1 / 3) * np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    assert np.allclose(d0_from_011, expected)

    d0_from_01m1 = transition_electric_moments(mol, np.identity(4), h=-1, from_states=0)
    expected = np.sqrt(1 / 3) * np.array([0, 0, 0, 1])
    assert np.allclose(d0_from_01m1, expected)

    d0_from_011 = transition_electric_moments(mol, np.identity(4), h=1, from_states=0)
    expected = np.sqrt(1 / 3) * np.array([0, 1, 0, 0])
    assert np.allclose(d0_from_011, expected)

    d0_from_011 = transition_electric_moments(
        mol, np.identity(4), h=1, from_states=[0, 1]
    )
    expected = np.sqrt(1 / 3) * np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    assert np.allclose(d0_from_011, expected)

    d0_from_011 = transition_electric_moments(
        mol, np.eye(4, 2), h=1, from_states=[0, 1]
    )
    expected = np.sqrt(1 / 3) * np.array([[0, 1], [1, 0]])
    assert np.allclose(d0_from_011, expected)

    eigvecs = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    d0_from_011 = transition_electric_moments(mol, eigvecs, h=-1, from_states=[0, 1])
    expected = np.sqrt(1 / 3) * np.array([[0, 1], [1, 0]])
    assert np.allclose(d0_from_011, expected)

    d0_from_011 = transition_electric_moments(mol, eigvecs, h=1, from_states=[0, 1])
    expected = np.sqrt(1 / 3) * np.array([[0, 0], [0, 0]])
    assert np.allclose(d0_from_011, expected)


def test_transition_electric_moments_from_states_matches_full_rows():
    mol = SingletSigmaMolecule.from_preset("Testing")
    mol.Nmax = 1
    mol.Ii = (0, 0)

    states = np.identity(4)

    for h in [-1, 1]:
        full = transition_electric_moments(mol, states, h=h)
        for from_state in range(states.shape[-1]):
            subset = transition_electric_moments(
                mol, states, h=h, from_states=from_state
            )
            np.testing.assert_allclose(subset, full[from_state : from_state + 1])


def test_transition_electric_moments_to_states_matches_full_submatrix():
    mol = SingletSigmaMolecule.from_preset("Testing")
    mol.Nmax = 1
    mol.Ii = (0, 0)

    states = np.identity(4)
    from_states = [0, 3]
    to_states = [1, 2]

    for h in [-1, 0, 1]:
        full = transition_electric_moments(mol, states, h=h)
        subset = transition_electric_moments(
            mol, states, h=h, from_states=from_states, to_states=to_states
        )

        np.testing.assert_allclose(subset, full[np.ix_(from_states, to_states)])


def test_transition_electric_moments_from_to_symmetry():
    mol = SingletSigmaMolecule.from_preset("Testing")
    mol.Nmax = 1
    mol.Ii = (0, 0)

    states = np.identity(4)
    from_states = [0, 1]
    to_states = [1, 0]

    for h in [-1, 0, 1]:
        forward = transition_electric_moments(
            mol, states, h=h, from_states=from_states, to_states=to_states
        )
        reverse = transition_electric_moments(
            mol, states, h=h, from_states=to_states, to_states=from_states
        )

        np.testing.assert_allclose(forward, reverse.T)


def test_diagonal_2d():
    np.random.seed(0)  # Setting a seed for reproducibility
    A = np.random.rand(5, 5)
    B = np.random.rand(5, 5)

    for d in range(-2, 2):
        expected_diag = np.diag(A @ B, k=d)
        result_diag = _matrix_prod_diagonal(A, B, d)
        np.testing.assert_array_almost_equal(result_diag, expected_diag)


def test_diagonal_multidimensional():
    np.random.seed(0)
    A = np.random.rand(3, 6, 6)
    B = np.random.rand(3, 6, 6)

    for d in range(-2, 2):
        expected_diag = np.array([np.diag(a @ b, k=d) for a, b in zip(A, B)])
        result_diag = _matrix_prod_diagonal(A, B, d)
        np.testing.assert_array_almost_equal(result_diag, expected_diag)


def test_diagonal_multidimensional_not_square():
    np.random.seed(0)
    A = np.random.rand(3, 6, 4)
    B = np.random.rand(3, 4, 6)

    for d in range(-2, 2):
        expected_diag = np.array([np.diag(a @ b, k=d) for a, b in zip(A, B)])
        result_diag = _matrix_prod_diagonal(A, B, d)
        np.testing.assert_array_almost_equal(result_diag, expected_diag)
