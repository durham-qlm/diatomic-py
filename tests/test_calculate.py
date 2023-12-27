import numpy as np
from diatomic.systems import SingletSigmaMolecule
import diatomic.operators as operators

from diatomic.calculate import (
    _get_prior_repeat_indices,
    label_states,
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
        operators.num_proj_with_below(mol.Nmax)
        * operators.num_proj(mol.Ii[0])
        * operators.num_proj(mol.Ii[1])
    )

    states = np.zeros((basis_size, 3))

    N_state = np.zeros((operators.num_proj_with_below(mol.Nmax)))
    N_state[0] = 1  # N=0,MN=0

    i1_state = np.zeros((operators.num_proj(mol.Ii[0])))
    i1_state[1] = 1  # MI1=1/2

    i2_state = np.zeros((operators.num_proj(mol.Ii[1])))
    i2_state[1] = 1  # MI2 = 0

    state = np.kron(N_state, np.kron(i1_state, i2_state))

    states[:, 0] = state

    ##############

    N_state = np.zeros((operators.num_proj_with_below(mol.Nmax)))
    N_state[1] = 1  # N=1,MN=1

    i1_state = np.zeros((operators.num_proj(mol.Ii[0])))
    i1_state[0] = 1  # MI1=3/2

    i2_state = np.zeros((operators.num_proj(mol.Ii[1])))
    i2_state[2] = 1  # MI2 = -1

    state = np.kron(N_state, np.kron(i1_state, i2_state))

    states[:, 1] = state

    ################

    N_state = np.zeros((operators.num_proj_with_below(mol.Nmax)))
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
