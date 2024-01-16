import numbers
import numpy as np
import scipy.constants
from scipy.linalg import block_diag
from scipy.special import sph_harm
from sympy.physics.wigner import wigner_3j

from diatomic import log_time

"""
Basis function labels and order
"""


class HalfInt:
    """
    Container to encapsulate half-integer spin & projection lengths. Values will be
    automatically downcast to an integer if the numerator of <of>/2 is an even number
    as soon as possible, even upon initialisation.
    """

    def __new__(cls, *, of):
        if not isinstance(of, numbers.Integral):
            raise TypeError("The argument 'of' must be an integer.")
        elif of % 2 == 0:
            return of // 2
        else:
            return super().__new__(cls)

    def __init__(self, *, of):
        self._double = of

    def _fraction_str(self):
        return f"({self._double}/2)"

    def __repr__(self):
        return f"({self._fraction_str()} : {self.__class__.__name__})"

    def __str__(self):
        return self._fraction_str()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._double == other._double
        elif isinstance(other, int):
            return self._double == 2 * other
        elif isinstance(other, float):
            return self.__float__() == other
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._double < other._double
        elif isinstance(other, int):
            return self._double < 2 * other
        elif isinstance(other, float):
            return self.__float__() < other
        else:
            return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self._double > other._double
        elif isinstance(other, int):
            return self._double > 2 * other
        elif isinstance(other, float):
            return self.__float__() > other
        else:
            return NotImplemented

    def __ge__(self, other):
        return self > other or self == other

    def __float__(self):
        return self._double / 2

    def __int__(self):
        return int(self._double / 2)

    def __abs__(self):
        return self.__class__(of=abs(self._double))

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(of=self._double + other._double)
        elif isinstance(other, int):
            return self.__class__(of=self._double + 2 * other)
        elif isinstance(other, float):
            return self.__float__() + other
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(of=self._double - other._double)
        elif isinstance(other, int):
            return self.__class__(of=self._double - 2 * other)
        elif isinstance(other, float):
            return self.__float__() - other
        else:
            return NotImplemented

    def __rsub__(self, other):
        return -(self - other)

    def __neg__(self):
        return self.__class__(of=-self._double)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__float__() * float(other)
        elif isinstance(other, int):
            return self.__class__(of=self._double * other)
        elif isinstance(other, float):
            return self.__float__() * other
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other


def proj_iter(angmom):
    """
    Generator function that yields the projections of angular momentum quantum numbers.

    This function takes an angular momentum value (angmom) and generates a sequence of
    projection quantum numbers (m) from +angmom to -angmom in steps of 1 angular
    momentum unit, represented as HalfInt objects.

    Args:
        angmom (int | HalfInt | float): The angular momentum quantum number,
            which can be an integer, HalfInt, or float.

    Yields:
        HalfInt: A projection quantum number of the angular momentum starting from
            +angmom to -angmom in decrements of 1 (in terms of angular momentum units).
    """
    return (HalfInt(of=dm) for dm in range(int(2 * angmom), -(int(2 * angmom) + 1), -2))
    # return (HalfInt(double=dm) for dm in range(-2 * angmom, 2 * angmom + 1, 2))


def sph_iter(Nmax):
    """
    Generator function that yields tuples of spherical harmonic angular momentum
    quantum numbers up to a specified maximum angular momentum Nmax.

    Args:
        Nmax (int): The maximum (inclusive) angular momentum quantum number to
            iterate up to.

    Yields:
        tuple: A tuple of two elements where the first element is the angular momentum
            quantum number (N) and the second element is the projection quantum number
            (MN) as yielded by the proj_iter function for that N.
    """
    return ((N, MN) for N in range(0, Nmax + 1) for MN in proj_iter(N))


def uncoupled_basis_iter(Nmax, I1, I2):
    """
    Generator function that yields a Cartesian product of tuples representing the
    uncoupled basis states  for two angular momenta up to a specified maximum angular
    momentum.

    Args:
        Nmax (int): The maximum angular momentum quantum number for the orbital angular
            momentum.
        I1 (int | HalfInt | float): The angular momentum quantum number for the first
            intrinsic angular momentum of the nucleus.
        I2 (int | HalfInt | float): The angular momentum quantum number for the second
            intrinsic angular momentumof the nucleus.

    Yields:
        tuple: A tuple of four elements (N, MN, M1, M2), where N is an angular momentum
            quantum number from 0 to Nmax (inclusive), MN is the projection of N,
            and M1 and M2 are the projections of I1 and I2, respectively.
    """
    return (
        (N, MN, M1, M2)
        for (N, MN) in sph_iter(Nmax)
        for M1 in proj_iter(I1)
        for M2 in proj_iter(I2)
    )


def uncoupled_basis_pos(N, MN, MI1, MI2, I1, I2):
    """
    Calculates the position (index) of a given uncoupled basis state in a linear array
    that would be produced by uncoupled_basis_iter.

    Args:
        N (int | HalfInt | float): The angular momentum quantum number for the first
            component of the basis state.
        MN (int | HalfInt | float): The projection quantum number of N.
        MI1 (int | HalfInt | float): The projection quantum number of the first
            intrinsic angular momentum I1.
        MI2 (int | HalfInt | float): The projection quantum number of the second
            intrinsic angular momentum I2.
        I1 (int | HalfInt | float): The angular momentum quantum number for the
            first intrinsic angular momentum.
        I2 (int | HalfInt | float): The angular momentum quantum number for the
            second intrinsic angular momentum.

    Returns:
        int: The index position of the uncoupled basis state in a linear array.
    """
    return int(
        (I2 - MI2) + (2 * I2 + 1) * ((I1 - MI1) + (2 * I1 + 1) * (N**2 + N - MN))
    )


def num_proj_with_below(Nmax):
    """
    Calculates the total number of projection quantum numbers for all angular momentum
    quantum numbers from 0 up to a specified maximum value, Nmax.

    The function sums analyticalls the number of possible projections for each
    angular momentum quantum number, which follows the formula 2 * N + 1.
    i.e. `sum([2 * x + 1 for x in range(0, Nmax + 1)])`

    Args:
        Nmax (int): The maximum angular momentum quantum number to include in the sum.

    Returns:
        int: The cumulative number of projection quantum numbers for each angular
            momentum 0 to Nmax.
    """
    # sum([2 * x + 1 for x in range(0, Nmax + 1)]) =
    return int((Nmax + 1) ** 2)


def num_proj(angmom):
    """
    Computes the number of possible projection quantum numbers for a given angular
    momentum quantum number.

    Args:
        angmom (int | HalfInt | float): The angular momentum quantum number for which
            to calculate the number of possible projections.

    Returns:
        int: The number of possible projection quantum numbers for the specified
            angular momentum.
    """
    return int(2 * angmom + 1)


def vector_dot(A, B):
    """Cartesian dot product of two vectors of (matrix) operators A, B

    Args:
        A,B (np.ndarray): Cartesian length-3 vectors of operators (3xNxN)

    Returns:
        np.ndarray: result of the dot product, NxN array.
    """
    return np.einsum("cij,cjk->ik", A, B, optimize="optimal")


def tensor_dot(T1, T2):
    """Product of two rank-2 spherical tensors T1, T2

    A function to calculate the scalar product of two spherical tensors
    T1 and T2 are lists or np arrays that represent the spherical tensors
    lists are indexed from lowest m to highests

    Args:
        T1,T2 (list of np.ndarray) - length-5 list of np.ndarray

    Returns:
        X (np.ndarray) - scalar product of spherical tensors
    """
    x = np.zeros(T1[0].shape, dtype=np.complex128)
    for i, q in enumerate(range(-2, 2 + 1)):
        x += ((-1) ** q) * np.dot(T1[i], T2[-(i + 1)])
    return x


"""
Angular momentum operators
"""


def raising_operator(j):
    """Creates the angular momentum raising operator for j

    In the j,mj basis running from max(mj) to min (mj) creates a matrix that
    represents the operator j+\\|j,mj> = \\|j,mj+1>

    Args:
        j (float) : value of the angular momentum

    Returns:
        J+ (np.ndarray) : Array representing the operator J+,
            has shape ((2j+1),(2j+1))

    """
    matrix_width = num_proj(j)
    J = np.zeros((matrix_width, matrix_width), dtype=float)
    for x, MJ1 in enumerate(proj_iter(j)):
        for y, MJ2 in enumerate(proj_iter(j)):
            if MJ1 == MJ2 + 1:
                J[x, y] = np.sqrt((j - MJ2) * (j + MJ2 + 1))
    return J


# produce the three generalised projections of angular momentum:
# for S=1/2 these should return the Pauli matrices.
# for the source of these definitions see any good QM textbook e.g.
# Bransden & Joachain (or wikipedia)


def x_operator(J):
    """operator for X component of J

    Creates the Cartesian operator Jx for a given J

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jx (np.ndarray) : 2J+1 square np array
    """
    J_plus = raising_operator(J)
    J_minus = np.transpose(J_plus)
    return 0.5 * (J_plus + J_minus)


def y_operator(J):
    """operator for Y component of J

    Creates the Cartesian operator Jy for a given J

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jy (np.ndarray) : 2J+1 square np array
    """
    J_plus = raising_operator(J)
    J_minus = np.transpose(J_plus)
    return 0.5j * (J_minus - J_plus)


def z_operator(J):
    """operator for Z component of J

    Creates the Cartesian operator Jz for a given J. This is diagonal in the j,mj basis
    such that jz|j,mj> = mj|j,mj>

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jz (np.ndarray) : 2J+1 square np array
    """
    J_plus = raising_operator(J)
    J_minus = np.transpose(J_plus)
    return 0.5 * (J_plus @ J_minus - J_minus @ J_plus)


def generate_vecs(Nmax, I1, I2):
    """Build N, I1, I2 angular momentum vectors

    Generate the vectors of the angular momentum operators which we need
    to be able to produce the Hamiltonian

    Args:
        Nmax (float): maximum rotational level to include in calculations
        I1,I2 (float): Nuclear spins of nuclei 1 and 2
    Returns:
        N_vec,I1_vec,I2_vec (list of np.ndarray): length-3 list of
            (2Nmax+1)*(2I1+1)*(2I2+1) square np arrays
    """

    shapeN = num_proj_with_below(Nmax)
    shape1 = num_proj(I1)
    shape2 = num_proj(I2)

    Nx = np.array([[]])
    Ny = np.array([[]])
    Nz = np.array([[]])

    for n in range(0, Nmax + 1):
        Nx = block_diag(Nx, x_operator(n))
        Ny = block_diag(Ny, y_operator(n))
        Nz = block_diag(Nz, z_operator(n))

    # remove the first element of the N vectors, which are empty
    Nx = Nx[1:, :]
    Ny = Ny[1:, :]
    Nz = Nz[1:, :]

    # Each of the following corresponds to the product [N x 1Rb x 1Cs]
    # This gives the operators for N in the full hyperfine space.

    # np.kron is the function for the Kronecker product, often also called
    # the tensor product.

    N_vec = np.array(
        [
            np.kron(Nx, np.kron(np.identity(shape1), np.identity(shape2))),
            np.kron(Ny, np.kron(np.identity(shape1), np.identity(shape2))),
            np.kron(Nz, np.kron(np.identity(shape1), np.identity(shape2))),
        ]
    )

    # we also have to repeat for the nuclear spins
    I1_vec = np.array(
        [
            np.kron(
                np.identity(shapeN),
                np.kron(x_operator(I1), np.identity(shape2)),
            ),
            np.kron(
                np.identity(shapeN),
                np.kron(y_operator(I1), np.identity(shape2)),
            ),
            np.kron(
                np.identity(shapeN),
                np.kron(z_operator(I1), np.identity(shape2)),
            ),
        ]
    )

    I2_vec = np.array(
        [
            np.kron(
                np.identity(shapeN),
                np.kron(np.identity(shape1), x_operator(I2)),
            ),
            np.kron(
                np.identity(shapeN),
                np.kron(np.identity(shape1), y_operator(I2)),
            ),
            np.kron(
                np.identity(shapeN),
                np.kron(np.identity(shape1), z_operator(I2)),
            ),
        ]
    )

    return N_vec, I1_vec, I2_vec


def wigner_D(l, m, alpha, beta, gamma):  # noqa: E741
    """The Wigner D matrix with labels l and m.

    Calculates the Wigner D Matrix for the given Alpha,beta,gamma in radians.
    The wigner-D matrices represent rotations of angular momentum operators.
    The indices l and m determine the value of the matrix.
    The second index (m') is always zero.

    The input angles are the x-z-x euler angles

    Args:
        l (int) : order of wigner Matrix
        m (float): first index of Wigner Matrix
        alpha,beta,gamma (float) : x,z,x Euler angles in radians
    Returns:
        D (float) : Value of the wigner-D matrix
    """
    prefactor = np.sqrt((4 * np.pi) / (2 * l + 1))
    function = np.conj(sph_harm(m, l, alpha, beta))
    return prefactor * function


def T2_C(Nmax, I1, I2):
    """
    The irreducible spherical tensors for the spherical harmonics in the
    rotational basis.

    Args:
        Nmax (int) : Maximum rotational state to include
        I1,I2 (float) :  The nuclear spins of nucleus 1 and 2

    Returns:
        T (list of np.ndarray) : spherical tensor T^2(C). Each element is a
            spherical operator
    """
    matrix_width = num_proj_with_below(Nmax)

    T = np.zeros((5, matrix_width, matrix_width))

    for x, (N, MN) in enumerate(sph_iter(Nmax)):
        for y, (Np, MNp) in enumerate(sph_iter(Nmax)):
            for i, q in enumerate(range(-2, 2 + 1)):
                T[i][x, y] = (
                    ((-1) ** MN)
                    * np.sqrt((2 * N + 1) * (2 * Np + 1))
                    * wigner_3j(N, 2, Np, 0, 0, 0)
                    * wigner_3j(N, 2, Np, -MN, q, MNp)
                )

    # Expand into full Hyperfine basis
    nuclear_identity_width = num_proj(I1) * num_proj(I2)
    nuclear_identity = np.identity(nuclear_identity_width)
    expanded_matrix_width = matrix_width * nuclear_identity_width
    T_expanded = np.zeros((5, expanded_matrix_width, expanded_matrix_width))
    for i, q in enumerate(range(-2, 2 + 1)):
        T_expanded[i] = np.kron(T[i], nuclear_identity)

    return T_expanded


def makeT2(I1_vec, I2_vec):
    """Construct the spherical tensor T2 from two cartesian vectors of operators.

    Args:
        I1,I2 (list of np.ndarray): Length-3 list of cartesian angular momentum
            operators: the output of makevecs
    Returns:
        T (list of np.ndarray): T^2(I1,I2) length-5 list of spherical angular
            momentum operators
    """
    T2m2 = 0.5 * (
        np.dot(I1_vec[0], I2_vec[0])
        - 1.0j * np.dot(I1_vec[0], I2_vec[1])
        - 1.0j * np.dot(I1_vec[1], I2_vec[0])
        - np.dot(I1_vec[1], I2_vec[1])
    )
    T2p2 = 0.5 * (
        np.dot(I1_vec[0], I2_vec[0])
        + 1.0j * np.dot(I1_vec[0], I2_vec[1])
        + 1.0j * np.dot(I1_vec[1], I2_vec[0])
        - np.dot(I1_vec[1], I2_vec[1])
    )

    T2m1 = 0.5 * (
        np.dot(I1_vec[0], I2_vec[2])
        - 1.0j * np.dot(I1_vec[1], I2_vec[2])
        + np.dot(I1_vec[2], I2_vec[0])
        - 1.0j * np.dot(I1_vec[2], I2_vec[1])
    )
    T2p1 = -0.5 * (
        np.dot(I1_vec[0], I2_vec[2])
        + 1.0j * np.dot(I1_vec[1], I2_vec[2])
        + np.dot(I1_vec[2], I2_vec[0])
        + 1.0j * np.dot(I1_vec[2], I2_vec[1])
    )

    T20 = -np.sqrt(1 / 6) * (
        np.dot(I1_vec[0], I2_vec[0]) + np.dot(I1_vec[1], I2_vec[1])
    ) + np.sqrt(2 / 3) * np.dot(I1_vec[2], I2_vec[2])

    T = [T2m2, T2m1, T20, T2p1, T2p2]

    return T


"""
Molecular operators
"""


def unit_dipole_operator(Nmax, h):
    """
    Generates the induced dipole moment operator for a rigid rotor molecule in
    the spherical harmonic basis.

    This function constructs the dipole moment matrix by iterating over spherical
    harmonics for rotational states up to Nmax. It uses the Wigner 3j symbols to
    calculate the matrix elements, which represent transitions between rotational
    states induced by an external dipole field of specified helicity (h).

    Args:
        Nmax (int): The maximum rotational quantum number to include.
        h (float): The helicity of the dipole field.

    Returns:
        np.ndarray: The induced dipole moment matrix for transitions between
                    rotational states up to Nmax.
    """
    matrix_width = num_proj_with_below(Nmax)
    dmat = np.zeros((matrix_width, matrix_width))

    for i, (N1, M1) in enumerate(sph_iter(Nmax)):
        for j, (N2, M2) in enumerate(sph_iter(Nmax)):
            dmat[i, j] = (
                (-1) ** M1
                * np.sqrt((2 * N1 + 1) * (2 * N2 + 1))
                * wigner_3j(N1, 1, N2, -M1, h, M2)
                * wigner_3j(N1, 1, N2, 0, 0, 0)
            )

    return dmat


def expanded_unit_dipole_operator(mol, h):
    """
    Expands a dipole operator originally the spherical harmonic basis
    to cover states in the uncoupled hyperfine basis of the molecule.

    The dipole matrix is expanded to the basis including nuclear spin states by taking
    the Kronecker product with the identity matrix of the appropriate size.

    Args:
        mol: A molecular object with Nmax and nuclear spins (Ii) attributes.
        h (float): The helicity of the dipole field.

    Returns:
        np.ndarray: The expanded dipole operator matrix.
    """
    dmat = unit_dipole_operator(mol.Nmax, h)

    nuc_spin_identity = np.identity(num_proj(mol.Ii[0]) * num_proj(mol.Ii[1]))
    dmat_expanded = np.kron(dmat, nuc_spin_identity)

    return dmat_expanded


def electric_gradient(Nmax):
    """
    Calculates the electric field gradient at the nucleus for rotational states.

    Constructs a spherical tensor representing the electric field gradient at the
    nucleus, which depends on the rotational states but not the nuclear spin states.

    Args:
        Nmax (int): The maximum rotational state to include.

    Returns:
        list of np.ndarray: A length-5 list of arrays representing the electric field
                            gradient tensor components.
    """
    matrix_width = num_proj_with_below(Nmax)
    T = np.zeros((5, matrix_width, matrix_width))

    for i, (N1, M1) in enumerate(sph_iter(Nmax)):
        for j, (N2, M2) in enumerate(sph_iter(Nmax)):
            for n, q in enumerate(range(-2, 2 + 1)):
                T[n][i, j] = (
                    (-1) ** M1
                    * np.sqrt((2 * N1 + 1) * (2 * N2 + 1))
                    * wigner_3j(N1, 2, N2, -M1, q, M2)
                    * wigner_3j(N1, 2, N2, 0, 0, 0)
                )

    return T


def expanded_electric_gradient(mol):
    """Expands the electric gradient tensor to include nuclear spin states.

    Uses the Kronecker product to expand the electric gradient tensor calculated for
    rotational states into a basis that also includes nuclear spin states.

    Args:
        mol: A molecular object with Nmax and nuclear spins (Ii) attributes.

    Returns:
        np.ndarray: The expanded electric gradient tensor.
    """
    elec_mat = electric_gradient(mol.Nmax)

    nuc_spin_basis_size = num_proj(mol.Ii[0]) * num_proj(mol.Ii[1])
    nuc_spin_identity = np.identity(nuc_spin_basis_size)

    expanded_matrix_width = elec_mat[0].shape[0] * nuc_spin_basis_size
    T_expanded = np.empty((5, expanded_matrix_width, expanded_matrix_width))

    for i in range(5):
        T_expanded[i] = np.kron(elec_mat[i], nuc_spin_identity)

    return T_expanded


def quad_moment(I_nuc):
    """Calculates the nuclear electric quadrupole moments of nuclear spin.

    Constructs a spherical tensor representing the nuclear quadrupole moment,
    which depends on the nuclear spin states, not the rotational states.

    Args:
        I_nuc (float): The nuclear spin quantum number.

    Returns:
        list of np.ndarray: A length-5 list of arrays representing the nuclear
            quadrupole moment tensor components.
    """
    matrix_width = num_proj(I_nuc)
    T = np.zeros((5, matrix_width, matrix_width))

    for i, M1 in enumerate(proj_iter(I_nuc)):
        for j, M2 in enumerate(proj_iter(I_nuc)):
            for n, q in enumerate(range(-2, 2 + 1)):
                T[n][i, j] = (
                    (-1) ** int(I_nuc - M1)
                    * wigner_3j(I_nuc, 2, I_nuc, -M1, q, M2)
                    / wigner_3j(I_nuc, 2, I_nuc, -I_nuc, 0, I_nuc)
                )

    return T


def expanded_quad_moment(mol, nucleus):
    """Expands the quadrupole moment tensor into the full hyperfine basis.

    Args:
        mol: A molecular object with Nmax and nuclear spins (Ii) attributes.
        nucleus (int): Index of the nucleus (0 or 1) for which to calculate the
                       quadrupole moment tensor.

    Returns:
        np.ndarray: The expanded nuclear quadrupole moment tensor.
    """
    T_nucleus = quad_moment(mol.Ii[nucleus])

    # Expand into full hyperfine basis
    num_N_proj = num_proj_with_below(mol.Nmax)
    nuc_spin_basis_size = num_proj(mol.Ii[0]) * num_proj(mol.Ii[1])
    expanded_matrix_width = num_N_proj * nuc_spin_basis_size

    IdentityN = np.identity(num_N_proj)

    T_exp = np.empty((5, expanded_matrix_width, expanded_matrix_width))

    if nucleus == 0:
        Identity1 = np.identity(num_proj(mol.Ii[1]))
        for i, q in enumerate(range(-2, 2 + 1)):
            T_exp[i] = np.kron(IdentityN, np.kron(T_nucleus[i], Identity1))
    else:
        Identity0 = np.identity(num_proj(mol.Ii[0]))
        for i, q in enumerate(range(-2, 2 + 1)):
            T_exp[i] = np.kron(IdentityN, np.kron(Identity0, T_nucleus[i]))

    return T_exp


def quadrupole(mol):
    """Calculates the nuclear electric quadrupole interaction energy.

    Computes the quadrupole interaction terms for the full hyperfine Hamiltonian using
    spherical tensor algebra. Requires the nuclear quadrupole moments and
    electric field gradients.

    Args:
        mol: A molecular object with necessary attributes like nuclear quadrupole
            moments (Qi), nuclear spins (Ii), and maximum rotational quantum
            number (Nmax).

    Returns:
        np.ndarray: The quadrupole interaction term of the hyperfine Hamiltonian.
    """

    TdE = expanded_electric_gradient(mol)
    Tq0 = expanded_quad_moment(mol, 0)
    Tq1 = expanded_quad_moment(mol, 1)

    if mol.Ii[0] < 1:
        Hq0 = 0
    else:
        Hq0 = mol.Qi[0] * tensor_dot(Tq0, TdE)

    if mol.Ii[1] < 1:
        Hq1 = 0
    else:
        Hq1 = mol.Qi[1] * tensor_dot(Tq1, TdE)

    return (Hq0 + Hq1) / 4


def rotational(N, Brot, Drot):
    """Calculates the hyperfine-free rotational Hamiltonian for a rigid rotor molecule.

    Includes both the rotational constant (Brot) and the centrifugal distortion (Drot).

    Matrix is returned in the N,MN basis with MN going from maximum to minimum.

    Args:
        N (list of np.ndarray): length 3 list representing the Angular momentum
            vector for rotation.
        Brot (float): Rotational constant in joules.
        Drot (float): Centrifugal distortion constant in joules.

    Returns:
        np.ndarray: The rotational Hamiltonian in the N, MN basis.
    """

    N_squared = vector_dot(N, N)
    return Brot * N_squared - Drot * N_squared * N_squared


def zeeman(Cz, J):
    """Calculates the Zeeman effect Hamiltonian for a magnetic field along the z-axis.

    Linear Zeeman shift, fixed magnetic field along z so only need the
    last component of the angular momentum vector.

    Args:
        Cz (float): Zeeman coefficient/magnetic moment.
        J (list of np.ndarray): Angular momentum operator vector.

    Returns:
        np.ndarray: The Zeeman Hamiltonian.
    """
    Hzeeman = -Cz * J[2]
    return Hzeeman


def scalar_nuclear(Ci, J1, J2):
    """Calculate the scalar spin-spin interaction term

    Returns the scalar spin-spin term of the HF Hamiltonian

    Args:
        Ci(float) - Scalar spin coupling coefficient
        J1,J2 (list of np.ndarray) - Angular momentum vectors

    Returns:
        H (np.ndarray) - Hamiltonian for spin-spin interaction
    """
    return Ci * vector_dot(J1, J2)


def tensor_nuclear(C3, I1_vec, I2_vec, I1_val, I2_val, Nmax):
    """Calculate the tensor spin-spin interaction.

    This function is to calculate the tensor spin-spin interaction.
    This version uses spherical tensors to calculate the correct off-diagonal
    behaviour.

    Args:
        C3 (float) - spin-spin coupling constant
        I1,I2 (float) - Cartesian Angular momentum operator Vectors
        Nmax (int) - maximum rotational state to include

    Returns:
        Hss (np.ndarray) - Hamiltonian for tensor spin-spin interaction
    """

    # steps for maths, creates the spherical tensors
    T1 = T2_C(Nmax, I1_val, I2_val)
    T2 = makeT2(I1_vec, I2_vec)
    # return final Hamiltonian
    tensorss = np.sqrt(6) * C3 * tensor_dot(T1, T2)

    return tensorss


def unit_ac_iso(Nmax, I1, I2):
    """Calculate isotropic Stark shifts

    Generates the effect of the isotropic AC Stark shift for a rigid-rotor
    like molecule.

    This term is calculated differently to all of the others in this work
    and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
    N,MN,N',MN' to build a matrix without hyperfine structure then uses
    kronecker products to expand it into all of the hyperfine states.

    Args:
        Nmax (int) - maximum rotational quantum number to calculate (int)
        I1,I2 (float) - Nuclear spin of nucleus 1,2

    Returns:
        H (np.ndarray) - isotropic AC Stark Hamiltonian

    """
    matrix_width = num_proj_with_below(Nmax) * num_proj(I1) * num_proj(I2)
    HAC = -1 * np.identity(matrix_width)

    # return the matrix, in the full uncoupled basis.
    return (1.0 / (2 * scipy.constants.epsilon_0 * scipy.constants.c)) * HAC


def unit_ac_aniso(Nmax, I1, I2, Beta):
    """Calculate anisotropic ac stark shift.

    Generates the effect of the anisotropic AC Stark shift for a rigid-rotor
    like molecule.

    This term is calculated differently to all of the others in this work
    and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
    N,MN,N',MN' to build a matrix without hyperfine structure then uses
    kronecker products to expand it into all of the hyperfine states.

    Args:

        Nmax (int) - maximum rotational quantum number to calculate
        I1,I2 (float) - Nuclear spin of nucleus 1,2
        Beta (float) - polarisation angle of the laser in Radians

    Returns:
        H (np.ndarray): Hamiltonian in joules
    """

    matrix_width = num_proj_with_below(Nmax)
    HAC = np.zeros((matrix_width, matrix_width), dtype=complex)
    for i, (N1, M1) in enumerate(sph_iter(Nmax)):
        for j, (N2, M2) in enumerate(sph_iter(Nmax)):
            M = M2 - M1
            HAC[i, j] = -1 * (
                wigner_D(2, M, 0, Beta, 0)
                * (-1) ** M2
                * np.sqrt((2 * N1 + 1) * (2 * N2 + 1))
                * wigner_3j(N2, 2, N1, -M2, M, M1)
                * wigner_3j(N2, 2, N1, 0, 0, 0)
            )

    HAC[np.isnan(HAC)] = 0

    HAC_expanded = np.kron(HAC, np.identity(num_proj(I1) * num_proj(I2)))
    # return the matrix, in the full uncoupled basis.
    return (1.0 / (2 * scipy.constants.epsilon_0 * scipy.constants.c)) * HAC_expanded


# Now some functions to take these functions and assemble them into the physical
# Hamiltonians where necessary.


@log_time
def hyperfine_ham(mol):
    """Calculate the field-free Hyperfine hamiltonian

    Wrapper to call all of the functions that are appropriate for the singlet-sigma
    hyperfine hamiltonian.

    Args:
        Nmax (int) - Maximum rotational level to include
        I1_mag,I2_mag (float) - magnitude of the nuclear spins
        Consts (Dictionary): Dict of molecular constants
    Returns:
        H0 : Hamiltonian for the hyperfine structure in joules
    """
    N, I1, I2 = generate_vecs(mol.Nmax, mol.Ii[0], mol.Ii[1])
    rotational_part = rotational(N, mol.Brot, mol.Drot)
    scalar_part = (
        scalar_nuclear(mol.Ci[0], N, I1)
        + scalar_nuclear(mol.Ci[1], N, I2)
        + scalar_nuclear(mol.C4, I1, I2)
    )
    tensor_nuclear_part = tensor_nuclear(mol.C3, I1, I2, mol.Ii[0], mol.Ii[1], mol.Nmax)
    quadrupole_part = quadrupole(mol)
    H = rotational_part + scalar_part + tensor_nuclear_part + quadrupole_part
    return H


@log_time
def dc_ham(mol):
    """calculate HDC for a diatomic molecule

    Generates the effect of the dc Stark shift for a rigid-rotor like
    molecule.

    This term is calculated differently to all of the others in this work
    and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
    N,MN,N',MN' to build a matrix without hyperfine structure then uses
    kronecker products to expand it into all of the hyperfine states.


    Args:
        Nmax(int) -  maximum rotational quantum number to calculate
        d0 (float) - Permanent electric dipole momentum
        I1,I2 (float) - Nuclear spin of nucleus 1,2


    Returns:
        H (np.ndarray) - DC Stark Hamiltonian in joules
    """

    return -1 * mol.d0 * expanded_unit_dipole_operator(mol, 0)


@log_time
def zeeman_ham(mol):
    """Assembles the Zeeman term and generates operator vectors

    Calculates the Zeeman effect for a magnetic field on a singlet-sigma molecule.
    There is no electronic term and the magnetic field is fixed to be along the z axis.

    Args:
        Nmax (int) - Maximum rotational level to include
        I1_mag,I2_mag (float) - magnitude of the nuclear spins
        Consts (Dictionary): Dict of molecular constants

    Returns:
        Hz (np.ndarray): Hamiltonian for the zeeman effect
    """
    N, I1, I2 = generate_vecs(mol.Nmax, mol.Ii[0], mol.Ii[1])
    H = zeeman(mol.Mui[0], I1) + zeeman(mol.Mui[1], I2) + zeeman(mol.MuN, N)
    return H


@log_time
def ac_ham(mol, a02, beta=0):
    """
    Computes the AC Stark shift Hamiltonian for a molecule in an oscillating electric
    field.

    The function combines the isotropic and anisotropic AC Stark shifts to obtain the
    total AC Stark Hamiltonian, considering the polarization of the electric field.

    Args:
        mol: A molecule object containing the necessary attributes for calculation.
        a02 (tuple): A two-element tuple containing the isotropic and anisotropic
            polarizabilities.
        beta (float, optional): The polarization angle of the electric field.
            Defaults to 0.

    Returns:
        np.ndarray: The AC Stark shift Hamiltonian matrix.
    """
    Hac = a02[0] * unit_ac_iso(mol.Nmax, mol.Ii[0], mol.Ii[1]) + a02[1] * unit_ac_aniso(
        mol.Nmax, mol.Ii[0], mol.Ii[1], beta
    )
    return Hac
