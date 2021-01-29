import numpy
from sympy.physics.wigner import wigner_3j,wigner_9j
from scipy.linalg import block_diag,eig,eigvals
import scipy.constants
from scipy.special import sph_harm
import warnings

'''
This module contains the main code to calculate the hyperfine structure of
singlet -sigma molecules. In usual circumstances most of the functions within
are not user-oriented.
'''


###############################################################################
# Start by definining a bunch of constants that are needed for the code       #
###############################################################################

'''
    Important note!

    All units in this code are SI i.e. elements in the Hamiltonian have units
    of Joules. Outputs will be on the order of 1e-30

'''

h = scipy.constants.h
muN = scipy.constants.physical_constants['nuclear magneton'][0]
bohr = scipy.constants.physical_constants['Bohr radius'][0]
eps0 = scipy.constants.epsilon_0
c = scipy.constants.c


pi = numpy.pi

DebyeSI = 3.33564e-30

###############################################################################
# Bialkali Molecular Constants                                                    #
###############################################################################

#Constants are from
# https://doi.org/10.1103/PhysRevA.96.042506
#https://doi.org/10.1103/PhysRevA.78.033434
# and https://arxiv.org/pdf/1707.02168.pdf

# RbCs Constants are from https://doi.org/10.1103/PhysRevA.94.041403
# Polarisabilities are for 1064 nm

RbCs = {    "I1":1.5,
            "I2":3.5,
            "d0":1.225*DebyeSI,
            "binding":114268135.25e6*h,
            "Brot":490.173994326310e6*h,
            "Drot":207.3*h,
            "Q1":-809.29e3*h,
            "Q2":59.98e3*h,
            "C1":98.4*h,
            "C2":194.2*h,
            "C3":192.4*h,
            "C4":19.0189557e3*h,
            "MuN":0.0062*muN,
            "Mu1":1.8295*muN,
            "Mu2":0.7331*muN,
            "a0":2020*4*pi*eps0*bohr**3,
            "a2":1997*4*pi*eps0*bohr**3,
            "Beta":0}

K41Cs = {   "I1":1.5,
            "I2":3.5,
            "d0":1.84*DebyeSI,
            "Brot":880.326e6*h,
            "Drot":0*h,
            "Q1":-0.221e6*h,
            "Q2":0.075e6*h,
            "C1":4.5*h,
            "C2":370.8*h,
            "C3":9.9*h,
            "C4":628*h,
            "MuN":0.0*muN,
            "Mu1":0.143*(1-1340.7e-6)*muN,
            "Mu2":0.738*(1-6337.1e-6)*muN,
            "a0":7.783e6*h, #h*Hz/(W/cm^2)
            "a2":0,
            "Beta":0}

K40Rb = {   "I1":4,
            "I2":1.5,
            "d0":0.62*DebyeSI,
            "Brot":1113.4e6*h,
            "Drot":0*h,
            "Q1":0.311e6*h,
            "Q2":-1.483e6*h,
            "C1":-24.1*h,
            "C2":419.5*h,
            "C3":-48.2*h,
            "C4":-2028.8*h,
            "MuN":0.0140*muN,
            "Mu1":-0.324*(1-1321e-6)*muN,
            "Mu2":1.834*(1-3469e-6)*muN,
            "a0":5.33e-5*1e6*h, #h*Hz/(W/cm^2)
            "a2":6.67e-5*1e6*h,
            "Beta":0}
###############################################################################
# Functions for the calculations to use                                       #
###############################################################################

#first functions are mathematical and used to generate the structures that we
#will need to use

def Raising_operator(j):
    ''' Creates the angular momentum raising operator for j

    In the j,mj basis running from max(mj) to min (mj) creates a matrix that represents the operator j+|j,mj> = |j,mj+1>

    Inputs:
        j (float) : value of the angular momentum
    Outputs:
        J+ (numpy.ndarray) : Array representing the operator J+, has shape ((2j+1),(2j+1))

    '''
    dimension = numpy.rint(2.0*j+1).astype(int)
    J = numpy.zeros((dimension,dimension))
    for m_j in range(numpy.rint(2.0*j).astype(int)):
        J[m_j,m_j+1]=numpy.sqrt(j*(j+1)-(j-m_j)*(j-m_j-1))
    return J

#produce the three generalised projections of angular momentum:
# for S=1/2 these should return the Pauli matrices.
# for the source of these definitions see any good QM textbook e.g.
# Bransden & Joachain (or wikipedia)

def X_operator(J):
    ''' operator for X component of J

        Creates the Cartesian operator Jx for a given J

        input arguments:
            J (float): Magnitude of angular momentum
        output:
            Jx (numpy.ndarray) : 2J+1 square numpy array
    '''
    J_plus = Raising_operator(J)
    J_minus = numpy.transpose(J_plus)
    return 0.5*(J_plus+J_minus)

def Y_operator(J):
    ''' operator for Y component of J

        Creates the Cartesian operator Jy for a given J

        input arguments:
            J (float): Magnitude of angular momentum
        output:
            Jy (numpy.ndarray) : 2J+1 square numpy array
    '''
    J_plus = Raising_operator(J)
    J_minus = numpy.transpose(J_plus)
    return 0.5j*(J_minus - J_plus)

def Z_operator(J):
    ''' operator for Z component of J

        Creates the Cartesian operator Jz for a given J. This is diagonal in the j,mj basis such that jz|j,mj> = mj|j,mj>

        input arguments:
            J (float): Magnitude of angular momentum
        output:
            Jz (numpy.ndarray) : 2J+1 square numpy array
    '''
    J_plus = Raising_operator(J)
    J_minus = numpy.transpose(J_plus)
    return 0.5*(numpy.dot(J_plus,J_minus)-numpy.dot(J_minus,J_plus))

def vector_dot(x,y):
    '''Cartesian dot product of two vectors of operators x,y

        A function that can do the dot product of a vector of matrices default
        behaviour of numpy.dot does the elementwise product of the matrices.

        input arguments:
            x,y (numpy.ndarray): length-3 Vectors of Angular momentum operators, each element is a JxJ arrays

        output:
            Z (numpy.ndarray): result of the dot product, JxJ array
    '''
    X_Y = numpy.zeros(x[0].shape,dtype=numpy.complex)
    for i in range(x.shape[0]):
        X_Y += numpy.dot(x[i],y[i])
    return X_Y

def Generate_vecs(Nmax,I1,I2):
    ''' Build N, I1, I2 angular momentum vectors

        Generate the vectors of the angular momentum operators which we need
        to be able to produce the Hamiltonian

        input arguments:
            Nmax (float): maximum rotational level to include in calculations
            I1,I2 (float): Nuclear spins of nuclei 1 and 2
        output arguments:
            N_vec,I1_vec,I2_vec (list of numpy.ndarray): length-3 list of (2Nmax+1)*(2I1+1)*(2I2+1) square numpy arrays
    '''

    shapeN = int(numpy.sum([2*x+1 for x in range(0,Nmax+1)]))
    shape1 = int(2*I1+1)
    shape2 = int(2*I2+1)

    Nx = numpy.array([[]])
    Ny=numpy.array([[]])
    Nz= numpy.array([[]])

    for n in range(0,Nmax+1):
        Nx = block_diag(Nx,X_operator(n))
        Ny = block_diag(Ny,Y_operator(n))
        Nz = block_diag(Nz,Z_operator(n))

    #remove the first element of the N vectors, which are empty
    Nx = Nx[1:,:]
    Ny = Ny[1:,:]
    Nz = Nz[1:,:]

    #Each of the following corresponds to the product [N x 1Rb x 1Cs]
    #This gives the operators for N in the full hyperfine space.

    # numpy.kron is the function for the Kronecker product, often also called
    # the tensor product.

    N_vec = numpy.array([numpy.kron(Nx,numpy.kron(numpy.identity(shape1),
                                                    numpy.identity(shape2))),
                        numpy.kron(Ny,numpy.kron(numpy.identity(shape1),
                                                    numpy.identity(shape2))),
                        numpy.kron(Nz,numpy.kron(numpy.identity(shape1),
                                                    numpy.identity(shape2)))])

    # we also have to repeat for the nuclear spins
    I1_vec = numpy.array([numpy.kron(numpy.identity(shapeN),
                        numpy.kron(X_operator(I1),numpy.identity(shape2))),
                        numpy.kron(numpy.identity(shapeN),
                        numpy.kron(Y_operator(I1),numpy.identity(shape2))),
                        numpy.kron(numpy.identity(shapeN),
                        numpy.kron(Z_operator(I1),numpy.identity(shape2)))])

    I2_vec = numpy.array([numpy.kron(numpy.identity(shapeN),
                        numpy.kron(numpy.identity(shape1),X_operator(I2))),
                        numpy.kron(numpy.identity(shapeN),
                        numpy.kron(numpy.identity(shape1),Y_operator(I2))),
                        numpy.kron(numpy.identity(shapeN),
                        numpy.kron(numpy.identity(shape1),Z_operator(I2)))])

    return N_vec,I1_vec,I2_vec

def Wigner_D(l,m,alpha,beta,gamma):
    ''' The Wigner D matrix with labels l and m.

    Calculates the Wigner D Matrix for the given Alpha,beta,gamma in radians.
    The wigner-D matrices represent rotations of angular momentum operators.
    The indices l and m determine the value of the matrix.
    The second index (m') is always zero.

    The input angles are the x-z-x euler angles

    Inputs:
        l (int) : order of wigner Matrix
        m (float): first index of Wigner Matrix
        alpha,beta,gamma (float) : x,z,x Euler angles in radians
    outputs:
        D (float) : Value of the wigner-D matrix
    '''
    prefactor = numpy.sqrt((4*numpy.pi)/(2*l+1))
    function = numpy.conj(sph_harm(m,l,alpha,beta))
    return prefactor*function

def T2_C(Nmax,I1,I2):
    '''
    The irreducible spherical tensors for the spherical harmonics in the
    rotational basis.

    input arguments:
        Nmax (int) : Maximum rotational state to include
        I1,I2 (float) :  The nuclear spins of nucleus 1 and 2

    outputs:
        T (list of numpy.ndarray) : spherical tensor T^2(C). Each element is a spherical operator

    '''
    shape = sum([2*x+1 for x in range(0,Nmax+1)])
    shape = (shape,shape)
    Identity1 = numpy.identity(int(2*I1+1))
    Identity2 = numpy.identity(int(2*I2+1))

    Identity = numpy.kron(Identity1,Identity2)

    T = [numpy.zeros(shape),numpy.zeros(shape),
        numpy.zeros(shape),
        numpy.zeros(shape),numpy.zeros(shape)]

    x=-1
    for N in range(0,Nmax+1):
        for MN in range(N,-(N+1),-1):
            x+=1
            y=-1
            for Np in range(0,Nmax+1):
                for MNp in range(Np,-(Np+1),-1):
                    y+=1
                    for i,q in enumerate(range(-2,2+1)):
                        T[i][x,y]=((-1)**MN)*numpy.sqrt((2*N+1)*(2*Np+1))*\
                            wigner_3j(N,2,Np,0,0,0)*wigner_3j(N,2,Np,-MN,q,MNp)

    for i,q in enumerate(range(-2,2+1)):
        T[i] = numpy.kron(T[i],Identity)
    return T

def MakeT2(I1,I2):
    ''' Construct the spherical tensor T2 from two cartesian vectors of operators.

    Inputs
        I1,I2 (list of numpy.ndarray) - Length-3 list of cartesian angular momentum operators: the output of makevecs
    Outputs
        T (list of numpy.ndarray) - T^2(I1,I2) length-5 list of spherical angular momentum operators
    '''
    T2m2 = 0.5*(numpy.dot(I1[0],I2[0])-1.0j*numpy.dot(I1[0],I2[1])-1.0j*numpy.dot(I1[1],I2[0])-numpy.dot(I1[1],I2[1]))
    T2p2 = 0.5*(numpy.dot(I1[0],I2[0])+1.0j*numpy.dot(I1[0],I2[1])+1.0j*numpy.dot(I1[1],I2[0])-numpy.dot(I1[1],I2[1]))

    T2m1 = 0.5*(numpy.dot(I1[0],I2[2])-1.0j*numpy.dot(I1[1],I2[2])+numpy.dot(I1[2],I2[0])-1.0j*numpy.dot(I1[2],I2[1]))
    T2p1 = -0.5*(numpy.dot(I1[0],I2[2])+1.0j*numpy.dot(I1[1],I2[2])+numpy.dot(I1[2],I2[0])+1.0j*numpy.dot(I1[2],I2[1]))

    T20 = -numpy.sqrt(1/6)*(numpy.dot(I1[0],I2[0])+numpy.dot(I1[1],I2[1]))+numpy.sqrt(2/3)*numpy.dot(I1[2],I2[2])

    T = [T2m2,T2m1,T20,T2p1,T2p2]

    return T

def TensorDot(T1,T2):
    ''' Product of two rank-2 spherical tensors T1, T2

     A function to calculate the scalar product of two spherical tensors
    T1 and T2 are lists or numpy arrays that represent the spherical tensors
    lists are indexed from lowest m to highests

    Input:
        T1,T2 (list of numpy.ndarray) - length-5 list of numpy.ndarray
    output:
        X (numpy.ndarray) - scalar product of spherical tensors
    '''
    x = numpy.zeros(T1[0].shape,dtype=numpy.complex128)
    for i,q in enumerate(range(-2,2+1)):
        x += ((-1)**q)*numpy.dot(T1[i],T2[-(i+1)])
    return x


# From here the functions will calculate individual terms in the Hamiltonian,
# I have split them up for two reasons 1) readability and 2) so that its obvious
# what is doing what.


def ElectricGradient(Nmax,I1,I2):
    '''Calculate electric field gradient at the nucleus.

    spherical tensor for the electric field gradient at nucleus i. Depends
    on the rotational states not the nuclear spin states. Returns a spherical
    tensor.

    input arguments are:
        Nmax (int) - Maximum rotational state to include
        I1,I2 (float)- The nuclear spins of nucleus 1 and 2
    Output:
        T (list of numpy.ndarray) - length-5 list of numpy.ndarrays
    '''
    shape = sum([2*x+1 for x in range(0,Nmax+1)])
    shape = (shape,shape)
    Identity1 = numpy.identity(int(2*I1+1))

    Identity2 = numpy.identity(int(2*I2+1))

    Identity = numpy.kron(Identity1,Identity2)

    T = [numpy.zeros(shape),numpy.zeros(shape),
        numpy.zeros(shape),
        numpy.zeros(shape),numpy.zeros(shape)]

    x=-1
    for N in range(0,Nmax+1):
        for MN in range(N,-(N+1),-1):
            x+=1
            y=-1
            for Np in range(0,Nmax+1):
                for MNp in range(Np,-(Np+1),-1):
                    y+=1
                    for i,q in enumerate(range(-2,2+1)):
                        T[i][x,y]=(-1)**(N-MN)*wigner_3j(N,2,Np,-MN,q,MNp)*\
                        (-1)**N*numpy.sqrt((2*N+1)*(2*Np+1))*\
                        wigner_3j(N,2,Np,0,0,0)

    for i,q in enumerate(range(-2,2+1)):
        T[i] = numpy.kron(T[i],Identity)
    return T

def QuadMoment(Nmax,I1,I2):
    ''' Calculate the nuclear electric quadrupole moments of nuclei 1 and 2.

    spherical tensor for the nuclear quadrupole moment of both nuclei. Depends
    on the nuclear spin states not the rotational states.
    input arguments are:
        Nmax (int) - Maximum rotational state to include
        I1,I2 (float) - The nuclear spins of nucleus 1 and 2
    Output:
        T (list of numpy.ndarray) - length-5 list of numpy.ndarrays

    '''
    shape1 = int(2*I1+1)
    shape1 = (shape1,shape1)

    T1 = [numpy.zeros(shape1),numpy.zeros(shape1),
        numpy.zeros(shape1),
        numpy.zeros(shape1),numpy.zeros(shape1)]

    shape2 = int(2*I2+1)
    shape2 = (shape2,shape2)

    T2 = [numpy.zeros(shape2),numpy.zeros(shape2),
        numpy.zeros(shape2),
        numpy.zeros(shape2),numpy.zeros(shape2)]

    ShapeN = int(sum([2*x+1 for x in range(0,Nmax+1)]))

    IdentityN = numpy.identity(ShapeN)
    Identity1 = numpy.identity(int(2*I1+1))
    Identity2 = numpy.identity(int(2*I2+1))

    x=-1
    for M1 in numpy.arange(I1,-(I1+1),-1):
        x+=1
        y=-1
        for M1p in numpy.arange(I1,-(I1+1),-1):
            y+=1
            for i,q in enumerate(range(-2,2+1)):
                T1[i][x,y]=(-1)**(I1-M1)*wigner_3j(I1,2,I1,-M1,q,M1p)/\
                wigner_3j(I1,2,I1,-I1,0,I1)
    x=-1
    for M2 in numpy.arange(I2,-(I2+1),-1):
        x+=1
        y=-1
        for M2p in numpy.arange(I2,-(I2+1),-1):
            y+=1
            for i,q in enumerate(range(-2,2+1)):
                T2[i][x,y]=(-1)**(I2-M2)*wigner_3j(I2,2,I2,-M2,q,M2p)/\
                wigner_3j(I2,2,I2,-I2,0,I2)

    for i,q in enumerate(range(-2,2+1)):
        T1[i] = numpy.kron(IdentityN,numpy.kron(T1[i],Identity2))
        T2[i] = numpy.kron(IdentityN,numpy.kron(Identity1,T2[i]))
    return T1,T2

def Quadrupole(Q,I1,I2,Nmax):
    ''' Calculate Hquad, the nuclear electric quadrupole interaction energy

    Calculates the Quadrupole terms for the hyperfine Hamiltonian using
    spherical tensor algebra. Requires the nuclear quadrupole moments and
    electric field gradients.

    input arguments are:
        Q (tuple of floats) - two-tuple of nuclear electric quadrupole moments in Joules
        Nmax (int) - Maximum rotational state to include
        I1,I2  (float) - The nuclear spins of nucleus 1 and 2

    Outputs:
        Hquad (numpy.ndarray) - numpy array with shape (2I1+1)*(2I2+1)*sum([(2*x+1) for x in range(Nmax+1)])
    '''
    Q1,Q2 = Q

    TdE = ElectricGradient(Nmax,I1,I2)
    Tq1,Tq2 = QuadMoment(Nmax,I1,I2)

    Hq = Q1*TensorDot(Tq1,TdE)+Q2*TensorDot(Tq2,TdE)

    return Hq/4


def Rotational(N,Brot,Drot):
    ''' Rigid rotor rotational structure

        Generates the hyperfine-free hamiltonian for the rotational levels of
        a rigid-rotor like molecule. Includes the centrifugal distortion term.

        Matrix is returned in the N,MN basis with MN going from maximum to minimum.

        input arguments:
            N (list of numpy.ndarray) - length 3 list representing the Angular momentum vector for rotation
            Brot(float) - Rotational constant coefficient in joules
            Drot (float) - Centrifugal distortion coefficient in joules

        output arguments:
            Hrot (numpy.ndarray) - hamiltonian for rotation in the N,MN basis
    '''
    N_squared = vector_dot(N,N)
    return Brot*N_squared-Drot*N_squared*N_squared

def Zeeman(Cz,J):
    '''Calculate the Zeeman effect for a magnetic field along z

        Linear Zeeman shift, fixed magnetic field along z so only need the
        last component of the angular momentum vector.

        input arguments:
            Cz (float) - Zeeman Coefficient/magnetic moment
            J (list of numpy.ndarray) - Angular momentum vector
        outputs:
            Hz (numpy.ndarray) - Zeeman Hamiltonian
    '''
    Hzeeman = -Cz*J[2]
    return Hzeeman

def scalar_nuclear(Ci,J1,J2):
    ''' Calculate the scalar spin-spin interaction term

        Returns the scalar spin-spin term of the HF Hamiltonian

        Input arguments:
            Ci(float) - Scalar spin coupling coefficient
            J1,J2 (list of numpy.ndarray) - Angular momentum vectors

        returns:
            H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''
    return Ci*vector_dot(J1,J2)

def tensor_nuclear(C3,I1,I2,Nmax):
    ''' Calculate the tensor spin-spin interaction.

        This function is to calculate the tensor spin-spin interaction.
        This version uses spherical tensors to calculate the correct off-diagonal
        behaviour.

        Inputs:
            C3 (float) - spin-spin coupling constant
            I1,I2 (float) - Cartesian Angular momentum operator Vectors
            Nmax (int) - maximum rotational state to include

        Outputs:
            Hss (numpy.ndarray) - Hamiltonian for tensor spin-spin interaction
    '''
    #find the value of I1 and I2 with less input arguments
    I1_val = numpy.round(numpy.amax(I1[2]),1).real
    I2_val = numpy.round(numpy.amax(I2[2]),1).real

    #steps for maths, creates the spherical tensors
    T1 = T2_C(Nmax,I1_val,I2_val)
    T2 = MakeT2(I1,I2)
    #return final Hamiltonian
    tensorss = numpy.sqrt(6)*C3*TensorDot(T1,T2)

    return tensorss

def DC(Nmax,d0,I1,I2):
    ''' calculate HDC for a diatomic molecule

        Generates the effect of the dc Stark shift for a rigid-rotor like
        molecule.

        This term is calculated differently to all of the others in this work
        and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
        N,MN,N',MN' to build a matrix without hyperfine structure then uses
        kronecker products to expand it into all of the hyperfine states.


        input arguments:
            Nmax(int) -  maximum rotational quantum number to calculate
            d0 (float) - Permanent electric dipole momentum
            I1,I2 (float) - Nuclear spin of nucleus 1,2


        returns:
            H (numpy.ndarray) - DC Stark Hamiltonian in joules
     '''

    shape = numpy.sum(numpy.array([2*x+1 for x in range(0,Nmax+1)]))
    HDC = numpy.zeros((shape,shape),dtype= numpy.complex)

    I1shape = int(2*I1+1)
    I2shape = int(2*I2+1)

    i =0
    j =0
    for N1 in range(0,Nmax+1):
        for M1 in range(N1,-(N1+1),-1):
            for N2 in range(0,Nmax+1):
                for M2 in range(N2,-(N2+1),-1):
                    HDC[i,j]=-d0*numpy.sqrt((2*N1+1)*(2*N2+1))*(-1)**(M1)*\
                    wigner_3j(N1,1,N2,-M1,0,M2)*wigner_3j(N1,1,N2,0,0,0)
                    j+=1
            j=0
            i+=1
    return (numpy.kron(HDC,numpy.kron(numpy.identity(I1shape),
            numpy.identity(I2shape))))

def AC_iso(Nmax,a0,I1,I2):
    ''' Calculate isotropic Stark shifts

        Generates the effect of the isotropic AC Stark shift for a rigid-rotor
        like molecule.

        This term is calculated differently to all of the others in this work
        and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
        N,MN,N',MN' to build a matrix without hyperfine structure then uses
        kronecker products to expand it into all of the hyperfine states.

        input arguments:
            Nmax (int) - maximum rotational quantum number to calculate (int)
            a0 (float) - isotropic polarisability in joules/ W/m^2
            I1,I2 (float) - Nuclear spin of nucleus 1,2


        returns:
            H (numpy.ndarray) - isotropic AC Stark Hamiltonian

     '''
    shape = numpy.sum(numpy.array([2*x+1 for x in range(0,Nmax+1)]))
    I1shape = int(2*I1+1)
    I2shape = int(2*I2+1)
    HAC = numpy.zeros((shape,shape),dtype= numpy.complex)
    i=0
    j=0
    for N1 in range(0,Nmax+1):
        for M1 in range(N1,-(N1+1),-1):
            for N2 in range(0,Nmax+1):
                for M2 in range(N2,-(N2+1),-1):
                    if N1==N2 and M1 ==M2:
                        HAC[i,j]=-a0
                    j+=1
            j=0
            i+=1
    #final check for NaN errors, mostly this is due to division by zero or
    # multiplication by a small prefactor. it is safe to set these terms to 0
    HAC[numpy.isnan(HAC)] =0

    #return the matrix, in the full uncoupled basis.
    return (numpy.kron(HAC,numpy.kron(numpy.identity(I1shape),
                                                    numpy.identity(I2shape))))

def AC_aniso(Nmax,a2,Beta,I1,I2):
    ''' Calculate anisotropic ac stark shift.

        Generates the effect of the anisotropic AC Stark shift for a rigid-rotor
        like molecule.

        This term is calculated differently to all of the others in this work
        and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
        N,MN,N',MN' to build a matrix without hyperfine structure then uses
        kronecker products to expand it into all of the hyperfine states.

        input arguments:

            Nmax (int) - maximum rotational quantum number to calculate
            a2 (float) - anisotropic polarisability
            Beta (float) - polarisation angle of the laser in Radians
            I1,I2 (float) - Nuclear spin of nucleus 1,2

        returns:
            H (numpy.ndarray): Hamiltonian in joules
     '''
    I1shape = int(2*I1+1)
    I2shape = int(2*I2+1)
    shape = numpy.sum(numpy.array([2*x+1 for x in range(0,Nmax+1)]))
    HAC = numpy.zeros((shape,shape),dtype= numpy.complex)
    i=0
    j=0
    for N1 in range(0,Nmax+1):
        for M1 in range(N1,-(N1+1),-1):
            for N2 in range(0,Nmax+1):
                for M2 in range(N2,-(N2+1),-1):
                    M = M2-M1
                    HAC[i,j]= -a2*(Wigner_D(2,M,0,Beta,0)*(-1)**M2*\
                                numpy.sqrt((2*N1+1)*(2*N2+1))*\
                                wigner_3j(N2,2,N1,0,0,0)*\
                                wigner_3j(N2,2,N1,-M2,M,M1))
                    j+=1
            j=0
            i+=1
    #final check for NaN errors, mostly this is due to division by zero or
    # multiplication by a small prefactor. it is safe to set these terms to 0
    HAC[numpy.isnan(HAC)] =0

    #return the matrix, in the full uncoupled basis.
    return (numpy.kron(HAC,numpy.kron(numpy.identity(I1shape),
            numpy.identity(I2shape))))

#Now some functions to take these functions and assemble them into the physical
#Hamiltonians where necessary.


def Hyperfine_Ham(Nmax,I1_mag,I2_mag,Consts):
    '''Calculate the field-free Hyperfine hamiltonian

        Wrapper to call all of the functions that are appropriate for the singlet-sigma hyperfine hamiltonian.

        Input arguments:
            Nmax (int) - Maximum rotational level to include
            I1_mag,I2_mag (float) - magnitude of the nuclear spins
            Consts (Dictionary): Dict of molecular constants
        returns:
            H0 : Hamiltonian for the hyperfine structure in joules
    '''
    N,I1,I2 = Generate_vecs(Nmax,I1_mag,I2_mag)
    H = Rotational(N,Consts['Brot'],Consts['Drot'])+\
    scalar_nuclear(Consts['C1'],N,I1)+scalar_nuclear(Consts['C2'],N,I2)+\
    scalar_nuclear(Consts['C4'],I1,I2)+tensor_nuclear(Consts['C3'],I1,I2,Nmax)+\
    Quadrupole((Consts['Q1'],Consts['Q2']),I1_mag,I2_mag,Nmax)
    return H

def Zeeman_Ham(Nmax,I1_mag,I2_mag,Consts):
    '''Assembles the Zeeman term and generates operator vectors

        Calculates the Zeeman effect for a magnetic field on a singlet-sigma molecule.
        There is no electronic term and the magnetic field is fixed to be along the z axis.

        Input arguments:
        Nmax (int) - Maximum rotational level to include
        I1_mag,I2_mag (float) - magnitude of the nuclear spins
        Consts (Dictionary): Dict of molecular constants

        returns:
        Hz (numpy.ndarray): Hamiltonian for the zeeman effect
    '''
    N,I1,I2 = Generate_vecs(Nmax,I1_mag,I2_mag)
    H = Zeeman(Consts['Mu1'],I1)+Zeeman(Consts['Mu2'],I2)+\
                Zeeman(Consts['MuN'],N)
    return H
# This is the main build function and one that the user will actually have to
# use.

def Build_Hamiltonians(Nmax,Constants,zeeman=False,EDC=False,AC=False):
    ''' Return the hyperfine hamiltonian.

        This function builds the hamiltonian matrices for evalutation so that
        the user doesn't have to rebuild them every time and we can benefit from
        numpy's ability to do distributed multiplcation.

        Input arguments:
            Nmax (int) - Maximum rotational level to include
            I1_mag,I2_mag (float) - magnitude of the nuclear spins
            Constants (Dictionary) - Dict of molecular constants
            zeeman,EDC,AC (Boolean) - Switches for turning off parts of the total Hamiltonian can save significant time on calculations where DC and AC fields are not required due to nested for loops

        returns:
            H0,Hz,HDC,HAC (numpy.ndarray): Each of the terms in the Hamiltonian.
    '''
    I1 = Constants['I1']
    I2 = Constants['I2']

    H0 = Hyperfine_Ham(Nmax,I1,I2,Constants)
    if zeeman:
        Hz = Zeeman_Ham(Nmax,I1,I2,Constants)
    else:
        Hz =0.
    if EDC:
        HDC = DC(Nmax,Constants['d0'],I1,I2)
    else:
        HDC =0.
    if AC:
        HAC = (1./(2*eps0*c))*(AC_iso(Nmax,Constants['a0'],I1,I2)+\
        AC_aniso(Nmax,Constants['a2'],Constants['Beta'],I1,I2))
    else:
        HAC =0.
    return H0,Hz,HDC,HAC


if __name__=="__main__":

    #This code only executes if the module is directly executed so acts as a
    #simple test.

    import matplotlib.pyplot as pyplot
    import scipy.constants
    import time
    ''' My test is building a zeeman structure plot for N<= 5 takes ~20 mins'''
    B = 0
    I = 0
    E = 0

    h = scipy.constants.h
    muN = scipy.constants.physical_constants['nuclear magneton'][0]
    bohr = scipy.constants.physical_constants['Bohr radius'][0]
    eps0 = scipy.constants.epsilon_0
    c = scipy.constants.c

    DebyeSI = 3.33564e-30 #C m/Debye

    Nmaximum = 3

    Steps = 100

    bvary = numpy.linspace(0,500e-4,Steps)
    start = time.time()
    Hamiltonian = Build_Hamiltonians(Nmaximum,RbCs,zeeman=True)
    end = time.time()

    print("Creating the Hamiltonian took {:.3f} s".format(end-start))
    print("######################################")
    start = end
    energy = Vary_magnetic(Hamiltonian,(B,I,E),bvary)
    end = time.time()

    print("Evaluating the Hamiltonian took {:.3f} s".format(end-start))
    print("######################################")
    pfig = pyplot.figure()

    for i in range(len(energy[:,0])):
        pyplot.plot(bvary,1e-6*energy[i,:]/h,color='k')

    pyplot.xlabel("Magnetic Field (G)")
    pyplot.ylabel("Energy/$h$ (MHz)")
    pyplot.show()
