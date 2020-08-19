import numpy
from sympy.physics.wigner import wigner_3j,wigner_9j
from sympy.physics.quantum.spin import Rotation
from scipy.linalg import block_diag,eig,eigvals
import scipy.constants
import warnings

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
            "Drot":213*h,
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
    #produce the angular momentum raising operator J+
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
    '''
        input arguments:
        J: Magnitude of angular momentum (float)
    '''
    J_plus = Raising_operator(J)
    J_minus = numpy.transpose(J_plus)
    return 0.5*(J_plus+J_minus)

def Y_operator(J):
    '''
        input arguments:
        J: Magnitude of angular momentum (float)
    '''
    J_plus = Raising_operator(J)
    J_minus = numpy.transpose(J_plus)
    return 0.5j*(J_minus - J_plus)

def Z_operator(J):
    '''
        input arguments:
        J: Magnitude of angular momentum (float)
    '''
    J_plus = Raising_operator(J)
    J_minus = numpy.transpose(J_plus)
    return 0.5*(numpy.dot(J_plus,J_minus)-numpy.dot(J_minus,J_plus))

def vector_dot(x,y):
    '''
        A function that can do the dot product of a vector of matrices default
        behaviour of numpy.dot does the elementwise product of the matrices.
        input arguments:
        x,y: Vectors of Angular momentum operators, each element is a JxJ arrays
             (numpy.ndarray)
    '''
    X_Y = numpy.zeros(x[0].shape,dtype=numpy.complex)
    for i in range(x.shape[0]):
        X_Y += numpy.dot(x[i],y[i])
    return X_Y

def Generate_vecs(Nmax,I1,I2):
    '''
        Generate the vectors of the angular momentum operators which we need
        to be able to produce the Hamiltonian

        input arguments:
        Nmax: maximum rotational level to include in calculations (float)
        I1,I2: Nuclear spins of nuclei 1 and 2 (float)

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

# From here the functions will calculate individual terms in the Hamiltonian,
# I have split them up for two reasons 1) readability and 2) so that its obvious
# what is doing what.

def Rotational(N,Brot,Drot):
    '''
        Generates the hyperfine-free hamiltonian for the rotational levels of
        a rigid-rotor like molecule. Includes the centrifugal distortion term

        input arguments:
        N: Angular momentum vector for rotation (numpy.ndarry)
        Brot: Rotational constant (float)
        Drot: Centrifugal distortion (float)
    '''
    N_squared = vector_dot(N,N)
    return Brot*N_squared-Drot*N_squared*N_squared

def Zeeman(Cz,J):
    '''
        Linear Zeeman shift, fixed magnetic field along z so only need the
        last component of the angular momentum vector.

        input arguments:
        Cz: Zeeman Coefficient (float)
        J: Angular momentum vector (numpy.ndarray)
    '''
    Hzeeman = -Cz*J[2]
    return Hzeeman

def scalar_nuclear(Ci,J1,J2):
    '''
        Returns the scalar spin-spin term of the HF Hamiltonian
        Input arguments:
        Ci: Scalar spin coupling coefficient (float)
        J1,J2: Angular momentum vector (numpy.ndarray)

        returns:
        Quad: (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.
    '''
    return Ci*vector_dot(J1,J2)

def tensor_nuclear(C3,I1,I2,N):
    '''
        The tensor - nuclear spin spin interaction
        input arguments:
        C3: Tensor spin-spin coupling coefficient (float)
        I1,I2,N: Angular momentum Vectors (numpy.ndarry)
        returns:
        Quad: (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.
    '''
    with warnings.catch_warnings():
        # this is a statement to make the code nicer to use, python wants to
        # warn the user whenever the data type is changed from Complex. But we
        # know that it will always be real so it doesn't matter.
        warnings.filterwarnings("ignore",category=numpy.ComplexWarning)
        #find max values for angular momentum from their projections onto z
        Nmax = int(numpy.round(numpy.real(numpy.amax(N[2])),1))
        I1max = numpy.real(numpy.round(numpy.amax(I1[2]),1))
        I2max = numpy.real(numpy.round(numpy.amax(I2[2]),1))

    I1shape = int(2*I1max+1)
    I2shape = int(2*I2max+1)

    # The tensor nuclear spin-spin interaction depends on the rotational level
    # not its projection, so we have to create a new matrix that contains the
    # values of N. Thankfully the terms are block-diagonal in N so we don't have
    # to worry what the term <N,MN|I1 dot T dot I|N',MN'> looks like
    Narray = numpy.zeros((1,1))

    for n in range(0,Nmax+1):
        #this loop iterates over all the values for N (indexed as n) allowed and
        # builds an nxn matrix of only one value.

        shape = int((2*n+1)*(2*I1max+1)*(2*I2max+1))
        nsub = numpy.zeros((shape,shape))+n
        Narray = block_diag(Narray,nsub)

    #first element is fixed to be zero - get rid of it
    Narray = Narray[1:,1:]

    #Now calculate the terms as shown earlier
    prefactor = C3/((2*Narray+3)*(2*Narray-1))
    term1 = 3*numpy.dot(vector_dot(I1,N),vector_dot(I2,N))
    term2 = 3*numpy.dot(vector_dot(I2,N),vector_dot(I1,N))
    term3 = -2*vector_dot(I1,I2)*Narray*(Narray+1)
    return prefactor*(term1+term2+term3)


def Quadrupole(Q,I1,I2,N):
    '''
        from 10.1103/PhysRev.91.1403, which quotes the quadrupole interaction
         for KBr
         input arguments:

         Q:Tuple or list of the nuclear quadrupole moments as (Q1,Q2)  (tuple)
         I1,I2,N: Nuclear spin of nucleus 1,2 and rotational angular momentum
                  vectory (numpy.ndarray)
        returns:
        Quad: (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.

    '''
    Q1,Q2 = Q
    with warnings.catch_warnings():
        # this is a statement to make the code nicer to use, python wants to
        # warn the user whenever the data type is changed from Complex. But we
        # know that it will always be real so it doesn't matter.
        warnings.filterwarnings("ignore",category=numpy.ComplexWarning)
        #find max values for angular momentum from their projections onto z
        Nmax = int(numpy.round(numpy.real(numpy.amax(N[2])),1))
        I1max = numpy.round(numpy.real(numpy.amax(I1[2])),1)
        I2max = numpy.round(numpy.real(numpy.amax(I2[2])),1)

    Narray = numpy.array([])
    Narray=numpy.zeros((1,1))

    for n in range(Nmax+1):
        # this loop iterates over all the values for N (indexed as n) allowed &
        # builds an (2*I1+1)*(2*I2+1)*(2*n+1)x(2*I1+1)*(2*I2+1)*(2*n+1) matrix
        # of only one value.
        shape = int((2*I1max+1)*(2*I2max+1)*(2*n+1))
        subarray = numpy.zeros((shape,shape))+n
        Narray= scipy.linalg.block_diag(Narray,subarray)
    Narray = Narray[1:,1:]
    # there is the possibility for division by zero here, so define a machine
    # epsilon to avoid NaN errors. Epsilon is insignificantly small,
    # particularly on modern 64-bit machines.
    epsilon = (numpy.finfo(float).eps)

    prefactor1 = numpy.zeros(Narray.shape)
    prefactor2 = numpy.zeros(Narray.shape)

    # Calculate the terms as earlier. This is presented in Sigma notation in the
    # text but is actually just two terms.
    prefactor1 = -Q1/(2*I1max*(2*I1max-1)*(2*Narray-1)\
                        *(2*Narray+3))

    term1_1= 3*(numpy.dot(vector_dot(I1,N),vector_dot(I1,N)))
    term2_1 = 1.5*vector_dot(I1,N)
    term3_1 = -1*numpy.dot(vector_dot(I1,I1),vector_dot(N,N))
    Quad1 = prefactor1*(term1_1 +term2_1+term3_1)

    prefactor2 = -Q2/(2*I2max*(2*I2max-1)*(2*Narray-1)*\
                        (2*Narray+3))

    term1_2= 3*(numpy.dot(vector_dot(I2,N),vector_dot(I2,N)))
    term2_2 = 1.5*vector_dot(I2,N)
    term3_2 = -1*numpy.dot(vector_dot(I2,I2),vector_dot(N,N))
    Quad2 = prefactor2*(term1_2 +term2_2+term3_2)

    return Quad1+Quad2

def DC(Nmax,d0,I1,I2):
    '''
        Generates the effect of the dc Stark shift for a rigid-rotor like
        molecule.

        This term is calculated differently to all of the others in this work
        and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
        N,MN,N',MN' to build a matrix without hyperfine structure then uses
        kronecker products to expand it into all of the hyperfine states.


        input arguments:

        Nmax: maximum rotational quantum number to calculate (int)
        d0: Permanent electric dipole momentum (float)
        I1,I2: Nuclear spin of nucleus 1,2 (float)


        returns:
        H: Hamiltonian, (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.
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
    '''
        Generates the effect of the isotropic AC Stark shift for a rigid-rotor
        like molecule.

        This term is calculated differently to all of the others in this work
        and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
        N,MN,N',MN' to build a matrix without hyperfine structure then uses
        kronecker products to expand it into all of the hyperfine states.

        input arguments:

        Nmax: maximum rotational quantum number to calculate (int)
        a0: isotropic polarisability (float)
        I1,I2: Nuclear spin of nucleus 1,2 (float)


        returns:
        H: Hamiltonian, (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.

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
    '''
        Generates the effect of the anisotropic AC Stark shift for a rigid-rotor
        like molecule.

        This term is calculated differently to all of the others in this work
        and is based off Jesus Aldegunde's FORTRAN 77 code. It iterates over
        N,MN,N',MN' to build a matrix without hyperfine structure then uses
        kronecker products to expand it into all of the hyperfine states.

        input arguments:

        Nmax: maximum rotational quantum number to calculate (int)
        a2: anisotropic polarisability (float)
        Beta: polarisation angle of the laser in Radians (float)
        I1,I2: Nuclear spin of nucleus 1,2 (float)

        returns:
        H: Hamiltonian, (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.
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
                    HAC[i,j]= -a2*(Rotation.d(2,M,0,Beta).doit()*(-1)**M2*\
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
    '''
        The field-free Hyperfine hamiltonian

        Input arguments:
        Nmax: Maximum rotational level to include (float)
        I1_mag,I2_mag, magnitude of the nuclear spins (float)
        Consts: Dict of molecular constants (Dict of floats)

        returns:
        H: Hamiltonian, (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.
    '''
    N,I1,I2 = Generate_vecs(Nmax,I1_mag,I2_mag)
    H = Rotational(N,Consts['Brot'],Consts['Drot'])+\
    scalar_nuclear(Consts['C1'],N,I1)+scalar_nuclear(Consts['C2'],N,I2)+\
    scalar_nuclear(Consts['C4'],I1,I2)+tensor_nuclear(Consts['C3'],I1,I2,N)+\
    Quadrupole((Consts['Q1'],Consts['Q2']),I1,I2,N)
    return H

def Zeeman_Ham(Nmax,I1_mag,I2_mag,Consts):
    '''
        assembles the Zeeman term and generates operator vectors

        Input arguments:
        Nmax: Maximum rotational level to include (float)
        I1_mag,I2_mag, magnitude of the nuclear spins (float)
        Consts: Dict of molecular constants (Dict of floats)

        returns:
        H: Hamiltonian, (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.
    '''
    N,I1,I2 = Generate_vecs(Nmax,I1_mag,I2_mag)
    H = Zeeman(Consts['Mu1'],I1)+Zeeman(Consts['Mu2'],I2)+\
                Zeeman(Consts['MuN'],N)
    return H

# This is the main build function and one that the user will actually have to
# use.

def Build_Hamiltonians(Nmax,Constants,zeeman=False,EDC=False,AC=False):
    '''
        This function builds the hamiltonian matrices for evalutation so that
        the user doesn't have to rebuild them every time and we can benefit from
        numpy's ability to do distributed multiplcation.



        Input arguments:
        Nmax: Maximum rotational level to include (float)
        I1_mag,I2_mag, magnitude of the nuclear spins (float)
        Constants: Dict of molecular constants (Dict of floats)
        zeeman,EDC,AC :Switches for turning off parts of the total Hamiltonian
                        can save significant time on calculations where DC and
                        AC fields are not required due to nested for loops
                        (bool)

        returns:
        H0,Hz,HDC,HAC: Each is a (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1)x
           (2*Nmax+1)*(2*I1_mag+1)*(2*I2_mag+1) array.
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

#These are the functions that the user will use to generate any interesting maps
#obviously these can be added to by writing custom scripts but these should
# cover most needs

def Vary_magnetic(Hams,fields0,Bz,return_states = False):

    '''
        find Eigenvalues (and optionally Eigenstates) of the total Hamiltonian

        input arguments:
        Hams: list or tuple of hamiltonians. Should all be the same size
        fields0: initial field conditions, allows for zeeman + Stark effects
        Bz: magnetic field to be iterated over
        return_states: Switch to return EigenStates as well as Eigenenergies

        returns:
        energy:array of Eigenenergies, sorted from smallest to largest along
               the 0 axis
        states:array of Eigenstates, sorted as in energy.
    '''

    H0,Hz,HDC,HAC = Hams
    E,B,I = fields0

    #warn the user if they've done something silly, so they don't waste time
    if type(Hz) != numpy.ndarray:
        warnings.warn("Hamiltonian is zero: nothing will change!")
    else:
        EigenValues = numpy.zeros((H0.shape[0],len(Bz)))
        if return_states:
            States = numpy.zeros((H0.shape[0],H0.shape[0],len(Bz)))
        for i,b in enumerate(Bz):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=numpy.ComplexWarning)
                H = H0+E*HDC+I*HAC+b*Hz
                if return_states:
                    Eigen = eig(H)
                    order = numpy.argsort(Eigen[0])
                    EigenValues[:,i]=Eigen[0][order]
                    States[:,:,i] = Eigen[1][:,order]
                else:
                    Eigen = eigvals(H)
                    EigenValues[:,i]=numpy.sort(Eigen)
        if return_states:
            return EigenValues,States
        else:
            return EigenValues

def Vary_ElectricDC(Hams,fields0,Ez,return_states = False):

    '''
        find Eigenvalues (and optionally Eigenstates) of the total Hamiltonian

        input arguments:
        Hams: list or tuple of hamiltonians. Should all be the same size
        fields0: initial field conditions, allows for zeeman + Stark effects
        Ez: Electric field to be iterated over
        return_states: Switch to return EigenStates as well as Eigenenergies

        returns:
        energy:array of Eigenenergies, sorted from smallest to largest along
               the 0 axis
        states:array of Eigenstates, sorted as in energy.
    '''
    E,B,I = fields0
    H0,Hz,HDC,HAC = Hams
    EigenValues = numpy.zeros((H0.shape[0],len(Ez)))

    #warn the user if they've done something silly, so they don't waste time

    if type(HDC) != numpy.ndarray:
        warnings.warn("Hamiltonian is zero: nothing will change!")

    else:
        if return_states:
            States = numpy.zeros((H0.shape[0],H0.shape[0],len(Ez)))
        for i,e in enumerate(Ez):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=numpy.ComplexWarning)
                H = H0+e*HDC+I*HAC+B*Hz
                if return_states:
                    Eigen = eig(H)
                    order = numpy.argsort(Eigen[0])
                    EigenValues[:,i]=Eigen[0][order]
                    States[:,:,i] = Eigen[1][:,order]
                else:
                    Eigen = eigvals(H)
                    EigenValues[:,i]=numpy.sort(Eigen)
        if return_states:
            return EigenValues,States
        else:
            return EigenValues

def Vary_Intensity(Hams,fields0,I_app,return_states = False):
    '''
        find Eigenvalues (and optionally Eigenstates) of the total Hamiltonian

        input arguments:
        Hams: list or tuple of hamiltonians. Should all be the same size
        fields0: initial field conditions, allows for zeeman + Stark effects
        I_app: Laser
        return_states: Switch to return EigenStates as well as Eigenenergies

        returns:
        energy:array of Eigenenergies, sorted from smallest to largest along
               the 0 axis
        states:array of Eigenstates, sorted as in energy.
    '''

    H0,Hz,HDC,HAC = Hams
    E,B,I = fields0

    #warn the user if they've done something silly, so they don't waste time

    if type(HAC) != numpy.ndarray:
        warnings.warn("Hamiltonian is zero: nothing will change")
    else:
        EigenValues = numpy.zeros((H0.shape[0],len(I_app)))
        if return_states:
            States = numpy.zeros((H0.shape[0],H0.shape[0],len(I_app)))
        else:
            for i,Int in enumerate(I_app):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                            category=numpy.ComplexWarning)
                    H = H0+E*HDC+Int*HAC+B*Hz
                    if return_states:
                        Eigen = eig(H)
                        order = numpy.argsort(Eigen[0])
                        EigenValues[:,i]=Eigen[0][order]
                        States[:,:,i] = Eigen[1][:,order]
                    else:
                        Eigen = eigvals(H)
                        EigenValues[:,i]=numpy.sort(Eigen)
            if return_states:
                return EigenValues,States
            else:
                return EigenValues

def Vary_Beta(Hams,fields0,Angles,Molecule_pars,return_states = False):
    '''
        find Eigenvalues (and optionally Eigenstates) of the total Hamiltonian
        This function works differently to the applied field ones. Because beta
        changes the matrix elements in the Hamiltonian we cannot simply
        multiply it through. Therefore we have to recalculate the matrix
        elements on each interation. This makes the function slower.

        input arguments:
        Hams: list or tuple of hamiltonians. Should all be the same size
        fields0: initial field conditions, allows for zeeman + Stark effects
        Angles: Polarisation angles to iterate over

        Molecule_pars: Nmax,I1,I2,a2, arguments to feed to regenerate the
                        anisotropic Stark shift matrix.

        return_states: Switch to return EigenStates as well as Eigenenergies

        returns:
        energy:array of Eigenenergies, sorted from smallest to largest along
               the 0 axis
        states:array of Eigenstates, sorted as in energy.
    '''

    Nmax,I1,I2,a2 = Molecule_pars
    H0,Hz,HDC,HAC = Hams
    E,B,I = fields0

    #warn the user if they've done something silly, so they don't waste time

    if I == 0:
        warnings.warn("Intensity is zero: nothing will change")
    else:
        EigenValues = numpy.zeros((H0.shape[0],len(Angles)))
        if return_states:
            States = numpy.zeros((H0.shape[0],H0.shape[0],len(Angles)))
        for i,beta in enumerate(Angles):
            HAC = AC_aniso(Nmax,a2,beta,I1,I2)/(2*eps0*c)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=numpy.ComplexWarning)
                H = H0+E*HDC+I*HAC+B*Hz
                if return_states:
                    Eigen = eig(H)
                    order = numpy.argsort(Eigen[0])
                    EigenValues[:,i]=Eigen[0][order]
                    States[:,:,i] = Eigen[1][:,order]
                else:
                    Eigen = eigvals(H)
                    EigenValues[:,i]=numpy.sort(Eigen)
        if return_states:
            return EigenValues,States
        else:
            return EigenValues


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
