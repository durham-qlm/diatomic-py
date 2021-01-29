from diatom.Hamiltonian import vector_dot
import numpy
from scipy.linalg import block_diag

'''
This module contains code that is incorrect beyond the diagonal elements of the
Hamiltonian in N,MN and is left purely for legacy purposes. In almost all
cicumstances the code in Hamiltonian is better.
'''


def tensor_nuclear(C3,I1,I2,N):
    ''' The tensor - nuclear spin spin interaction.

        This version uses cartesian angular momentum matrices and is incorrect.
        Correct version has off-diagonal terms in N. this only has the diagonals.
        It is close but only suitable where high-performance requirements replace
        accuracy requirements.

        Args:
            C3 (float): Tensor spin-spin coupling coefficient
            I1,I2,N (lists of numpy.ndarray): Angular momentum Vectors

        Returns:
            H (numpy.ndarray): Tensor spin-spin term
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
    ''' Legacy Quadrupole moment calculation

        This form of the quadrupole moments is only accurate on the diagonal.
        it comes from doi:10.1103/PhysRev.91.1403, which quotes the quadrupole interaction
        for KBr

        Args:
            Q (tuple of floats) : Tuple or list of the nuclear quadrupole moments as (Q1,Q2)
            I1,I2,N (lists of numpy.ndarray): Angular momentum Vectors

        Returns:
            Quad (numpy.ndarray) - Quadrupole term
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
#These are the functions that the user will use to generate any interesting maps
#obviously these can be added to by writing custom scripts but these should
# cover most needs

def Vary_magnetic(Hams,fields0,Bz,return_states = False):
    ''' Vary magnetic field

    find Eigenvalues (and optionally Eigenstates) of the total Hamiltonian
    This function works differently to the applied field ones. Because beta
    changes the matrix elements in the Hamiltonian we cannot simply
    multiply it through. Therefore we have to recalculate the matrix
    elements on each interation. This makes the function slower.

    Args:
        Hams: list or tuple of hamiltonians. Should all be the same size
        fields0: initial field conditions, allows for zeeman + Stark effects
        Bz: magnetic fields to iterate over
        return_states: Switch to return EigenStates as well as Eigenenergies

    Returns:
        energy:array of Eigenenergies, sorted from smallest to largest along the 0 axis
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
    ''' vary electric field DC

    find Eigenvalues (and optionally Eigenstates) of the total Hamiltonian
    This function works differently to the applied field ones. Because beta
    changes the matrix elements in the Hamiltonian we cannot simply
    multiply it through. Therefore we have to recalculate the matrix
    elements on each interation. This makes the function slower.

    Args:
        Hams: list or tuple of hamiltonians. Should all be the same size
        fields0: initial field conditions, allows for zeeman + Stark effects
        Ez: Electric fields to iterate over
        return_states: Switch to return EigenStates as well as Eigenenergies

    Returns:
        energy:array of Eigenenergies, sorted from smallest to largest along the 0 axis
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
    ''' vary intensity of off-resonant laser field

    find Eigenvalues (and optionally Eigenstates) of the total Hamiltonian
    This function works differently to the applied field ones. Because beta
    changes the matrix elements in the Hamiltonian we cannot simply
    multiply it through. Therefore we have to recalculate the matrix
    elements on each interation. This makes the function slower.

    Args:
        Hams: list or tuple of hamiltonians. Should all be the same size
        fields0: initial field conditions, allows for zeeman + Stark effects
        Intensity: Intensities to iterate over
        return_states: Switch to return EigenStates as well as Eigenenergies

    Returns:
        energy:array of Eigenenergies, sorted from smallest to largest along the 0 axis
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
    ''' vary polarisation of laser field

    find Eigenvalues (and optionally Eigenstates) of the total Hamiltonian
    This function works differently to the applied field ones. Because beta
    changes the matrix elements in the Hamiltonian we cannot simply
    multiply it through. Therefore we have to recalculate the matrix
    elements on each interation. This makes the function slower.

    Args:
        Hams: list or tuple of hamiltonians. Should all be the same size
        fields0: initial field conditions, allows for zeeman + Stark effects
        Angles: Polarisation angles to iterate over
        Molecule_pars: Nmax,I1,I2,a2, arguments to feed to regenerate the anisotropic Stark shift matrix.
        return_states: Switch to return EigenStates as well as Eigenenergies

    Returns:
        energy: array of Eigenenergies, sorted from smallest to largest along the 0 axis
        states: array of Eigenstates, sorted as in energy.

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
