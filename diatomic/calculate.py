from . import hamiltonian
import numpy
import scipy.constants
from sympy.physics.wigner import wigner_3j

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


def solve_quadratic(a,b,c):
    ''' Solve a quadratic equation

    for a*x^2+b*x+c=0 this is a simple function to solve the quadratic formula for x. returns the most
    positive value of x supported.

    Args:
        a,b,c (floats) - coefficients in quadratic

    Returns:
        x (float) - maximum value of x supported by equation

    '''
    x1 = (-b+numpy.sqrt((b**2)-(4*(a*c))))/(2*a)
    x2 = (-b-numpy.sqrt((b**2)-(4*(a*c))))/(2*a)
    return max([x1,x2])

def label_states_N_MN(states,Nmax,I1,I2,locs=None):
    ''' Label states by N,MN

    This function returns two lists: the input states labelled by N and MN
    in the order that they are provided. The returned numbers will only be good
    if the state is well -represented in the decoupled basis.

    Optionally can return the quantum  numbers for a subset if the locs kwarg
    is provided. Each element in the list locs corresponds to the index for the
    states to label.

    Args:

        States (Numpy.ndarray) - array of eigenstates, from linalg.eig
        Nmax (int) - maximum rotational state in calculation
        I1 , I2 (float) - nuclear spin quantum numbers

    kwargs:
        locs (list of ints) - list of indices of states to label

    Returns:
        Nlabels,MNlabels (list of ints) - list of values of N,MN

    '''
    if locs != None:
        states = states[:,locs]
    N, I1,I2 = hamiltonian.generate_vecs(Nmax,I1,I2)#change

    N2 = hamiltonian.vector_dot(N,N)#change
    Nz = N[2]

    Nlabels = numpy.einsum('ik,ij,jk->k',numpy.conj(states),N2,states)
    Nlabels = numpy.round([solve_quadratic(1,1,-1*x)  for x in Nlabels],0).real

    MNlabels = numpy.round(numpy.einsum('ik,ij,jk->k',
                                    numpy.conj(states),Nz,states),0).real

    return Nlabels,MNlabels

def label_states_I_MI(states,Nmax,I1,I2,locs = None):
    ''' Label states by I,MI

    This function returns two lists: the input states labelled by I and MI
    in the order that they are provided. The returned numbers will only be good
    if the state is well -represented in the decoupled basis.

    Optionally can return the quantum  numbers for a subset if the locs kwarg
    is provided. Each element in the list locs corresponds to the index for the
    states to label.

    Args:
        States (Numpy.ndarray) - array of eigenstates, from linalg.eig
        Nmax (int) - maximum rotational state in calculation
        I1 , I2 (float) - nuclear spin quantum numbers

    kwargs:
        locs (list of ints) - list of indices of states to label

    Returns:
        Ilabels,MIlabels (list of ints) - list of values of I,MI

    '''
    if locs != None:
        states = states[:,locs]

    #I = I1 + I2 change

    N, I1,I2 = hamiltonian.generate_vecs(Nmax,I1,I2)#change
    
    I = I1 + I2#change

    I2 = hamiltonian.vector_dot(I,I)#change

    Iz = I[2]

    Ilabels = numpy.einsum('ik,ij,jk->k',numpy.conj(states),I2,states)
    Ilabels = numpy.round([solve_quadratic(1,1,-1*x)  for x in Ilabels],1).real

    MIlabels = numpy.round(numpy.einsum('ik,ij,jk->k',
                                    numpy.conj(states),Iz,states),1).real

    return Ilabels,MIlabels

def label_states_F_MF(states,Nmax,I1,I2,locs=None):
    ''' Label states by F,MF

    This function returns two lists: the input states labelled by F and MF
    in the order that they are provided. The returned numbers will only be good
    if the state is well -represented in the decoupled basis.

    Optionally can return the quantum  numbers for a subset if the locs kwarg
    is provided. Each element in the list locs corresponds to the index for the
    states to label.

    Args:
        States (Numpy.ndarray) - array of eigenstates, from linalg.eig
        Nmax (int) - maximum rotational state in calculation
        I1 , I2 (float) - nuclear spin quantum numbers

    kwargs:
        locs (list of ints) - list of indices of states to label

    Returns:
        Flabels,MFlabels (list of ints) - list of values of F,MF

    '''

    if locs != None:
        states = states[:,locs]

    N, I1,I2 = hamiltonian.generate_vecs(Nmax,I1,I2)

    F = N + I1 + I2

    F2 = hamiltonian.vector_dot(F,F)

    Fz = F[2]

    Flabels = numpy.einsum('ik,ij,jk->k',numpy.conj(states),F2,states)
    Flabels = numpy.round([solve_quadratic(1,1,-1*x)  for x in Flabels],1).real

    MFlabels = numpy.round(numpy.einsum('ik,ij,jk->k',
                                    numpy.conj(states),Fz,states),1).real

    return Flabels,MFlabels

def dipole(Nmax,I1,I2,d,M):
    ''' Generates the induced dipole moment operator for a Rigid rotor.
    Expanded to cover state  vectors in the uncoupled hyperfine basis.

    Args:
        Nmax (int) - maximum rotational states
        I1,I2 (float) - nuclear spin quantum numbers
        d (float) - permanent dipole moment
        M (float) - index indicating the helicity of the dipole field

    Returns:
        Dmat (numpy.ndarray) - dipole matrix
    '''
    shape = numpy.sum(numpy.array([2*x+1 for x in range(0,int(Nmax+1))]))
    dmat = numpy.zeros((shape,shape), dtype=complex)
    i =0
    j =0
    for N1 in range(0,int(Nmax+1)):
        for M1 in range(N1,-(N1+1),-1):
            for N2 in range(0,int(Nmax+1)):
                for M2 in range(N2,-(N2+1),-1):
                    dmat[i,j]=d*numpy.sqrt((2*N1+1)*(2*N2+1))*(-1)**(M1)*\
                    wigner_3j(N1,1,N2,-M1,M,M2)*wigner_3j(N1,1,N2,0,0,0)
                    j+=1
            j=0
            i+=1

    shape1 = int(2*I1+1)

    shape2 = int(2*I2+1)

    dmat = numpy.kron(dmat,numpy.kron(numpy.identity(shape1),
                                                    numpy.identity(shape2)))

    return dmat

def transition_dipole_moment(Nmax,I1,I2,M,states,gs,locs=None):
    ''' calculate TDM between gs and States

    Function to calculate the Transition Dipole Moment between a state  gs
    and a range of states. Returns the TDM in units of the permanent dipole
    moment (d0).

    Args:
        Nmax (int): Maximum rotational quantum number in original calculations
        I1,I2 (float): nuclear spin quantum numbers
        M (float): Helicity of Transition, -1 = S+, 0 = Pi, +1 = S-
        States (numpy.ndarray): matrix for eigenstates of problem output from numpy.linalg.eig
        gs (int): index of ground state.

    kwargs:
        locs (list of ints): optional argument to calculate for subset of States, should be an
                array-like.

    Outputs:
        TDM(list of floats) - transition dipole moment between gs and States
    
    '''

    dipole_op = dipole(Nmax,I1,I2,1,M)
    if type(gs) == tuple:#change
        '''
        gs is usually produced by np.where() or np.nonzero(). And the outputs of those functions are
        tuples. This if condition ensures gs is an int.
        If there are more than one index in gs, only take the first one
        '''
        gs=gs[0][0]#change

    gs = numpy.conj(states[:,gs])
    if locs != None :
        states =  states[:,locs]

    tdm =  numpy.einsum('i,ij,jk->k',gs,dipole_op,states).real

    return tdm


def magnetic_moment(States, Nmax, Consts):
    '''Returns the magnetic moments of each eigenstate
    
    Args:
        States (numpy.ndarray): matrix for eigenstates of problem output from numpy.linalg.eig
        Nmax (int): Maximum rotational quantum number in original calculations
        Consts: Dictionary of constants for the molecular to be calculated
        
    '''
    
    muz = -1*hamiltonian.zeeman_ham(Nmax,Consts['I1'],Consts['I2'],Consts)
    
    mu =numpy.einsum('ijk,jl,ilk->ik',
            numpy.conjugate(States),muz,
            States)
    return mu


def electric_moment(States, Nmax, Consts):
    '''Returns the electric dipole moments of each eigenstate
    
    Args:
        States (numpy.ndarray): matrix for eigenstates of problem output from numpy.linalg.eig
        Nmax (int): Maximum rotational quantum number in original calculations
        Consts: Dictionary of constants for the molecular to be calculated
        
    '''
    
    dz = -1*hamiltonian.dc(Nmax,Consts['d0'],Consts['I1'],Consts['I2'])
    
    d =numpy.einsum('ijk,jl,ilk->ik',
            numpy.conjugate(States),dz,
            States)
    return d



def sort_by_state(energies, states, Nmax, Consts):
    '''Sort states by (N, M_F)_k where k is an index labelling states of
    a given N, MF in ascending energy. k is determined for the last element in 
    the array, this is usually the highest value of the varied parameter. 
    
    Args:
        energies (numpy.ndarray): eigenvalues output from numpy.linalg.eigh
        states (numpy.ndarray): eigenstates output from numpy.linalg.eigh
        Nmax: Maximum rotational quantum number in original calculations
        Consts: Molecular constants used in original calculations
        
    Output:
        energies_sorted, states_sorted: eigenenergies and eigenvalues sorted 
        by the ordering of states given in labels.
        labels: labels of the states in order
    '''
    #Find state labels
    N, I1,I2 = hamiltonian.generate_vecs(Nmax,Consts['I1'],Consts['I2'])
    
    N2 = hamiltonian.vector_dot(N,N)
    Nlabels = numpy.einsum('lik,ij,ljk->lk',numpy.conj(states),N2,states)
    Nlabels = numpy.round((-1+numpy.sqrt(1+4*1*Nlabels))/2).real
    
    F = N + I1 + I2
    Fz = F[2]
    MFlabels = numpy.round(numpy.einsum('lik,ij,ljk->lk',
                                        numpy.conj(states),Fz,states),1).real                            
                
    #Loop required to figure out k for each state                        
    labels = numpy.empty((numpy.shape(energies)[1],3))
    labels[:] = numpy.NaN
    for i in range(numpy.shape(energies)[1]):
        k = 0
        for j in range(numpy.shape(energies)[1]):
            if labels[j,0] == Nlabels[-1,i] and labels[j,1] == MFlabels[-1,i]:
                k+=1
        labels[i,:] = numpy.array([Nlabels[-1,i], MFlabels[-1,i], k])          
    
    #Now loop over energies and sort into order given by labels
    energies_sorted = numpy.zeros(numpy.shape(energies), dtype='complex')
    states_sorted = numpy.zeros(numpy.shape(states), dtype='complex')
    for i in range(numpy.shape(energies)[0]): #B field loop
        for j in  range(numpy.shape(energies)[1]): #State loop
            x = 0
            for k in range(numpy.shape(energies)[1]):
                N, MF = Nlabels[i,j], MFlabels[i,j] #N, MF for given index
                if N == labels[k,0] and MF == labels[k,1] and energies_sorted[i,k]==0 and x==0:
                    energies_sorted[i,k] = energies[i,j]
                    states_sorted[i,:,k] = states[i,:,j]
                    x += 1
                    
    return energies_sorted, states_sorted, labels


def sort_smooth(energy, states):
    ''' Sort states to remove false avoided crossings.

    This is a function to ensure that all eigenstates plotted change
    adiabatically, it does this by assuming that step to step the eigenstates
    should vary by only a small amount (i.e. that the  step size is fine) and
    arranging states to maximise the overlap one step to the next.

    Args:
        Energy (numpy.ndarray) : numpy.ndarray containing the eigenergies, as from numpy.linalg.eig
        States (numpy.ndarray): numpy.ndarray containing the states, in the same order as Energy
    Returns:
        Energy (numpy.ndarray) : numpy.ndarray containing the eigenergies, as from numpy.linalg.eig
        States (numpy.ndarray): numpy.ndarray containing the states, in the same order as Energy E[x,i] -> States[x,:,i]
    '''
    ls = numpy.arange(states.shape[2],dtype="int")
    number_iterations = len(energy[:,0])
    for i in range(1,number_iterations):
        '''
        This loop sorts the eigenstates such that they maintain some
        continuity. Each eigenstate should be chosen to maximise the overlap
        with the previous.
        '''
        #calculate the overlap of the ith and jth eigenstates
        overlaps = numpy.einsum('ij,ik->jk',
                                numpy.conjugate(states[i-1,:,:]),states[i,:,:])
        orig2 = states[i,:,:].copy()
        orig1 = energy[i,:].copy()
        #insert location of maximums into array ls
        numpy.argmax(numpy.abs(overlaps),axis=1,out=ls)
        for k in range(states.shape[2]):
            l = ls[k]
            if l!=k:
                energy[i,k] = orig1[l].copy()
                states[i,:,k] = orig2[:,l].copy()
    return energy, states
