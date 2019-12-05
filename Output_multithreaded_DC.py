from diatom import Hamiltonian
from multiprocessing import Pool,cpu_count
from itertools import product
import numpy
import time
import os
from scipy import constants
from functools import partial
from sympy.physics.wigner import wigner_3j,wigner_9j
from sympy.physics.quantum.spin import Rotation

h = constants.h

def Solve(H0,Hext,var):
    H = H0 +Hext*var
    eigs = numpy.linalg.eig(H)
    order = numpy.argsort(eigs[0])
    eigvals = eigs[0][order]
    eigstates = eigs[1][:,order]
    eigvals = numpy.insert(eigvals,0,var)
    return eigvals,eigstates

def DC(Nmax,Const):
    ''' Generates the effect of the dc Stark shift for a rigid-rotor like
    molecule '''
    shape = numpy.sum(numpy.array([2*x+1 for x in range(0,Nmax+1)]))
    HDC = numpy.zeros((shape,shape),dtype= numpy.complex)
    HR = HDC.copy()
    Brot = Const['Brot']
    Drot = Const["Drot"]
    d0 = Const['d0']
    i =0
    j =0
    for N1 in range(0,Nmax+1):
        for M1 in range(N1,-(N1+1),-1):
            for N2 in range(0,Nmax+1):
                for M2 in range(N2,-(N2+1),-1):
                    if N1 == N2 and M1 == M2:
                        HR[i,j] += Brot*N1*(N1+1)-Drot*N1**2*(N1+1)**2+M1*h
                    HDC[i,j]=-d0*numpy.sqrt((2*N1+1)*(2*N2+1))*(-1)**(M1)*\
                    wigner_3j(N1,1,N2,-M1,0,M2)*wigner_3j(N1,1,N2,0,0,0)
                    j+=1
            j=0
            i+=1
    return HDC,HR

if __name__ =="__main__":
    import matplotlib.pyplot as pyplot

    cwd = os.path.dirname(os.path.abspath(__file__))
    #calculation constants
    Const = Hamiltonian.RbCs
    Nmax = 6
    Int = 0
    sort =True #do you want to do the sorting on output?

    then = time.time()
    HDC,H0 = DC(Nmax,Const)

    now = time.time()
    print("setup took {:.4f} seconds".format(now-then))
    then = now
    fn = partial(Solve,H0,HDC)
    #set up multiprocessing to leave one cpu core free (this stops OS crashes)
    pool = Pool(processes = cpu_count()-1)

    Fields = numpy.linspace(0,2000*1e2,500)

    out = numpy.zeros((HDC.shape[0]+1,Fields.shape[0]))
    out2 = numpy.zeros((HDC.shape[0],HDC.shape[0],Fields.shape[0]),
                            dtype='complex128')

    print("Starting to work...")
    then = time.time()
    x=0

    '''
    implement multi-processing to calculate all the fields simulataneously. It
    doesn't matter what order they are processed in so we can send it unordered
    this saves a small amount of memory and processor time, but we are short of
    both!
    '''

    for r in pool.imap_unordered(fn,Fields):
        out[:,x] = r[0].real
        out2[:,:,x] = r[1]
        x+=1

    now = time.time()
    print("all calculations took {:.4f} seconds".format(now-then))
    then = now

    order = numpy.argsort(out[0,:])
    out = out[:,order]
    out2 = out2[:,:,order]
    locs = range(out2.shape[0])
    #multiprocessing can't help us from here. So give the cores back to the OS.
    pool.close()
    pool.join()

    numpy.savetxt(cwd+"\\Unsorted\\RR_N{:.0f}_unsrt_test.csv".format(Nmax),out,delimiter=',')
    numpy.save(cwd+"\\Unsorted\\RR_N{:.0f}_states_unsrt_test".format(Nmax),out2)

    if sort:
        L0 = numpy.arange(out2.shape[0],dtype="int")
        ls = L0.copy()

        for i in range(1,len(Fields)):
            '''
            This loop sorts the eigenstates such that they maintain some
            continuity. Each eigenstate should be chosen to maximise the overlap
            with the previous.
            '''
            print("iteration {:.0f} of {:.0f}".format(i,len(Fields)))

            #calculate the overlap of the ith and jth eigenstates
            overlaps = numpy.einsum('ij,ik->jk',out2[:,:,i-1],out2[:,:,i])

            orig2 = out2[:,:,i].copy()
            orig1 = out[:,i].copy()

            numpy.argmax(numpy.abs(overlaps),axis=1,out=ls)

            for k in range(out2.shape[0]):
                l = ls[k]
                if l!=k:
                    out[k+1,i] = orig1[l+1].copy()
                    out2[:,k,i] = orig2[:,l].copy()

        now = time.time()
        print("output prep took {:.4f} seconds".format(now-then))
        then = now
        numpy.savetxt(cwd+"\\Sorted\\RR_N{:.0f}_2kV.csv".format(Nmax),out,delimiter=',')
        numpy.save(cwd+"\\Sorted\\RR_N{:.0f}_states_2kV".format(Nmax),out2)


    for i in range(1,out.shape[0]):
        pyplot.plot(out[0,:],out[i,:])
    pyplot.show()
