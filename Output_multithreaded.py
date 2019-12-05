from diatom import Hamiltonian
from multiprocessing import Pool,cpu_count
from itertools import product
import numpy
import time
import os

from functools import partial

def Solve(H0,Hext,var):
    H = H0 +Hext*var
    eigs = numpy.linalg.eig(H)
    order = numpy.argsort(eigs[0])
    eigvals = eigs[0][order]
    eigstates = eigs[1][:,order]
    eigvals = numpy.insert(eigvals,0,var)
    return eigvals,eigstates

def Overlap(A,B,n):
    i,j = n
    state0 = A[:,i]
    state1 = B[:,j]
    return numpy.abs(numpy.dot(state1,state0)),i,j

if __name__ =="__main__":
    import matplotlib.pyplot as pyplot

    cwd = os.path.dirname(os.path.abspath(__file__))
    #calculation constants
    Const = Hamiltonian.RbCs
    Nmax = 5
    B = 181.5*1e-4 #T
    E = 0*1e2 #V/cm
    Int = 10*1e7 #kW/cm^2
    Const['Beta']= numpy.deg2rad(90)#54.735610317245345685)
    sort =True #do you want to do the sorting on output?

    then = time.time()
    H0,Hz,HDC,HAC = Hamiltonian.Build_Hamiltonians(Nmax,Const,
                                            zeeman=True,EDC=True,AC=True)

    H0 = H0 + Hz*B + HDC*E
    now = time.time()
    print("setup took {:.4f} seconds".format(now-then))
    then = now
    fn = partial(Solve,H0,HAC)
    #set up multiprocessing to leave one cpu core free (this stops OS crashes)
    pool = Pool(processes = cpu_count()-1)

    Fields = numpy.linspace(0,Int,250)

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

    numpy.savetxt(cwd+\
    "\\AC Stark Data\\Unsorted\\multithread_N{:.0f}_unsrt_test.csv".format(Nmax),
                                                        out,delimiter=',')
    numpy.save(cwd+\
    "\\AC Stark Data\\Unsorted\\multithread_N{:.0f}_states_unsrt_test".format(Nmax),
                                                        out2)

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
        numpy.savetxt(cwd+\
        "\\AC Stark Data\\Sorted\\N{:.0f}_B{:.0f}_E{:.0f}.csv".format(Nmax,
                                        numpy.rad2deg(Const['Beta']),E*1e-2),
                                        out,delimiter=',')
        numpy.save(cwd+\
        "\\AC Stark Data\\Sorted\\N{:.0f}_B{:.0f}_E{:.0f}_states".format(Nmax,
                                        numpy.rad2deg(Const['Beta']),E*1e-2),
                                        out2)

    for i in range(1,out.shape[0]):
        pyplot.plot(out[0,:],out[i,:])
    pyplot.show()
