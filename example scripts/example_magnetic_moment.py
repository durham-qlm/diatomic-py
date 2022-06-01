import numpy
import matplotlib.pyplot as pyplot
import hamiltonian
from constants import Rb87Cs133
from numpy.linalg import eigh
import calculate
import scipy.constants 

muN = scipy.constants.physical_constants['nuclear magneton'][0]

Nmax=2
H0,Hz,Hdc,Hac = \
    hamiltonian.build_hamiltonians(Nmax,Rb87Cs133,zeeman=True,Edc=True,ac=True)


I = 0 #W/m^2
E = 0 #V/m
B = numpy.linspace(1, 600, int(60))*1e-4 #T

H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I 
H = H.transpose(2,0,1)

energies, states = eigh(H)
energies, states, labels = calculate.sort_by_state(energies, states, Nmax, Rb87Cs133)

mu = calculate.magnetic_moment(states, Nmax, Rb87Cs133)

pyplot.figure(figsize=(5,4), dpi=400)
i = numpy.where(labels[:,0]==0)[0]
pyplot.plot(B*1e4, numpy.real(mu[:,i])/muN)
pyplot.ylim(6, 2)
pyplot.xlim(0, 600)
pyplot.ylabel("Magnetic moment ($\\mu/ \\mu_N$)")
pyplot.xlabel("Magnetic field (G)")
pyplot.show()



