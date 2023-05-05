'''Generates a simple Zeeman plot for RbCs as an example'''

import numpy
import matplotlib.pyplot as pyplot
import diatomic.hamiltonian as hamiltonian
from diatomic.constants import Rb87Cs133
from scipy.constants import h
from numpy.linalg import eigh

Nmax=2
H0,Hz,Hdc,Hac = \
    hamiltonian.build_hamiltonians(Nmax,Rb87Cs133,zeeman=True,Edc=True,ac=True)

I = 0 #W/m^2
E = 0 #V/m
B = numpy.linspace(1, 300, int(60))*1e-4 #T

H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I 
H = H.transpose(2,0,1)

energies, states = eigh(H)

pyplot.figure(figsize=(5,4), dpi=400)
pyplot.plot(B*1e4, energies*1e-6/h, color='k')
pyplot.ylim(-1.0, 1.0)
pyplot.xlim(0, 300)
pyplot.ylabel("Energy/h (MHz)")
pyplot.xlabel("Magnetic Field (G)")
pyplot.show()



