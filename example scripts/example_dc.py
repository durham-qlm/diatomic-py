'''Generates a simple dc Stark plot for RbCs as an example
         !Takes a while to run due to high Nmax!
'''

import numpy
import matplotlib.pyplot as pyplot
import diatom.hamiltonian as hamiltonian
from diatom.constants import Rb87Cs133
from scipy.constants import h
from numpy.linalg import eigh

Nmax=4
H0,Hz,Hdc,Hac = \
    hamiltonian.build_hamiltonians(Nmax,Rb87Cs133,zeeman=True,Edc=True,ac=True)

I = 0 #W/m^2
E = numpy.linspace(0, 5, int(60))*1e5 #V/m
B = 0 #T

H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I 
H = H.transpose(2,0,1)

energies, states = eigh(H)

pyplot.figure(figsize=(5,4), dpi=400)
pyplot.plot(E*1e-5, energies*1e-6/h, color='k')
pyplot.ylim(-1000, 4000)
pyplot.xlim(0, 5)
pyplot.ylabel("Energy/h (MHz)")
pyplot.xlabel("Electric Field (kV/cm)")
pyplot.show()



