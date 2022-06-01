'''Generates plot showing available transitions with transition dipole 
moments as shown in Fig.5 of arXiv:2205.05686

Note, this uses the transition_plot function from the plotting.py module!
'''

from numpy.linalg import eigh
import plotting as plotting
import hamiltonian as hamiltonian
from constants import Rb87Cs133
import matplotlib.pyplot as pyplot

Nmax=4
H0,Hz,Hdc,Hac = \
    hamiltonian.build_hamiltonians(Nmax,Rb87Cs133,zeeman=True,Edc=True,ac=True)

I = 0 #W/m^2
E = 0 #V/m
B = 181.5e-4 #T

H = H0+Hz*B+Hdc*E+Hac*I 

energies, states = eigh(H)

fig = pyplot.figure()
plotting.transition_plot(energies,states,0,Nmax,Rb87Cs133['I1'],Rb87Cs133['I2'],Offset=980,prefactor=1e-6, maxf=0.8)
pyplot.show()
