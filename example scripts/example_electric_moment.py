import numpy
import matplotlib.pyplot as pyplot
import diatomic.hamiltonian as hamiltonian
from diatomic.constants import Rb87Cs133
from numpy.linalg import eigh
import diatomic.calculate as calculate
import scipy.constants 

muN = scipy.constants.physical_constants['nuclear magneton'][0]

Nmax=4
H0,Hz,Hdc,Hac = \
    hamiltonian.build_hamiltonians(Nmax,Rb87Cs133,zeeman=True,Edc=True,ac=True)


I = 0 #W/m^2
E = numpy.linspace(0, 5e5, int(50)) #V/m
B = 181.5e-4 #T

H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I 
H = H.transpose(2,0,1)

energies, states = eigh(H)
#energies, states, labels = calculate.sort_by_state(energies, states, Nmax, Rb87Cs133)
energies, states = calculate.sort_smooth(energies, states)

d = calculate.electric_moment(states, Nmax, Rb87Cs133)

pyplot.figure(figsize=(5,4), dpi=400)
#i = numpy.where(labels[:,0]==0)[0]
pyplot.plot(E/1e5, d/Rb87Cs133['d0'])
#pyplot.ylim(-0.5, 1.0)
pyplot.xlim(0, 5)
pyplot.ylabel("Electric dipole moment ($\\mu/ \\mu_0$)")
pyplot.xlabel("Electric Field (kV cm$^{-1}$)")
pyplot.show()



