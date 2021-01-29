from diatom.Hamiltonian import *
from diatom.Calculate import *
from matplotlib import pyplot
import os
import diatom.Legacy


cwd = os.path.dirname(os.path.abspath(__file__))

Nmax =3

H0,Hz,HDC,HAC = Build_Hamiltonians(Nmax,RbCs,zeeman=True)

I1 = RbCs['I1']
I2 = RbCs['I2']

H = H0+181.5e-4*Hz

eigvals,States = numpy.linalg.eigh(H)

Nlabel, MNlabel = LabelStates_N_MN(States,Nmax,I1,I2,locs=None)

Flabel, MFlabel = LabelStates_F_MF(States,Nmax,I1,I2,locs=None)

file = cwd+"\\test.csv"
Pretty_Export_State_Comp(file,Nmax,I1,I2,States,labels=[Nlabel,MFlabel],headers=["N","MF"])

pyplot.imshow(H0.real,vmax=6.63e-34*2e3,vmin=-6.63e-34*2e3,cmap="RdBu_r")
pyplot.colorbar(extend='both')
pyplot.show()
