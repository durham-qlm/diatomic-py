from diatom import Hamiltonian
import matplotlib.pyplot as pyplot
import scipy.constants
import numpy
import time
import os
from jqc import jqc_plot
from scipy import sparse

jqc_plot.plot_style('Large')
cwd = os.path.dirname(os.path.abspath(__file__))

B = 0
I = 0
E = 0
Beta = 0

h = scipy.constants.h
muN = scipy.constants.physical_constants['nuclear magneton'][0]
bohr = scipy.constants.physical_constants['Bohr radius'][0]
eps0 = scipy.constants.epsilon_0
c = scipy.constants.c
pi = numpy.pi
DebyeSI = 3.33564e-30
Constants =    {"IRb":1.5,
                "ICs":3.5,
                "d0":1.225*DebyeSI,
                "D0":114268135.25e6*h,
                "Brot":490.173994e6*h,
                "Drot":213*h,
                "QRb":-809.29e3*h,
                "QCs":59.98e3*h,
                "CRb":29.4*h,
                "CCs":196.8*h,
                "C3":192.4*h,
                "C4":19.019e3*h,
                "MuN":0.0062*muN,
                "MuRb":1.8295*muN,
                "MuCs":0.7331*muN,
                "a0":2020*4*pi*eps0*bohr**3,
                "a2":1997*4*pi*eps0*bohr**3,
                "Beta":0}

now = time.time()
Hams = Hamiltonian.Build_Hamiltonians(2,3/2,7/2,Constants)
print(time.time()-now)

Magnetic = numpy.linspace(0,300e-4,150)

Fields = (0,0,0) #Electric DC, Magnetic, Intensity
now = time.time()
Energy = Hamiltonian.Vary_magnetic(Hams,(0,0,0),Magnetic)
print(time.time()-now)

fig = pyplot.figure("Hyperfine")
ax = fig.add_subplot(111)
for i in range(len(Energy[:,0])):
    ax.plot(Magnetic*1e4,1e-6*Energy[i,:]/h,ls='--',color='k',zorder=1.)


for mF in [0,1,2,3,4,5,6,7]:
    if mF != 0:
    	Dat_m = numpy.genfromtxt(cwd+"\\TestData\\Jesus_Zeeman\\MFm0"+str(mF)+"\\results.dat")
    	Dat_p = numpy.genfromtxt(cwd+"\\TestData\\Jesus_Zeeman\\MFp0"+str(mF)+"\\results.dat")

    	for i in range(2,len(Dat_m[0,:])):
    		ax.plot(Dat_p[:,1]*1e4,Dat_p[:,i]*1e-3,color=jqc_plot.colours['grayblue'],zorder=0.5)
    		ax.plot(Dat_m[:,1]*1e4,Dat_m[:,i]*1e-3,color=jqc_plot.colours['grayblue'],zorder=0.5)
    else:
        Dat_p = numpy.genfromtxt(cwd+"\\TestData\\Jesus_Zeeman\\MFp0"+str(mF)+"\\results.dat")
        for i in range(2,len(Dat_p[0,:])):
            ax.plot(Dat_p[:,1]*1e4,Dat_p[:,i]*1e-3,color=jqc_plot.colours['grayblue'],zorder=0.5)

ax.set_xlabel("magnetic field (G)")
ax.set_ylabel("Energy (MHz)")

ax.set_ylim(975,985)

pyplot.show()
