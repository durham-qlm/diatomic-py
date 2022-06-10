import numpy
from matplotlib import pyplot
import diatom.hamiltonian as hamiltonian
from diatom.constants import Rb87Cs133
from numpy.linalg import eigh
from scipy import constants
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

h = constants.h
pi = numpy.pi

Consts = Rb87Cs133
Nmax = 2
Consts['a0'] = 0 #Looking at transition energy so set isotropic component to zero
Consts['Beta'] = 1*pi/2


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


def colorline(x, y, z=None, cmap=pyplot.get_cmap('copper'), norm=pyplot.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,legend=False):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = numpy.array([z])
        
    z = numpy.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth)
    
    ax = pyplot.gca()
    ax.add_collection(lc)
    
    return lc




print("Building Hamiltonian...")
H0,Hz,Hdc,Hac = hamiltonian.build_hamiltonians(Nmax,Consts,zeeman=True,Edc=True,ac=True)

I = numpy.linspace(0, 10e7, int(2e2))
E = 0
B = 181.6e-4 #Set magnetic field range here

H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I 
H = H.transpose(2,0,1)

print("Diagonalizing Hamiltonian...")
energies, states = eigh(H)
                                       


#Plot the figure
pyplot.figure(figsize=(5,4), dpi=600)
for i in range(numpy.shape(energies)[1]):
    #colour each line as base grey
    pyplot.plot(I/1e7, (energies[:,i]-numpy.amin(energies))/(1e6*h), linestyle='solid', color='lightgray', zorder=0)
    
    #add colour on top to indicate the component that has N=1, MN=1, IRb=3/2, ICs=7/2
    cl=colorline(I/1e7,(energies[:,i]-numpy.amin(energies))/(1e6*h),z=abs(numpy.real(states[:,32,i])),cmap='RbCs_map_blue',norm=LogNorm(vmin=1e-2,vmax=1),linewidth=2.0)

   

pyplot.ylim(980.1, 980.6)
pyplot.xlim(0, 10)
pyplot.xlabel("Laser intensity (kW cm$^{-2}$)")
pyplot.ylabel("Transition energy from $N=0, M_F=5$,  $E$ / $h$ (MHz)")
pyplot.show()


