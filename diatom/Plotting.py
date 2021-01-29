from matplotlib import pyplot,gridspec,colors,patches
import numpy
import os
from diatom import Calculate
import warnings
from scipy import constants

h = constants.h

cwd = os.path.dirname(os.path.abspath(__file__))

def make_segments(x, y):
    ''' segment x and y points

    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array

    Args:
        x,y (numpy.ndarray -like ) - points on lines

    Returns:
        segments (numpy.ndarray) - array of numlines by points per line by 2

    '''

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)

    return segments

def colorline(x, y, z=None, cmap=pyplot.get_cmap('copper'),
                norm=pyplot.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,
                legend=False,ax=None):
    '''Plot a line shaded by an extra value.


    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width

    Args:
        x,y (list-like): x and y coordinates to plot

    kwargs:
        z (list): Optional third parameter to colour lines by
        cmap (matplotlib.cmap): colour mapping for z
        norm (): Normalisation function for mapping z values to colours
        linewidth (float): width of plotted lines (default =3)
        alpha (float): value of alpha channel (default = 1)
        legend (Bool): display a legend (default = False)
        ax (matplotlib.pyplot.axes): axis object to plot on

    Returns:
        lc (Collection) - collection of lines
        
    '''
    if ax == None:
        ax = pyplot.gca()

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = numpy.array([z])

    z = numpy.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                    linewidth=linewidth,zorder=1.25)

    ax.add_collection(lc)

    return lc

def TDM_plot(energies,States,gs,Nmax,I1,I2,TDMs=None,
            pm = +1, Offset=0,fig=pyplot.gcf(),
            log=False,minf=None,maxf=None,prefactor=1e-3,col=None):

    ''' Create a TDM plot

    this function plots a series of energy levels and their transition dipole
    moments from a given ground state. In this version a lot of the plotting style
    is fixed.

    Args:
        energies (numpy.ndarray) - array of energy levels
        states (numpy.ndarray) - array of states corresponding to energies such that E[i] -> States[:,i]
        gs (int) - index for ground state of interest
        Nmax (int) - maximum rotational quantum number to include
        I1, I2 (float) - nuclear spins of nuclei 1 and 2
    Kwargs:
        TDMs (list of numpy.ndarray) - optional precomputed transition dipole moments in [sigma-,pi,sigma+] order
        pm (float) - flag for if the transition increases or decreases N (default = 1)
        Offset (float) - yaxis offset (default = 0)
        fig (matplotlib.pyplot.figure) - figure object to draw on
        log (bool) - use logarithmic scaling for TDM plots
        minf (float) - minimum frequency to show
        maxf (float) - maximum frequency to show
        prefactor (float) - scaling factor for all energies
        col (list) - list of colours for lines (must be at least length 3 )

    '''

    gray ='xkcd:grey'
    if col == None:
        green ='xkcd:darkgreen'
        red ='xkcd:maroon'
        blue ='xkcd:azure'

        col=[red,blue,green]

    if TDMs == None and (Nmax == None or I1 == None or  I2 == None):
        raise RuntimeError("TDMs  or Quantum numbers must be supplied")

    elif (Nmax == None or I1 == None or  I2 == None):
        TDMs = numpy.array(TDMs)
        dm = TDMs[0,:]
        dz = TDMs[1,:]
        dp = TDMs[2,:]
    elif TDMs == None:
        dm = numpy.round(Calculate.TDM(Nmax,I1,I2,+1,States,gs),6)
        dz = numpy.round(Calculate.TDM(Nmax,I1,I2,0,States,gs),6)
        dp = numpy.round(Calculate.TDM(Nmax,I1,I2,-1,States,gs),6)

    if abs(pm)>1:
        pm = int(pm/abs(pm))

    widths = numpy.zeros(4)+1
    widths[-1] = 1.4

    fig.set_figheight(8)
    fig.set_figwidth(6)

    grid= gridspec.GridSpec(2,4,width_ratios=widths)

    N,MN = Calculate.LabelStates_N_MN(States,Nmax,I1,I2)
    #find the ground state that the user has put in

    N0 = N[gs]

    gs_E = energies[gs]
    lim =10

    l1 = numpy.where(N==N0)[0]

    min_gs = prefactor*numpy.amin(energies[l1]-gs_E)/h
    max_gs = prefactor*numpy.amax(energies[l1]-gs_E)/h

    l2 = numpy.where(N==N0+pm)[0]


    if minf ==None:

        emin = numpy.amin(energies[l2])
        minf = 10e4

        f = prefactor*(emin-gs_E)/h - Offset
        minf = min([minf,f])

    if maxf ==None:

        emax = numpy.amax(energies[l2])
        maxf = 0

        f = prefactor*(emax-gs_E)/h - Offset
        maxf = max([maxf,f])

    if pm == 1:
        ax0 = fig.add_subplot(grid[1,:-1])
        ax = []
        for j in range(3):
            if j ==0:
                ax.append(fig.add_subplot(grid[0,j],zorder=1))
            else:
                ax.append(fig.add_subplot(grid[0,j],sharey=ax[0],zorder=1))

    elif pm == -1:
        ax0 = fig.add_subplot(grid[0,:-1])
        ax = []
        for j in range(3):
            if j ==0:
                ax.append(fig.add_subplot(grid[1,j],zorder=1))
            else:
                ax.append(fig.add_subplot(grid[1,j],sharey=ax[0],zorder=1))


    #plotting the energy levels for ground state
    for l in l1:
        f =prefactor*(energies[l]-gs_E)/h #- Offset
        if l ==gs:
            ax0.plot([-lim,lim],[f,f],color='k',zorder=1.2)
        else:
            ax0.plot([-lim,lim],[f,f],color=gray,zorder=0.8)
    lbl = ['$\sigma_-$',"$\pi$","$\sigma_+$"]

    for j,axis in enumerate(ax):
    #plotting for excited state
        for l in l2:
            f = prefactor*(energies[l]-gs_E)/h - Offset
            if dz[l]!=0 and j==1:
                axis.plot([-lim,lim],[f,f],color=blue,zorder=1.2)
            elif dp[l] !=0 and j ==2:
                axis.plot([-lim,lim],[f,f],color=green,zorder=1.2)
            elif dm[l] !=0 and j ==0:
                axis.plot([-lim,lim],[f,f],color=red,zorder=1.2)
            else:
                axis.plot([-lim,lim],[f,f],color=gray,zorder=0.8)
        if j ==0 :
            axis.tick_params(labelbottom=False,bottom=False,which='both')
        else:
            axis.tick_params(labelleft=False,left=False,labelbottom=False,
                        bottom=False,which='both')
        axis.set_xlim(-lim,lim)
        axis.set_title(lbl[j],color=col[j])

    # set the ticks so that only the left most has a frequency/energy axis
    # and none have an x axis

    ax0.tick_params(labelbottom=False,bottom=False,which='both')
    ax0.set_xlim(-lim,lim)

    #add the bar plot axis
    ax_bar = fig.add_subplot(grid[0,-1],sharey = ax[0])
    ax_bar.tick_params(labelleft=False,left=False, which='both')


    #fix the ROI to be 300 kHz around the state the user has chosen
    ax0.set_ylim(min_gs,max_gs)
    f = prefactor*(energies-gs_E)/h-Offset

    #normalise function, returns a number between 0 and 1
    Norm = colors.LogNorm(vmin=1e-3,vmax=1,clip=True)
    #how thick should a line be?
    max_width = 2

    #setting where and how far apart the lines should all be in data coords

    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]

    disp = ax2.transData.transform((-lim,0))
    x1a = ax0.transData.inverted().transform(disp)[0]

    disp = ax2.transData.transform((lim,0))
    x1b = ax0.transData.inverted().transform(disp)[0]

    Nz = len(numpy.where(dz!=0)[0])
    iz = 0

    deltax = (x1b-x1a)/(Nz+1)
    x0 = x1a+deltax

    disp = ax3.transData.transform((-lim,0))
    y1a = ax0.transData.inverted().transform(disp)[0]

    disp = ax3.transData.transform((lim,0))
    y1b = ax0.transData.inverted().transform(disp)[0]

    Np = len(numpy.where(dp!=0)[0])
    ip =0

    deltay = (y1b-y1a)/(Np+1)
    y0 = y1a+deltay

    disp = ax1.transData.transform((-lim,0))
    z1a = ax0.transData.inverted().transform(disp)[0]

    disp = ax1.transData.transform((lim,0))
    z1b = ax0.transData.inverted().transform(disp)[0]

    Nm = len(numpy.where(dm!=0)[0])
    im = 0

    deltaz = (z1b-z1a)/(Nm+1)
    z0 = z1a+deltaz

    for j,d in enumerate(dz):
        #this block of code plots the dipole moments (or transition strengths)
        if abs(d)>0:
            width = max_width*Norm(3*numpy.abs(d)**2)
            x = x0 +iz*deltax
            # makes sure that the line is perfectly vertical in display coords
            disp = ax0.transData.transform((x,0))
            x2 = ax2.transData.inverted().transform(disp)[0]

            p = patches.ConnectionPatch((x,0),(x2,f[j]),coordsA='data',coordsB='data',
                                            axesA=ax0,axesB=ax2,zorder=5,color='k',
                                            lw=width) #line object
            ax2.add_artist(p) # add line to axes
            iz+=1
            #bar plot for transition strengths. Relative to spin-stretched TDM
            ax_bar.barh(f[j],numpy.abs(d),color=blue,height=5)

        d=dp[j]
        if abs(d)>0:
            width = max_width*Norm(3*numpy.abs(d)**2)
            y= y0 +ip*deltay
            # makes sure that the line is perfectly vertical in display coords

            disp = ax0.transData.transform((y,0))
            y2 = ax3.transData.inverted().transform(disp)[0]

            p = patches.ConnectionPatch((y,0),(y2,f[j]),coordsA='data',coordsB='data',
                                            axesA=ax0,axesB=ax3,zorder=5,color='k',
                                            lw=width) #line object
            ax3.add_artist(p)
            ip+=1
            #bar plot for transition strengths. Relative to spin-stretched TDM
            ax_bar.barh(f[j],numpy.abs(d),color=green,height=5)

        d=dm[j]
        if abs(d)>0:
            width = max_width*Norm(3*numpy.abs(d)**2)
            z = z0 +im*deltaz
            # makes sure that the line is perfectly vertical in display coords

            disp = ax0.transData.transform((z,0))
            z2 = ax1.transData.inverted().transform(disp)[0]

            p = patches.ConnectionPatch((z,0),(z2,f[j]),coordsA='data',coordsB='data',
                                            axesA=ax0,axesB=ax1,zorder=5,color='k',
                                            lw=width)#line object
            ax1.add_artist(p)
            im +=1
            #bar plot for transition strengths. Relative to spin-stretched TDM
            ax_bar.barh(f[j],numpy.abs(d),color=red,height = 5)

    #setup log axes for axis 4 (bar plots)
    if log:
        ax_bar.set_xscale('log')

        ax_bar.set_xticks([1e-6,1e-3,1])
        ax_bar.set_xticks([1e-5,1e-4,1e-2,1e-1],minor=True)

        ax_bar.set_xticklabels(["10$^{-6}$","10$^{-3}$","1"])
        ax_bar.set_xticklabels(["","","",""],minor=True)

    # now to rescale the other axes so that they have the same y scale
    ax1.set_ylim(minf-20,maxf+20)
    grid.set_height_ratios([(maxf-minf)+40,300])
    pyplot.subplots_adjust(hspace=0.1)
    grid.update()

    #add some axis labels
    ax0.set_ylabel("Energy/$h$ (kHz)")

    if Offset != 0:
        ax[0].set_ylabel("Energy/$h$ (kHz) - {:.1f} MHz".format(Offset))
    else:
        ax[0].set_ylabel("Energy/$h$ (Hz)")

    ax_bar.set_xlabel("TDM ($d_0$)")

if __name__ == '__main__':
    from diatom import Hamiltonian,Calculate


    H0,Hz,HDC,HAC = Hamiltonian.Build_Hamiltonians(3,Hamiltonian.RbCs,zeeman=True)

    eigvals,eigstate = numpy.linalg.eigh(H0+181.5e-4*Hz)

    TDM_plot(eigvals,eigstate,1,
    Nmax = 3,I1 = Hamiltonian.RbCs['I1'], I2 = Hamiltonian.RbCs['I2'],
    Offset=980e3,prefactor=1e-3)

    fig = pyplot.figure(2)

    loc = 0
    TDM_pi = Calculate.TDM(3,Hamiltonian.RbCs['I1'],Hamiltonian.RbCs['I2'],0,eigstate,loc)
    TDM_Sigma_plus = Calculate.TDM(3,Hamiltonian.RbCs['I1'],Hamiltonian.RbCs['I2'],-1,eigstate,loc)
    TDM_Sigma_minus = Calculate.TDM(3,Hamiltonian.RbCs['I1'],Hamiltonian.RbCs['I2'],+1,eigstate,loc)


    TDMs =[TDM_Sigma_minus,TDM_pi,TDM_Sigma_plus]

    TDM_plot(eigvals,eigstate,loc,3,Hamiltonian.RbCs['I1'],Hamiltonian.RbCs['I2'],Offset=980e3,fig=fig)

    pyplot.show()
