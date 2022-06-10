from matplotlib import pyplot,gridspec,colors,patches,collections
import numpy
import warnings
import diatom.calculate as calculate
import diatom.hamiltonian as hamiltonian
from scipy import constants

h = constants.h

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
    lc = collections.LineCollection(segments, array=z, cmap=cmap, norm=norm,#change
                                    linewidth=linewidth,zorder=1.25)

    ax.add_collection(lc)

    return lc

def export_energy(fname,energy,fields=None,labels=None,
                                headers=None,dp=6,format=None):
    ''' Export Energies in spreadsheet format.

    This exports the energy of the states for a calculation in a human-readable spreadsheet format.

    Currently only saves .csv files.

    Args:
        fname (string) - file name to save, appends .csv if not present.
        Energy (numpy.ndarray) - Energies to save

    Kwargs:
        Fields (numpy.ndarray) - Field variables used in calculation
        labels (numpy.ndarray) - labels for states
        headers (list of strings) - header for each of the labels in labels
        dp (float) - number of decimal places to use for output (default =6)
        format (list of strings) - list of formats passed to numpy.savetxt for labels
    '''
    # some input sanitisation, ensures that the fname includes an extension
    if fname[-4:]!=".csv":
        fname = fname+".csv"
    dp = int(numpy.round(dp))

    # check whether the user has given labels and headers or  just one
    lflag = False
    hflag = False

    if labels != None:
        labels = numpy.array(labels)
        lflag = True
    else:
        labels=[]

    if headers != None:
        hflag = True
    else:
        headers = []

    # all this is just checking whether there are headers and labels, just
    #labels, just headers or neither.

    if not hflag and lflag:
        warnings.warn("using default headers for labels",UserWarning)
        headers = ["Label {:.0f}".format(x) for x in range(len(labels[0,:]))]

    elif hflag and not lflag:
        warnings.warn("headers given without labels",UserWarning)
        headers =[]

    if len(headers) != labels.shape[0]:
        warnings.warn("Not enough headers given for chosen labels",UserWarning)
        headers = ["Label {:.0f}".format(x) for x in range(len(labels[:,0]))]

    # Now to write a string to make the output look nice. For simplicity we say
    # that all the  labels must be given to 1 dp
    if format==None:
        format = ["%.1f" for x in range(len(labels[:,0]))]

    # now just make the one for the main body of the output file. Specified by
    # the dp argument.
    if len(energy.shape)>1:
        format2 = ["%."+str(dp)+"f" for x in range(len(energy[:,0]))]
    else:
        format2 = ["%."+str(dp)+"f"]
    #numpy needs only one format argument
    format.extend(format2)

    headers =','.join(headers)


    headers = ','.join(["Labels" for l in range(labels.shape[0])])+",Energy (Hz)\n"+headers

    if type(fields) != type(None):
        energy = numpy.insert(energy,0,fields.real,axis=1)
        labels = numpy.insert(labels,0,[-1 for x in range(labels.shape[0])],axis=1)

    output = numpy.row_stack((labels,energy))
    numpy.savetxt(fname,output.T,delimiter=',',header = headers,fmt=format)

def export_state_comp(fname,Nmax,I1,I2,states,labels=None,
                                headers=None,dp=6,format=None):
    ''' function to export state composition in a human-readable format
    along the first row are optional headers and the labels for the basis States
    in the uncoupled basis.

    the user can supply optional labels for the states in a (x,y) list or array
    where y is the number of states and x is the number of unique labels, for
    instance a list of the N quantum  number for each state.

    they can also (optionally) supply a (x,1) list to include custom headers
    in the first row. If the labels kwarg is included and headers is not,
    then non-descriptive labels are used to ensure correct output.

    by default the output is given to 6 decimal places (truncated) this can be
    adjusted using the kwarg dp

    Args:
        fname (string) : the filename and path to save the output file
        Nmax (int/float) : the maximum value of N used in the calculation
        I1,I2 (float) : the nuclear spin quantum numbers of nucleus 1 and 2
        States (N,M) ndarray : eigenstates stored in an (N,M) ndarray, N is the
                                number of eigenstates. M is the number of basis
                                states.
    kwargs:
        labels (N,X) ndarray : ndarray containing X labels for each of the N states
        headers (X) ndarray-like : Ndarray-like containing descriptions of the labels
        dp (int) : number of decimal places to output the file to [default = 6]
        format (list) :  list of strings for formatting the headers. Defaults to 1 dp.

    '''

    # some input sanitisation, ensures that the fname includes an extension
    if fname[-4:]!=".csv":
        fname = fname+".csv"
    dp = int(numpy.round(dp))

    # check whether the user has given labels and headers or  just one
    lflag = False
    hflag = False

    if labels != None:
        labels = numpy.array(labels)
        lflag = True

    if headers != None:
        hflag = True
    else:
        headers = []


    # create labels for basis states from Generate_vecs
    # first step is to recreate the angular momentum operators
    N, I1,I2 = hamiltonian.generate_vecs(Nmax,I1,I2)

    # each basis state is an eigenstate of N^2, so N^2 is diagonal in our basis
    # with eigenvalues N(N+1)

    N2 = numpy.round([calculate.solve_quadratic(1,1,-1*x) for x in numpy.diag(hamiltonian.vector_dot(N,
                                                                N))],0).real

    # they are also eigenstates of Nz, I1z and I2z which are diagonal
    # in the basis that we constructed.

    MN = numpy.round(numpy.diag(N[2]),0).real
    M1 = numpy.round(numpy.diag(I1[2]),1).real
    M2 = numpy.round(numpy.diag(I2[2]),1).real

    # Now we create a list of each of the values in the right place
    state_list = ["({:.0f} : {:.0f} : {:.1f} : {:.1f})".format(N2[i],
                                    MN[i],M1[i],M2[i]) for i in range(len(MN))]
    # all this is just checking whether there are headers and labels, just
    #labels, just headers or neither.

    if not hflag and lflag:
        warnings.warn("using default headers for labels",UserWarning)
        headers = ["Label {:.0f}".format(x) for x in range(len(labels[0,:]))]

    elif hflag and not lflag:
        warnings.warn("headers given without labels",UserWarning)
        headers =[]

    if len(headers) != labels.shape[0]:
        warnings.warn("Not enough headers given for chosen labels",UserWarning)
        headers = ["Label {:.0f}".format(x) for x in range(len(labels[:,0]))]

    # Now to write a string to make the output look nice. For simplicity we say
    # that all the  labels must be given to 1 dp
    if format==None:
        format = ["%.1f" for x in range(len(headers))]

    # now just make the one for the main body of the output file. Specified by
    # the dp argument.
    format2 = ["%."+str(dp)+"f" for x in range(len(state_list))]

    #numpy needs only one format argument
    format.extend(format2)

    headers.extend(state_list)
    headers =','.join(headers)
    headers = ','.join(["Labels" for l in range(labels.shape[0])])+",States in (N:MN:M1:M2) basis\n"+headers
    states=numpy.transpose(states)#changde	
    output = numpy.insert(states.real,0,labels.real,axis=0)
    numpy.savetxt(fname,output.T,delimiter=',',header = headers,fmt=format)


def transition_plot(energies,states,gs,Nmax,I1,I2,TDMs=None,
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

    gray ='xkcd:lightgold'
    gray='#fddc5c'
    if col == None:
        cc=['#61e160','#fe2f4a','#155084']
        green=cc[0]
        red=cc[1]
        blue=cc[2]
        #green ='xkcd:darkgreen'
        #red ='xkcd:maroon'
        #blue ='xkcd:azure'

        col=[red,blue,green]

    if TDMs == None and (Nmax == None or I1 == None or  I2 == None):
        raise RuntimeError("TDMs  or Quantum numbers must be supplied")

    elif (Nmax == None or I1 == None or  I2 == None):
        TDMs = numpy.array(TDMs)
        dm = TDMs[0,:]
        dz = TDMs[1,:]
        dp = TDMs[2,:]
    elif TDMs == None:
        dm = numpy.round(calculate.transition_dipole_moment(Nmax,I1,I2,+1,states,gs),6)
        dz = numpy.round(calculate.transition_dipole_moment(Nmax,I1,I2,0,states,gs),6)
        dp = numpy.round(calculate.transition_dipole_moment(Nmax,I1,I2,-1,states,gs),6)

    if abs(pm)>1:
        pm = int(pm/abs(pm))

    widths = numpy.zeros(4)+1
    widths[-1] = 1.4

    fig.set_figheight(8)
    fig.set_figwidth(6)

    grid= gridspec.GridSpec(2,4,width_ratios=widths)

    N,MN = calculate.label_states_N_MN(states,Nmax,I1,I2)
    #find the ground state that the user has put in

    N0 = N[gs]

    energies = energies-energies[gs]
    lim =10

    l1 = numpy.where(N==N0)[0]
    l2 = numpy.where(N==N0+pm)[0]


    if minf == None:

        emin = numpy.amin(energies[l2])
        minf = prefactor*(emin)/h - Offset

    if maxf == None:

        emax = numpy.amax(energies[l2])
        maxf = prefactor*(emax)/h - Offset

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
        f =prefactor*(energies[l])/h 
        if l ==gs:
            ax0.plot([-lim,lim],[f,f],color='k',zorder=1.2)
        else:
            ax0.plot([-lim,lim],[f,f],color=gray,zorder=0.8)
    lbl = ['$\sigma_-$',"$\pi$","$\sigma_+$"]

    for j,axis in enumerate(ax):        
    #plotting for excited state
        for l in l2:
            f = prefactor*(energies[l])/h - Offset
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

    #fix the ROI to be 200 kHz around the state the user has chosen
    if gs == 0:
        ax0.set_ylim(-20e3*prefactor, 180e3*prefactor)
    else:
        ax0.set_ylim(-20e3*prefactor, 180e3*prefactor)

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

    Nz = len(numpy.where(dz[l2]!=0)[0])
    iz = 0

    deltax = (x1b-x1a)/(Nz+1)
    x0 = x1a+deltax

    disp = ax3.transData.transform((-lim,0))
    y1a = ax0.transData.inverted().transform(disp)[0]

    disp = ax3.transData.transform((lim,0))
    y1b = ax0.transData.inverted().transform(disp)[0]

    Np = len(numpy.where(dp[l2]!=0)[0])
    ip =0

    deltay = (y1b-y1a)/(Np+1)
    y0 = y1a+deltay

    disp = ax1.transData.transform((-lim,0))
    z1a = ax0.transData.inverted().transform(disp)[0]

    disp = ax1.transData.transform((lim,0))
    z1b = ax0.transData.inverted().transform(disp)[0]

    Nm = len(numpy.where(dm[l2]!=0)[0])
    im = 0

    deltaz = (z1b-z1a)/(Nm+1)
    z0 = z1a+deltaz
    f = prefactor*(energies)/h-Offset
    
    for j,d in enumerate(dz):
        #this block of code plots the dipole moments (or transition strengths
        if j<max(l2) and f[j]<maxf:
            if abs(d)>0 and j<max(l2):
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
                ax_bar.barh(f[j],numpy.abs(d),color=blue,height=1e4*prefactor)
    
            d=dp[j]
            if abs(d)>0 and j<max(l2):
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
                ax_bar.barh(f[j],numpy.abs(d),color=green,height=1e4*prefactor)
    
            d=dm[j]
            if abs(d)>0 and j<max(l2):
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
                ax_bar.barh(f[j],numpy.abs(d),color=red,height = 1e4*prefactor)

    #setup log axes for axis 4 (bar plots)
    if log:
        ax_bar.set_xscale('log')

        ax_bar.set_xticks([1e-6,1e-3,1])
        ax_bar.set_xticks([1e-5,1e-4,1e-2,1e-1],minor=True)

        ax_bar.set_xticklabels(["10$^{-6}$","10$^{-3}$","1"])
        ax_bar.set_xticklabels(["","","",""],minor=True)

    # now to rescale the other axes so that they have the same y scale
    ax1.set_ylim(minf-20e3*prefactor,maxf+20e3*prefactor)
    grid.set_height_ratios([(maxf-minf)+40e3*prefactor,200e3*prefactor])
    pyplot.subplots_adjust(hspace=0.1)
    grid.update()

    #add some axis labels
    ax0.set_ylabel("Energy/$h$ (kHz)")

    if Offset != 0:
        ax[0].set_ylabel("Energy/$h$ (kHz) - {:.3f} MHz".format(Offset+minf-20*1e3*prefactor))
    else:
        ax[0].set_ylabel("Energy/$h$ (Hz)")

    ax_bar.set_xlabel("TDM ($d_0$)")

