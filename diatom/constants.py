import scipy.constants
from numpy import pi

###############################################################################
# Bialkali Molecular Constants
###############################################################################
#Here are some starting dictionaries for various bialkali molecules. 
#References given, but check up to date if precision needed!

h = scipy.constants.h
muN = scipy.constants.physical_constants['nuclear magneton'][0]
bohr = scipy.constants.physical_constants['Bohr radius'][0]
eps0 = scipy.constants.epsilon_0
c = scipy.constants.c
DebyeSI = 3.33564e-30

# Most recent Rb87Cs133 Constants are given in the supplementary 
#of Gregory et al., Nat. Phys. 17, 1149-1153 (2021)
#https://www.nature.com/articles/s41567-021-01328-7
# Polarisabilities are for 1064 nm reported 
#in Blackmore et al., PRA 102, 053316 (2020)
#https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.053316
Rb87Cs133 = {"I1":1.5,
            "I2":3.5,
            "d0":1.225*DebyeSI,
            "binding":114268135.25e6*h,
            "Brot":490.173994326310e6*h,
            "Drot":207.3*h,
            "Q1":-809.29e3*h,
            "Q2":59.98e3*h,
            "C1":98.4*h,
            "C2":194.2*h,
            "C3":192.4*h,
            "C4":19.0189557e3*h,
            "MuN":0.0062*muN,
            "Mu1":1.8295*muN,
            "Mu2":0.7331*muN,
            "a0":2020*4*pi*eps0*bohr**3, #1064nm
            "a2":1997*4*pi*eps0*bohr**3, #1064nm
            "Beta":0}

#K41Cs133 values are from theory:
#Vexiau et al., Int. Rev. Phys. Chem. 36, 709-750 (2017)
#https://www.tandfonline.com/doi/full/10.1080/0144235X.2017.1351821
#Aldegunde et al., PRA 96, 042506 (2017)
#https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
K41Cs133 = {"I1":1.5,
            "I2":3.5,
            "d0":1.84*DebyeSI,
            "Brot":880.326e6*h,
            "Drot":0*h,
            "Q1":-0.221e6*h,
            "Q2":0.075e6*h,
            "C1":4.5*h,
            "C2":370.8*h,
            "C3":9.9*h,
            "C4":628*h,
            "MuN":0.0*muN,
            "Mu1":0.143*(1-1340.7e-6)*muN,
            "Mu2":0.738*(1-6337.1e-6)*muN,
            "a0":7.783e6*h, #h*Hz/(W/cm^2)
            "a2":0, #Not reported
            "Beta":0}#


#For K40Rb87:
#Brot, Q1, Q2 are from Ospelkaus et al., PRL 104, 030402 (2010)
#https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.104.030402
#d0 is from Ni et al., Science 322, 231-235 (2008)
#https://www.science.org/doi/10.1126/science.1163861
#a0, a2 are from Neyenhuis et al., PRL 109, 230403 (2012)
#https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.109.230403
#All other parameters are from Aldegunde et al., PRA 96, 042506 (2017)
#https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
K40Rb87 = { "I1":4,
            "I2":1.5,
            "d0":0.566*DebyeSI,
            "Brot":1113.950e6*h,
            "Drot":0*h,
            "Q1":0.45e6*h,
            "Q2":-1.41e6*h,
            "C1":-24.1*h,
            "C2":419.5*h,
            "C3":-48.2*h,
            "C4":-2028.8*h,
            "MuN":0.0140*muN,
            "Mu1":-0.324*(1-1321e-6)*muN,
            "Mu2":1.834*(1-3469e-6)*muN,
            "a0":5.53e-5*1e6*h, #h*Hz/(W/cm^2) #1064nm
            "a2":4.47e-5*1e6*h, #1064nm
            "Beta":0}


#For Na23K40
#Parameters from Will et al., PRL 116, 225306 (2016)
#https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.225306
#and from Aldegunde et al., PRA 96, 042506 (2017)
#https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
Na23K40 = { "I1":1.5,
            "I2":4,
            "d0":2.72*DebyeSI,
            "Brot":2.8217297e9*h,
            "Drot":0*h,
            "Q1":-0.187e6*h,
            "Q2":0.899e6*h,
            "C1":117.4*h,
            "C2":-97.0*h,
            "C3":-48.4*h,
            "C4":-409*h,
            "MuN":0.0253*muN,
            "Mu1":1.477*(1-624.4e-6)*muN,
            "Mu2":-0.324*(1-1297.4e-6)*muN,
            "a0":0*h, #Not reported
            "a2":0*h, #Not reported
            "Beta":0}


#For Na23Rb87
#Parameters from Guo et al., PRA 97, 020501(R) (2018)
#https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.020501
#and from Aldegunde et al., PRA 96, 042506 (2017)
#https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
Na23Rb87 = {"I1":1.5,
            "I2":1.5,
            "d0":3.2*DebyeSI,
            "Brot":2.0896628e9*h,
            "Drot":0*h,
            "Q1":-0.139e6*h,
            "Q2":-3.048e6*h,
            "C1":60.7*h,
            "C2":983.8*h,
            "C3":259.3*h,
            "C4":6.56e3*h,
            "MuN":0.001*muN,
            "Mu1":1.484*muN,
            "Mu2":1.832*muN,
            "a0":0*h, #Not reported
            "a2":0*h, #Not reported
            "Beta":0}