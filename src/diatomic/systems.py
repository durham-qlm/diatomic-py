import scipy.constants
from diatomic.operators import HalfInt

pi = scipy.constants.pi
h = scipy.constants.h
muN = scipy.constants.physical_constants["nuclear magneton"][0]
bohr = scipy.constants.physical_constants["Bohr radius"][0]
eps0 = scipy.constants.epsilon_0
DebyeSI = 3.33564e-30  # C m


class SingletSigmaMolecule:
    """
    A container to store all the necessary constants of a molecule. This can be passed
    around functions that need to know molecular constants to build hamiltonians, for
    example.
    """

    def __init__(
        self,
        Ii: tuple[int | HalfInt, int | HalfInt] = (0, 0),
        Nmax: int = 1,
        Brot: float = 1e9 * h,
        Drot: float = 0.0,
        Ci: tuple[float, float] = (0.0, 0.0),
        C4: float = 0.0,
        C3: float = 0.0,
        Qi: tuple[float, float] = (0.0, 0.0),
        d0: float = 1 * DebyeSI,
        MuN: float = 0.0,
        Mui: tuple[float, float] = (0.0, 0.0),
        a02: dict[float, tuple[float, float]] = {},
    ):
        # Nuclear Spin magnitudes
        self.Ii = Ii

        # Maximum rotational basis components considered
        self.Nmax = Nmax

        # Rotational Constants
        self.Brot = Brot
        self.Drot = Drot

        # Hyperfine Constants
        self.Ci = Ci  # Scalar Spin-Rotation
        self.C4 = C4  # Scalar Spin-Spin
        self.C3 = C3  # Tensor Spin-Spin
        self.Qi = Qi  # Quadrupole

        # Electric Dipole Moment (DC Stark)
        self.d0 = d0

        # Magnetic Dipole Moments (Zeeman)
        self.MuN = MuN
        self.Mui = Mui

        # Polarisability, function of wavelength (AC Stark)
        self.a02 = a02  # (iso, aniso)

    @classmethod
    def from_preset(cls, str_name):
        """
        Loads the molecule class from a preset of constants defined in the `presets`
        dictionary.
        """
        if str_name in cls.presets:
            return cls(**cls.presets[str_name])
        else:
            raise KeyError("Preset name does not exist.")

    presets = {
        "RigidRotor": {
            "Ii": (0, 0),
            "d0": 1 * DebyeSI,
            "Brot": 1e9 * h,
            "Drot": 0,
            "Qi": (0, 0),
            "Ci": (0, 0),
            "C3": 0,
            "C4": 0,
            "MuN": 0.01 * muN,
            "Mui": (0, 0),
        },
        "Testing": {
            "Ii": (HalfInt(of=3), 1),
            "d0": 1,
            "Brot": 1,
            "Drot": 1,
            "Qi": (1, 1),
            "Ci": (1, 1),
            "C3": 1,
            "C4": 1,
            "MuN": 1,
            "Mui": (1, 1),
            "a02": {1: (1, 1)},
        },
        "Rb87Cs133": {
            # Most recent Rb87Cs133 Constants are given in the supplementary
            # of Gregory et al., Nat. Phys. 17, 1149-1153 (2021)
            # https://www.nature.com/articles/s41567-021-01328-7
            # Polarisabilities are for 1064 nm reported
            # in Blackmore et al., PRA 102, 053316 (2020)
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.053316
            "Ii": (HalfInt(of=3), HalfInt(of=7)),
            "d0": 1.225 * DebyeSI,
            "Brot": 490.173994326310e6 * h,
            "Drot": 207.3 * h,
            "Qi": (-809.29e3 * h, 59.98e3 * h),
            "Ci": (29.4 * h, 196.8 * h),
            "C3": 192.4 * h,
            "C4": 19.0189557e3 * h,
            "MuN": 0.0062 * muN,
            "Mui": (1.8295 * muN, 0.7331 * muN),
            "a02": {
                1064: (
                    2020 * 4 * pi * eps0 * bohr**3,
                    1997 * 4 * pi * eps0 * bohr**3,
                ),
                1065: (
                    1825 * 4 * pi * eps0 * bohr**3,
                    1981 * 4 * pi * eps0 * bohr**3,
                ),
                817: (
                    443 * 4 * pi * eps0 * bohr**3,
                    -2816 * 4 * pi * eps0 * bohr**3,
                ),
            },
        },
        "K41Cs133": {
            # K41Cs133 values are from theory:
            # Vexiau et al., Int. Rev. Phys. Chem. 36, 709-750 (2017)
            # https://www.tandfonline.com/doi/full/10.1080/0144235X.2017.1351821
            # Aldegunde et al., PRA 96, 042506 (2017)
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
            "Ii": (HalfInt(of=3), HalfInt(of=7)),
            "d0": 1.84 * DebyeSI,
            "Brot": 880.326e6 * h,
            "Drot": 0 * h,
            "Qi": (-0.221e6 * h, 0.075e6 * h),
            "Ci": (4.5 * h, 370.8 * h),
            "C3": 9.9 * h,
            "C4": 628 * h,
            "MuN": 0.0 * muN,
            "Mui": (0.143 * (1 - 1340.7e-6) * muN, 0.738 * (1 - 6337.1e-6) * muN),
        },
        "K40Rb87": {
            # For K40Rb87:
            # Brot, Q1, Q2 are from Ospelkaus et al., PRL 104, 030402 (2010)
            # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.104.030402
            # d0 is from Ni et al., Science 322, 231-235 (2008)
            # https://www.science.org/doi/10.1126/science.1163861
            # a0, a2 are from Neyenhuis et al., PRL 109, 230403 (2012)
            # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.109.230403
            # All other parameters are from Aldegunde et al., PRA 96, 042506 (2017)
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
            "Ii": (HalfInt(of=4), HalfInt(of=3)),
            "d0": 0.566 * DebyeSI,
            "Brot": 1113.950e6 * h,
            "Drot": 0 * h,
            "Qi": (0.45e6 * h, -1.41e6 * h),
            "Ci": (-24.1 * h, 419.5 * h),
            "C3": -48.2 * h,
            "C4": -2028.8 * h,
            "MuN": 0.0140 * muN,
            "Mui": (-0.324 * (1 - 1321e-6) * muN, 1.834 * (1 - 3469e-6) * muN),
        },
        "Na23K40": {
            # For Na23K40
            # Parameters from Will et al., PRL 116, 225306 (2016)
            # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.225306
            # and from Aldegunde et al., PRA 96, 042506 (2017)
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
            "Ii": (HalfInt(of=3), HalfInt(of=8)),
            "d0": 2.72 * DebyeSI,
            "Brot": 2.8217297e9 * h,
            "Drot": 0 * h,
            "Qi": (-0.187e6 * h, 0.899e6 * h),
            "Ci": (117.4 * h, -97.0 * h),
            "C3": -48.4 * h,
            "C4": -409 * h,
            "MuN": 0.0253 * muN,
            "Mui": (1.477 * (1 - 624.4e-6) * muN, -0.324 * (1 - 1297.4e-6) * muN),
        },
        "Na23Rb87": {
            # Parameters from Guo et al., PRA 97, 020501(R) (2018)
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.020501
            # and from Aldegunde et al., PRA 96, 042506 (2017)
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
            "Ii": (HalfInt(of=3), HalfInt(of=3)),
            "d0": 3.2 * DebyeSI,
            "Brot": 2.0896628e9 * h,
            "Drot": 0 * h,
            "Qi": (-0.139e6 * h, -3.048e6 * h),
            "Ci": (60.7 * h, 983.8 * h),
            "C3": 259.3 * h,
            "C4": 6.56e3 * h,
            "MuN": 0.001 * muN,
            "Mui": (1.484 * muN, 1.832 * muN),
        },
        "Na23Cs133": {
            # from Aldegunde et al., PRA 96, 042506 (2017)
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.042506
            "Ii": (HalfInt(of=3), HalfInt(of=7)),
            "d0": 4.53 * DebyeSI,  # Theory value
            "Brot": 0.962e9 * h,
            "Drot": 0 * h,
            "Qi": (-0.097e6 * h, 0.150e6 * h),
            "Ci": (14.2 * h, 854.5 * h),
            "C3": 105.6 * h,
            "C4": 3941.8 * h,
            "MuN": 0.0 * muN,  # No value
            "Mui": (1.478 * (1 - 639.2e-6) * muN, 0.738 * (1 - 6278.7e-6) * muN),
        },
    }
