import scipy.constants as const
import numpy as np

"""
See `bulkrodesolver.py` for bibliography. 

This file contains hard-coded material objects for InSb, InAs, and InP
as described in Rode #3

Everything is stored in S.I. units.
"""


class Material(object):
    """
    template for the material object; see instances below
    """
    DEFAULT_NUM = 0  # for numbering unnamed materials

    def __init__(self):
        # a name for the material
        self.name = None

        # basic valence band description
        # ------------------------------
        self.mh_DOS = None  # [kg] DOS effective mass of holes

        # parameters from TABLE 1. in Rode #3
        # -----------------------------------
        self.me0 = None  # [kg] Gamma valley effective mass
        self.Eg0 = None  # [J] effective mass energy gap at 0 K
        self.slope_Eg0 = None  # [J/K] energy-gap temperature coefficient
        self.eps_lo = None  # [epsilon_0] low-frequency dielectric permittivity
        self.eps_hi = None  # [epsilon_0] high-frequency dielectric permittivity
        self.Tpo = None  # [K] polar-phonon Debye temperature
        self.E1 = None  # [J] acoustic-deformation potential
        self.cl = None  # [N/m^2] longitudinal elastic constant
        self.ct = None  # [N/m^2] transverse elastic constant
        self.Pz = None  # [1] piezoelectric coefficient

        # Kane band parameters (from Vurgaftman)
        # --------------------------------------
        self.Ep = None  # used to calculate `Psq`
        self.Eso = None  # [J] spin-orbit splitting energy
        self.F = None  # [1] a band parameter

        # Varshni parameters for Eg(T) (from Vurgaftman)
        # ----------------------------------------------
        self.Varshni_alpha = None  # [J/K]
        self.Varshni_beta = None  # [K]

        # calculated parameters
        # ---------------------
        self.wpo = None  # [Hz] polar-optical phonon frequency

        # flags
        # -----
        self.meG_expression = 'Vurgaftman'  # specifies equation used for effective mass temperature-dependence
        self.Eg_expression = 'Vurgaftman'  # specifies equation used for Band gap temperature-dependence

    def __str__(self):
        return "<materials.Material object: '{}'>".format(self.name)

    def modified(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                raise ValueError("{} has not attribute {}"
                                 .format(self, key))
        return self

    @classmethod
    def create(cls, **params):
        mat = cls()

        # copy over parameters from input dictionary
        if 'name' not in params:
            mat.name = 'Unnamed' + str(Material.DEFAULT_NUM)
            Material.DEFAULT_NUM += 1

        for key, value in params.items():
            if hasattr(mat, key):
                mat.__setattr__(key, value)
            else:
                print("unrecognized parameter '{}' = `{}` for material '{}'"
                      .format(key, value, mat.name))

        # print which (if any) are left blank
        for key, value in mat.__dict__.items():
            if value is None and key != 'wpo':
                print("parameter `{}` is `None` for material '{}'".format(key, mat.name))

        # calculate parameters not provided directly
        mat.wpo = const.k * mat.Tpo / const.hbar

        # set this to zero if wasn't provided so things don't break
        if mat.mh_DOS is None:
            mat.mh_DOS = 0.

        return mat

    def get_meG(self, T):
        """
        temperature-dependent Gamma-valley electron effective mass;

        NOTE: I am not quite sure what Rode's '𝒫' parameter is in Eqs. (33) and (35).

              '𝒫'-squared should have units of energy, which leads me to believe it's
              the same as 'Ep' defined in Vurgaftman (2001) Eq. (2.4).

              The default effective mass expression will therefore be the more modern:
              Vurgaftman (2001) Eq. (2.15)
        """
        if self.meG_expression == 'Vurgaftman':
            Eg = self.get_Eg(T)
            Ep = self.Ep
            Eso = self.Eso
            F = self.F
            return const.m_e / ((1. + 2. * F) + Ep * (Eg + 2. * Eso / 3.) / Eg / (Eg + Eso))
        elif self.meG_expression == '33':
            # Use Rode #3, Eq. (33)
            Eg = self.get_Eg(T)
            return const.m_e / (1. + self.Ep / Eg)
        elif self.meG_expression == '35':
            # Use Rode #3, Eq. (35)
            Eg = self.get_Eg(T)
            return const.m_e / (1. + 1 / 3 * self.Ep * (2. / Eg + 1. / (Eg + self.Eso)))
        else:
            raise ValueError("unrecognized or invalid `meG_expression` '{}'"
                             .format(self.meG_expression))

    def get_Eg(self, T):
        """
        temperature-dependent effective mass band gap;
        Rode #3, equation (34)
        """
        if self.Eg_expression in ['Vurgaftman', 'Varshni']:
            a = self.Varshni_alpha
            b = self.Varshni_beta
            return self.Eg0 - a * T**2 / (T + b)
        elif self.Eg_expression == '34':
            # Use Rode #3, Eq. (34)
            return self.Eg0 - self.slope_Eg0 * T
        else:
            raise ValueError("unrecognized or invalid `Eg_expression` '{}'"
                             .format(self.Eg_expression))


# INSTANCES REPRESENTING THE MATERIAL MODELS IN RODE #3
# -----------------------------------------------------
InSb = Material.create(
    name='InSb',

    mh_DOS=0.43 * const.m_e,  # http://www.ioffe.ru/SVA/NSM/Semicond/InSb/bandstr.html

    me0=0.0155 * const.m_e,
    Eg0=0.265 * const.e,
    slope_Eg0=0.97 * const.e * 1e-4,
    eps_lo=17.64 * const.epsilon_0,
    eps_hi=15.75 * const.epsilon_0,
    Tpo=274.,
    E1=9.5 * const.e,
    cl=7.89 * 1e10,
    ct=2.42 * 1e10,
    Pz=0.027,  # Rode (1975)

    Varshni_alpha=0.32 * 1e-3 * const.e,
    Varshni_beta=170,

    Ep=23.3 * const.e,
    Eso=0.81 * const.e,
    F=-0.23
)

InAs = Material.create(
    name='InAs',

    mh_DOS=0.41 * const.m_e,  # http://www.ioffe.ru/SVA/NSM/Semicond/InAs/bandstr.html

    me0=0.025 * const.m_e,
    Eg0=0.46 * const.e,
    slope_Eg0=0.69 * const.e * 1e-4,
    eps_lo=14.54 * const.epsilon_0,
    eps_hi=12.25 * const.epsilon_0,
    Tpo=337.,
    E1=11.5 * const.e,
    cl=9.98 * 1e10,
    ct=3.14 * 1e10,
    Pz=0.017,  # Rode (1975)

    Varshni_alpha=0.276 * 1e-3 * const.e,
    Varshni_beta=93,

    Eso=0.39 * const.e,
    Ep=21.5 * const.e,
    F=-2.90
)

InP = Material.create(
    name='InP',

    mh_DOS=0.6 * const.m_e,  # http://www.ioffe.ru/SVA/NSM/Semicond/InP/bandstr.html

    me0=0.072 * const.m_e,
    Eg0=1.42 * const.e,
    slope_Eg0=0.41 * const.e * 1e-4,
    eps_lo=12.38 * const.epsilon_0,
    eps_hi=9.55 * const.epsilon_0,
    Tpo=497.,
    E1=14.5 * const.e,
    cl=12.10 * 1e10,
    ct=3.65 * 1e10,
    Pz=0.013,  # Rode (1975)

    Varshni_alpha=0.363 * 1e-3 * const.e,
    Varshni_beta=162,

    Eso=0.11 * const.e,  # Given in Rode #3 pg. 3296; Vurgaftman's value is `0.108` [eV]
    Ep=20.7 * const.e,
    F=-1.31
)

GaAs = Material.create(
    # there parameters are from Rode #1, Rode (1975), and Vurgaftman (2001)
    name='GaAs',

    mh_DOS=0.53 * const.m_e,  # http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/bandstr.html

    me0=0.0655 * const.m_e,
    Eg0=1.58 * const.e,
    slope_Eg0=1.2 * const.e * 1e-4,
    eps_lo=12.90 * const.epsilon_0,
    eps_hi=10.92 * const.epsilon_0,
    Tpo=420.,
    E1=8.6 * const.e,  # Rode (1975)
    cl=14.0 * 1e10,  # Rode (1975)
    ct=4.89 * 1e10,  # Rode (1975)
    Pz=0.052,  # Rode (1975)

    Varshni_alpha=0.5405 * 1e-3 * const.e,
    Varshni_beta=204,

    Eso=0.341 * const.e,  # Vurgaftman (2001)
    Ep=28.8 * const.e,  # Vurgaftman (2001)
    F=-1.94  # Vurgaftman (2001)
)


# BULK INTRINSIC CARRIER CONCENTRATIONS FROM LITERATURE
# ------------------------------------------------

# although there is a `calculate_Ei` method in `RodeSolver`,
# these are for replicating Rode's results as closely as possible


def ni_InSb_Hrostowski(T):
    """
    intrinsic carrier concentration of InSb above 200 K;
    taken from Rode #3 Ref 33:

         Hrostowski, H. J., et al. "Hall effect and conductivity of InSb."
         Physical Review 100.6 (1955): 1672.
    """

    return np.sqrt(
        3.6e29 * T**3 * np.exp(-0.26 * const.e / const.k / T)
    ) * 1e6  # [m^-3]


def ni_InAs_Folberth(T):
    """
    intrinsic carrier concentration of InAs;
    taken from Rode #3, Ref 48:

        Folberth, O. G., O. Madelung, and H. Weiss.
        "Die elektrischen Eigenschaften von Indiumarsenid II."
        Zeitschrift für Naturforschung A 9.11 (1954): 954-958.
    """
    alpha = -4.5e-4  # band gap temperature coefficient [ev/K]
    Eg = (0.47 + alpha * T) * const.e  # [J] band gap
    m_eff = 0.10  # [1] effective mass

    return np.sqrt(
        2.4e31 * T**3 * m_eff**3 * np.exp(-Eg / const.k / T)
    ) * 1e6  # [m^-3]


def ni_InP_Folberth(T):
    """
    intrinsic carrier concentration of InP;
    taken from Rode #3, Ref 65

        Folberth, O. G., and H. Weiss.
        "Herstellung und elektrische Eigenschaften von InP und GaAs."
        Zeitschrift für Naturforschung A 10.8 (1955): 615-619.
    """
    Eg = 1.34 * const.e

    return np.sqrt(
        7.e31 * T**3 * np.exp(-Eg / const.k / T)
    ) * 1e6  # [m^-3]
