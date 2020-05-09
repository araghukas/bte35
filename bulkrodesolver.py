import numpy as np
import scipy.constants as const

from scipy.special import expit  # 'expit(x) = 1/(1+exp(-x))'; no warnings, unlike `np.exp`
from scipy.optimize import fmin
from scipy.interpolate import interp1d
from scipy.integrate import quad

from materials import Material

"""
The main references are:

    "Rode #1" --> Rode, D. L. "Electron mobility in direct-gap polar semiconductors."
                  Physical Review B 2.4 (1970): 1012.

    "Rode #2" --> Rode, D. L., and S. Knight. "Electron transport in GaAs."
                  Physical Review B 3.8 (1971): 2534.

    "Rode #3" --> Rode, D. L. "Electron transport in InSb, InAs, and InP."
                  Physical Review B 3.10 (1971): 3287.

See also:

    Rode, D. L. "Low-field transport in semiconductors."
    Semiconductors and Semimetals 10 (1975): 1-90.

    Kane, Evan O. "Band structure of indium antimonide."
    Journal of Physics and Chemistry of Solids 1.4 (1957): 249-261.

    Vurgaftman, I., J. R. Meyer, and L. R. Ram-Mohan.
    "Band parameters for III–V compound semiconductors and their alloys."
    Journal of applied physics 89.11 (2001): 5815-5875.

    Broido, D. A., and T. L. Reinecke.
    "Theory of thermoelectric power factor in quantum well and quantum wire superlattices."
    Physical review B 64.4 (2001): 045324.

    Pichanusakorn, Paothep, and Prabhakar Bandaru. "Nanostructured thermoelectrics."
    Materials Science and Engineering: R: Reports 67.2-4 (2010): 19-63.

"""


class RodeSolver(object):
    """
    Calculates non-equilibrium distribution function (solution of B.T.E.) using Rode's iteration method;
    Uses non-equilibrium distribution function to compute thermoelectric transport coefficients.
    """

    # temperature gradient (arbitrary, cancelled out)
    dTdx = 1e3  # [K/m]

    # field strength (arbitrary, cancelled out)
    eps_field = 1e4  # [V/m]

    def __init__(self, mat, T, Rc, n=None, p=None, num_k=10000, k_MIN=2e6):

        # assign material
        if isinstance(mat, Material):
            self.mat = mat
        else:
            raise TypeError("`mat` argument must be an instance of `materials.Material`")

        # set temperature
        self._T = T  # protecting this to keep things simple

        # threshold electron energy and wave number for optical phonon emission
        self.Epo = const.k * self.mat.Tpo
        self.kpo = self.k_CB(self.Epo)

        # polar optical phonon occupation number
        self.Npo = RodeSolver.bE(self.Epo, self.T)

        # dictionary of arrays
        self.SPACES = {}

        # some defaults
        self._p = p if p is not None else 0.  # [1/m^3] hole concentration
        self.k_MIN = k_MIN  # [1/m] minimum (magnitude of) wave number, effectively zero

        # maximum `k` value (set by `get_k_MAX` method)
        self.k_MAX = None  # [1/m] maximum wave number, effectively infinity

        # average energy (set by `E_J` method)
        self._EJ = None

        # dopant compensation ratio
        self._Rc = Rc

        # number of points in k-space
        self.num_k = num_k

        if n is not None:
            self._n = n  # [1/m^3] electron concentration
            self._Ef = self.calculate_Ef(n)  # [J] Fermi energy

            self.compute_spaces(n, self._p, Rc, num_k)
        else:
            self._n = None
            self._Ef = None

    # PROPERTIES
    # ----------
    @property
    def T(self):
        # temperature setting [K]
        return self._T

    @T.setter
    def T(self, newT):
        self._T = newT

        # update Fermi energy--assuming electron concentration somehow stays constant
        self._Ef = self.calculate_Ef(self.n)

        # update phonon occupation number
        self.Npo = RodeSolver.bE(self.Epo, newT)

        # update pre-computed arrays
        self.compute_spaces(self.n, self.p, self.Rc, self.num_k)

    @property
    def n(self):
        # electron concentration [1/m^3]
        return self._n

    @n.setter
    def n(self, new_n):
        self._n = new_n

        # update Fermi energy--assuming no temperature change has occurred
        self._Ef = self.calculate_Ef(new_n)

        # update hole concentration
        self._p = self.calculate_p(self._Ef)

        # update pre-computed arrays
        self.compute_spaces(new_n, self.p, self.Rc, self.num_k)

    @property
    def p(self):
        # hole concentration [1/m^3]
        return self._p

    @p.setter
    def p(self, new_p):
        self._p = new_p

        # update ionized impurity scattering rate
        self.SPACES['r_ii'] = self.r_ii(self.n, new_p, self.Rc)

    @property
    def Rc(self):
        # dopant compensation ratio
        return self._Rc

    @Rc.setter
    def Rc(self, newR):
        self._Rc = newR

        # update ionized impurity scattering rate
        self.SPACES['r_ii'] = self.r_ii(self.n, self.p, newR)

    @property
    def Ef(self):
        # No setter for this! Manipulate `T` or `n` instead. Keeping things simple.
        return self._Ef

    # MAIN ITERATOR FUNCTIONS
    # -----------------------
    def g_dist(self, i, xi):
        """
        Iterative solver of the B.T.E. with assumptions in Rode #3;
        Rode #3, Eq. (16)

        Parameters:
            i : (int)
                number of iterations to perform
            xi : (array)
                value of perturbation at each `k`

        Returns:
            gk : (numpy.array)
                the perturbation that applied to the equilibrium distribution
                function (the Fermi-Dirac distribution) at wave number `k`
        """
        # zeroth iteration is 0 for all `k`
        if i <= 0:
            return np.zeros(self.num_k)

        # pre-computed values
        fk = self.SPACES['f']
        fk_p = self.SPACES['f+']
        fk_m = self.SPACES['f-']

        lo_p = self.SPACES['lo+']
        lo_m = self.SPACES['lo-']
        li_p = self.SPACES['li+']
        li_m = self.SPACES['li-']

        # values over k+ or k- of the previous solution
        g_previous = self.g_dist(i - 1, xi)  # previous solution over all k
        g_prev = interp1d(self.SPACES['k'], g_previous, bounds_error=False, fill_value=0.)

        g_p = g_prev(self.SPACES['k+'])
        g_m = g_prev(self.SPACES['k-'])

        # numerator terms
        S_in_em = (self.Npo + fk) * li_m * g_m
        S_in_ab = (self.Npo + 1. - fk) * li_p * g_p

        # denominator terms
        S_out_em = (self.Npo + 1. - fk_m) * lo_m
        S_out_ab = (self.Npo + fk_p) * lo_p
        r_el = self.SPACES['r_el']

        return (S_in_em + S_in_ab - xi) / (S_out_em + S_out_ab + r_el)

    # TRANSPORT COEFFICIENTS
    # ----------------------
    def mu(self, i=30):
        """
        Electron drift mobility [m^2/V/s];
        Rode #3, Eq. (32)

        Formula is equivalent to: sigma / (e * n)
        """

        v = self.SPACES['v']
        k = self.SPACES['k']
        f = self.SPACES['f']
        dfdk = self.SPACES['dfdk']

        xi = -const.e * self.eps_field / const.hbar * dfdk
        g = self.g_dist(i, xi)

        I1 = np.trapz(g * v * k**2, x=k)
        I2 = np.trapz(f * k**2, x=k)

        return -1. / 3. / self.eps_field * I1 / I2

    def sigma(self, i=30):
        """
        Electrical conductivity (electron contribution only) [S/m];
        Derived from Broido (2001) Eqs. (14-17) or similar;
        equivalent to `n * e * mu`
        """
        v = self.SPACES['v']
        k = self.SPACES['k']
        dfdk = self.SPACES['dfdk']

        xi = -const.e * self.eps_field / const.hbar * dfdk
        g = self.g_dist(i, xi)

        I1 = np.trapz(g * v * k**2, x=k)
        return -const.e / (3. * np.pi**2 * self.eps_field) * I1

    def S(self, i=30):
        """
        Seebeck Coefficient [V/K]
        Rode #3, Eq. (A5)
        """
        dfdk = self.SPACES['dfdk']
        dEfdx = self.SPACES['dEfdx']
        E = self.SPACES['E']

        sigma = self.sigma(i)

        xi = -1. / const.hbar * dfdk * (dEfdx + (E - self.Ef) / self.T * self.dTdx)
        g = self.g_dist(i, xi)
        Jsc = self.J_e(g)

        return 1. / const.e * dEfdx / self.dTdx - Jsc / (sigma * self.dTdx)

    def kappa_e(self, i=30):
        """
        Open-circuit electronic thermal conductivity;
        """
        dfdk = self.SPACES['dfdk']
        dEfdx = self.SPACES['dEfdx']
        E = self.SPACES['E']

        xi = -1. / const.hbar * dfdk * (dEfdx + (E - self.Ef) / self.T * self.dTdx)
        g = self.g_dist(i, xi)
        J_Q_oc = self.J_Q(g)

        return J_Q_oc / self.dTdx

    # AVERAGES
    # --------
    def EJ(self):
        """
        The average energy of conduction electrons [J];
        This is kT times the integrals ratio in Rode #3, Eqs. (A5) and (A6)
        """
        k = self.SPACES['k']
        f0 = self.SPACES['f']
        E = self.SPACES['E']

        I1 = np.trapz(k**2 * f0 * (1. - f0) * E, x=k)
        I2 = np.trapz(k**2 * f0 * (1. - f0), x=k)

        return I1 / I2

    # CURRENT DENSITIES
    # -----------------
    def J_e(self, g):
        """
        Electrical current density [A/m^2];
        """
        k = self.SPACES['k']
        v = self.SPACES['v']

        j = 1. / (3. * const.pi**2) * np.trapz(k**2 * v * g, x=k)
        return -const.e * j

    def J_Q(self, g):
        """
        Electronic component of heat flux density [W/m^2];
        """
        k = self.SPACES['k']
        v = self.SPACES['v']
        E = self.E_CB(k)

        return 1. / (3. * const.pi**2) * np.trapz(k**2 * v * g * (E - self.Ef), x=k)

    # CONDUCTION BAND MODEL (KANE BANDS)
    # ---------------------------------
    def E_CB(self, k):
        """
        Sub-parabolic conduction band energy model;
        Rode #1, Eq. (3)
        """
        Eg = self.mat.get_Eg(self.T)

        E0 = const.hbar**2 * k**2 / 2. / const.m_e
        _alpha = self.alpha(k)
        return E0 + (_alpha - 1.) * Eg / 2.

    def k_CB(self, E):
        """
        Analytical inverse of `E_CB`
        """
        Eg = self.mat.get_Eg(self.T)
        meG = self.mat.get_meG(self.T)

        a = const.hbar**2 / 2. / const.m_e
        b = Eg / 2.
        c = 2. * const.hbar**2 / const.m_e * (const.m_e - meG) / (meG * Eg)

        # solve resulting equation by substituting x**2 = 1 - c*k**2, then use quadratic formula:
        x1 = (-b + np.sqrt(b**2 + 4. * a / c * (b + a / c + E))) / (2. * a / c)

        return np.sqrt((x1**2 - 1.) / c)

    def v_CB(self, k):
        """
        Group velocity of electrons;

        From Rode #1, Eq. (5), we have: 'dEdk = ℏ**2 * k / (m_e * d)',
        where 'd' is the output of `augmented_dos(k)`.

        The group velocity is defined as: 'v = dω/dk = 1/ℏ * dEdk'.

        Therefore, we have: 'v = ℏ * k / (m_e * d)'
        """
        return const.hbar * k / const.m_e / self.augmented_dos(k)

    def alpha(self, k):
        """
        The non-parabolic parameter;
        Rode #1, Eq. (4)
        """
        Eg = self.mat.get_Eg(self.T)
        meG = self.mat.get_meG(self.T)

        E0 = const.hbar**2 * k**2 / 2. / const.m_e
        return np.sqrt(1. + 4. * E0 * (const.m_e - meG) / meG / Eg)

    def psi_coeffs(self, k):
        """
        Coefficients of s- and p-type wave function components;
        Rode #1, Eqs. (8) and (9)

        see also: Kane (1957), Eq. (17)
        """
        a_coeff = np.sqrt(.5 + .5 / self.alpha(k))
        c_coeff = np.sqrt(1. - a_coeff**2)
        return a_coeff, c_coeff

    def augmented_dos(self, k):
        """
        Augmented density of states--used to scale effective mass;
        Rode #1, Eq. (6)
        """
        meG = self.mat.get_meG(self.T)

        _alpha = self.alpha(k)
        return meG * _alpha / (const.m_e + meG * (_alpha - 1.))

    # CARRIER CONCENTRATIONS
    # ----------------------

    def calculate_n(self, Ef):
        """
        Calculates electron concentration;
        Rode #3, Eq. (30)
        """
        k_MAX = self.get_k_MAX(Ef, self.T)
        ks = np.linspace(self.k_MIN, k_MAX, self.num_k)
        return 1. / const.pi**2 * np.trapz(ks**2 * self.fk(ks, Ef, self.T), x=ks)  # [1/m^3]

    # a simple description of the valence band using the D.O.S. effective mass
    def calculate_p(self, Ef):
        """
        Integrate hole distribution over energy to find concentration;
        """
        Eg = self.mat.get_Eg(self.T)

        # N.B. f(-E, -Ef, T) == [1 - f(E, Ef, T)]
        return quad(lambda E: self.fE(-E, -Ef, self.T) * self.g_VB(E), -Eg - 5. * const.e, -Eg)[0]

    def g_VB(self, E):
        """
        Valence band density of states; assuming parabolic bands;
        Pichanusakorn (2010), Eq. (16)
        """
        Eg = self.mat.get_Eg(self.T)

        if -Eg - E < 0:
            return 0

        return (1. / (2. * np.pi**2)
                * (2. * self.mat.mh_DOS / const.hbar**2)**(3 / 2) * (-Eg - E)**(1 / 2))

    # STATISTICS
    # ----------
    def calculate_Ef(self, n):  # `n` in [1/m^3] !!
        """
        Numerically inverts `calculate_n` to determine Fermi energy
        provide input `n` in [1/m^3]
        """
        guess = -self.mat.get_Eg(self.T) / 2.
        return fmin(lambda Ef: abs(np.log(self.calculate_n(Ef[0]) / n)),
                    x0=guess, maxiter=100, disp=False)[0]

    def calculate_Ei(self):
        """
        Finds `Ef` such that `n == p`
        """
        guess = self.mat.get_Eg(self.T) / 2.
        return fmin(lambda Ef: abs(self.calculate_n(Ef[0]) - self.calculate_p(Ef[0])),
                    x0=guess, maxiter=100, disp=False)[0]

    @staticmethod
    def bE(E, T):
        """
        The Bose-Einstein distribution, determines polar-optical phonon occupation number;
        Rode #3, Eq. (10)
        """
        return 1. / (np.exp(E / const.k / T) - 1.)

    @staticmethod
    def fE(E, Ef, T):
        """
        The Fermi-Dirac distribution--function of energy;
        Rode #3, Eq. (2)
        """
        return expit((Ef - E) / const.k / T)

    @staticmethod
    def dfdE(E, Ef, T):
        """
        Energy-derivative of the Fermi-Dirac distribution
        """
        return -RodeSolver.fE(E, Ef, T)**2 * (1 / const.k / T) * (1. / expit((Ef - E) / const.k / T) - 1.)

    def fk(self, k, Ef, T):
        """
        The Fermi-Dirac distribution--function of wave number;
        Rode #3, Eq. (2)
        """
        E = self.E_CB(k)
        return RodeSolver.fE(E, Ef, T)

    def dfdk(self, k, Ef):
        """
        k-derivative of Fermi-Dirac distribution
        """
        meG = self.mat.get_meG(self.T)
        dfdE = self.dfdE(self.E_CB(k), Ef, self.T)

        # apply chain rule: 'dfdk = dfdE * dEdk'
        return dfdE * (const.hbar**2 * k / const.m_e) * (1. + 1. / self.alpha(k) * (const.m_e / meG - 1.))

    # ELASTIC SCATTERING RATES
    # ------------------------
    def calculate_beta_sq(self):
        """
        Calculates square of inverse screening length for ionized impurities;
        Rode #3, Eq. (29)
        """
        k = self.SPACES['k']
        f = self.SPACES['f']

        factor = const.e**2 / (np.pi**2 * const.k * self.T * self.mat.eps_lo)
        integral = np.trapz(k**2 * f * (1. - f), x=k)
        return factor * integral

    def r_ac(self):
        """ acoustic deformation potential scattering relaxation rate """
        c = self.SPACES['c']
        d = self.SPACES['d']
        k = self.SPACES['k']
        rate_ac = (
                (const.k * self.T * self.mat.E1**2 * const.m_e * d * k)
                / (3. * np.pi * const.hbar**3 * self.mat.cl)
                * (3. - 8. * c**2 + 6. * c**4)
        )  # NOTE: Rode has an extra factor of `e**2` in this expression, because he inputs `E1` in units of eV
        return rate_ac

    def r_pe(self):
        """
        Piezoelectric scattering relaxation rate;
        Rode #3, Eq. (23)
        """
        c = self.SPACES['c']
        d = self.SPACES['d']
        k = self.SPACES['k']
        rate_pe = (
                (const.e**2 * const.k * self.T * self.mat.Pz**2 * const.m_e * d)
                / (6. * np.pi * const.hbar**3 * self.mat.eps_lo * k)
                * (3. - 6. * c**2 + 4. * c**4)
        )
        return rate_pe

    def r_ii(self, n, p, Rc):
        """
        Ionized impurity scattering relaxation rate;
        Rode #3, Eqs. (25-29)
        """
        c = self.SPACES['c']
        d = self.SPACES['d']
        k = self.SPACES['k']
        beta_sq = self.calculate_beta_sq()

        N = Rc * (n + p) + p  # density of ionized impurity scattering centres

        D = 1. + (2. * beta_sq * c**2 / k**2) + (3. * beta_sq**2 * c**4 / 4. / k**4)

        B = (
                (
                        (4. * k**2)
                        + (8. * (beta_sq + 2. * k**2) * c**2)
                        + (3. * beta_sq**2 + 6. * beta_sq * k**2 - 8. * k**4) * c**4 / k**2
                )
                / (beta_sq + 4. * k**2)
        )

        rate_ii = (
                (const.e**4 * N * const.m_e * d)
                / (8. * np.pi * self.mat.eps_lo**2 * const.hbar**3 * k**3)
                * (D * np.log(1 + 4. * k**2 / beta_sq) - B)
        )
        return rate_ii

    # INELASTIC SCATTERING RATES
    # --------------------------
    def lambda_inout(self, pm):
        """
        Rode's 'lambda' parameters related to inelastic in/out-scattering rates;
        Rode #3, Eqs. (17-20)
        """
        if pm not in ['+', '-']:
            raise ValueError("`pm` must be one of '+' or '-'")

        a = self.SPACES['a']
        c = self.SPACES['c']
        k = self.SPACES['k']

        k_pm = self.SPACES['k' + pm]
        a_pm = self.SPACES['a' + pm]
        c_pm = self.SPACES['c' + pm]
        d_pm = self.SPACES['d' + pm]

        A_pm = a * a_pm + (k_pm**2 + k**2) / (2. * k_pm * k) * c * c_pm
        beta_pm = (
                (const.e**2 * self.mat.wpo * const.m_e * d_pm)
                / (4. * np.pi * const.hbar**2 * k)
                * (1. / self.mat.eps_hi - 1. / self.mat.eps_lo)
        )

        lambda_in = beta_pm * (

                (k_pm**2 + k**2) / (2. * k_pm * k) * A_pm**2 * np.log(abs((k_pm + k) / (k_pm - k)))
                - A_pm**2
                - c**2 * c_pm**2 / 3.
        )

        lambda_out = beta_pm * (

                A_pm**2 * np.log(abs((k_pm + k) / (k_pm - k)))
                - (A_pm * c * c_pm)
                - (a * a_pm * c * c_pm)
        )

        if pm == '+':
            # absorption
            return lambda_in, lambda_out
        else:
            # no emission possible below optical phonon energy
            # hence 0 in/out scattering via emission in those cases (see `self.k_step` method)
            lambda_in[k < self.kpo] = 0.
            lambda_out[k < self.kpo] = 0.
            return lambda_in, lambda_out

    # UTILITIES
    # ---------
    def get_k_MAX(self, Ef, T):
        """
        Given the Fermi energy, determine `k_MAX` such that 'k**2 * f(k)' is approx. 0
        """
        E_MAX = 50. * const.k * T + Ef
        while E_MAX < 0.3 * const.e:
            dE = 5. * const.k * T
            E_MAX += dE  # increase maximum energy

        return self.k_CB(E_MAX)

    def k_step(self, ks, pm):
        """
        Calculates the wave vector of new state at plus/minus the optical phonon energy
        """
        if pm == '+':
            Es = self.E_CB(ks) + self.Epo  # E + ℏω
            return self.k_CB(Es)
        elif pm == '-':
            out = np.zeros(ks.shape)
            for i, k in enumerate(ks):
                newE = self.E_CB(k) - self.Epo  # E - ℏω
                if newE >= 0:
                    out[i] = self.k_CB(newE)
                else:
                    out[i] = -1  # use -1 as placeholder for no solution
            return out
        else:
            raise ValueError("`pm` should be either '+' or '-'")

    def compute_spaces(self, n, p, Rc, num_k):
        """
        Pre-computes a number of arrays that don't need to be updated during iteration process.
        --> There is some spaghetti going on here, but it's much faster this way.
        """
        # set maximum wave number
        self.k_MAX = self.get_k_MAX(self.Ef, self.T)

        # discrete linear space of `k` values
        ks = np.linspace(self.k_MIN, self.k_MAX, num_k, dtype=np.double)
        self.SPACES['k'] = ks
        self.SPACES['k+'] = self.k_step(ks, '+')
        self.SPACES['k-'] = self.k_step(ks, '-')

        # electron energies
        self.SPACES['E'] = self.E_CB(ks)

        # carrier velocities
        self.SPACES['v'] = self.v_CB(ks)

        # Fermi-Dirac (equilibrium) distribution function
        self.SPACES['f'] = self.fk(ks, self.Ef, self.T)
        self.SPACES['f+'] = self.fk(self.SPACES['k+'], self.Ef, self.T)
        self.SPACES['f-'] = self.fk(self.SPACES['k-'], self.Ef, self.T)

        # average energy of mobile electrons
        self._EJ = self.EJ()

        # space derivative of Fermi-Energy
        self.SPACES['dEfdx'] = 1. / self.T * (self.Ef - self._EJ) * self.dTdx

        # k-derivative of Fermi-Dirac distribution function
        self.SPACES['dfdk'] = self.dfdk(ks, self.Ef)

        # coefficients of wave function with s-type and p-type basis
        self.SPACES['a'], self.SPACES['c'] = self.psi_coeffs(ks)
        self.SPACES['a+'], self.SPACES['c+'] = self.psi_coeffs(self.SPACES['k+'])
        self.SPACES['a-'], self.SPACES['c-'] = self.psi_coeffs(self.SPACES['k-'])

        # augmented density of states (multiplies m_e to make effective mass)
        self.SPACES['d'] = self.augmented_dos(ks)
        self.SPACES['d+'] = self.augmented_dos(self.SPACES['k+'])
        self.SPACES['d-'] = self.augmented_dos(self.SPACES['k-'])

        # total elastic scattering relaxation rate
        self.SPACES['r_pe'] = self.r_pe()
        self.SPACES['r_ac'] = self.r_ac()
        self.SPACES['r_ii'] = self.r_ii(n, p, Rc)
        self.SPACES['r_el'] = self.SPACES['r_pe'] + self.SPACES['r_ac'] + self.SPACES['r_ii']

        # lambda parameters for inelastic in/out scattering rates
        self.SPACES['li+'], self.SPACES['lo+'] = self.lambda_inout('+')
        self.SPACES['li-'], self.SPACES['lo-'] = self.lambda_inout('-')
