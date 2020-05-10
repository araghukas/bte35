import numpy as np
import scipy.constants as const

from scipy.integrate import quad
from scipy.optimize import fmin
from scipy.special import iv, kv

from fdint import fdk

from materials import Material
from _bessel_roots import alphas


def I0(x):
    # modified Bessel function of 1st kind of order 0
    return iv(0, x)


def I1(x):
    # modified Bessel function of 1st kind of order 1
    return iv(1, x)


def K0(x):
    # modified Bessel function of 2nd kind of order 1
    return kv(0, x)


def K1(x):
    # modified Bessel function of 2nd kind of order 1
    return kv(1, x)


class NWRTAsolver(object):
    """
        Calculates non-equilibrium distribution function (solution of B.T.E.) using
        the relaxation time approximation (RTA);
        Uses non-equilibrium distribution function to compute thermoelectric transport coefficients.

        Bands are assumed to be parabolic.

        Only lowest sub-band is occupied (so explicit scattering rates can be used)

        Everything in S.I. units!
    """

    # temperature gradient (arbitrary, cancelled out)
    dTdx = 1e3  # [K/m]

    # field strength (arbitrary, cancelled out)
    eps_field = 1e4  # [V/m]

    def __init__(self, mat, T, R, n=None, p=None, n_subs=20):

        if isinstance(mat, Material):
            self.mat = mat
        else:
            raise TypeError("`mat` argument must be an instance of `materials.Material`")

        # set temperature [k]
        self._T = T

        # calculate effective mass [kg]
        self.meG = self.mat.get_meG(T)

        # calculate maximum wave vector (integration limit)
        self.k_max = np.sqrt(2. * self.meG * 2. * const.e) / const.hbar

        # set nanowire radius [m]
        self._R = R

        # number of sub-bands
        self.n_subs = n_subs

        # calculate confinement energy [J]
        self.E_lns_CB = self.E_conf(self.meG, idxs=[i for i in range(n_subs)])
        self.E_lns_VB = self.E_conf(self.mat.mh_DOS, idxs=[i for i in range(n_subs)])

        # optical phonon energy
        self.Epo = const.k * self.mat.Tpo
        self.kpo = np.sqrt(2. * self.meG * self.Epo) / const.hbar

        # calculate optical phonon occupation number
        self.Npo = NWRTAsolver.bE(self.Epo, T)

        # set electron concentration [m^-3]
        self._p = 0. if p is None else p
        if n is not None:
            self.n = n  # [1/m^3] electron concentrations
        else:
            self._n = None
            self._Ef = None

    @property
    def Ef(self):
        # No setter for `Ef`! Manipulate `n` or `T` instead.
        return float(self._Ef)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, newT):
        self._T = newT

        # update temperature dependent values
        self.meG = self.mat.get_meG(newT)
        E_max = self.Ef + 200 * const.k * self.T
        self.k_max = np.sqrt(2. * self.meG * E_max) / const.hbar
        self.Npo = NWRTAsolver.bE(self.Epo, newT)
        self._Ef = self.calculate_Ef(self.n, newT)  # adjust `Ef` for same electron concentration
        self.E_lns_CB = self.E_conf(self.meG, idxs=[i for i in range(self.n_subs)])
        self.E_lns_VB = self.E_conf(self.mat.mh_DOS, idxs=[i for i in range(self.n_subs)])

    @property
    def R(self):
        # No setter for `R`! Create another solver.
        return self._R

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, new_n):
        self._n = new_n

        # update `n`-dependent values
        self._Ef = self.calculate_Ef(new_n, self.T)
        self._p = self.calculate_p(self.Ef, self.T)

        # update `k_max`
        E_max = self.Ef + 10. * const.k * self.T
        self.k_max = np.sqrt(2. * E_max * self.meG) / const.hbar

    @property
    def p(self):
        # No setter for `p`!
        return self._p

    # TRANSPORT
    # ---------
    def sigma(self):
        sigma_ = 0.
        for j in range(self.n_subs):
            sigma_ += self.sigma_j(j)
        return sigma_

    def S(self):
        sum_ = 0.
        for j in range(self.n_subs):
            sum_ += self.sigma_j(j) * self.S_j(j)
        sigma_ = self.sigma()
        S_ = sum_ / sigma_
        return S_

    def kappa_e(self):
        kappa_e_ = 0.
        for j in range(self.n_subs):
            kappa_e_ += self.kappa_e_j(j)
        return kappa_e_

    # TRANSPORT (EACH BAND)
    # ---------------------
    def sigma_j(self, j):
        Ef = self.Ef
        T = self.T
        c = -2. * const.e**2 / (np.pi**2 * self.R**2) / const.hbar
        sigma_ = c * (
            quad(lambda k: self.v_CB(k) * self.tau(j, k) * self.dfdk(j, k, Ef, T),
                 0, self.k_max, points=[self.kpo])[0]
        )
        return sigma_

    def S_j(self, j):
        Ef = self.Ef
        T = self.T
        c = -1. / T / const.e
        S_ = c * (self.EJ(j) - Ef)
        return S_

    def kappa_e_j(self, j):
        Ef = self.Ef
        T = self.T
        S_j = self.S_j(j)
        c = 2. / (np.pi * self.R)**2 / const.hbar
        i1 = quad(lambda k: (
                (self.E_CB(j, k) - self.Ef) * self.v_CB(k) * self.tau(j, k) * self.dfdk(j, k, Ef, T)),
                  0, self.k_max, points=[self.kpo])[0]
        i2 = quad(lambda k: (
                (self.E_CB(j, k) - self.Ef)**2 * self.v_CB(k) * self.tau(j, k) * self.dfdk(j, k, Ef, T)),
                  0, self.k_max, points=[self.kpo])[0]
        kappa_e_ = c * (-const.e * S_j * i1 - 1. / T * i2)
        return kappa_e_

    def EJ(self, j):
        """
        The average energy of conduction electrons [J];
        """
        k_low, k_peak, k_hi = self.get_klims(j)

        i1 = quad(
            lambda k: self.E_CB(j, k) * self.v_CB(k) * self.tau(j, k) * self.dfdk(j, k, self.Ef, self.T),
            k_low, k_hi, points=[self.kpo, k_peak])[0]

        i2 = quad(
            lambda k: self.v_CB(k) * self.tau(j, k) * self.dfdk(j, k, self.Ef, self.T),
            k_low, k_hi, points=[self.kpo, k_peak])[0]

        EJ_ = i1 / i2
        return EJ_

    # CONDUCTION BAND MODEL (PARABOLIC)
    # ---------------------------------
    def E_conf(self, m, idxs):
        E_lns = []
        for idx in idxs:
            k_ln = alphas[idx] / self.R  # root of ordinary Bessel function
            E_lns.append(const.hbar**2 * k_ln**2 / 2. / m)
        return E_lns

    def E_CB(self, j, k):
        return self.E_lns_CB[j] + const.hbar**2 * k**2 / 2. / self.meG

    def kx_CB(self, E):
        return np.sqrt(2. * self.meG * E) / const.hbar

    def v_CB(self, k):
        return const.hbar * k / self.meG

    # DISTRIBUTION FUNCTIONS
    # ----------------------
    @staticmethod
    def bE(E, T):
        return 1. / (np.exp(E / const.k / T) - 1)

    @staticmethod
    def fE(E, Ef, T):
        """
        Fermi-Dirac distribution;
         transformed to avoid overflow: 1/(1 + exp(x)) = exp(-x)/(1 + exp(-x))
        """
        exp_val = np.exp((Ef - E) / const.k / T)
        return exp_val / (1. + exp_val)

    @staticmethod
    def dfdE(E, Ef, T):
        f0 = NWRTAsolver.fE(E, Ef, T)
        return - 1. / const.k / T * f0 * (1 - f0)

    def fk(self, j, k, Ef, T):
        E = self.E_CB(j, k)
        return NWRTAsolver.fE(E, Ef, T)

    def dfdk(self, j, k, Ef, T):
        f0 = self.fk(j, k, Ef, T)
        return -1. / const.k / T * (const.hbar**2 * k / self.meG) * f0 * (1. - f0)

    def dEfdx(self, j):
        return 1. / self.T * (self.Ef - self.EJ(j)) / self.dTdx

    # CARRIER CONCENTRATIONS [m^-3]
    # -----------------------------
    def calculate_n(self, Ef):
        c = 1. / (np.pi**2 * self.R**2) * np.sqrt(2. * const.k * self.T * self.meG) / const.hbar
        n_ = 0.
        for E_ln in self.E_lns_CB:
            eta = (float(Ef) - E_ln) / const.k / self.T
            n_ += c * fdk(-1 / 2, eta)

        return n_

    def calculate_p(self, Ef, T):
        c = 1. / (np.pi**2 * self.R**2) * np.sqrt(2. * const.k * T * self.mat.mh_DOS) / const.hbar
        p_ = 0.
        Eg = self.mat.get_Eg(T)
        for E_ln in self.E_lns_VB:
            eta = -(float(Ef) - E_ln - Eg) / const.k / T
            p_ += c * fdk(-1 / 2, eta)

        return p_

    def calculate_Ef(self, n, T):
        guess = -self.mat.get_Eg(T) / 3.
        Ef_ = fmin(lambda Ef: abs(np.log(n / self.calculate_n(Ef))),
                   x0=guess, maxiter=500, ftol=1e-8, disp=False)
        check_n = self.calculate_n(Ef_)
        err = abs(n - check_n) / n
        if err >= 0.01:
            print("large Ef finding error (n = {:.2e}/cm^-3): {:.2f} %"
                  .format(n / 1e6, err * 1e2))
        return Ef_

    # SCATTERING RATES
    # ----------------
    def r_ac(self, k):
        k = np.abs(k)
        rate_ac = (
                2 * self.meG * self.mat.E1**2 * const.k * self.T
                / (const.hbar**3 * self.mat.cl * np.pi * self.R**2 * k)
        )
        return rate_ac

    def r_pe(self, k):
        k = np.abs(k)
        rate_pe = (
                8 * self.meG * const.e**2 * self.mat.Pz**2 * const.k * self.T
                / (const.hbar**3 * self.mat.eps_lo)
                * (1. - 2. * I1(2. * k * self.R) * K1(2. * k * self.R))
                / (k * (2 * k * self.R)**2)
        )
        return rate_pe / (4. * np.pi)

    def r_po(self, k):
        k = np.abs(k)
        q0 = np.sqrt(2. * self.meG * self.Epo) / const.hbar
        qp = k + np.sqrt(k**2 + q0**2)
        qm = k + np.sqrt((k**2 - q0**2) * np.heaviside(k - q0, 1))

        rate = (
                8. * const.e**2 * self.meG * self.Epo
                / const.hbar**3
                * (1. / self.mat.eps_hi - 1. / self.mat.eps_lo)
                * (
                        self.Npo * (1. - 2. * I1(qp * self.R) * K1(qp * self.R))
                        / (qp * self.R * np.sqrt(k**2 + q0**2))

                        +

                        (self.Npo + 1) * (1. - 2. * I1(qm * self.R) * K1(qm * self.R))
                        / (qm * self.R * np.sqrt((k**2 - q0**2) * np.heaviside(k - q0, 1))
                           + np.heaviside(q0 - k, 1))
                        * np.heaviside(k - q0, 1)
                )
        )
        return rate / (4. * np.pi)**2

    def r_ii(self, j, k, Z=1, screen=False):
        """
        Ionized impurity scattering rate (background) at finite temperature;
        Lee, Eq (23)
        """
        k = np.abs(k)
        Q = 2. * k * self.R
        N = (self.n + self.p)

        if screen:
            eps = self.eps1D(j, Q / self.R)
        else:
            eps = self.mat.eps_lo

        rate_ii = (
                          8 * np.pi * Z**2 * const.e**4 * self.meG * N
                          / (const.hbar**3 * k**3 * eps**2)
                          * (2. / Q * K1(Q) * I0(Q)
                             - 4. / Q**2 * K1(Q) * I1(Q)
                             - I1(Q)**2 * (K1(Q)**2 - K0(Q)**2))
                  ) / (4. * np.pi)**2
        return rate_ii

    def r_ii0(self, k, Z=1.):
        """
        Ionized impurity scattering rate (background) at zero temperature;
        Lee, Eq (23)
        """
        k = np.abs(k)
        Q = 2. * k * self.R
        N = (self.n + self.p)
        eps = self.eps1D0(Q / self.R)

        rate_ii0 = (
                           8 * np.pi * Z**2 * const.e**4 * self.meG * N
                           / (const.hbar**3 * k**3 * eps**2)
                           * (2. / Q * K1(Q) * I0(Q)
                              - 4. / Q**2 * K1(Q) * I1(Q)
                              - I1(Q)**2 * (K1(Q)**2 - K0(Q)**2))
                   ) / (4. * np.pi)**2
        return rate_ii0

    def r_tot(self, j, k):
        return self.r_ac(k) + self.r_pe(k) + self.r_po(k) + self.r_ii(j, k)

    def tau(self, j, k):
        return 1. / self.r_tot(j, k)

    # DIELECTRIC RESPONSE / SCREENING
    # -------------------------------
    # def Fq(self, q, Ef, T):
    #     """
    #     Static Lindhard function at finite temperature;
    #     Lee, Eq. (12)
    #
    #     NOTE: take `L=1` since it's cancelled anyway
    #     """
    #     if np.isscalar(q):
    #         q = np.asarray(q)
    #         if q.ndim == 0:
    #             q = q[None]
    #
    #     output = np.zeros(q.shape)
    #     for i, q_ in enumerate(q):
    #         output[i] = quad(
    #             lambda k: ((self.fk(k + q_, Ef, T) - self.fk(k, Ef, T))
    #                        / (self.E_CB(k + q_) - self.E_CB(k))),
    #             -self.k_max, self.k_max)[0]
    #     return output * 1 / np.pi

    def Fq(self, j, q, Ef, T):
        """
        Static Lindhard function at finite temperature;
        Lee, Eq. (12)

        NOTE: take `L=1` since it's cancelled anyway
        """
        if np.isscalar(q):
            q = np.asarray(q)
            if q.ndim == 0:
                q = q[None]

        output = np.zeros(q.shape)
        ks = np.linspace(-self.k_max, self.k_max, 1000)
        for i, q_ in enumerate(q):
            f_k = self.fk(j, ks, Ef, T)
            f_kq = self.fk(j, ks + q_, Ef, T)
            E_k = self.E_CB(j, ks)
            E_kq = self.E_CB(j, ks + q_)
            output[i] = np.trapz((f_kq - f_k) / (E_kq - E_k), x=ks)
        return output * 1 / np.pi

    def eps1D(self, j, q):
        """
        Dimensionless dielectric function of quantum wire [1];
        (finite temperature)
        Lee, Eq. (13)
        """
        Fq_ = self.Fq(j, q, self.Ef, self.T)
        return (
                self.mat.eps_lo
                +
                (8 * const.e**2 / (q**2 * self.R**2)
                 * (K1(q * self.R) * I1(q * self.R) - .5)
                 * Fq_) / (4. * np.pi)
        )

    def eps1D0(self, q):
        """
        Dimensionless dielectric function of quantum wire [1];
        (zero temperature)
        Lee, Eq. (13)

        `q` is a wave vector [1/m]

        NOTE: second term divided by 4*Pi to work with out units (S.I.)
        """
        Fq0_ = self.Fq0(q)
        return (
                self.mat.eps_lo
                +
                (8 * const.e**2 / (q**2 * self.R**2)
                 * (K1(q * self.R) * I1(q * self.R) - .5)
                 * Fq0_) / (4. * np.pi)  # dividing by 4Pi here gives you Lee's figure
        )

    def Fq0(self, q):
        """
        Static Lindhard function at 0 temperature;
        Lee, Eq. (17)

        NOTE: take `L=1` since it's cancelled anyway

        `q` is a wave vector [1/m]
        """
        n_linear = self.n * np.pi * self.R**2
        kf = np.pi / 2 * n_linear
        return (
                2. * self.meG / (np.pi * const.hbar**2 * q)
                * np.log(np.abs((2 * kf - q) / (2 * kf + q)))
        )

    # UTILITIES
    # ---------
    def get_klims(self, j, delta=15.):
        Ef = self.Ef
        T = self.T
        m = self.meG

        ks = np.linspace(1., self.k_max, 1000)
        i1_vals = self.E_CB(j, ks) * self.v_CB(ks) * self.tau(j, ks) * self.dfdk(j, ks, Ef, T)
        min_i1 = np.min(i1_vals)
        min_idx = int(np.where(i1_vals == min_i1)[0])
        k_peak = ks[min_idx]

        E_peak = const.hbar**2 * k_peak**2 / 2. / m
        E_low = E_peak - delta * const.k * T
        E_hi = E_peak + delta * const.k * T

        k_low = np.sqrt(2. * m * E_low) / const.hbar if E_low > 0. else 0.
        k_hi = np.sqrt(2. * m * E_hi) / const.hbar

        return k_low, k_peak, k_hi
