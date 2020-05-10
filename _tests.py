from materials import *
from bulkrodesolver import RodeSolver
from nwrta import NWRTAsolver

from scipy.integrate import quad

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from matplotlib.ticker import LogLocator, LinearLocator

mpl.rcParams['figure.figsize'] = [6.8, 6.2]
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 8.
mpl.rcParams['xtick.minor.size'] = 5.
mpl.rcParams['ytick.major.size'] = 8.
mpl.rcParams['ytick.minor.size'] = 5.
mpl.rcParams['lines.linewidth'] = 1.5  # is in fact the default, but just to be explicit


def log10(x):
    return np.log(x) / np.log(10)


def sized_plot(xy_sizes, xy_pads, scale=1.):
    """
    a function to create plots where the axes size and location is specified exactly;
    the `xy_pads` parameter moves the bottom left corner of the axes to the desired location,
    the figure is sized to centre the axes.

    NOTE: `ax` output does not work well with ax.twinx() or ax.twiny();
           use .secondary_xaxis() or .secondary_yaxis() instead.
    """
    x_size, y_size = xy_sizes
    x_pad, y_pad = xy_pads

    # scale values
    x_size *= scale
    y_size *= scale
    x_pad *= scale
    y_pad *= scale

    # create figure that will fit desired axes
    fig = plt.figure(figsize=(x_size + 2 * x_pad, y_size + 2 * y_pad))

    # use helper classes to prepare dimensions
    h = [Size.Fixed(x_pad), Size.Fixed(x_size)]
    v = [Size.Fixed(y_pad), Size.Fixed(y_size)]
    divider = Divider(fig, (0.0, 0.0, x_pad, y_pad), h, v, aspect=False)

    # create axes
    ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    # add axes to figure
    fig.add_axes(ax)

    return fig, ax


def g_convergence_test(mat, T, R, n, typ='F', num_k=10000, i_max=15):
    solver = RodeSolver(mat, T, R, n, num_k, k_MIN=1e5)
    k = solver.SPACES['k']

    x = const.hbar * k / np.sqrt(2. * mat.get_meG(T) * const.k * solver.T)

    plt.figure()
    plt.xlim(0, 2.5)
    for i in range(i_max + 1):
        gk = solver.g_dist(i, typ)
        plt.plot(x, gk)
        plt.pause(.01)
    plt.show()


# RECREATING RODE'S FIGURES
# -------------------------
#
# mu(n) curves are the most difficult to reproduce.


# InSb, mu(T), intrinsic
def R3F1(i, num_pts):
    solver = RodeSolver(InSb, 200, Rc=0.)
    Ts = np.linspace(200, 800, num_pts)
    mus = []

    for T in Ts:
        ni = ni_InSb_Hrostowski(T)
        solver.n = ni
        solver.p = ni
        solver.T = T

        mu = solver.mu(i)
        mus.append(mu * 1e4)

        print("ni = {:.2e} cm^-3".format(ni / 1e6))
        print("mu({:.2f}) = {:.2e} cm^2/V/s".format(T, mu * 1e4))
        print()

    fig, ax = sized_plot((4.4, 7.0), (.8, .5))

    ax.plot(Ts, np.array(mus), color='k')

    ax.set_xscale('log')
    ax.set_xticks([200, 300, 400, 500, 600, 700, 800])
    ax.set_xticklabels([200, '', 400, '', 600, '', 800])
    # cant figure out second x-axis for this example; labels persist for some reason

    ax.set_yscale('log')
    ax.set_yticks([1e4, 1e5])
    ax.set_yticks([2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5, 2e5], minor=True)
    ax.set_yticklabels([r'$2\times10^4$', '', '', r'$5\times10^4$', '', '', '', '', r'$2\times10^5$'], minor=True)
    secyax = ax.secondary_yaxis('right')  # twinx() does not work on log scale
    secyax.tick_params('y', labelright=False)
    # couldn't figure out how to set minor ticks on log scale

    ax.set_xlim(200, 800)
    ax.set_ylim(1e4, 2e5)

    ax.set_xlabel(r'TEMPERATURE, T ($^\circ$K)')
    ax.set_ylabel(r'MOBILITY, $\mu$ (cm$^2$/V sec)')
    ax.text(410, 1.4e5, 'InSb')
    ax.set_title('R3F1')

    plt.show()


# InSb, mu(T), intrinsic
def R3F2(i, num_pts=30):
    solver = RodeSolver(InSb, T=300, Rc=0.)
    Ts = np.logspace(log10(20), 3, num_pts)
    mus = []

    for T in Ts:
        ni = ni_InSb_Hrostowski(T)
        solver.p = ni
        solver.n = ni
        solver.T = T

        mu = solver.mu(i)
        mus.append(mu * 1e4)

        print("ni = {:.2e} cm^-3".format(ni / 1e6))
        print("mu({}) = {:.2e} cm^2/V/s".format(T, mu * 1e4))
        print()

    fig, ax = sized_plot((5.9, 6.4), (.8, .5))

    ax.plot(Ts, mus, color='k')

    ax.set_xscale('log')
    ax.set_xticks([20, 40, 60, 80, 100, 200, 400, 600, 800, 1000])
    ax.set_xticklabels([20, 40, 60, '', 100, 200, 400, 600, '', 1000])

    ax.set_yscale('log')
    ax.set_yticks([1e4, 1e5, 1e6, 1e7])
    ax.set_yticks([2e4, 4e4, 6e4, 8e4, 2e5, 4e5, 6e5, 8e5, 2e6, 4e6, 6e6, 8e6, 2e7, 4e7], minor=True)
    ax.set_yticklabels([r'$2\times10^4$', '', '', r'$5\times10^4$', '', '', '', '', r'$2\times10^5$'], minor=True)

    ax.set_xlim(20, 1000)
    ax.set_ylim(1e4, 4e7)

    ax.set_xlabel(r'TEMPERATURE, T ($^\circ$K)')
    ax.set_ylabel(r'MOBILITY, $\mu$ (cm$^2$/V sec)')
    ax.set_title('R3F2')

    plt.show()


# InSb, mu(n, R) @ 300 K
def R3F3(i, num_pts=30):
    """
    mu(n) over R = 1, 2, 5 (at T = 300 K)
    """

    solver = RodeSolver(InSb, 300., Rc=1.)
    # ns = np.logspace(log10(2e22), log10(7e24), num_pts)
    ns = np.logspace(22, 25, num_pts)
    Rs = [1., 2., 5.]

    fig, ax = sized_plot((5.9, 5.), (.9, .5))

    for R in Rs:
        mu_R = []
        solver._Rc = R
        for n in ns:
            solver.n = n
            mu = solver.mu(i)
            mu_R.append(mu * 1e4)
            print("mu({:.2e}, {:.2e}, {}) = {:.2e} cm^2/V/s"
                  .format(solver.n / 1e6, solver.p / 1e6, int(R), mu * 1e4))
            print("Ef = {:.5f}".format(solver.Ef / const.e))
            print()
        ax.plot(ns / 1e6, mu_R, color='k')

    ax.set_xscale('log')
    ax.set_xticks([1e16, 1e17, 1e18, 1e19])
    ax.xaxis.set_minor_locator(LogLocator(numticks=5, subs=[2, 4, 6, 8]))  # this does not work on secondary axis below
    secxax = ax.secondary_xaxis('top')
    secxax.tick_params('x', labeltop=False)  # this somehow is working

    ax.set_yticks([0, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4])
    ax.set_yticklabels(['', '', '', '', r'$4\times10^4$', '', '', '', r'$8\times10^4$'])
    secyax = ax.secondary_yaxis('right')
    secyax.tick_params('y', labelright=False)

    ax.set_xlim(1e16, 1e19)
    ax.set_ylim(0, 8e4)

    ax.set_xlabel(r'FREE-ELECTRON CONCENTRATION, n (cm$^{-3}$)')
    ax.set_ylabel(r'MOBILITY, $\mu$ (cm$^2$/V sec)')
    ax.text(5e17, 6.5e4, r'InSb, 300$^\circ$K')
    ax.set_title("R3F3")

    plt.show()


# InSb, S(n) @ 300 K
def R3F5(i, num_pts=30):
    solver = RodeSolver(InSb, 300, Rc=1.)
    ns = np.logspace(22, 25, num_pts)
    Ss = []

    for n in ns:
        solver.n = n
        S = solver.S(i)
        Ss.append(-S * 1e6)
        print("S({:.2e}) = {:.2f} uV/K".format(solver.n / 1e6, S * 1e6))

    fig, ax = sized_plot((6.1, 5.75), (.8, .5))

    ax.plot(ns / 1e6, Ss, color='black')

    ax.set_xscale('log')
    ax.set_xticks([1e16, 1e17, 1e18, 1e19])
    ax.xaxis.set_minor_locator(LogLocator(numticks=5, subs=[2, 4, 6, 8]))  # this does not work on secondary axis below
    secxax = ax.secondary_xaxis('top')
    secxax.tick_params('x', labeltop=False)  # this somehow is working

    ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
    ax.set_yticklabels(['', '', r'$-100$', '', r'$-200$', '', r'$-300$', ''])
    ax.yaxis.set_minor_locator(LinearLocator(numticks=1))
    secyax = ax.secondary_yaxis('right')
    secyax.tick_params('y', labelright=False)

    ax.set_xlim(1e16, 1e19)
    ax.set_ylim(0, 350)

    ax.set_xlabel(r'FREE-ELECTRON CONCENTRATION, n (cm$^{-3}$)')
    ax.set_ylabel(r'THERMOELECTRIC POWER, Q ($\mu V / ^\circ K$)')
    ax.text(2.5e17, 250, r'InSb, 300$^\circ$K')
    ax.set_title('R3F5')

    plt.show()


# InSb, S(T), intrinsic
def R3F6(i, num_pts=30):
    """
    S(T) (intrinsic)
    """
    solver = RodeSolver(InSb, 300, Rc=0.)
    Ts = np.linspace(200, 800, num_pts)
    Ss = []

    for T in Ts:
        n_i = ni_InSb_Hrostowski(T)
        solver.n = n_i
        solver.p = n_i
        solver.T = T
        S = solver.S(i)
        Ss.append(-S * 1e6)
        print("S({:.2f}) = {:.2f} uV/K".format(T, -S * 1e6))

    fig, ax = sized_plot((5.9, 5.95), (.8, .5))

    ax.plot(Ts, Ss, color='black')

    ax.set_xticks([200, 300, 400, 500, 600, 700, 800])
    ax.set_xticklabels([200, '', '', 500, '', '', 800])
    secxax = ax.secondary_xaxis('top')
    secxax.tick_params('x', labeltop=False)  # this somehow is working

    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_yticklabels(['', '', r'$-200$', '', r'$-400$', ''])
    secyax = ax.secondary_yaxis('right')
    secyax.tick_params('y', labelright=False)

    ax.set_xlim(200, 800)
    ax.set_ylim(0, 500)

    ax.set_xlabel(r'TEMPERATURE, T ($^\circ K$)')
    ax.set_ylabel(r'THERMOELECTRIC POWER, Q ($\mu V / ^\circ K$)')
    ax.text(520, 380, 'InSb')
    ax.set_title('R3F6')

    plt.show()


# InAs, mu(T), intrinsic
def R3F7(i, num_pts=30):
    solver = RodeSolver(InAs, 300, Rc=0.)
    Ts = np.linspace(100, 1000, num_pts)
    mus = []

    for T in Ts:
        n_i = ni_InAs_Folberth(T)
        solver.n = n_i
        solver._p = n_i
        solver.T = T
        mu = solver.mu(i)
        mus.append(mu * 1e4)
        print("mu({:.2f}) = {:.2e} cm^2/V/s".format(T, mu * 1e4))

    fig, ax = sized_plot((5.75, 6.2), (.7, .5))

    ax.plot(Ts, mus, color='k')
    ax.set_xlim(100, 1000)
    ax.set_ylim(2e3, 6e5)

    ax.set_xscale('log')
    ax.set_xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ax.set_xticklabels([100, 200, '', 400, '', 600, '', '', '', 1000])
    # same second x-axis problem as R3F1()

    ax.set_yscale('log')
    ax.set_yticks([1e4, 1e5])
    ax.yaxis.set_minor_locator(LogLocator(numticks=5, subs=[2, 4, 6, 8]))
    secyax = ax.secondary_yaxis('right')  # twinx() does not work on log scale
    secyax.tick_params('y', labelright=False)

    ax.set_xlabel(r'TEMPERATURE, T ($^\circ$K)')
    ax.set_ylabel(r'MOBILITY, $\mu$ (cm$^2$/V sec)')
    ax.text(300, 1.4e5, 'InAs')
    ax.set_title("R3F7")

    plt.show()


# InAs, mu(n, R) @ 300 K (not great)
def R3F9(i, num_pts=30):
    solver = RodeSolver(InAs, 300, Rc=0.)
    ns = np.logspace(22, 25, num_pts)
    Rs = [1., 2., 5.]

    fig, ax = sized_plot((5.7, 4.65), (.9, .5))

    for R in Rs:
        mu_R = []
        solver._Rc = R
        for n in ns:
            solver.n = n
            mu = solver.mu(i)
            mu_R.append(mu * 1e4)
            print("mu({:.2e}, {:.2e}, {}) = {:.2e} cm^2/V/s"
                  .format(solver.n / 1e6, solver.p / 1e6, int(R), mu * 1e4))
            print("Ef = {:.5f}".format(solver.Ef / const.e))
            print()

        ax.plot(ns / 1e6, mu_R, label=str(R))

    ax.legend()
    ax.set_xscale('log')
    ax.set_xticks([1e16, 1e17, 1e18, 1e19])
    ax.xaxis.set_minor_locator(LogLocator(numticks=5, subs=[2, 4, 6, 8]))  # this does not work on secondary axis below
    secxax = ax.secondary_xaxis('top')
    secxax.tick_params('x', labeltop=False)  # this somehow is working

    ax.set_yticks([0, .5e4, 1e4, 1.5e4, 2e4, 2.5e4, 3e4])
    ax.set_yticklabels([0, '', r'$10^4$', '', r'$2\times10^4$', '', r'$3\times10^4$'])
    secyax = ax.secondary_yaxis('right')
    secyax.tick_params('y', labelright=False)

    ax.set_xlim(1e16, 1e19)
    ax.set_ylim(0, 3e4)

    ax.set_xlabel(r'FREE-ELECTRON CONCENTRATION, n (cm$^{-3}$)')
    ax.set_ylabel(r'MOBILITY, $\mu$ (cm$^2$/V sec)')
    ax.text(5e17, 2.6e4, r'InAs, 300$^\circ$K')
    ax.set_title("R3F9")
    plt.show()


# InAs, S(n, R) @ 300 K
def R3F11(i, num_pts=30):
    solver = RodeSolver(InAs, 300., Rc=1.)
    ns = np.logspace(22, 25, num_pts)
    Rs = [1., 10.]

    fig, ax = sized_plot((6.0, 5.7), (.9, .5))

    for R in Rs:
        S_R = []
        solver._Rc = R
        for n in ns:
            solver.n = n
            S = solver.S(i)
            S_R.append(-S * 1e6)
            print("S({:.2e}, {}) = {:.2f} uV/K".format(solver.n, int(R), -S * 1e6))

        ax.plot(ns / 1e6, S_R, color='k', linestyle='--' if R == 10 else '-')

    ax.set_xscale('log')
    ax.set_xticks([1e16, 1e17, 1e18, 1e19])
    ax.xaxis.set_minor_locator(LogLocator(numticks=5, subs=[2, 4, 6, 8]))  # this does not work on secondary axis below
    secxax = ax.secondary_xaxis('top')
    secxax.tick_params('x', labeltop=False)  # this somehow is working

    ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_yticklabels(['0', '', '', '', '-200', '', '', '', '-400'])
    secyax = ax.secondary_yaxis('right')
    secyax.tick_params('y', labelright=False)

    ax.set_xlim(1e16, 1e19)
    ax.set_ylim(0, 425)

    ax.set_xlabel(r'FREE-ELECTRON CONCENTRATION, n (cm$^{-3}$)')
    ax.set_ylabel(r'THERMOELECTRIC POWER, Q ($\mu V / ^\circ K$)')
    ax.text(5e17, 2.6e4, r'InAs, 300$^\circ$K')
    ax.set_title("R3F11")

    plt.show()


# InP, mu(T), intrinsic
def R3F13(i, num_pts=30):
    solver = RodeSolver(InP, 300, Rc=0.)
    Ts = np.logspace(log10(20), 3, num_pts)
    mus = []

    for T in Ts:
        n_i = ni_InP_Folberth(T)
        solver.n = n_i
        solver._p = n_i
        solver.T = T
        mu = solver.mu(i)
        mus.append(mu * 1e4)
        print("mu({:.2f}) = {:.2e} cm^2/V/s".format(T, mu * 1e4))

    fig, ax = sized_plot((5.75, 6.2), (.7, .5))

    ax.plot(Ts, mus, color='k')
    ax.set_xlim(100, 1200)
    ax.set_ylim(1e3, 2e6)

    ax.set_xscale('log')
    ax.set_xticks([20, 40, 60, 80, 100, 200, 400, 600, 1000])
    ax.set_xticklabels([20, 40, "", "", 100, 200, 400, "", 1000])
    # same second x-axis problem as R3F1()

    ax.set_yscale('log')
    ax.set_yticks([1e3, 1e4, 1e5, 1e6])
    ax.yaxis.set_minor_locator(LogLocator(numticks=5, subs=[2, 4, 6, 8]))
    secyax = ax.secondary_yaxis('right')  # twinx() does not work on log scale
    secyax.tick_params('y', labelright=False)

    ax.set_xlabel(r'TEMPERATURE, T ($^\circ$K)')
    ax.set_ylabel(r'MOBILITY, $\mu$ (cm$^2$/V sec)')
    ax.text(200, 2.5e5, 'InP')
    ax.set_title("R3F13")

    plt.show()


# InP, S(n, R)
def R3F16(i, num_pts=30):
    solver = RodeSolver(InP, 300., Rc=1.)
    ns = np.logspace(22, 25, num_pts)
    Rs = [1., 10.]

    fig, ax = sized_plot((5.7, 5.4), (.9, .5))

    for R in Rs:
        S_R = []
        solver._Rc = R
        for n in ns:
            solver.n = n
            S = solver.S(i)
            S_R.append(-S * 1e6)
            print("S({:.2e}, {}) = {:.2f} uV/K".format(solver.n, int(R), -S * 1e6))

        ax.plot(ns / 1e6, S_R, color='k', linestyle='-')

    ax.set_xscale('log')
    ax.set_xticks([1e16, 1e17, 1e18, 1e19])
    ax.xaxis.set_minor_locator(LogLocator(numticks=5, subs=[2, 4, 6, 8]))  # this does not work on secondary axis below
    secxax = ax.secondary_xaxis('top')
    secxax.tick_params('x', labeltop=False)  # this somehow is working

    ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
    ax.set_yticklabels(['0', '', '-200', '', '-400', '', '-600'])
    secyax = ax.secondary_yaxis('right')
    secyax.tick_params('y', labelright=False)

    ax.set_xlim(1e16, 1e19)
    ax.set_ylim(0, 700)

    ax.set_xlabel(r'FREE-ELECTRON CONCENTRATION, n (cm$^{-3}$)')
    ax.set_ylabel(r'THERMOELECTRIC POWER, Q ($\mu V / ^\circ K$)')
    ax.text(2e17, 550, r'InP, 300$^\circ$K')
    ax.set_title("R3F16")

    plt.show()


# NWRTA rests
# -----------
def scatter_rates(n_linear, T, R):
    n = n_linear / np.pi / R**2
    nwsolver = NWRTAsolver(GaAs.modified(E1=12. * const.e), T=T, R=R, n=n)
    ks = np.linspace(1, nwsolver.k_CB(.25 * const.e), 1000)
    Es = nwsolver.E_CB(ks)
    E_po = nwsolver.Epo

    fig, ax = plt.subplots(tight_layout=True)

    rate_PE = nwsolver.r_pe(ks)
    rate_DP = nwsolver.r_ac(ks)
    rate_LO = nwsolver.r_po(ks)
    rate_TOT = 1. / (1. / rate_PE + 1. / rate_DP + 1. / rate_LO)

    ax.plot(Es / E_po, rate_PE, label='PE')
    ax.plot(Es / E_po, rate_DP, label='DP')
    ax.plot(Es / E_po, rate_LO, label='LO', color='r')
    ax.plot(Es / E_po, rate_TOT, linewidth=1., linestyle=':', label='TOT')

    ax.set_yscale('log')
    ax.set_ylabel(r'$\tau^{-1}$', fontsize=20)
    ax.set_xlabel(r'$E/\hbar\omega_0$', fontsize='20')
    ax.set_xlim(0, 5)
    ax.set_ylim(1e9, 1e14)

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(*ax.get_ylim())

    ax.text(0.5, .9, r"n = {:.2e}".format(n / 1e6) + " cm$^{-3}$",
            transform=ax.transAxes)
    ax.text(0.5, .85, r"R = {:.2f} A".format(R * 1e10),
            transform=ax.transAxes)
    ax.text(0.5, .8, r"T = {:.0f} K".format(T),
            transform=ax.transAxes)

    ax.legend()
    plt.show()


def fishman_fig3(n_linear, R):
    Ts = np.logspace(1, log10(300), 30)

    mu_PE = np.zeros(Ts.shape)
    mu_DP = np.zeros(Ts.shape)
    mu_LO = np.zeros(Ts.shape)
    mu_TOT = np.zeros(Ts.shape)

    n = n_linear / np.pi / R**2
    nwsolver = NWRTAsolver(GaAs.modified(E1=12. * const.e), T=300, R=R, n=n)

    k_min = 0.
    k_max = nwsolver.k_max
    for i, T in enumerate(Ts):
        nwsolver.T = T

        Ef = nwsolver.Ef
        check_n = nwsolver.calculate_n(Ef, T)
        err_n = abs(n - check_n) / n
        print("Ef({:.1f} K) = {} eV".format(T, Ef / const.e))
        print("n error: {:.5f}".format(err_n))
        if abs(check_n - n) / n >= .1:
            print("----------------------------> Ef finding Error!")

        norm = quad(lambda k: k * nwsolver.dfdk(k, Ef, T),
                    k_min, k_max)[0]

        if norm == 0:
            mu_PE[i] = np.nan
            mu_DP[i] = np.nan
            mu_LO[i] = np.nan
            mu_TOT[i] = np.nan
            print("norm was 0 as T = {:.1f}".format(T))
            continue

        t_PE = quad(lambda k: k * nwsolver.dfdk(k, Ef, T) * 1. / nwsolver.r_pe(k),
                    k_min, k_max)[0] / norm
        t_DP = quad(lambda k: k * nwsolver.dfdk(k, Ef, T) * 1. / nwsolver.r_ac(k),
                    k_min, k_max)[0] / norm
        t_LO = quad(lambda k: k * nwsolver.dfdk(k, Ef, T) * 1. / nwsolver.r_po(k),
                    k_min, k_max)[0] / norm

        mu_PE[i] = const.e / nwsolver.meG * t_PE
        mu_DP[i] = const.e / nwsolver.meG * t_DP
        mu_LO[i] = const.e / nwsolver.meG * t_LO

        print("done T = {:.1f} K".format(T))
        print()

    fig, ax = sized_plot((5.9, 8.27), (1., 1.), scale=.5)
    ax.set_xticks([10, 30, 100, 300])
    ax.set_xticklabels([10, 30, 100, 300])
    ax.set_title("Fishman (1987) - Figure 3")

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e4, 1e7)
    ax.set_xlim(10, 300)

    ax.plot(Ts, mu_PE * 1e4, linestyle=':', label='PE')
    ax.plot(Ts, mu_LO * 1e4, linestyle=':', label='LO')
    ax.plot(Ts, mu_DP * 1e4, linestyle=':', label='DP')
    ax.plot(Ts, mu_TOT * 1e4, label='TOT')

    ax.legend()

    ax2 = ax.secondary_yaxis("right")
    ax3 = ax.secondary_xaxis("top")
    plt.show()


def Lee_fig2(T):
    n_linear = 1e5 * 1e2  # [m^-1]
    kf = np.pi * n_linear / 2.
    R = 1. / kf
    n = n_linear / (np.pi * R**2)

    nwsolver = NWRTAsolver(GaAs, T=T, R=R, n=n)
    Qs = np.linspace(.01, 7, 1000)

    fig, ax = sized_plot((7., 10.1,), (.8, .5), scale=1 / 2)

    eps0_vals = nwsolver.eps1D0(Qs * kf) / nwsolver.mat.eps_lo
    eps_vals = nwsolver.eps1D(Qs * kf) / nwsolver.mat.eps_lo
    ax.plot(Qs, eps0_vals, color='k')
    ax.plot(Qs, eps_vals, color='r')

    ax.set_ylim(0.9, 1e2)
    ax.set_xlim(1, 7)
    ax.set_yscale('log')

    # y_cusp = nwsolver.eps1D0(
    #     kf * Qs[np.where(np.abs(Qs - 2.) == np.min(np.abs(Qs - 2.)))[0]]
    # ) / nwsolver.mat.eps_lo
    # print(y_cusp)
    # ax.vlines(x=2, ymin=y_cusp, ymax=1e3, color='k')

    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(['1', '10', r'$10^2$'])

    ax.set_xticks([i for i in range(8)])

    plt.show()


def Lee_fig5(T):
    n_linear = 1e5 * 1e2  # [m^-1]
    kf = np.pi * n_linear / 2.
    R = 1. / kf
    n = n_linear / (np.pi * R**2)

    nwsolver = NWRTAsolver(GaAs, T=T, R=R, n=n)
    Qs = np.linspace(.01, 7, 1000)

    fig, ax = sized_plot((7., 10.1,), (.8, .5), scale=1 / 2)

    rate_b0 = nwsolver.r_ii0(Qs * kf / 2)
    ax.plot(Qs, rate_b0, color='k')

    rate_b = nwsolver.r_ii(Qs * kf / 2)
    ax.plot(Qs, rate_b, color='r')

    nwsolver = NWRTAsolver(GaAs, T=2 * T, R=R, n=n)
    rate_b2 = nwsolver.r_ii(Qs * kf / 2)
    ax.plot(Qs, rate_b2, color='g')

    ax.set_ylim(1e10, 1e17)
    ax.set_xlim(-0.01, 7)
    ax.set_yscale('log')

    ax.set_xticks([i for i in range(8)])

    plt.show()


def screen_test(T1, T2):
    n_linear = 1e5 * 1e2  # [m^-1]
    kf = np.pi * n_linear / 2.
    R = 1. / kf
    n = n_linear / (np.pi * R**2)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    nwsolver = NWRTAsolver(GaAs, T=T1, R=R, n=n)
    Qs = np.linspace(.01, 7, 1000)
    rate_screen = nwsolver.r_ii(Qs * kf / 2, screen=True)
    ax.plot(Qs, rate_screen, color='k', label='screened')
    rate_unscreen = nwsolver.r_ii(Qs * kf / 2, screen=False)
    ax.plot(Qs, rate_unscreen, color='r', label='unscreened')

    ax2 = fig.add_subplot(122)
    nwsolver = NWRTAsolver(GaAs, T=T2, R=R, n=n)
    rate_screen = nwsolver.r_ii(Qs * kf / 2, screen=True)
    ax2.plot(Qs, rate_screen, color='k', label='screened')
    rate_unscreen = nwsolver.r_ii(Qs * kf / 2, screen=False)
    ax2.plot(Qs, rate_unscreen, color='r', label='unscreened')

    ax.set_ylim(1e10, 1e19)
    ax.set_xlim(-0.01, 7)
    ax.set_yscale('log')
    ax.set_xticks([i for i in range(8)])
    ax2.text(.6, .8, r"$T$" + " = {:d} K".format(T1), transform=ax.transAxes)

    ax2.set_ylim(1e10, 1e19)
    ax2.set_xlim(-0.01, 7)
    ax2.set_yscale('log')
    ax2.set_xticks([i for i in range(8)])
    ax2.text(.6, .8, r"$T$" + " = {:d} K".format(T2), transform=ax2.transAxes)

    ax.set_ylabel(r'$\tau^{-1}$', fontsize=20)
    ax.legend()
    ax2.legend()

    plt.show()


def nw_sigma_test():
    T = 300
    R = 15e-9
    n = 1e18 * 1e6
    nws = NWRTAsolver(GaAs, T, R, n)
    print("sigma = {:.5e}".format(nws.sigma()))


def nw_S_test():
    T = 300
    R = 15e-9
    n = 1e18 * 1e6
    nws = NWRTAsolver(GaAs, T, R, n)
    print("S = {:.5e}".format(nws.S()))


def nw_kappa_test():
    T = 300
    R = 15e-9
    n = 1e18 * 1e6
    nws = NWRTAsolver(GaAs, T, R, n)
    print("S = {:.5e}".format(nws.kappa_e()))


def nw_Ef_test():
    Efs = np.linspace(-2, 2, 1000) * const.e
    nws = NWRTAsolver(InAs, R=15e-9, T=300, n_subs=5)
    n_target = 3.73e16 * 1e6

    ns = np.zeros(Efs.shape)
    for i, Ef in enumerate(Efs):
        ns[i] = nws.calculate_n(Ef)

    plt.plot(Efs / const.e, abs(np.log(ns / n_target)))
    plt.yscale('log')
    plt.show()


def EJ_integrand_test(mat, j, n, R=10e-9, T=300):
    nws = NWRTAsolver(mat, T=T, R=R, n=n)
    print("n = {:.5e}".format(nws.n / 1e6))
    print("Ef = {} eV".format(nws.Ef / const.e))
    i_lo = 0
    i_hi = 0
    for i in range(nws.n_subs):
        if nws.Ef > nws.E_lns_CB[i]:
            i_lo = i
        else:
            i_hi = i
            break
    print("E_ln[{}] < Ef < E_ln[{}]"
          .format(i_lo, i_hi))

    ks = np.linspace(1, nws.k_max, 1000)

    def i1(j_, k):
        return nws.E_CB(j_, k) * nws.v_CB(k) * nws.tau(j_, k) * nws.dfdk(j_, k, nws.Ef, nws.T)

    def i2(j_, k):
        return nws.v_CB(k) * nws.tau(j_, k) * nws.dfdk(j_, k, nws.Ef, nws.T)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    i1vals = i1(j, ks)
    ax1.plot(ks / nws.kpo, i1vals, label=str(j))

    i2vals = i2(j, ks)
    ax2.plot(ks / nws.kpo, i2vals, label=str(j))

    k_low, k_peak, k_hi = nws.get_klims(j)

    ax2.axvline(x=k_hi / nws.kpo, linestyle=':', color='r')
    ax2.axvline(x=k_peak / nws.kpo, linestyle=':', color='k')
    ax2.axvline(x=k_low / nws.kpo, linestyle=':', color='b')

    ax1.legend()
    plt.show()


if __name__ == "__main__":
    # g_convergence_test(GaAs, 300, 0., 1e22, typ='F', i_max=20)

    # R3F1(30, 50)
    # R3F2(30, 100)
    # R3F3(30, 50)
    # R3F5(30, 50)
    # R3F6(30, 50)
    # R3F7(30, 50)
    # R3F9(30, 30)
    # R3F11(30, 50)
    # R3F13(30, 50)
    # R3F16(30, 50)

    # scatter_rates(n_linear=1e6 * 1e2, R=100e-10, T=300)
    # fishman_fig3(1e6 * 1e2, 100e-10)
    # Lee_fig2(T=50)
    # Lee_fig5(T=20)
    # screen_test(T1=12, T2=300)
    # nw_sigma_test()
    # nw_S_test()
    # nw_kappa_test()
    # nw_Ef_test()
    EJ_integrand_test(GaAs, j=15, n=1e19 * 1e6)
    pass
