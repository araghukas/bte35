import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator
from os import path

from materials import *
from bulkrodesolver import *
from nwrta import *

# plotting global settings
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['legend.frameon'] = False

# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['r', 'g', 'b', 'c', 'k'])

label_fontsize = 17
label_texfontsize = 17


def log10(x):
    return np.log(x) / np.log(10)


def check_rcParam(string):
    for key, value in mpl.rcParams.items():
        if string in key:
            print("{} : {}".format(key, value))


# def nanowire_scattering_rates(show=True, save=False,
#                               mat=GaAs, T=300, R=15e-9, n=1e17 * 1e6):
#     nwsolver = NWRTAsolver(mat, T, R, n=n)
#     Es = np.linspace(1e-4 * const.e, .2 * const.e, 1000)
#     ks = nwsolver.k_CB(Es)
#
#     rate_pe = nwsolver.r_pe(ks)
#     rate_ac = nwsolver.r_ac(ks)
#     rate_po = nwsolver.r_po(ks)
#     rate_ii = nwsolver.r_ii(ks)
#     rate_tot = rate_ac + rate_pe + rate_po + rate_ii
#
#     fig, ax = plt.subplots(tight_layout=True)
#     ax.set_yscale('log')
#
#     E_po = nwsolver.Epo
#     ax.plot(Es / E_po, rate_pe, label='pe')
#     ax.plot(Es / E_po, rate_ac, label='ac')
#     ax.plot(Es / E_po, rate_po, label='po')
#     ax.plot(Es / E_po, rate_ii, label='ii')
#     ax.plot(Es / E_po, rate_tot, label='total', linewidth=1.5, linestyle='--', color='k')
#
#     ax.set_xticks([0, 1, 2, 3, 4])
#     ax.set_xlim(0, 4)
#     ax.set_xlabel(r'$E/E_\mathrm{po}$', fontsize=label_texfontsize)
#
#     ax.set_ylim((5e8, 1e14))
#     ax.set_ylabel(r'Scattering Relaxation Rate [s$^{-1}$]', fontsize=label_fontsize)
#
#     # descriptions
#     ax.text(.35, .9, "{}".format(mat.name), fontsize=label_fontsize,
#             transform=ax.transAxes)
#     ax.text(.35, .82, r"$R = {:.1f}$".format(R * 1e9) + " nm", fontsize=label_fontsize,
#             transform=ax.transAxes)
#     ax.text(.35, .74, r"$T = {:d}$".format(T) + " K", fontsize=label_fontsize,
#             transform=ax.transAxes)
#
#     ax.legend()
#
#     if save:
#         plt.savefig(path.expanduser("~/Dropbox/Code/bte35/figures/nw_scatt_elastic.pdf"))
#     if show:
#         plt.show()


def nwrta_tau_and_gk(j1, j2, show=True, save=False, mat=GaAs, T=300, R=10e-9, n=1e17 * 1e6):
    nwsolver = NWRTAsolver(mat, T, R, n=n)
    Ef = nwsolver.Ef
    m = nwsolver.meG
    kpo = np.sqrt(2 * m * nwsolver.Epo) / const.hbar

    fig = plt.figure(figsize=(8, 7))
    plt.subplots_adjust(bottom=.12, wspace=.3, hspace=.2)

    ks = np.linspace(1e4, 4. * kpo, 1000)
    Es = nwsolver.E_CB(j1, ks)
    dfdks = nwsolver.dfdk(j1, ks, Ef, T)

    ax1 = fig.add_subplot(221)
    ax1.set_yscale('log')
    rate_pe = nwsolver.r_pe(ks)
    rate_ac = nwsolver.r_ac(ks)
    rate_po = nwsolver.r_po(ks)
    rate_ii = nwsolver.r_ii(j1, ks)
    rate_tot = rate_ac + rate_pe + rate_po + rate_ii
    # ax1.plot(ks / kpo, rate_pe, label='pe', linewidth=1.)
    # ax1.plot(ks / kpo, rate_ac, label='ac', linewidth=1.)
    ax1.plot(ks / kpo, rate_po, label='po')
    ax1.plot(ks / kpo, rate_ii, label='ii')
    ax1.plot(ks / kpo, rate_tot, label=r'po$+$ac$+$pe$+$ii', color='k', linestyle='--')
    ax1.set_ylim(1e9, 1e16)
    ax1.set_xlim(0, 4)
    ax1.set_ylabel(r"Scattering Rate [s$^{-1}$]", fontsize=15)
    ax1.text(.9, .9, "a.", fontsize=16, transform=ax1.transAxes)
    # ax1.legend()

    ax2 = fig.add_subplot(223)
    # g(k) in general is:
    # tau(k)/hbar * dfdk * (e*epslion + dEfdx + (E-Ef)/T * dTdx)
    tau = 1. / rate_tot
    g_dTdx_is_0 = tau / const.hbar * dfdks * const.e * nwsolver.eps_field
    g_eps_is_0 = tau / const.hbar * dfdks * (nwsolver.dEfdx(j1) + (Es - Ef) / T * nwsolver.dTdx)
    vk = const.hbar * ks / m
    ax2.plot(ks / kpo, -vk * g_dTdx_is_0 / np.max(np.abs(vk * g_dTdx_is_0)),
             label=r'$\frac{dT}{dx}=0$', color=u'#4b0082')
    ax2.plot(ks / kpo, -vk * g_eps_is_0 / np.max(np.abs(vk * g_eps_is_0)),
             label=r'$\varepsilon=0$', color=u'#d2691e')
    # ax2.text(.47, .45, r"$\varepsilon=0$", transform=ax2.transAxes, fontsize=15)
    # ax2.text(.65, .25, r"$\frac{dT}{dx}=0$", transform=ax2.transAxes, fontsize=15)
    # ax2.arrow(x=.63, y=.23, dx=-.1, dy=-.1, head_width=.015, fc='k',
    #           transform=ax2.transAxes)
    ax2.set_xlim(0, 4)
    ax2.set_ylabel(r"$-v(k)g(k)$ (normalized)", fontsize=15)
    ax2.text(.9, .9, "c.", fontsize=16, transform=ax2.transAxes)
    # ax2.legend(loc=(.4, .7))

    # -------------------------------------------------------------------------------
    Es = nwsolver.E_CB(j2, ks)
    dfdks = nwsolver.dfdk(j2, ks, Ef, T)

    ax3 = fig.add_subplot(222)
    ax3.set_yscale('log')
    rate_pe = nwsolver.r_pe(ks)
    rate_ac = nwsolver.r_ac(ks)
    rate_po = nwsolver.r_po(ks)
    rate_ii = nwsolver.r_ii(j2, ks)
    rate_tot = rate_ac + rate_pe + rate_po + rate_ii
    ax3.plot(ks / kpo, rate_po, label='po')
    ax3.plot(ks / kpo, rate_ii, label='ii')
    ax3.plot(ks / kpo, rate_tot, label=r'po$+$ac$+$pe$+$ii', color='k', linestyle='--')
    ax3.set_ylim(1e9, 1e16)
    ax3.set_xlim(0, 4)
    # ax3.set_ylabel(r"Scattering Rate [s$^{-1}$]", fontsize=15)
    ax3.text(.9, .9, "b.", fontsize=16, transform=ax3.transAxes)
    ax3.legend()

    ax4 = fig.add_subplot(224)
    tau = 1. / rate_tot
    g_dTdx_is_0 = tau / const.hbar * dfdks * const.e * nwsolver.eps_field
    g_eps_is_0 = tau / const.hbar * dfdks * (nwsolver.dEfdx(j2) + (Es - Ef) / T * nwsolver.dTdx)
    vk = const.hbar * ks / m
    ax4.plot(ks / kpo, -vk * g_dTdx_is_0 / np.max(np.abs(vk * g_dTdx_is_0)),
             label=r'$\frac{dT}{dx}=0$', color=u'#4b0082')
    ax4.plot(ks / kpo, -vk * g_eps_is_0 / np.max(np.abs(vk * g_eps_is_0)),
             label=r'$\varepsilon=0$', color=u'#d2691e')
    ax4.set_xlim(0, 4)
    # ax4.set_ylabel(r"$-v(k)g(k)$ (normalized)", fontsize=15)
    ax4.text(.9, .9, "d.", fontsize=16, transform=ax4.transAxes)
    ax4.legend(loc=(.4, .7))

    fig.text(.5, .025, r"Electron Wave Vector [$\sqrt{2 m^{*} E_\mathrm{po}} / \hbar$]",
             va='bottom', ha='center', fontsize=15)

    if save:
        plt.savefig(path.expanduser("~/Dropbox/Code/bte35/figures/nw_total_tau_vkgk_sb{}_sb{}.pdf"
                                    .format(j1, j2)))
    if show:
        plt.show()


def bulk_tau_and_gk(i, show=True, save=False, mat=GaAs, T=300, n=1e17 * 1e6):
    solver = RodeSolver(mat, T=300, Rc=1, n=n)
    m = solver.mat.get_meG(T)
    f = solver.SPACES['f']
    Npo = solver.Npo
    kpo = np.sqrt(2. * m * solver.Epo) / const.hbar
    ks = solver.SPACES['k']

    fig = plt.figure(figsize=(8, 3.5))
    plt.subplots_adjust(bottom=.2, wspace=.3, hspace=.1)

    ax1 = fig.add_subplot(121)
    ax1.set_yscale('log')
    rate_el = solver.SPACES['r_el']
    S_in_minus = (Npo + f) * solver.SPACES['li-']
    S_in_plus = (Npo + 1 - f) * solver.SPACES['li+']
    S_out_minus = (Npo + 1 - f) * solver.SPACES['lo-']
    S_out_plus = (Npo + f) * solver.SPACES['lo+']
    ax1.plot(ks / kpo, rate_el, label=r'ac$+$pe$+$ii', linestyle='--')
    ax1.plot(ks / kpo, S_in_minus, label=r'$S_\mathrm{in}^{-}$')
    ax1.plot(ks / kpo, S_out_minus, label=r'$S_\mathrm{out}^{-}$')
    ax1.plot(ks / kpo, S_in_plus, label=r'$S_\mathrm{in}^{+}$')
    ax1.plot(ks / kpo, S_out_plus, label=r'$S_\mathrm{out}^{+}$')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(1e8, 1e14)
    ax1.set_ylabel(r"Scattering Rate [s$^{-1}$]", fontsize=15)
    ax1.text(.9, .9, "a.", fontsize=16, transform=ax1.transAxes)
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(122)
    dfdk = solver.SPACES['dfdk']
    dEfdx = solver.SPACES['dEfdx']
    E = solver.SPACES['E']

    # xi in general is:
    # -1/hbar * dfdk * (e*epslion + dEfdx + (E-Ef)/T * dTdx)
    xi_eps_is_0 = -1. / const.hbar * dfdk * (dEfdx + (E - solver.Ef) / solver.T * solver.dTdx)
    xi_dTdx_is_0 = -const.e / const.hbar * dfdk * solver.eps_field
    # xi_F_is_0 = -1 / const.hbar * dfdk * (E - solver.Ef) / solver.T * solver.dTdx
    gF = solver.g_dist(i, xi_dTdx_is_0)
    gT = solver.g_dist(i, xi_eps_is_0)
    vk = solver.SPACES['v']
    ax2.plot(ks / kpo, -ks**2 * vk * gF / np.max(np.abs(-ks**2 * vk * gF)), label='F',
             color=u'#4b0082')
    ax2.plot(ks / kpo, -ks**2 * vk * gT / np.max(np.abs(-ks**2 * vk * gT)), label='gradT',
             color=u'#d2691e')
    ax2.set_xlim(0, 4)
    ax2.set_ylabel(r"-$k^{2}v(k)g(k)$ (normalized)", fontsize=15)
    ax2.text(.53, .82, r"$\varepsilon=0$", transform=ax2.transAxes, fontsize=15)
    ax2.text(.70, .45, r"$\frac{dT}{dx}=0$", transform=ax2.transAxes, fontsize=15)
    ax2.arrow(x=.68, y=.43, dx=-.1, dy=-.1, head_width=.015, fc='k', transform=ax2.transAxes)
    ax2.text(.9, .9, "b.", fontsize=16, transform=ax2.transAxes)
    # ax2.legend()

    fig.text(.5, .025, r"Electron Wave Vector [$\sqrt{2 m^{*} E_\mathrm{po}} / \hbar$]",
             va='bottom', ha='center', fontsize=15)

    if save:
        plt.savefig(path.expanduser("~/Dropbox/Code/bte35/figures/bulk_total_tau_vkgk.pdf"))
    if show:
        plt.show()


def nwrta_transport_n(R, num_n, show=True, save=False, T=300):
    mats = [GaAs, InAs, InSb, InP]

    # nmax_mat = {GaAs: 19.5, InAs: 18.55, InSb: 18., InP: 19.5}
    # nmax_mat = {GaAs: 20., InAs: 19., InSb: 18.5, InP: 20.}
    ns_mat = {mat: np.logspace(16, 21, num_n) * 1e6 for mat in mats}

    nsubs_mat = {GaAs: 50, InAs: 50, InSb: 50, InP: 50}
    solvers = {mat: NWRTAsolver(mat, T, R, n_subs=nsubs_mat[mat]) for mat in mats}
    sigmas = {}
    Ss = {}
    kappas = {}
    ZTs = {}

    for mat in mats:
        print("working on {}".format(mat.name))
        nws = solvers[mat]
        ns = ns_mat[mat]
        S_vals = np.zeros(ns.shape)
        sigma_vals = np.zeros(ns.shape)
        kappa_vals = np.zeros(ns.shape)
        for i, n in enumerate(ns):
            nws.n = n

            S = nws.S()
            sigma = nws.sigma()
            kappa_e = nws.kappa_e()

            if abs(S) > 1e-3:
                print("S = {:.4e} @ n = {:.4e}".format(S, n))
                S_vals[i] = np.nan
                sigma_vals[i] = sigma
                kappa_vals[i] = np.nan
            else:
                S_vals[i] = S
                sigma_vals[i] = sigma
                kappa_vals[i] = kappa_e

        Ss[mat] = S_vals
        sigmas[mat] = sigma_vals
        kappas[mat] = kappa_vals
        ZTs[mat] = S_vals**2 * sigma_vals * T / (kappa_vals + mat.kappa_bulk)

    colors = {GaAs: 'r', InAs: 'g', InSb: 'b', InP: u'#daa520'}

    fig = plt.figure(figsize=(8, 7))
    plt.subplots_adjust(bottom=.12, wspace=.3, hspace=.2)

    # Seebeck plot
    ax1 = fig.add_subplot(221)
    for mat in mats:
        ax1.plot(ns_mat[mat] / 1e6, -Ss[mat] * 1e6, color=colors[mat], label=mat.name)
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_xticks([1e16, 1e17, 1e18, 1e19, 1e20])
    ax1.set_ylabel(r"$|S|$ [$\mu$V/K]", fontsize=15)
    ax1.text(.1, .1, "a.", fontsize=16, transform=ax1.transAxes)

    # sigma plot
    ax2 = fig.add_subplot(222)
    for mat in mats:
        ax2.plot(ns_mat[mat] / 1e6, sigmas[mat] / 1e2, color=colors[mat])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks([1e16, 1e17, 1e18, 1e19, 1e20])
    ax2.set_ylabel(r"$\sigma$ [S/cm]", fontsize=15)
    ax2.text(.1, .9, "b.", fontsize=16, transform=ax2.transAxes)

    # kappa plot
    ax3 = fig.add_subplot(223)
    for mat in mats:
        ax3.plot(ns_mat[mat] / 1e6, kappas[mat], color=colors[mat])
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xticks([1e16, 1e17, 1e18, 1e19, 1e20])
    ax3.set_ylabel(r"$\kappa_e$ [W/m/K]", fontsize=15)
    ax3.text(.1, .9, "c.", fontsize=16, transform=ax3.transAxes)

    # ZT plot
    ax4 = fig.add_subplot(224)
    for mat in mats:
        ax4.plot(ns_mat[mat] / 1e6, ZTs[mat], color=colors[mat])
    ax4.set_xscale('log')
    ax4.set_xticks([1e16, 1e17, 1e18, 1e19, 1e20])
    ax4.set_ylabel(r"$ZT$")
    ax4.text(.1, .9, "d.", fontsize=16, transform=ax4.transAxes)

    fig.text(.5, .025, r"Electron Concentration [cm$^{-3}$]",
             va='bottom', ha='center', fontsize=15)

    if save:
        plt.savefig(path.expanduser("~/Dropbox/Code/bte35/figures/nwrta_transport.pdf"))
    if show:
        plt.show()


def bulk_transport_n(num_n, T=300, save=False, show=True):
    mats = [GaAs, InAs, InSb, InP]
    ns = np.logspace(16, 20, num_n) * 1e6
    solvers = {mat: RodeSolver(mat, T, 1.) for mat in mats}
    sigmas = {}
    Ss = {}
    kappas = {}
    ZTs = {}

    for mat in mats:
        print("working on {}".format(mat.name))
        nws = solvers[mat]
        S_vals = np.zeros(ns.shape)
        sigma_vals = np.zeros(ns.shape)
        kappa_vals = np.zeros(ns.shape)
        for i, n in enumerate(ns):
            nws.n = n

            S = nws.S()
            sigma = nws.sigma()
            kappa_e = nws.kappa_e()

            S_vals[i] = S
            sigma_vals[i] = sigma
            kappa_vals[i] = kappa_e

        Ss[mat] = S_vals
        sigmas[mat] = sigma_vals
        kappas[mat] = kappa_vals
        ZTs[mat] = S_vals**2 * sigma_vals * T / (kappa_vals + mat.kappa_bulk)

    colors = {GaAs: 'r', InAs: 'g', InSb: 'b', InP: u'#daa520'}

    fig = plt.figure(figsize=(8, 7))
    plt.subplots_adjust(bottom=.12, wspace=.3, hspace=.2)

    # Seebeck plot
    ax1 = fig.add_subplot(221)
    for mat in mats:
        ax1.plot(ns / 1e6, -Ss[mat] * 1e6, color=colors[mat], label=mat.name)
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_xticks([1e16, 1e17, 1e18, 1e19, 1e20])
    ax1.set_ylabel(r"$|S|$ [$\mu$V/K]", fontsize=15)
    ax1.text(.1, .1, "a.", fontsize=16, transform=ax1.transAxes)

    # sigma plot
    ax2 = fig.add_subplot(222)
    for mat in mats:
        ax2.plot(ns / 1e6, sigmas[mat] / 1e2, color=colors[mat])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks([1e16, 1e17, 1e18, 1e19, 1e20])
    ax2.set_ylabel(r"$\sigma$ [S/cm]", fontsize=15)
    ax2.text(.1, .9, "b.", fontsize=16, transform=ax2.transAxes)

    # kappa plot
    ax3 = fig.add_subplot(223)
    for mat in mats:
        ax3.plot(ns / 1e6, kappas[mat], color=colors[mat])
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xticks([1e16, 1e17, 1e18, 1e19, 1e20])
    ax3.set_ylabel(r"$\kappa_e$ [W/m/K]", fontsize=15)
    ax3.text(.1, .9, "c.", fontsize=16, transform=ax3.transAxes)

    # ZT plot
    ax4 = fig.add_subplot(224)
    for mat in mats:
        ax4.plot(ns / 1e6, ZTs[mat], color=colors[mat])
    ax4.set_xscale('log')
    ax4.set_xticks([1e16, 1e17, 1e18, 1e19, 1e20])
    ax4.set_ylabel(r"$ZT$")
    ax4.text(.1, .9, "d.", fontsize=16, transform=ax4.transAxes)

    fig.text(.5, .025, r"Electron Concentration [cm$^{-3}$]",
             va='bottom', ha='center', fontsize=15)

    if save:
        plt.savefig(path.expanduser("~/Dropbox/Code/bte35/figures/bulk_transport.pdf"))
    if show:
        plt.show()


if __name__ == '__main__':
    # nanowire_scattering_rates(show=True, save=True)
    # nwrta_tau_and_gk(j1=0, j2=4, show=True, save=True)
    # bulk_tau_and_gk(i=30, show=True, save=False)
    nwrta_transport_n(R=5e-9, num_n=20, save=False, show=True)
    # bulk_transport_n(num_n=100, save=True, show=True)
    pass
