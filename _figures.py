import matplotlib as mpl
import matplotlib.pyplot as plt

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


def nanowire_scattering_rates(show=True, save=False,
                              mat=GaAs, T=300, R=15e-9, n=1e17 * 1e6):
    nwsolver = NWRTAsolver(mat, T, R, n=n)
    Es = np.linspace(1e-4 * const.e, .2 * const.e, 1000)
    ks = nwsolver.k_CB(Es)

    rate_pe = nwsolver.r_pe(ks)
    rate_ac = nwsolver.r_ac(ks)
    rate_po = nwsolver.r_po(ks)
    rate_ii = nwsolver.r_ii(ks)
    rate_tot = rate_ac + rate_pe + rate_po + rate_ii

    fig, ax = plt.subplots(tight_layout=True)
    ax.set_yscale('log')

    E_po = nwsolver.Epo
    ax.plot(Es / E_po, rate_pe, label='pe')
    ax.plot(Es / E_po, rate_ac, label='ac')
    ax.plot(Es / E_po, rate_po, label='po')
    ax.plot(Es / E_po, rate_ii, label='ii')
    ax.plot(Es / E_po, rate_tot, label='total', linewidth=1.5, linestyle='--', color='k')

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlim(0, 4)
    ax.set_xlabel(r'$E/E_\mathrm{po}$', fontsize=label_texfontsize)

    ax.set_ylim((5e8, 1e14))
    ax.set_ylabel(r'Scattering Relaxation Rate [s$^{-1}$]', fontsize=label_fontsize)

    # descriptions
    ax.text(.35, .9, "{}".format(mat.name), fontsize=label_fontsize,
            transform=ax.transAxes)
    ax.text(.35, .82, r"$R = {:.1f}$".format(R * 1e9) + " nm", fontsize=label_fontsize,
            transform=ax.transAxes)
    ax.text(.35, .74, r"$T = {:d}$".format(T) + " K", fontsize=label_fontsize,
            transform=ax.transAxes)

    ax.legend()

    if save:
        plt.savefig(path.expanduser("~/Dropbox/Code/bte35/figures/nw_scatt_elastic.pdf"))
    if show:
        plt.show()


def nw_tau_and_gk(show=True, save=False, mat=GaAs, T=300, R=15e-9, n=1e17 * 1e6):
    nwsolver = NWRTAsolver(mat, T, R, n=n)
    m = nwsolver.meG
    kpo = np.sqrt(2 * m * nwsolver.Epo) / const.hbar

    fig = plt.figure(figsize=(8, 3.5))
    plt.subplots_adjust(bottom=.2, wspace=.3, hspace=.1)

    ks = np.linspace(1e4, 4. * kpo, 1000)

    ax1 = fig.add_subplot(121)
    ax1.set_yscale('log')
    rate_pe = nwsolver.r_pe(ks)
    rate_ac = nwsolver.r_ac(ks)
    rate_po = nwsolver.r_po(ks)
    rate_ii = nwsolver.r_ii(ks)
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
    ax1.legend()

    ax2 = fig.add_subplot(122)
    gF = nwsolver.g_dist(ks, gradT=0, F=1e4)
    gT = nwsolver.g_dist(ks, gradT=1e3, F=0)
    vk = const.hbar * ks / m
    ax2.plot(ks / kpo, -vk * gF / np.max(np.abs(vk * gF)), label='F', color=u'#4b0082')
    ax2.plot(ks / kpo, -vk * gT / np.max(np.abs(vk * gT)), label='T', color=u'#d2691e')
    ax2.text(.47, .45, r"$\mathrm{F}=0$", transform=ax2.transAxes, fontsize=15)
    ax2.text(.65, .25, r"$\frac{dT}{dx}=0$", transform=ax2.transAxes, fontsize=15)
    ax2.arrow(x=.63, y=.23, dx=-.1, dy=-.1, head_width=.015, fc='k', transform=ax2.transAxes)
    ax2.set_xlim(0, 4)
    ax2.set_ylabel(r"$-v(k)g(k)$ (normalized)", fontsize=15)
    ax2.text(.9, .9, "b.", fontsize=16, transform=ax2.transAxes)
    # ax2.legend()

    fig.text(.5, .025, r"Electron Wave Vector [$\sqrt{2 m^{*} E_\mathrm{po}} / \hbar$]",
             va='bottom', ha='center', fontsize=15)

    if save:
        plt.savefig(path.expanduser("~/Dropbox/Code/bte35/figures/nw_total_tau_vkgk.pdf"))
    if show:
        plt.show()


def bulk_tau_and_gk(show=True, save=False, mat=GaAs, T=300, R=15e-9, n=1e17 * 1e6):
    solver = RodeSolver(mat, T=300, Rc=1, n=1e17 * 1e6)
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
    gF = solver.g_dist(30, 'F')
    gT = solver.g_dist(30, 'gradT')
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


if __name__ == '__main__':
    # nanowire_scattering_rates(show=True, save=True)
    nw_tau_and_gk(show=True, save=True)
    # bulk_tau_and_gk(show=True, save=True)
    pass
