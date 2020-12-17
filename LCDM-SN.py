import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d
from numba import jit
from SN import mu_teo_f2py
from Data import *


plt.rcParams['figure.figsize'] =(14,12)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 29
plt.rcParams['legend.fancybox'] = True


light_speed = 299792.458 #light speed
Z_MAX = 1600. # max redshift to integrate 
N_INTERP = 2000 #number of step for interpolation



@jit
def H(a, H0, Obh2, Och2):
    h = H0 / 100.0
    Om = (Obh2 + Och2) / h ** 2
    Or = (2.469e-5) * (h ** (-2.0)) * (1.0 + ((7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)) * 3.046e0)

    return H0*np.sqrt((Om*(a)**(-3) + Or*(a)**(-4) + 1.0 - Om - Or))


@jit
def chi(a, H0, Obh2, Och2):
    aa = np.concatenate(
        (np.logspace(np.log10(1. / (1. + Z_MAX)), np.log10(0.7), N_INTERP),
         np.logspace(np.log10(0.71), 0., N_INTERP)))

    integ = 1. / (aa ** 2.) / (H(aa, H0, Obh2, Och2))
    intchi = -cumtrapz(integ[::-1], aa[::-1], initial=0.)[::-1]

    
    interp_intchi = interp1d(aa, intchi, kind='linear', bounds_error=False)

    return light_speed * interp_intchi(a)

@jit
def DLz(z_hel, z_cmb, H0, Obh2, Och2):
    a1 = 1.0 / (1.0 + z_hel) # z_hel is heliocentric redfhit
    a2 = 1.0 / (1.0 + z_cmb) # z_cmb is CMB frame redshift
    return chi(a2, H0, Obh2, Och2)/a1


@jit
def model(z, H0, Obh2, Och2):
    a = 1.0 / (1.0 + z)
    return 1.0/H(a, H0, Obh2, Och2)


@jit
def DL(z_hel, z_cmb, H0, Obh2, Och2):
    return light_speed *(1.0 + z_hel)* quad(model, 0.0, z_cmb, args=(H0, Obh2, Och2))[0]

DL = np.vectorize(DL)

@jit
def mu_teo_quad(z_hel, z_cmb, H0, Obh2, Och2):
    return 5.0*np.log10(DL(z_hel, z_cmb, H0, Obh2, Och2)) + 25.0


@jit
def mu_teo(z_hel, z_cmb, H0, Obh2, Och2):
    return 5.0*np.log10(DLz(z_hel, z_cmb, H0, Obh2, Och2)) + 25.0



H0, Obh2, Och2 = 67.4, 0.0224, 0.120 

z = np.linspace(np.min(zcmb), np.max(zcmb), 2000)



plt.errorbar(zcmb, mb + 19.10, yerr=dmu, fmt = 'ko',
             markersize=4, ecolor='grey', elinewidth = 2, capsize=5, capthick=2)

plt.plot(z, mu_teo(z, z, H0, Obh2, Och2), 'k-', linewidth='4', label='scipy.integrate.cumptraz')
plt.plot(z, mu_teo_quad(z, z, H0, Obh2, Och2), 'r:', linewidth='4', label='scipy.integrate.quad')
plt.plot(z, mu_teo_f2py(z, z, H0, Obh2, Och2), 'b--', linewidth='4', label='F2py')

plt.ylabel(r'$\mu(z)$')
plt.xlabel(r'$z$')
plt.tick_params(which='major',axis='x', length=14, width=4, direction='in')
plt.tick_params(which='minor',axis='x', length=7, width=2, direction='in')
plt.tick_params(which='major',axis='y', length=14, width=4, direction='in')
plt.tick_params(pad=20)
plt.tight_layout()
plt.legend(fontsize=25)
plt.savefig('figures/SNeIa.pdf', dpi=800)
#plt.show()

