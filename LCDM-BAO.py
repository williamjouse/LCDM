import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d
from numba import jit
from BAO2D import bao2d
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
def DAz(z, H0, Obh2, Och2):
    a = 1.0/(1.0 + z)
    _chi = chi(a, H0, Obh2, Och2)
    return  a * _chi


@jit
def model(z, H0, Obh2, Och2):
    a = 1.0 / (1.0 + z)
    return 1.0/H(a, H0, Obh2, Och2)


@jit
def rs(a, H0, Obh2, Och2):
    aa = np.concatenate(
        (np.logspace(np.log10(1. / (1. + Z_MAX)), np.log10(0.7), N_INTERP),
         np.logspace(np.log10(0.71), 0., N_INTERP)))

    Ob = Obh2* (100.0/H0)**(2.0)
    Og = 2.469e-5 * (100.0/H0)**(2.0)

    R = (3.e0 * Ob) / (4.e0 * Og * aa)
    cs = light_speed / np.sqrt(3.e0 * (1.e0 + R)) / aa / aa
    integ =cs/H(aa, H0, Obh2, Och2)


    intchi = -cumtrapz(integ[::-1], aa[::-1], initial=0.)[::-1]

    interp_intchi = interp1d(aa, intchi, kind='linear', bounds_error=False)


    return interp_intchi(a)

@jit
def fun(z, H0, Obh2, Och2):
    Ob = Obh2* (100.0/H0)**(2.0)
    Og = 2.469e-5 * (100.0/H0)**(2.0)

    R=(3.0*Obh2* (100.0/H0)**(2.0))/(4.0*2.469e-5* (100.0/H0)**2*(1.e0+z))

    cs=light_speed/np.sqrt(3.0*(1.e0+R))

    return cs*model(z, H0, Obh2, Och2)

@jit
def rss(z, H0, Obh2, Och2):
    return quad(fun,z,np.inf,args=(H0, Obh2, Och2))[0]

@jit
def DA(z, H0, Obh2, Och2):
    return (light_speed / (1.0 + z)) * (quad(model, 0.0, z, args=(H0, Obh2, Och2))[0])

DA = np.vectorize(DA)

@jit
def ang_quad(z, H0, Obh2, Och2):###Params
    b1 = 0.313e0 * (Obh2 + Och2) ** (-0.419e0) * (1.e0 + 0.607 * (Obh2 + Och2) ** (0.674e0))
    b2 = 0.238e0 * (Obh2 + Och2) ** (0.223e0)

    zd = (1291e0 * (Obh2 + Och2) ** (0.251e0) * (1.e0 + b1 * (Obh2) ** (b2))) / (
		1.e0 + 0.659e0 * (Obh2 + Och2) ** (0.828))

    return rss(zd,H0, Obh2, Och2)/((1.0+z)*DA(z,H0, Obh2, Och2))


@jit
def ang(z, H0, Obh2, Och2):
    a = 1.0/(1.0+z)
    b1 = 0.313e0*(Obh2+Och2)**(-0.419e0)*(1.e0 + 0.607*(Obh2+Och2)**(0.674e0))
    b2 = 0.238e0*(Obh2+Och2)**(0.223e0)

    zd = (1291e0*(Obh2+Och2)**(0.251e0)*(1.e0+b1*(Obh2)**(b2)))/(1.e0+0.659e0*(Obh2+Och2)**(0.828))
    ad = 1.0/(1.0+zd)

    return rs(ad, H0, Obh2, Och2)/chi(a, H0, Obh2, Och2)




H0, Obh2, Och2 = 67.4, 0.0222, 0.120

z = np.linspace(0.22, 2.25, 2000)

plt.errorbar(z_theta,theta, yerr=theta_err, fmt = 'ko',
             markersize=4, ecolor='grey', elinewidth = 2, capsize=5, capthick=2)

plt.plot(z, ang(z, H0, Obh2, Och2), 'k-', linewidth='4', label='scipy.integrate.cumptraz')
plt.plot(z,ang_quad(z, H0, Obh2, Och2), 'r:', linewidth='4', label='scipy.integrate.quad')
plt.plot(z,bao2d(z, H0, Obh2, Och2), 'b--', linewidth='4', label='F2py')

plt.ylabel(r'$\mu(z)$')
plt.xlabel(r'$z$')
plt.tick_params(which='major',axis='x', length=14, width=4, direction='in')
plt.tick_params(which='minor',axis='x', length=7, width=2, direction='in')
plt.tick_params(which='major',axis='y', length=14, width=4, direction='in')
plt.tick_params(pad=20)
plt.tight_layout()
plt.legend(fontsize=25)
plt.savefig('figures/BAO.pdf', dpi=800)
#plt.show()

