import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d
import json
from priors import Priors
import pymultinest
import time
from numba import jit
from SN import mu_teo_f2py
from BAO2D import bao2d
from Data import *



t0 = time.time() #initial time

light_speed = 299792.458 #light speed
Z_MAX = 1600. # max redshift to integrate 
N_INTERP = 2000 #number of step for interpolation



@jit
def H(a, H0, Obh2, Och2): 

    h = H0 / 100.0
    Om = (Obh2 + Och2) / h ** 2
    Or = ((2.469e-5) * (1.0 + ((7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)) * 3.046))*(h ** (-2.0)) 

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
def DLz(z_hel, z_cmb, H0, Obh2, Och2):

    a1 = 1.0 / (1.0 + z_hel) # z_hel is heliocentric redfhit
    a2 = 1.0 / (1.0 + z_cmb) # z_cmb is CMB frame redshift

    return chi(a2, H0, Obh2, Och2)/a1


@jit
def inverse_H(z, H0, Obh2, Och2):

    a = 1.0 / (1.0 + z)

    return 1.0/H(a, H0, Obh2, Och2)


@jit
def DL(z_hel, z_cmb, H0, Obh2, Och2):
    return light_speed *(1.0 + z_hel)* quad(inverse_H, 0.0, z_cmb, args=(H0, Obh2, Och2))[0]

DL = np.vectorize(DL)


@jit
def rs(a, H0, Obh2, Och2):

    aa = np.concatenate(
        (np.logspace(np.log10(1e-14), np.log10(0.7), N_INTERP),
         np.logspace(np.log10(0.71), 0., N_INTERP)))

    Ob = Obh2* (100.0/H0)**(2.0)
    Og = 2.469e-5 * (100.0/H0)**(2.0)


    R = (3.0 * Ob) / (4.0 * Og/aa)
    cs = light_speed / np.sqrt(3.0 * (1.0 + R)) /aa/aa
    integ =cs/H(aa, H0, Obh2, Och2)


    intchi = cumtrapz(integ, aa, initial=0.)

    interp_intchi = interp1d(aa, intchi, kind='linear', bounds_error=False)


    return interp_intchi(a)

@jit
def fun(z, H0, Obh2, Och2):

    Ob = Obh2 * (100.0/H0)**(2.0)
    Og = 2.469e-5 * (100.0/H0)**(2.0)

    R=(3.0*Ob)/(4.0*Og*(1.0+z))
    cs=light_speed/np.sqrt(3.0*(1.0+R))

    return cs*inverse_H(z, H0, Obh2, Och2)

@jit
def rss(z, H0, Obh2, Och2):
    return quad(fun,z,np.inf,args=(H0, Obh2, Och2))[0]

@jit
def DA(z, H0, Obh2, Och2):
    return (light_speed / (1.0 + z)) * (quad(inverse_H, 0.0, z, args=(H0, Obh2, Och2))[0])

DA = np.vectorize(DA)

@jit
def ang_quad(z, H0, Obh2, Och2):

    b1 = 0.313 * (Obh2 + Och2) ** (-0.419) * (1.0 + 0.607 * (Obh2 + Och2) ** (0.674))
    b2 = 0.238 * (Obh2 + Och2) ** (0.223)

    zd = (1345.0 * (Obh2 + Och2) ** (0.251) * (1.0 + b1 * (Obh2) ** (b2))) / (
		1.0 + 0.659 * (Obh2 + Och2) ** (0.828))


    return rss(zd,H0, Obh2, Och2)/((1.0+z)*DA(z,H0, Obh2, Och2))


@jit
def ang(z, H0, Obh2, Och2):

    a = 1.0/(1.0+z)

    b1 = 0.313 * (Obh2 + Och2) ** (-0.419) * (1.0 + 0.607 * (Obh2 + Och2) ** (0.674))
    b2 = 0.238 * (Obh2 + Och2) ** (0.223)

    zd = (1345.0 * (Obh2 + Och2) ** (0.251) * (1.0 + b1 * (Obh2) ** (b2))) / (
		1.0 + 0.659 * (Obh2 + Och2) ** (0.828))

    ad = 1.0/(1.0+zd)

    return rs(ad, H0, Obh2, Och2)/chi(a, H0, Obh2, Och2)


@jit
def X2SN_cumtrapz(H0, Obh2, Och2):
    I = np.ones(1048)
    m_teo = 5.0*np.log10(DLz(zhel, zcmb, H0, Obh2, Och2))
    Dmu = mb - m_teo 
    S1 = np.dot(Dmu, np.dot(invSN, Dmu))
    S2 = np.dot(Dmu, np.dot(invSN, I))
    S3 = np.dot(I, np.dot(invSN, I))
    return S1 - S2**2.0/S3 + np.log(S3/(2.e0*np.pi))

@jit
def X2SN_quad(H0, Obh2, Och2):
    I = np.ones(1048)
    m_teo = 5.0*np.log10(DL(zhel, zcmb, H0, Obh2, Och2))
    Dmu = mb - m_teo 
    S1 = np.dot(Dmu, np.dot(invSN, Dmu))
    S2 = np.dot(Dmu, np.dot(invSN, I))
    S3 = np.dot(I, np.dot(invSN, I))
    return S1 - S2**2.0/S3 + np.log(S3/(2.e0*np.pi))

@jit
def X2SN_f2py(H0, Obh2, Och2):
    I = np.ones(1048)
    m_teo = mu_teo_f2py(zhel, zcmb, H0, Obh2, Och2)
    Dmu = mb - m_teo 
    S1 = np.dot(Dmu, np.dot(invSN, Dmu))
    S2 = np.dot(Dmu, np.dot(invSN, I))
    S3 = np.dot(I, np.dot(invSN, I))
    return S1 - S2**2.0/S3 + np.log(S3/(2.e0*np.pi))



@jit
def X2B2_cumtrapz(H0, Obh2, Och2):
    return np.array([sum(((ang(z_theta,H0, Obh2, Och2)-theta)/theta_err)**2.0)])


@jit
def X2B2_quad(H0, Obh2, Och2):
    return np.array([sum(((ang_quad(z_theta,H0, Obh2, Och2)-theta)/theta_err)**2.0)])

@jit
def X2B2_f2py(H0, Obh2, Och2):
    return np.array([sum(((bao2d(z_theta,H0, Obh2, Och2)-theta)/theta_err)**2.0)])


@jit
def prior(cube,ndim,nparams):
    cube[0] = Priors().GaussianPrior(cube[0], 74.03, 1.42)
    cube[1] = Priors().GaussianPrior(cube[1], 0.02235, 0.00016) 
    cube[2] = Priors().UniformPrior(cube[2], 0.001, 0.99)


@jit
def loglike(cube,ndim,nparams):
    H0, Obh2, Och2 = cube[0], cube[1],cube[2]
    loglikelihood = -0.5e0 * (X2SN_cumtrapz(H0, Obh2, Och2))
    return loglikelihood


parameters = ["H0", "Obh2", "Och2"]
n_params = len(parameters)



for i in range(1, 6):
    out=f'chains/SN/cumtrapz-run_{i}'


    pymultinest.run(loglike, prior, n_params, outputfiles_basename=str(out), verbose = True,n_live_points=1000, evidence_tolerance=0.1,sampling_efficiency=0.3,max_iter=500000,resume=False, init_MPI=False)

    json.dump(parameters, open(str(out)+'params.json', 'w')) # save parameter name

t1 = time.time()
time = (t1 - t0)

print('time = {0} sec'.format(time))













