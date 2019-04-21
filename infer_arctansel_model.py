import numpy as np
import h5py
import emcee
import os
import sys
from scipy.stats import truncnorm, skewnorm, multivariate_normal
from scipy.interpolate import splrep
from scipy.special import erf
import ndinterp
import pickle
import wl_cosmology
from wl_cosmology import Mpc, M_Sun, c, G
from wl_profiles import sersic, nfw
import ndinterp
from fixed_hyperparameters import *


kpc = Mpc/1000.

f = open('summary_table.txt', 'r')
lines = f.readlines()
f.close()

nsl = len(lines)-1

zs = []
tein_obs = []
tein_err = []

rein_phys = []

lmchab_obs = []
lmchab_err = []

nint = 100000

lmchab_samp = []
lm200_samp = []
laimf_samp = []
lc200_samp = []

ein_frac = []
rhoc = []

crosssect_grids = []
tein_grids = []
lm200_grids = []
lmstar_grids = []
lc200_grids = []

for line in lines[1:]:
    line = line.split()
    name = line[0]

    zd = float(line[1])
    zs_here = float(line[2])
    zs.append(zs_here)

    tein_here = float(line[3])
    tein_obs.append(tein_here)
    tein_err.append(0.1*tein_here)

    lmchab_obs.append(float(line[6]))
    lmchab_err.append(float(line[7]))

    dd = wl_cosmology.Dang(zd)
    ds = wl_cosmology.Dang(zs_here)
    dds = wl_cosmology.Dang(zs_here, zd)

    arcsec2kpc = np.deg2rad(1./3600.) * dd * 1000.
    rhoc.append(wl_cosmology.rhoc(zd))

    rein_phys_here = tein_here * arcsec2kpc
    rein_phys.append(rein_phys_here)

    s_cr = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2

    mein_here = np.pi * rein_phys_here**2 * s_cr
    mein_obs.append(mein_here)
    mein_err.append(0.2*mein_here)

    reff1 = float(line[4])
    nser1 = float(line[5])
    
    # defines the stellar profile
    ein_frac.append(sersic.M2d(tein_here, nser1, reff1))

    # draws a sample of values of lmchab, to be used for importance sampling and MC integration
    lmchab_samp_here = skewnorm.rvs(10.**mchab_logskew, mchab_mu, mchab_sig, nint)
    lmchab_samp.append(lmchab_samp_here)
    lm200_samp.append(np.random.normal(0., 1., nint))
    lc200_samp.append(np.random.normal(0., 1., nint))
    laimf_samp.append(np.random.normal(0., 1., nint))

    grid_file = h5py.File('grids/%s_tein_crosssect_grid.hdf5'%name, 'r')

    lmstar_grid = grid_file['lmstar_grid'].value.copy()
    nmstar = len(lmstar_grid)

    lm200_grid = grid_file['lm200_grid'].value.copy()
    nm200 = len(lm200_grid)

    lc200_grid = grid_file['lc200_grid'].value.copy()
    nc200 = len(lc200_grid)

    axes = {0: splrep(lm200_grid, np.arange(nm200)), 1: splrep(lmstar_grid, np.arange(nmstar)), 2: splrep(lc200_grid, np.arange(nc200))}

    tein_grid = ndinterp.ndInterp(axes, grid_file['tein_grid'].value, order=1)
    crosssect_grid = ndinterp.ndInterp(axes, grid_file['crosssect_grid'].value, order=1)

    tein_grids.append(tein_grid)
    crosssect_grids.append(crosssect_grid)

    lm200_grids.append(lm200_grid)
    lc200_grids.append(lc200_grid)
    lmstar_grids.append(lmstar_grid)

    grid_file.close()

# reads in the file with the cross-section calculated over an important sample with interim prior

nimpsamp = 10

cs_samp = {}
cs_pars = ['lm200', 'laimf', 'lmchab', 'zs', 'tein', 'crosssect']
for par in cs_pars:
    cs_samp[par] = []

for i in range(nimpsamp):
    crosssect_file = h5py.File('crosssect_impsamp_%02d.hdf5'%i, 'r')
    for par in cs_pars:
        cs_samp[par].append(crosssect_file['%s_samp'%par].value.copy())

for par in cs_pars:
    cs_samp[par] = np.array(cs_samp[par]).flatten()

# copies the values of the hyper-parameters describing the interim prior on m200, aimf, zs distribution
cs_ip_pars = {}
group = crosssect_file['prior-parameters']
for par in group:
    cs_ip_pars[par] = group[par].value.copy()

# reads in MCMC of weak lensing inference, to use as a prior on m200.
f = open('wl_prior.dat', 'r')
wl_chain = pickle.load(f)
f.close()

wl_burnin = 300

wl_samp = np.array([wl_chain['m200_mu'][:, wl_burnin:].flatten(), wl_chain['m200_sig'][:, wl_burnin:].flatten(), wl_chain['mchab_dep'][:, wl_burnin:].flatten()])
wl_cov = np.cov(wl_samp)
wl_mu = np.array([wl_samp[0, :].mean(), wl_samp[1, :].mean(), wl_samp[2, :].mean()])

nstep = 500
nwalkers = 50

m200_mu = {'name': 'm200_mu', 'lower': 12., 'upper': 14., 'guess': 12.8, 'step': 0.03}
m200_beta = {'name': 'm200_beta', 'lower': 0., 'upper': 3., 'guess': 1.8, 'step': 0.03}
m200_sig = {'name': 'm200_sig', 'lower': 0., 'upper': 3., 'guess': 0.4, 'step': 0.03}

zs_mu = {'name': 'zs_mu', 'lower': 0., 'upper': 3., 'guess': 1.5, 'step': 0.1}
zs_sig = {'name': 'zs_sig', 'lower': 0.03, 'upper': 3., 'guess': 0.6, 'step': 0.03}

aimf_mu = {'name': 'aimf_mu', 'lower': -0.5, 'upper': 0.5, 'guess': 0.1, 'step': 0.1}
aimf_sig = {'name': 'aimf_sig', 'lower': 0.01, 'upper': 1., 'guess': 0.1, 'step': 0.03}

logtsel_m = {'name': 'logtsel_m', 'lower': -1., 'upper': 3., 'guess': 0., 'step': 0.1}
tsel_t0 = {'name': 'tsel_t0', 'lower': 0., 'upper': 3., 'guess': 1., 'step': 0.1}

pars = [m200_mu, m200_beta, m200_sig, zs_mu, zs_sig, aimf_mu, aimf_sig, logtsel_m, tsel_t0]

npars = len(pars)

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

def tsel_func(tein, tsel_m, tsel_t0):
    return 1./np.pi * np.arctan(tsel_m*(tein - tsel_t0)) + 0.5

norm_min = 0.01

def norm_func(p):

    m200_mu, m200_beta, m200_sig, zs_mu, zs_sig, aimf_mu, aimf_sig, logtsel_m, tsel_t0 = p

    # prepares arrays for calculation of the norm of hyper-parameter distribution 
    m200_muhere = m200_mu + m200_beta * (cs_samp['lmchab'] - mchab_piv)

    cs_ip_m200_muhere = cs_ip_pars['m200_mu'] + cs_ip_pars['m200_beta']*(cs_samp['lmchab'] - mchab_piv)

    cs_m200_term = 1./m200_sig * np.exp(-0.5*(cs_samp['lm200'] - m200_muhere)**2/m200_sig**2)
    cs_m200_ip = 1./cs_ip_pars['m200_sig'] * np.exp(-0.5*(cs_samp['lm200'] - cs_ip_m200_muhere)**2/cs_ip_pars['m200_sig']**2)

    cs_aimf_term = 1./aimf_sig * np.exp(-0.5*(cs_samp['laimf'] - aimf_mu)**2/aimf_sig**2)
    cs_aimf_ip = 1./cs_ip_pars['aimf_sig'] * np.exp(-0.5*(cs_samp['laimf'] - cs_ip_pars['aimf_mu'])**2/cs_ip_pars['aimf_sig']**2)

    cs_zs_term = 1./zs_sig * np.exp(-0.5*(cs_samp['zs'] - zs_mu)**2/zs_sig**2)

    integrand = cs_m200_term*cs_aimf_term*cs_zs_term*cs_samp['crosssect']*tsel_func(cs_samp['tein'], 10.**logtsel_m, tsel_t0)/cs_m200_ip/cs_aimf_ip

    norm = integrand.mean()

    if norm < norm_min:
        norm = norm_min

    return norm

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300, -1e300

    m200_mu, m200_beta, m200_sig, zs_mu, zs_sig, aimf_mu, aimf_sig, logtsel_m, tsel_t0 = p

    prior_arr = np.array([m200_mu, m200_sig, m200_beta])

    sumlogp = multivariate_normal.logpdf(prior_arr, mean=wl_mu, cov=wl_cov) # prior on m200 from weak lensing

    norm = norm_func(p)

    # loops over the strong lenses
    for i in range(nsl):

        # rescales samples to their actual distribution
        laimf_here = laimf_samp[i]*aimf_sig + aimf_mu
        lm200_here = lm200_samp[i]*m200_sig + m200_beta * (lmchab_samp[i] - mchab_piv) + m200_mu
        lc200_here = lc200_samp[i]*c200_sig + c200_mu + c200_beta * (lm200_here - m200_piv)

        lm200_oob = lm200_here > lm200_grids[i][-1]
        lm200_here[lm200_oob] = lm200_grids[i][-1]

        lc200_oob = lc200_here > lc200_grids[i][-1]
        lc200_here[lc200_oob] = lc200_grids[i][-1]

        lmstar_here = lmchab_samp[i] + laimf_here
        lmstar_oob = lmstar_here > lmstar_grids[i][-1]
        lmstar_here[lmstar_oob] = lmstar_grids[i][-1]

        point = np.array((lm200_here, lmstar_here, lc200_here)).T

        crosssect_here = crosssect_grids[i].eval(point)
        tein_here = tein_grids[i].eval(point)

        sl_like_term = 1./tein_err[i]*np.exp(-0.5*(tein_obs[i] - tein_here)**2/tein_err[i]**2)

        lmchab_like_term = 1./lmchab_err[i] * np.exp(-0.5*(lmchab_obs[i] - lmchab_samp[i])**2/lmchab_err[i]**2)
        zs_term = 1./zs_sig*np.exp(-0.5*(zs_mu - zs[i])**2/zs_sig**2)

        sumlogp += np.log((lmchab_like_term*sl_like_term*zs_term*crosssect_here*tsel_func(tein_here, 10.**logtsel_m, tsel_t0)/norm).mean())

        if sumlogp != sumlogp:
            return -1e300, -1e300

    return sumlogp, norm

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=50)

start = []
if len(sys.argv) > 1:
    print 'using last step of %s to initialize walkers'%sys.argv[1]
    startfile = h5py.File('%s'%sys.argv[1], 'r')

    for i in range(nwalkers):
        tmp = np.zeros(npars)
        for n in range(npars):
            tmp[n] = startfile[pars[n]['name']][i, -1]
        start.append(tmp)
    startfile.close()

else:
    for i in range(nwalkers):
        tmp = np.zeros(npars)
        for j in range(npars):
            a, b = (bounds[j][0] - pars[j]['guess'])/pars[j]['step'], (bounds[j][1] - pars[j]['guess'])/pars[j]['step']
            p0 = truncnorm.rvs(a, b, size=1)*pars[j]['step'] + pars[j]['guess']
            tmp[j] = p0

        start.append(tmp)

print "Sampling"

sampler.run_mcmc(start, nstep)

blobchain = sampler.blobs

output_file = h5py.File('arctansel_inference.hdf5', 'w')
output_file.create_dataset('logp', data = sampler.lnprobability)
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, :, n])

