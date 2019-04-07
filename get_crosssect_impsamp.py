import numpy as np
import wl_cosmology
from wl_cosmology import G, M_Sun, c, Mpc
from wl_profiles import sersic, nfw
import h5py
from scipy.optimize import brentq, fminbound
from scipy.interpolate import splrep, splev, splint
from scipy.stats import truncnorm, skewnorm
from scipy.special import erf
import os
import sys
from fixed_hyperparameters import *
import sys


# calculates the strong lensing cross-section on a sample in multi-dimensional space defined by (mchab, nser, reff, zd, zs, aimf, m200, c200). This is needed for the computation of the normalization constant of the hyper-parameters probability distribution.

nsamp = 100000 # number of points in the sample (hopefully that's enough)

batchno = int(sys.argv[1])

# defines parameters of interim prior
zs_min = 0.
zs_max = 4.

m200_mu = 12.8
m200_sig = 0.5
m200_beta = 1.8

aimf_mu = 0.
aimf_sig = 0.2

minmag = 0.5 # minimum magnification of inner image for an object to be declared a strong lens.

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)

# range of allowed values for Einstein radius (in kpc)
xmin = 0.01
xmax = 1000.
nx = 1001
x_arr = np.logspace(np.log10(xmin), np.log10(xmax), nx)

# draws values of the lens and source redshift
a, b = (0. - zd_mu)/zd_sig, (np.inf - zd_mu)/zd_sig
zd_samp = truncnorm.rvs(a, b, size=nsamp)*zd_sig + zd_mu

zs_samp = np.random.rand(nsamp)*(zs_max - zs_min) + zs_min

# draws values of the stellar mass (mchab) from a skew Gaussian (as modeled in Sonnenfeld et al. 2019)
lmchab_samp = skewnorm.rvs(10.**mchab_logskew, mchab_mu, mchab_sig, nsamp)

# draws sample in m200, c200, aimf
lm200_samp = m200_mu + m200_beta * (lmchab_samp - mchab_piv) + np.random.normal(0., m200_sig, nsamp)
lc200_samp = c200_mu + c200_beta * (lm200_samp - m200_piv) + np.random.normal(0., c200_sig, nsamp)
laimf_samp = np.random.normal(aimf_mu, aimf_sig, nsamp)

# draws sample in nser, reff
lnser_samp = nser_mu + nser_beta * (lmchab_samp - mchab_piv)
lreff_samp = reff_mu + reff_nu * (lnser_samp - np.log10(4.)) + reff_beta * (lmchab_samp - mchab_piv)

# prepares arrays to store values of theta_ein and the lensing cross-section for each point of the sample
tein_samp = np.zeros(nsamp)
crosssect_samp = np.zeros(nsamp)

for i in range(nsamp): # loops over the sample

    if zs_samp[i] < zd_samp[i] + 0.01:
        # source is in front of the lens
        tein_samp[i] = 0.
        crosssect_samp[i] = 0.
    else:
        dd = wl_cosmology.Dang(zd_samp[i])
        rhoc = wl_cosmology.rhoc(zd_samp[i]) # critical density of the Universe in M_Sun/Mpc**3
     
        arcsec2kpc = arcsec2rad * dd * 1000.
       
        ds = wl_cosmology.Dang(zs_samp[i])
        dds = wl_cosmology.Dang(zs_samp[i], zd_samp[i])
    
        s_cr = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2 # critical surface mass density, in M_Sun/kpc**2
    
        r200 = (10.**lm200_samp[i]*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
        c200 = 10.**lc200_samp[i]
        rs = r200/c200
    
        lmstar = lmchab_samp[i] + laimf_samp[i]
    
        reff = 10.**lreff_samp[i]
        nser = 10.**lnser_samp[i]
    
        # defines lensing-related functions
        def alpha(x): 
            # deflection angle (in kpc)
            return (10.**lm200_samp[i] / nfw.M3d(r200, rs) * nfw.M2d(x, rs) + 10.**lmstar * sersic.M2d(x, nser, reff)) / np.pi/x/s_cr
    
        def kappa(x): 
            # dimensionless surface mass density
            return (10.**lmstar * sersic.Sigma(x, nser, reff) + 10.**lm200_samp[i] / nfw.M3d(r200, rs) * nfw.Sigma(x, rs))/s_cr
        
        def m(x):
            return alpha(x) * x
        
        def mu_r(x):
            # radial magnification
            return (1. + m(x)/x**2 - 2.*kappa(x))**(-1)
        
        def mu_t(x):
            # tangential magnification
            return (1. - m(x)/x**2)**(-1)
        
        def mu(x):
            # total magnification
            return mu_r(x) * mu_t(x)
        
        def zerofunc(x):
            return m(x) - x**2
            
        if zerofunc(xmin) < 0.:
            rein = xmin
        elif zerofunc(xmax) > 0.:
            rein = xmax
        else:
            rein = brentq(zerofunc, xmin, xmax)
    
        print i, rein/arcsec2kpc
    
        def minmagfunc(x):
            return abs(mu(x)) - minmag
    
        if minmagfunc(xmin) >= 0.:
            xminmag = xmin
        else:
            xminmag = brentq(minmagfunc, xmin, rein)
        
        def yfunc(x):
            return x - alpha(x)
    
        ycaust = -yfunc(fminbound(yfunc, xminmag, rein))
        ycaust_arcsec = ycaust / arcsec2kpc
        crosssect_samp[i] = ycaust_arcsec**2
        tein_samp[i] = rein/arcsec2kpc

outputfile = h5py.File('crosssect_impsamp_%02d.hdf5'%batchno, 'w')
outputfile.create_dataset('lmchab_samp', data=lmchab_samp)
outputfile.create_dataset('lm200_samp', data=lm200_samp)
outputfile.create_dataset('laimf_samp', data=laimf_samp)
outputfile.create_dataset('lc200_samp', data=lc200_samp)
outputfile.create_dataset('zd_samp', data=zd_samp)
outputfile.create_dataset('zs_samp', data=zs_samp)
outputfile.create_dataset('lnser_samp', data=lnser_samp)
outputfile.create_dataset('lreff_samp', data=lreff_samp)
outputfile.create_dataset('crosssect_samp', data=crosssect_samp)
outputfile.create_dataset('tein_samp', data=tein_samp)

hyperp = outputfile.create_group('prior-parameters')

hyperp.create_dataset('m200_mu', data=m200_mu)
hyperp.create_dataset('m200_sig', data=m200_sig)
hyperp.create_dataset('m200_beta', data=m200_beta)
hyperp.create_dataset('aimf_mu', data=aimf_mu)
hyperp.create_dataset('aimf_sig', data=aimf_sig)
hyperp.create_dataset('zs_min', data=zs_min)
hyperp.create_dataset('zs_max', data=zs_max)



