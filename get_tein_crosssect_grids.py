import numpy as np
import wl_cosmology
from wl_cosmology import G, M_Sun, c, Mpc
from wl_profiles import sersic, nfw
import h5py
import emcee
from scipy.optimize import brentq, fminbound
from scipy.interpolate import splrep, splev
import os
import sys


lensno = int(sys.argv[1]) # number of lens to sample (according to order in 'summary_table.txt', file)

minmag = 0.5 # minimum magnification of inner image for an object to be declared a strong lens (needed to calculate strong lensing cross-section)

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)

f = open('summary_table.txt', 'r')
lines = f.readlines()
f.close()

griddir = 'grids/'

line = lines[lensno+1].split()
name = line[0]

zd = float(line[1])
zs = float(line[2])
rein = float(line[3]) # Einstein radius in arcseconds

reff = float(line[4])
nser = float(line[5])

# defines the stellar profile
ein_frac = sersic.M2d(rein, nser, reff)
def bulge_M2d(R):
    return sersic.M2d(R, nser, reff*arcsec2kpc)
def bulge_Sigma(R):
    return sersic.Sigma(R, nser, reff*arcsec2kpc)

lmchab_obs = float(line[6])
lmchab_err = float(line[7])

dd = wl_cosmology.Dang(zd)
ds = wl_cosmology.Dang(zs)
dds = wl_cosmology.Dang(zs, zd)

rhoc = wl_cosmology.rhoc(zd) # critical density of the Universe in M_Sun/Mpc**3

s_cr = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2 # critical surface mass density in M_Sun/kpc**2

arcsec2kpc = arcsec2rad * dd * 1000.
rein_phys = rein * arcsec2kpc # Einstein radius in kpc

nm200 = 30
nmstar = 30
nc200 = 5

# defines the grid
lmstar_min = lmchab_obs-0.5
lmstar_max = lmchab_obs+0.5

def lm200_func(lmchab):
    return 12.8 + 1.8*(lmchab-11.4)

lm200_min = lm200_func(lmchab_obs) - 1.
lm200_max = lm200_func(lmchab_obs) + 1.

lc200_min = 0.2
lc200_max = 1.2

lmstar_grid = np.linspace(lmstar_min, lmstar_max, nmstar)
lm200_grid = np.linspace(lm200_min, lm200_max, nm200)
lc200_grid = np.linspace(lc200_min, lc200_max, nc200)

# range of allowed values for Einstein radius
xmin = 0.01
xmax = 10. * rein_phys

crosssect_grid = np.zeros((nm200, nmstar, nc200))
tein_grid = np.zeros((nm200, nmstar, nc200))

for i in range(nm200):
    r200 = (10.**lm200_grid[i]*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
    for j in range(nmstar):
        for k in range(nc200):
            rs = r200/10.**lc200_grid[k]
            # defines lensing-related functions
            def alpha(x): 
                # deflection angle (in kpc)
                return (10.**lm200_grid[i] * nfw.M2d(x, rs) / nfw.M3d(r200, rs) + 10.**lmstar_grid[j]*bulge_M2d(x))/np.pi/x/s_cr

            def kappa(x): 
                # dimensionless surface mass density
                return (10.**lm200_grid[i] * nfw.Sigma(x, rs)/nfw.M3d(r200, rs) + 10.**lmstar_grid[j]*bulge_Sigma(x))/s_cr

            def m(x):
                return alpha(x) * x
             
            def mu_r(x):
                # radial magnification
                return (1. + m(x)/x**2 - 2.*kappa(x))**(-1)
            
            def mu_t(x):
                # tangential magnification
                return (1. - m(x)/x**2)**(-1)
            
            def mu(x):
                # magnification
                return mu_r(x) * mu_t(x)
            
            def zerofunc(x):
                return m(x) - x**2
        
            if zerofunc(xmin) < 0.:
                rein_here = xmin
            elif zerofunc(xmax) > 0.:
                rein_here = xmax
            else:
                rein_here = brentq(zerofunc, xmin, xmax)
        
            def minmagfunc(x):
                return abs(mu(x)) - minmag
        
            if minmagfunc(xmin) >= 0.:
                xminmag = xmin
            else:
                xminmag = brentq(minmagfunc, xmin, rein_here)
            
            def yfunc(x):
                return x - alpha(x)
        
            ycaust = -yfunc(fminbound(yfunc, xminmag, rein_here))
            ycaust_arcsec = ycaust / arcsec2kpc

            tein_grid[i, j, k] = rein_here/arcsec2kpc
            crosssect_grid[i, j, k] = ycaust_arcsec**2

            print i, j, k, rein_here/arcsec2kpc

output_file = h5py.File(griddir+'%s_tein_crosssect_grid.hdf5'%name, 'w')
output_file.create_dataset('lm200_grid', data=lm200_grid)
output_file.create_dataset('lmstar_grid', data=lmstar_grid)
output_file.create_dataset('lc200_grid', data=lc200_grid)
output_file.create_dataset('tein_grid', data=tein_grid)
output_file.create_dataset('crosssect_grid', data=crosssect_grid)

