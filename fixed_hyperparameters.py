# this code lists the values of the hyper-parameters that are kept fixed


# halo mass-concentration relation from Maccio' et al. (2008)
c200_mu = 0.830
c200_beta = -0.098
m200_piv = 12.
c200_sig = 0.1

# stellar mass distribution (skew-Gaussian) from Sonnenfeld et al. (2019)
mchab_mu = 11.249
mchab_sig = 0.285
mchab_logskew = 0.43

# pivot stellar mass and lower bound on observed stellar mass
mchab_piv = 11.4
mchab_cut = 11.

# Sersic index distribution
nser_mu = 0.704
nser_sig = 0.163
nser_beta = 0.464

# half-light radius distribution
reff_mu = 0.817 # average Re at the pivot mass and average Sersic index
reff_sig = 0.133 # intrinsic scatter in Re
reff_beta = 1.184 # dependence of logRe on logM*
reff_nu = 0.383 # dependence of logRe on lognser

# redshift distribution
zd_mu = 0.558
zd_sig = 0.085

