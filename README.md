# Survey of Gravitationally-lensed Objects in HSC Imaging. SuGOHI. III. Statistical strong lensing constraints on the stellar IMF of CMASS galaxies.

This repository contains the main code used for the inference of the stellar IMF mismatch parameter of CMASS galaxies, as well as the MCMC chains describing the posterior probability distribution of the model.

The inference method can be summarized as follows.

Each CMASS galaxy is described by a set of parameters. These parameters are drawn from a probability distribution described by hyper-parameters that we wish to infer.
CMASS *strong lenses* are a subsample of CMASS galaxies. Their distribution is given by the product between the distribution of CMASS galaxies and a strong lensing selection term (which depends on the Einstein radius and the strong lensing cross-section of each galaxy-background source pair).

## Chains

The MCMC chains drawn from the posterior probability distribution of the three models (base, 'Arctan' and 'Gaussian') can be found in `.hdf5` format in the `chains/` directory.

## Hierarchical inference code

The practical steps needed to sample the posterior probability distribution of the hyper-parameters given the data are:
1. For each strong lens, calculate the Einstein radius and strong lensing cross-section on a grid of values of the model parameters (script `get_tein_crosssect_grids.py`).
2. Sample the strong lensing cross-section for any CMASS galaxy-background source pair across the parameter space (script `get_crosssect_impsamp.py`).
3. Sample the posterior probability distribution of the hyper-parameters with an MCMC. Use results from step 1 and 2 to speed up computations (one of the scripts `infer_base_model.py`, `infer_arctansel_model.py`, `infer_gausssel_model.py`).

### Requirements ###

- Python
- emcee
- [This](https://github.com/astrosonnen/bayesian_hierarchical_wl) package, normally used for weak lensing

