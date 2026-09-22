import os
import sys
import pickle as pkl
from pathlib import Path
import warnings

import numpy as np
from best import Sampler
import tensorflow as tf


model = sys.argv[1]
iteration = sys.argv[2]
param_file = sys.argv[3]
temperature = np.float64(sys.argv[4])
CONNECT_PATH = sys.argv[5]

sys.path.insert(1, CONNECT_PATH)
from source.default_module import Parameters

param = Parameters(param_file)

warn = False
for lkl in param.sampling_likelihoods:
    if not lkl.endswith('_tf'):
        warn = True
        break
if warn:
    warnings.warn("Best is only compatible with TensorFlow likelihoods. All other likelihoods will be ignored")

directory = os.path.join(CONNECT_PATH, f'data/{param.jobname}/number_{iteration}/')

sys.stdout = open(directory + 'best.log','a+')
sys.stderr = sys.stdout


likelihoods = []
nuisance_indices = []
nuisance_params = []
nuisance_limits = []
nuisance_bestfit_guesses = []
nuisance_sigma_guesses = []
nuisance_priors = []
use_planck_lite_highl = False
use_planck_lite_lowl_TT = False
use_planck_lite_lowl_EE = False
sys.path.insert(1, os.path.join(CONNECT_PATH, 'mcmc_plugin/mp_likelihoods/Planck_lite_connect/tf_planck_lite/'))
from tf_planck_lite import tf_planck2018_lite
for lkl in param.sampling_likelihoods:
    if lkl == 'Planck_lite_tf':
        use_planck_lite_highl = True
    elif lkl == 'Planck_lowl_tf':
        use_planck_lite_lowl_TT = True
        use_planck_lite_lowl_EE = True
    elif lkl == 'Planck_lowl_TT_tf':
        use_planck_lite_lowl_TT = True
    elif lkl == 'Planck_lowl_EE_tf':
        use_planck_lite_lowl_EE = True
    else:
        warnings.warn(f"For now, only the Planck_lite_tf and Planck_lowl_tf likelihoods are supported. {lkl} is not supported and will be ignored")

if use_planck_lite_highl or use_planck_lite_lowl_TT or use_planck_lite_lowl_EE:
    Lkl = tf_planck2018_lite(connect_ttteee_model=os.path.join(CONNECT_PATH,'trained_models',model),
                             tf_planck2018_lite_path=os.path.join(CONNECT_PATH, 'mcmc_plugin/mp_likelihoods/Planck_lite_connect/tf_planck_lite/'),
                             use_highl=use_planck_lite_highl,
                             use_lowl_TT=use_planck_lite_lowl_TT,
                             use_lowl_EE=use_planck_lite_lowl_EE)
    likelihoods.append(lambda x: tf.squeeze(Lkl.get_loglkl(x), axis=-1))
    if "A_planck" not in nuisance_params:
        nuisance_params.append("A_planck")
        nuisance_limits.append([0.5, 1.5])
        nuisance_bestfit_guesses.append(1.0)
        nuisance_sigma_guesses.append(0.01)
        nuisance_priors.append(lambda x: -0.5*((x-1)/0.0025)**2)
    nuisance_indices.append([nuisance_params.index("A_planck")])

num_nuisance = len(nuisance_params)
def log_prob_fn(x):
    x_cosmo = x[..., :-num_nuisance]
    x_nuisance = x[..., -num_nuisance:]
    loglkl = tf.zeros(tf.shape(x)[:-1], dtype=tf.float32)
    for lkl, nuisance_idxs in zip(likelihoods, nuisance_indices):
        x_nuisance_subset = tf.gather(x_nuisance, nuisance_idxs, axis=-1)
        loglkl += lkl(tf.concat([x_cosmo, x_nuisance_subset], axis=-1))
    for i, prior in enumerate(nuisance_priors):
        loglkl += prior(x_nuisance[..., i])
    return loglkl

names_sampled = []
limits = []
bestfit_guesses = []
sigma_guesses = []
for par,interval in param.parameters.items():
    if par in param.prior_ranges:
        if param.prior_ranges[par][0] != 'None':
            xmin = param.prior_ranges[par][0]
        else:
            xmin = -1e+32
        if param.prior_ranges[par][1] != 'None':
            xmax = param.prior_ranges[par][1]
        else:
            xmax = 1e+32
    else:
        xmin = interval[0]
        xmax = interval[1]
    if par in param.bestfit_guesses:
        guess = param.bestfit_guesses[par]
    else:
        guess = (interval[0] + interval[1])/2
    if par in param.sigma_guesses:
        sig = param.sigma_guesses[par]
    else:
        sig = abs((interval[1] - interval[0])/10)
    if par in param.log_priors:
        if xmin != -1e+32:
            xmin = np.log10(xmin)
        else:
            xmin = 0
        if xmax != 1e+32:
            xmax = np.log10(xmax)
        guess = np.log10(guess)
        if xmin != -1e+32 and xmax != 1e+32:
            sig = sig * (xmax-xmin)/(10**xmax-10**xmin)
        else:
            sig = sig * ( abs( np.log10( abs(interval[1]) ) - np.log10( abs(interval[0]) ) ) /
                          abs( interval[1] - interval[0] ) )
        par = par + '_log10_prior'
    names_sampled.append(par)
    limits.append([xmin, xmax])
    bestfit_guesses.append(guess)
    sigma_guesses.append(sig)

# add nuisance parameters and values to the lists
for i in range(len(nuisance_params)):
    names_sampled.append(nuisance_params[i])
    limits.append(nuisance_limits[i])
    bestfit_guesses.append(nuisance_bestfit_guesses[i])
    sigma_guesses.append(nuisance_sigma_guesses[i])

# create diagonal covmat from sigma_guesses
covmat = np.diag(np.array(sigma_guesses)**2)
covmat_tf = tf.convert_to_tensor(covmat, dtype=tf.float32)

lower_bounds = tf.convert_to_tensor(np.array([l[0] for l in limits]), dtype=tf.float32)
upper_bounds = tf.convert_to_tensor(np.array([l[1] for l in limits]), dtype=tf.float32)

sampler = Sampler(
    log_prob_fn,
    covmat=covmat_tf,
    initial_state=tf.convert_to_tensor(np.array(bestfit_guesses), dtype=tf.float32),
    initial_distribution="gaussian",
    bounds=(lower_bounds, upper_bounds),
)

results = sampler.sample(
    method="mh",
    n_steps=10000,
    n_chains=50,
    num_burnin_steps=1000,
    num_covmat_updates=3,
    temperature=temperature,
)

print("Sampling done")

samples = results.samples.numpy()
samples = np.transpose(samples, (1, 0, 2))  # shape: (n_chains, n_steps, n_params)

chains = []
n_samples = 0
n_chains = 0
for s in samples:
    chain = np.unique(s, axis=0)
    chains.append(chain)
    n_samples += chain.shape[0]
    n_chains += 1

print("Saving all chains")
all_chains = {'chains' : chains,
              'names' : names_sampled}
print(f"Samples collected: {n_samples}, number of chains: {n_chains}")
with open(directory + 'best_all_chains.pkl','wb') as f:
    pkl.dump(all_chains,f)
print("chains saved successfully")
