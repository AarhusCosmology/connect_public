import os
import sys
import pickle as pkl
from pathlib import Path

from cobaya.run import run
from cobaya.log import LoggedError
from mpi4py import MPI
import numpy as np

from source.default_module import Parameters

model = sys.argv[1]
iteration = sys.argv[2]
param_file = sys.argv[3]
temperature = np.float64(sys.argv[4])

param = Parameters(param_file)

path = {}
with open('mcmc_plugin/connect.conf','r') as f:
    for line in f:
        exec(line)

path_clik = os.path.join(Path(path['clik']).parents[2], 'baseline/plc_3.0/')
CONNECT_PATH = Path(path['cosmo']).parents[0]

directory = os.path.join(CONNECT_PATH, f'data/{param.jobname}/number_{iteration}/')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sys.stdout = open(directory + 'cobaya.log','a+')
sys.stderr = sys.stdout



lkls = {'Planck_highl_TTTEEE_lite': {'name': 'planck_2018_highl_plik.TTTEEE_lite',
                                     'clik': os.path.join(path_clik, 'hi_l/plik_lite')},
        'Planck_lowl_TT':           {'name': 'planck_2018_lowl.TT_clik',
                                     'clik': os.path.join(path_clik, 'low_l/commander')},
        'Planck_lowl_EE':           {'name': 'planck_2018_lowl.EE_clik',
                                     'clik': os.path.join(path_clik, 'low_l/simall')}}



info = {'likelihood': {},
        'params': {},
        'sampler': {'mcmc': {'Rminus1_cl_stop':  0.1,
                             'Rminus1_stop':     param.mcmc_tol,
                             'max_tries':        1e+5,
                             'covmat':          'auto',
                             'oversample_power': 0,
                             'proposal_scale':   2.1,
                             'temperature':      temperature}},
        'theory': {'CosmoConnect': {'ignore_obsolete': True,
                                    'path':            os.path.join(CONNECT_PATH, 'mcmc_plugin/cobaya'),
                                    'python_path':     os.path.join(CONNECT_PATH, 'mcmc_plugin/cobaya'),
                                    'extra_args':      {'connect_model': model}
                                }}}


for lkl in param.sampling_likelihoods:
    if lkl == 'Planck_lite':
        lkl = 'Planck_highl_TTTEEE_lite'
    if lkl in lkls:
        for name in os.listdir(lkls[lkl]['clik']):
            if 'TTTEEE' in lkl and name.endswith('TTTEEE.clik'):
                clik_file = os.path.join(lkls[lkl]['clik'], name)
                break
            elif 'EE' in lkl and name.endswith('.clik') and 'BB' not in name:
                clik_file = os.path.join(lkls[lkl]['clik'], name)
                break
            elif 'TT' in lkl and name.endswith('.clik'):
                clik_file = os.path.join(lkls[lkl]['clik'], name)
                break
        info['likelihood'][lkls[lkl]['name']] = {'clik_file':   clik_file}

    else:
        raise NotImplementedError(f"For now, only the following three likelihoods are available during training:\n{' '*4}Planck_highl_TTTEEE_lit, Planck_lowl_TT, Planck_lowl_EE\nYou can manually add extra likelihoods as a nested dictionary using Cobaya syntax in the parameter file, e.g.\n{' '*4}"+"extra_cobaya_lkls = {'Likelihood_name': {'path': path/to/likelihood,\n"+f"{' '*52}'options': other_options,\n{' '*52}"+"...},\n"+f"{' '*44}"+"...}")

for name, item in param.extra_cobaya_lkls.items():
    info['likelihood'][name] = item

print(info['likelihood'])

for par,interval in param.parameters.items():
    if par in param.prior_ranges:
        if param.prior_ranges[par][0] != 'None':
            xmin = param.prior_ranges[par][0]
        else:
            xmin = -1e+32
        if param.prior_ranges[par][1] != 'None':
            xmax = param.prior_ranges[par][1]
        else:
            xmax =  1e+32
    else:
        xmin = -1e+32
        xmax =  1e+32
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
    proposal = sig
    if par.startswith('ln10^{10}A_s'):
        info['params']['logA'] = {}
        info['params']['logA']['prior'] = {}
        info['params']['logA']['ref'] = {'dist': 'norm'}
        info['params']['logA']['prior']['min'] = xmin
        info['params']['logA']['prior']['max'] = xmax
        info['params']['logA']['ref']['loc']   = guess
        info['params']['logA']['ref']['scale'] = sig
        info['params']['logA']['proposal'] = proposal
        info['params']['logA']['latex'] = par
        info['params']['logA']['drop'] = True
        info['params']['A_s'] = {}
        info['params']['A_s']['latex'] = 'A_s'
        info['params']['A_s']['value'] = 'lambda logA: 1e-10*np.exp(logA)'
    elif par.startswith('100*theta_s'):
        info['params']['theta_s_100'] = {}
        info['params']['theta_s_100']['prior'] = {}
        info['params']['theta_s_100']['ref'] = {'dist': 'norm'}
        info['params']['theta_s_100']['prior']['min'] = xmin
        info['params']['theta_s_100']['prior']['max'] = xmax
        info['params']['theta_s_100']['ref']['loc']   = guess
        info['params']['theta_s_100']['ref']['scale'] = sig
        info['params']['theta_s_100']['proposal'] = proposal
        info['params']['theta_s_100']['latex'] = par
        info['params']['theta_s_100']['drop'] = True
        info['params']['100*theta_s'] = {}
        info['params']['100*theta_s']['latex'] = par
        info['params']['100*theta_s']['value'] = 'lambda theta_s_100: theta_s_100'
        info['params']['100*theta_s']['derived'] = False
    else:
        info['params'][par] = {}
        info['params'][par]['prior'] = {}
        info['params'][par]['ref'] = {'dist': 'norm'}
        info['params'][par]['prior']['min'] = xmin
        info['params'][par]['prior']['max'] = xmax
        info['params'][par]['ref']['loc']   = guess
        info['params'][par]['ref']['scale'] = sig
        info['params'][par]['proposal'] = proposal
        info['params'][par]['latex'] = par

for par in param.output_derived:
    if par == 'A_s' and 'ln10^{10}A_s' in param.parameters:
        info['params']['A'] = {}
        info['params']['A']['derived'] = 'lambda A_s: 1e9*A_s'
        info['params']['A']['latex'] = '10^9 A_s'
    elif par == '100*theta_s':
        info['params']['theta_s_100'] = {}
        info['params']['theta_s_100']['latex'] = par
    else:
        info['params'][par] = {}
        info['params'][par]['latex'] = par



updated_info, sampler = run(info)

list_of_chains = comm.gather(sampler.products()["sample"].to_numpy(), root=0)

if rank == 0:
    names_dict = {'sampled': sampler.products()["sample"].sampled_params,
                  'derived': sampler.products()["sample"].derived_params,
                  'all'    : sampler.products()["sample"].columns}
    all_chains = {'chains' : list_of_chains,
                  'names'  : names_dict}

    with open(directory + 'cobaya_all_chains.pkl','wb') as f:
        pkl.dump(all_chains,f)

