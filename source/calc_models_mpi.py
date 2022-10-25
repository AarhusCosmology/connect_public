import numpy as np
from classy import Class
from scipy.stats import qmc
import pickle as pkl
import sys
import os
from default_module import Parameters
from tools import get_computed_cls
from mpi4py import MPI
import time
import itertools

param_file   = sys.argv[1]
CONNECT_PATH = sys.argv[2]
sampling     = sys.argv[3]
param_file = os.path.join(CONNECT_PATH, param_file)

param        = Parameters(param_file)
param_names  = list(param.parameters.keys())

path = os.path.join(CONNECT_PATH, f'data/{param.jobname}')
if sampling == 'iterative':
    try:
        iteration = max([int(f.split('number_')[-1]) for f in os.listdir(path) if f.startswith('number')])
        directory = os.path.join(path, f'number_{iteration}')
    except:
        directory = os.path.join(path, f'N-{param.N}')
elif sampling == 'lhc':
    directory = os.path.join(path, f'N-{param.N}')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N_slaves = comm.Get_size()-1

get_slave = itertools.cycle(range(1,N_slaves+1))



## rank == 0 (master)
if rank == 0:
    if sampling == 'iterative':
        exec(f'from source.mcmc_samplers.{param.mcmc_sampler} import {param.mcmc_sampler}')
        _locals = {}
        exec(f'mcmc = {param.mcmc_sampler}(param, CONNECT_PATH)', locals(), _locals)
        mcmc = _locals['mcmc']

        data = mcmc.import_points_from_chains(iteration)

        
    elif sampling == 'lhc':
        with open(os.path.join(CONNECT_PATH, f'data/lhc_samples/{len(param.parameters.keys())}_{param.N}.sample'),'rb') as f:
            sample = pkl.load(f)
        data = sample.T
        for i, name in enumerate(param_names):
            if name in param.log_priors:
                data[i] *= np.log10(param.parameters[name][1]) - np.log10(param.parameters[name][0])
                data[i] += np.log10(param.parameters[name][0])
                data[i] = np.power(10.,data[i])
            else:
                data[i] *= param.parameters[name][1] - param.parameters[name][0]
                data[i] += param.parameters[name][0]
        data = data.T


    sleep_short = 0.0001
    sleep_long = 0.1
    sleep_dict = {}
    for r in range(1,N_slaves+1):
        sleep_dict[r] = 1

    data_idx = 0
    while len(data) > data_idx:
        r = next(get_slave)
        if comm.iprobe(r):
            useless_info = comm.recv(source=r)
            comm.send(data[data_idx], dest=r)
            data_idx += 1
            sleep_dict[r] = 1
        else:
            sleep_dict[r] = 0

        if all(value == 0 for value in sleep_dict.values()):
            time.sleep(sleep_long)
        else:
            time.sleep(sleep_short)

    for r in range(1,N_slaves+1):
        comm.send('Done', dest=r)



## rank > 0 (slaves)
else:
    # Directories for input (model parameters) and output (Cl data) data
    in_dir           = os.path.join(directory, f'model_params_data/model_params_{rank}.txt')
    out_dirs_Cl      = []
    out_dirs_Pk      = []
    out_dirs_bg      = []
    out_dirs_th      = []
    for Cl in param.output_Cl:
        out_dirs_Cl.append(os.path.join(directory, f'Cl_{Cl}_data/Cl_{Cl}_data_{rank}.txt'))
    for Pk in param.output_Pk:
        out_dirs_Pk.append(os.path.join(directory, f'Pk_{Pk}_data/Pk_{Pk}_data_{rank}.txt'))
    if len(param.output_Pk) > 0:
        out_dir_kz_array = os.path.join(directory, f'Pk_kz_array.txt')
    for bg in param.output_bg:
        out_dirs_bg.append(os.path.join(directory, f'{bg}_data/{bg}_data_{rank}.txt'))
    for th in param.output_th:
        out_dirs_th.append(os.path.join(directory, f'{th}_data/{th}_data_{rank}.txt'))
    if len(param.output_derived) > 0:
        out_dir_derived = os.path.join(directory, f'derived_data/derived_data_{rank}.txt')

    param_header = '# '
    for par_name in param_names:
        if par_name == param_names[-1]:
            param_header += par_name+'\n'
        else:
            param_header += par_name+'\t'

    derived_header = '# '
    for der_name in param.output_derived:
        if der_name == param.output_derived[-1]:
            derived_header += der_name+'\n'
        else:
            derived_header += der_name+'\t'

    # Initialize data files
    with open(in_dir, 'w') as f:
        f.write(param_header)

    for out_dir in out_dirs_Cl + out_dirs_Pk + out_dirs_bg + out_dirs_th:
        with open(out_dir, 'w') as f:
            f.write('')
    try:
        with open(out_dir_kz_array, 'w') as f:
            f.write('')
    except:
        pass
    try:
        with open(out_dir_derived, 'w') as f:
            f.write(derived_header)
    except:
        pass


    # Iterate over each model
    while True:
        comm.send('I am done', dest=0)
        model = comm.recv(source=0)
        if type(model).__name__ == 'str':
            break

        # Set required CLASS parameters
        params = {}
        if len(param.output_Cl) > 0:
            params['output']            = 'tCl,lCl'
            params['lensing']           = 'yes'
            if any("b" in s or "e" in s for s in param.output_Cl):
                params['output']       += ',pCl'

        if len(param.output_Pk) > 0:
            if 'output' in params.keys():
                params['output']       += ',mPk'
            else:
                params['output']        = 'mPk'
            params['P_k_max_h/Mpc']     = 2.5*max(param.output_Pk_grid[0])
            params['z_max_pk']          = max(param.output_Pk_grid[1])
            if any("nonlinear" in s for s in param.output_Pk):
                params['non linear']    = 'halofit'

        if 'sigma8' in param.output_derived:
            if not 'mPk' in params['output']:
                if len(params['output']) > 0:
                    params['output']   += ',mPk'
                else:
                    params['output']    = 'mPk'
            if not 'P_k_max_h/Mpc' in params.keys():
                params['P_k_max_h/Mpc'] = 1.

        params.update(param.extra_input)
        for i, par_name in enumerate(param_names):
            params[par_name] = model[i]

        try:
            cosmo = Class()
            cosmo.set(params)
            cosmo.compute()
            if len(param.output_bg) > 0:
                bg = cosmo.get_background()
            if len(param.output_th) > 0:
                th = cosmo.get_thermodynamics()
            if len(param.output_derived) > 0:
                der = cosmo.get_current_derived_parameters(param.output_derived)
            if len(param.output_Cl) > 0:
                try:
                    cls = cosmo.lensed_cl_computed() # Only available in CLASS++
                except:
                    cls = get_computed_cls(cosmo)
                ell = cls['ell'][2:]
            success = True
        except:
            print('The following model failed in CLASS:', flush=True)
            print(params, flush=True)
            success = False

        if success:
            # Write data to data files
            for out_dir, output in zip(out_dirs_Cl, param.output_Cl):
                par_out = cls[output][2:]*ell*(ell+1)/(2*np.pi)
                with open(out_dir, 'a') as f:
                    for i, l in enumerate(ell):
                        if i != len(ell)-1:
                            f.write(str(l)+'\t')
                        else:
                            f.write(str(l)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_Pk, param.output_Pk):
                if output == 'm_nonlinear':
                    par_out = cosmo.pk_array(param.output_Pk_grid[0], param.output_Pk_grid[1])
                elif output == 'm_linear':
                    par_out = cosmo.pk_lin_array(param.output_Pk_grid[0], param.output_Pk_grid[1])
                elif output == 'cb_nonlinear':
                    par_out = cosmo.pk_cb_array(param.output_Pk_grid[0], param.output_Pk_grid[1])
                elif output == 'cb_linear':
                    par_out = cosmo.pk_cb_lin_array(param.output_Pk_grid[0], param.output_Pk_grid[1])
                else:
                    if '_nonlinear' in output:
                        try:
                            exec(f'par_out = cosmo.pk_{output.split("_nonlinear")[0]}' +
                                 '_array(param.output_Pk_grid[0], param.output_Pk_grid[1])')
                        except:
                            raise NotImplementedError(f'No method named "pk_{output.split("_nonlinear")[0]}' +
                                                      '_array" is implemented in the classy wrapper. Please ' +
                                                      'implement this before running the code again.')
                    elif '_linear' in output:
                        try:
                            exec(f'par_out = cosmo.pk_{output.split("_nonlinear")[0]}' +
                                 '_lin_array(param.output_Pk_grid[0], param.output_Pk_grid[1])')
                        except:
                            raise NotImplementedError(f'No method named "pk_{output.split("_linear")[0]}' +
                                                      '_array" is implemented in the classy wrapper. Please ' +
                                                      'implement this before running the code again.')

                with open(out_dir, 'a') as f:
                    for i, p in enumerate(par_out.flatten()):
                        if i != len(par_out.flatten())-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')
            try:
                with open(out_dir_kz_array, 'a') as f:
                    f.write('k\t')
                    for k in param.output_Pk_grid[0]:
                        f.write(f'{k}\t')
                    f.write('\nz\t')
                    for z in param.output_Pk_grid[1]:
                        f.write(f'{z}\t')
            except:
                pass

            if len(param.output_derived) > 0:
                par_out = []
                for output in param.output_derived:
                    par_out.append(der[output])
                with open(out_dir_derived, 'a') as f:
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_bg, param.output_bg):
                par_out = bg[output]
                with open(out_dir, 'a') as f:
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_th, param.output_th):
                par_out = th[output]
                with open(out_dir, 'a') as f:
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')
        
            with open(in_dir, 'a') as f:
                for i, m in enumerate(model):
                    if i != len(model)-1:
                        f.write(str(m)+'\t')
                    else:
                        f.write(str(m)+'\n')

        cosmo.struct_cleanup()

MPI.Finalize()
sys.exit(0)
