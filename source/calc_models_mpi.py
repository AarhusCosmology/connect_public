import os
import sys
import time
import itertools
import signal
import traceback

os.environ['UCX_LOG_LEVEL'] = 'error'
param_file   = sys.argv[1]
CONNECT_PATH = sys.argv[2]
sampling     = sys.argv[3]
recompute    = bool(sys.argv[4])
sys.path.insert(0,CONNECT_PATH)

import numpy as np
from mpi4py import MPI

from source.default_module import Parameters
from source.cosmic_solvers import ClassModule, CambModule

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
elif sampling in ['lhc','hypersphere','pickle']:
    directory = os.path.join(path, f'N-{param.N}')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N_slaves = comm.Get_size()-1

get_slave = itertools.cycle(range(1,N_slaves+1))


if param.cosmic_solver == 'class':
    cosmic_solver = ClassModule(param)
elif param.cosmic_solver == 'camb':
    cosmic_solver = CambModule(param)


def excepthook(etype, value, tb):
    if tb is not None and value not in [None, '']:
        print('Traceback (most recent call last):', file=sys.stderr)
        traceback.print_tb(tb, file=sys.stderr)
        print(f'{etype.__name__}: {value.args[0]}', file=sys.stderr)
    comm.Abort(1)
sys.excepthook = excepthook



## rank == 0 (master)
if rank == 0:
    if recompute:
        data = np.loadtxt(os.path.join(directory, 'model_params.txt'), skiprows=1)
    
    elif sampling == 'iterative':
        exec(f'from source.mcmc_samplers.{param.mcmc_sampler} import {param.mcmc_sampler}')
        _locals = {}
        exec(f'mcmc = {param.mcmc_sampler}(param, CONNECT_PATH)', locals(), _locals)
        mcmc = _locals['mcmc']
        data = mcmc.import_points_from_chains(iteration)
        
    elif sampling == 'lhc':
        from source.ini_samplers import LatinHypercubeSampler
        lhc = LatinHypercubeSampler(param)
        data = lhc.run()

    elif sampling == 'hypersphere':
        from source.ini_samplers import HypersphereSampler
        hs = HypersphereSampler(param)
        data = hs.run()

    elif sampling == 'pickle':
        from source.ini_samplers import PickleSample
        ps = PickleSampler(param)
        data = ps.run()

        
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
    out_dirs_ex      = []
    for Cl in param.output_Cl:
        out_dirs_Cl.append(os.path.join(directory, f'Cl_{Cl}_data/Cl_{Cl}_data_{rank}.txt'))
    for Pk in param.output_Pk:
        if type(Pk) == tuple:
            Pk = '-'.join(map(str,Pk))
        out_dirs_Pk.append(os.path.join(directory, f'Pk_{Pk}_data/Pk_{Pk}_data_{rank}.txt'))
    for bg in param.output_bg:
        bg = bg.replace('/','\\')
        out_dirs_bg.append(os.path.join(directory, f'bg_{bg}_data/{bg}_data_{rank}.txt'))
    for th in param.output_th:
        out_dirs_th.append(os.path.join(directory, f'th_{th}_data/{th}_data_{rank}.txt'))
    if len(param.output_derived) > 0:
        out_dir_derived = os.path.join(directory, f'derived_data/derived_data_{rank}.txt')
    for ex in param.extra_output:
        out_dirs_ex.append(os.path.join(directory, f'extra_{ex}_data/{ex}_data_{rank}.txt'))

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

    # Initialise data files
    with open(in_dir, 'w') as f:
        f.write(param_header)

    for out_dir in out_dirs_Cl + out_dirs_Pk + out_dirs_bg + out_dirs_th:
        with open(out_dir, 'w') as f:
            f.write('')
    try:
        with open(out_dir_derived, 'w') as f:
            f.write(derived_header)
    except:
        pass

    # Initialise timeout signal
    def timeout_handler(num, stack):
        raise Exception('timeout')
    signal.signal(signal.SIGALRM, timeout_handler)
    

    # Iterate over each model
    while True:
        comm.send('I am done', dest=0)
        model = comm.recv(source=0)
        if type(model).__name__ == 'str':
            break

        try:
            signal.alarm(param.max_time_for_theory) # CLASS computations must not take longer than this amount of seconds (default is 200)

            cosmic_solver.set_parameters(model)
            cosmic_solver.compute_outputs()
            outputs = cosmic_solver.get_outputs()

            success = True
        except cosmic_solver.computation_exception as e:
            print(f'The following model failed in {param.cosmic_solver.upper()}:', flush=True)
            print(cosmic_solver.params, flush=True)
            success = False
            print(e.message)
        except cosmic_solver.severe_exception as e:
            print(f'The following model failed in {param.cosmic_solver.upper()}:', flush=True)
            print(cosmic_solver.params, flush=True)
            success = False
            print(e.message)
        except Exception as e:
            if str(e) == 'timeout':
                print('The following model took too long to complete:', flush=True)
                print(cosmic_solver.params, flush=True)
                success = False
            else:
                raise e
        finally:
            signal.alarm(0)
                
        if success:
            # Write data to data files
            for out_dir, output in zip(out_dirs_Cl, param.output_Cl):
                par_out = outputs['Cl'][output]
                with open(out_dir, 'a') as f:
                    for i, l in enumerate(outputs['Cl']['ell']):
                        if i != len(outputs['Cl']['ell'])-1:
                            f.write(str(l)+'\t')
                        else:
                            f.write(str(l)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_Pk, param.output_Pk):
                with open(out_dir, 'a') as f:
                    for i, k in enumerate(outputs['Pk']['k']):
                        if i != len(outputs['Pk']['k'])-1:
                            f.write(str(k)+'\t')
                        else:
                            f.write(str(k)+'\n')
                    for z in param.z_Pk_list:
                        par_out = outputs['Pk'][output][z]
                        for i, p in enumerate(par_out):
                            if i != len(par_out)-1:
                                f.write(str(p)+'\t')
                            else:
                                f.write(str(p)+'\n')

            if len(param.output_derived) > 0:
                par_out = []
                for output in param.output_derived:
                    par_out.append(outputs['derived'][output])
                with open(out_dir_derived, 'a') as f:
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_bg, param.output_bg):
                par_out = outputs['bg'][output]
                with open(out_dir, 'a') as f:
                    for i, z in enumerate(outputs['bg']['z']):
                        if i != len(outputs['bg']['z'])-1:
                            f.write(str(z)+'\t')
                        else:
                            f.write(str(z)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_th, param.output_th):
                par_out = outputs['th'][output]
                with open(out_dir, 'a') as f:
                    for i, z in enumerate(outputs['th']['z']):
                        if i != len(outputs['th']['z'])-1:
                            f.write(str(z)+'\t')
                        else:
                            f.write(str(z)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')
        
            for out_dir, output in zip(out_dirs_ex, param.extra_output):
                par_out = outputs['extra'][output]
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


MPI.Finalize()
sys.exit(0)
