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
recompute    = eval(sys.argv[4])
sys.path.insert(0,CONNECT_PATH)

import numpy as np
from mpi4py import MPI

from source.default_module import Parameters
from source import cosmic_solvers

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


cosmic_solver = getattr(cosmic_solvers, cosmic_solvers.name_dict[param.cosmic_solver])(param)



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
        try:
            data = np.loadtxt(os.path.join(directory, 'model_params.txt'), skiprows=1)
        except FileNotFoundError:
            raise FileNotFoundError(f'No model_params.txt file found in {directory} to recompute from.')
    
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
    # Directories for input (model parameters) and output data
    in_dir           = os.path.join(directory, f'model_params_data/model_params_{rank}.txt')
    output_structure = cosmic_solver.output_structures()
    out_dirs_scalar  = []
    out_dirs_1D      = []
    out_dirs_2D      = []

    for out in output_structure['Scalar']['list']:
        out_name = out.replace('output_','').replace('_output','')
        out_dirs_scalar.append({'path': os.path.join(directory, f'{out_name}_data/{out_name}_data_{rank}.txt'),
                                'output_name': out})
    for out in output_structure['1D']['list']:
        out_name = out.replace('output_','').replace('_output','')
        for arr in getattr(param, out):
            array = arr.replace('/','\\')
            out_dirs_1D.append({'path': os.path.join(directory, f'{out_name}_{array}_data/{out_name}_{array}_data_{rank}.txt'),
                                'output_name': out,
                                'array': arr})
    for out in output_structure['2D']['list']:
        out_name = out.replace('output_','').replace('_output','')
        for arr in getattr(param, out):
            if type(arr) == tuple:
                arr = '-'.join(map(str,arr))
            array = arr.replace('/','\\')
            out_dirs_2D.append({'path': os.path.join(directory, f'{out_name}_{array}_data/{out_name}_{array}_data_{rank}.txt'),
                                'output_name': out,
                                'array': arr})

    param_header = '# '
    for par_name in param_names:
        if par_name == param_names[-1]:
            param_header += par_name+'\n'
        else:
            param_header += par_name+'\t'

    scalar_headers = []
    for out in output_structure['Scalar']['list']:
        scalar_header = '# '
        for name in getattr(param, out):
            if name == getattr(param, out)[-1]:
                scalar_header += name+'\n'
            else:
                scalar_header += name+'\t'
        scalar_headers.append(scalar_header)

    # Initialise data files
    with open(in_dir, 'w') as f:
        f.write(param_header)

    for i, out_dir in enumerate(out_dirs_scalar):
        with open(out_dir['path'], 'w') as f:
            f.write(scalar_headers[i])

    for i, out_dir in enumerate(out_dirs_1D):
        with open(out_dir['path'], 'w') as f:
            f.write('')

    for i, out_dir in enumerate(out_dirs_2D):
        with open(out_dir['path'], 'w') as f:
            f.write('')

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
            for out_dir in out_dirs_scalar:
                path = out_dir['path']
                output_name = out_dir['output_name']
                out_name = output_name.replace('output_','').replace('_output','')
                par_out = []
                for output in getattr(param, output_name):
                    par_out.append(outputs[out_name][output])
                with open(path, 'a') as f:
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')


            for out_dir in out_dirs_1D:
                path = out_dir['path']
                output_name = out_dir['output_name']
                out_name = output_name.replace('output_','').replace('_output','')
                array = out_dir['array']
                par_out = outputs[out_name][array]
                idx = output_structure['1D']['list'].index(output_name)
                x_arr_name = output_structure['1D']['x_arr'][idx]
                if x_arr_name is not None:
                    x_array = getattr(param, x_arr_name)
                else:
                    x_array = None
                with open(path, 'a') as f:
                    if x_array is not None:
                        for i, l in enumerate(x_array):
                            if i != len(x_array)-1:
                                f.write(str(l)+'\t')
                            else:
                                f.write(str(l)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')


            for out_dir in out_dirs_2D:
                path = out_dir['path']
                output_name = out_dir['output_name']
                out_name = output_name.replace('output_','').replace('_output','')
                array = out_dir['array']
                idx = output_structure['2D']['list'].index(output_name)
                x_arr_name = output_structure['2D']['x_arr'][idx]
                y_arr_name = output_structure['2D']['y_arr'][idx]
                if x_arr_name is not None:
                    x_array = getattr(param, x_arr_name)
                else:
                    x_array = None
                y_array = getattr(param, y_arr_name)
                with open(path, 'a') as f:
                    if x_array is not None:
                        for i, k in enumerate(x_array):
                            if i != len(x_array)-1:
                                f.write(str(k)+'\t')
                            else:
                                f.write(str(k)+'\n')
                    for y in y_array:
                        par_out = outputs[out_name][array][y]
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
