import os
import subprocess as sp
import fileinput
import shutil
import pickle as pkl

import numpy as np

from .lhc import LatinHypercubeSampler
from .join_output import CreateSingleDataFile
from .train_network import Training
from .default_module import Parameters
from .tools import create_output_folders, join_data_files, combine_iterations_data 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Sampling():
    def __init__(self, param_file, CONNECT_PATH):
        self.param_file = param_file
        self.param = Parameters(param_file)
        self.CONNECT_PATH = CONNECT_PATH
        self.N_tasks, self.N_cpus_per_task = np.int64(sp.run(['./source/shell_scripts/number_of_tasks.sh'], stdout=sp.PIPE).stdout.decode('utf-8').split(','))
        if self.param.sampling == 'lhc':
            self.data_path = f'data/{self.param.jobname}/N-{self.param.N}'
        elif self.param.sampling == 'iterative':
            self.data_path = f'data/{self.param.jobname}'

    def create_lhc_data(self):
        self.latin_hypercube_sampling()
        self.copy_param_file()
        self.call_calc_models(sampling='lhc')

    def create_iterative_data(self):
        exec(f'from source.mcmc_samplers.{self.param.mcmc_sampler} import {self.param.mcmc_sampler}')
        _locals = {}
        exec(f'mcmc = {self.param.mcmc_sampler}(self.param, self.CONNECT_PATH)', locals(), _locals)
        mcmc = _locals['mcmc']

        annealing = isinstance(self.param.temperature, list)
        if annealing:
            temp_len = len(self.param.temperature)
            temperature = self.param.temperature[0]
        else:
            temp_len = 0
            temperature = self.param.temperature

        mcmc.check_version()
        i_converged = 10000
        if self.param.resume_iterations:
            try:
                i = max([int(f.split('number_')[-1]) for f in os.listdir(self.data_path) if f.startswith('number')])
            except:
                raise NotImplementedError('You do not have any computed iterations to resume from. Please run again without resume_iterations=True')
            data_is_computed = os.path.isfile(os.path.join(self.data_path, f'number_{i}', 'training.log'))
            if data_is_computed:
                print('Resuming iterative sampling', flush=True)
                print(f'Retraining neural network from iteration {i}', flush=True)
            else:
                i -= 1
                if i==0:
                    raise NotImplementedError('You do not have any computed iterations to resume from. Please run again without resume_iterations=True')
                else:
                    os.system(f"rm -rf {self.data_path}/number_{i+1}")
                    print('Resuming iterative sampling', flush=True)
                    print(f'Retraining neural network from iteration {i}', flush=True)

            model = self.train_neural_network(sampling='iterative',
                                              output_file=os.path.join(self.data_path,
                                                                       f'number_{i}/training.log'))
            if not os.path.isdir(os.path.join(self.data_path, 'compare_iterations')):
                os.system(f"mkdir {self.data_path}/compare_iterations")
                mcmc.Gelman_Rubin_log_ini()
            with open(os.path.join(self.data_path, 'output.log'), 'r') as f:
                for line in f:
                    if "will be the last with same temperature since convergence in" in line:
                        i_converged = int(line.split(' ')[1])
                        break
            i += 1
        else:
            self.copy_param_file()
            if not os.path.isdir(self.data_path):
                os.system(f"mkdir {self.data_path}")
            os.system(f"rm -rf {self.data_path}/compare_iterations")
            os.system(f"mkdir {self.data_path}/compare_iterations")
            i = 1
            if self.param.initial_model == None:
                os.system(f"rm -rf {self.CONNECT_PATH}/{self.data_path}/number_*")
                mcmc.Gelman_Rubin_log_ini()
                print('No initial model given', flush=True)
                print(f'Calculating {self.param.N} initial CLASS models', flush=True)
                self.latin_hypercube_sampling()
                self.call_calc_models(sampling='lhc')
                print('Training neural network', flush=True)
                model = self.train_neural_network(sampling='lhc',
                                              output_file=os.path.join(self.data_path, 
                                                                       f'N-{self.param.N}/training.log'))
            else:
                model = self.param.initial_model
            print(f'Initial model is {model}', flush=True)

        kill_iteration = False
        while True:
            if i > i_converged and annealing:
                temperature = self.param.temperature[i-i_converged]
            print(f'\n\nBeginning iteration no. {i}', flush=True)
            print(f'Temperature is now {temperature}', flush=True)
            print(f'Running MCMC sampling no. {i}...', flush=True)
            mcmc.temperature = temperature
            mcmc.run_mcmc_sampling(model, i)
            print(f'MCMC sampling stopped since R-1 less than {self.param.mcmc_tol} has been reached.', flush=True)

            N_acc = mcmc.get_number_of_accepted_steps(i)
            print(f'Number of accepted steps: {N_acc}', flush=True)
            if i == 1 and not self.param.keep_first_iteration:
                N_keep = 5000
            else:
                N_keep = self.param.N_max_points
            N_keep = mcmc.filter_chains(N_keep,i)
            print(f'Keeping only last {N_keep} of the accepted Markovian steps', flush=True)
            print('Comparing latest iterations...', flush=True)
            if i > 1 and not kill_iteration:
                kill_iteration = mcmc.compare_iterations(i)
            if i > int(not self.param.keep_first_iteration) + 1 and i <= i_converged:
                N_accepted=mcmc.discard_oversampled_points(i)
                N_in_data_set = mcmc.get_number_of_data_points(i-1) + N_accepted
                print(f"Accepted {N_accepted} points out of {N_keep}", flush=True)
            else:
                N_accepted=N_keep
                N_in_data_set = 0

            if kill_iteration and N_accepted < 0.1*N_in_data_set:
                i_converged = i
                if annealing:
                    print(f"Iteration {i} will be the last with same temperature since convergence in", flush=True)
                    print(f"data has been reached and less than 10% of the data was added in this iteration.", flush=True)
                    print(f"Iterations will now continue according to temperature schedule.", flush=True)
                else:
                    print(f"Iteration {i} will be the last since convergence in data has been reached", flush=True)
                    print(f"and less than 10% of the data was added in this iteration.", flush=True)
            else:
                kill_iteration = False

            
            print(f'Calculating {N_accepted} CLASS models', flush=True)
            create_output_folders(self.param, iter_num=i, reset=False)
            self.call_calc_models(sampling='iterative')
            join_data_files(self.param)
            if i > int(not self.param.keep_first_iteration) + 1 and i <= i_converged:
                combine_iterations_data(self.param, i)
                print(f"Copied data from data/{self.param.jobname}/number_{i-1} into data/{self.param.jobname}/number_{i}", flush=True)

            print("Training neural network", flush=True)
            model = self.train_neural_network(sampling='iterative',
                                              output_file=os.path.join(self.data_path,
                                                                       f'number_{i}/training.log'))
            
            if kill_iteration and N_accepted < 0.1*N_in_data_set and not annealing:
                print(f"Final model is {model}", flush=True)
                break
            if i == i_converged+temp_len-1 and annealing:
                print(f"Final model is {model}", flush=True)
                break
            else:
                print(f"New model is {model}", flush=True)
                i += 1

    def copy_param_file(self):
        os.system(f"cp {self.CONNECT_PATH+'/'+self.param_file} {self.CONNECT_PATH+'/'+self.data_path+'/log_connect.param'}") 
        with open(self.CONNECT_PATH+'/'+self.data_path+'/log_connect.param', 'a+') as f:
            jobname_specified = False
            for line in f:
                if line.startswith('jobname'):
                    jobname_specified = True
            if not jobname_specified:
                f.write(f"\njobname = '{self.param.jobname}'")
        self.param.param_file = self.CONNECT_PATH+'/'+self.data_path+'/log_connect.param'

    def latin_hypercube_sampling(self):
        if not os.path.isfile(self.CONNECT_PATH+f'/data/lhc_samples/{len(self.param.parameters.keys())}_{self.param.N}.sample'):
            lhc = LatinHypercubeSampler(self.param)
            lhc.run(self.CONNECT_PATH)

    def call_calc_models(self, sampling='lhc'):
        os.environ["export OMP_NUM_THREADS"] = str({self.N_cpus_per_task})
        os.environ["PMIX_MCA_gds"] = "hash"
        sp.Popen(f"mpirun -np {self.N_tasks - 1} python {self.CONNECT_PATH}/source/calc_models_mpi.py {self.param.param_file} {self.CONNECT_PATH} {sampling}".split()).wait()
        os.environ["export OMP_NUM_THREADS"] = "1"

    def train_neural_network(self, sampling='lhc', output_file=None):
        if sampling == 'lhc':
            folder = ''
        elif sampling == 'iterative':
            i = max([int(f.split('number_')[-1]) for f in os.listdir(os.path.join(self.CONNECT_PATH, self.data_path)) if f.startswith('number')])
            folder = f'number_{i}'
        if not os.path.isfile(os.path.join(self.CONNECT_PATH,
                                           self.data_path,
                                           folder,
                                           'model_params.txt')):
            CSDF = CreateSingleDataFile(self.param, self.CONNECT_PATH)
            CSDF.join()
        tr = Training(self.param, self.CONNECT_PATH)
        tr.train_model(output_file = output_file)
        tr.save_model()
        tr.save_history()
        tr.save_test_data()

        if self.param.save_name != None:
            model_name = self.param.save_name
        else:
            model_name = f'{tr.param.jobname}_N{tr.N}_bs{tr.param.batchsize}_e{tr.param.epochs}'
        if not self.param.overwrite_model:
            M = 1
            if os.path.isdir(os.path.join('trained_models', model_name)):
                while os.path.isdir(os.path.join('trained_models', model_name+f'_{M}')):
                    M += 1
                if M-1 > 0:
                    model_name += f'_{M-1}'
        return model_name
