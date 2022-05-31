import numpy as np
import subprocess as sp
#from source.misc_functions import get_node_with_most_cpus
#from source.misc_functions import get_param_attributes
#from source.misc_functions import create_montepython_param
#from source.misc_functions import Gelman_Rubin_log
#from source.misc_functions import create_lhs
#from source.misc_functions import create_output_folders
#from source.misc_functions import filter_chains
#from source.misc_functions import discard_oversampled_points
#from source.misc_functions import create_output_folders
#from source.misc_functions import join_data_files
#from source.misc_functions import combine_iterations_data
from source.lhs import LatinHypercubeSampler
from source.join_output import CreateSingleDataFile
from source.train_network import Training
from source.default_module import Parameters
import source.misc_functions as misc
import os
import fileinput

class Sampling():
    def __init__(self, param_file, CONNECT_PATH):
        self.param_file = param_file
        self.param = Parameters(param_file)
        self.CONNECT_PATH = CONNECT_PATH
        self.N_tasks = 12#int(sp.run(['./source/shell_scripts/number_of_tasks.sh'], stdout=subprocess.PIPE).stdout)
        os.system(f"rm -rf data/{self.param.jobname}")
        os.system(f"mkdir data/{self.param.jobname}")
        if self.param.sampling == 'lhs':
            self.data_path = f'data/{self.param.jobname}/N-{self.param.N}'
        elif self.param.sampling == 'iterative':
            self.data_path = f'data/{self.param.jobname}'
        os.system(f"cp {self.CONNECT_PATH+'/'+param_file} {self.CONNECT_PATH+'/'+self.data_path+'/log_connect.param'}")
        with open(self.CONNECT_PATH+'/'+self.data_path+'/log_connect.param', 'a+') as f:
            jobname_specified = False
            for line in f:
                if line.startswith('jobname'):
                    jobname_specified = True
            if not jobname_specified:
                f.write(f"\njobname = '{self.param.jobname}'")
        slurm_bool = int(sp.run('if [ -z $SLURM_NPROCS ]; then echo 0; else echo 1; fi', shell=True, stdout=sp.PIPE).stdout.decode('utf-8')))
        self.mp_node = None
        if slurm_bool:
            self.mp_node = misc.get_node_with_most_cpus()

    def create_lhs_data(self):
        self.latin_hypercube_sampling()
        misc.create_output_folders(self.param_file)
        self.call_calc_models(sampling='lhs')

    def latin_hypercube_sampling(self):
        if not os.path.isfile(CONNECT_PATH+f'/data/lhs_samples/sample_models_{self.param.jobname}_{self.param.N}.txt'):
            lhs = LatinHypercubeSampler(self.param)
            lhs.run(self.CONNECT_PATH)

    def call_calc_models(self, sampling='lhs'):
        sp.run(f'mpirun -np {self.N_tasks - 1} python {self.CONNECT_PATH}/source/calc_models_mpi.py {self.data_path}/log_connect.param {self.CONNECT_PATH} {sampling}')

    def train_neural_network(self, sampling='lhs'):
        if sampling == 'lhs':
            folder = ''
        elif sampling == 'iterative':
            i = max([int(f.split('number_')[-1]) for f in os.listdir(self.CONNECT_PATH+'/'+self.data_path) if f.startswith('number')])
            folder = f'/number_{i}'
        if not os.path.isfile(self.CONNECT_PATH+'/'+self.data_path+folder+'/model_params.txt'):
            CSDF = CreateSingleDataFile(param, self.CONNECT_PATH)
            CSDF.join()
        tr = Training(self.param, self.CONNECT_PATH)
        tr.train_model()
        tr.save_model()
        tr.save_history()
        tr.save_test_data()

        if self.param.save_name != None:
            model_name = self.param.save_name
        else:
            model_name = f'{param.jobname}_N{param.N}_bs{param.batchsize}_e{param.epochs}_montepython'
        if not param.overwrite_model:
            M = 1
            if os.path.isdir('trained_models/' + model_path):
                while os.path.isdir('trained_models/' + model_path + f'_{M}'):
                    M += 1
                if M-1 > 0:
                    model_name += f'_{M-1}'
        return model_name

    def run_montepython_iteration(self, MP_param_file, model):
        with fileinput.input(MP_param_file, inplace=True) as file:
            for line in file:
                if "data.cosmo_arguments['connect_model']" in line:
                    line = f"data.cosmo_arguments['connect_model'] = '{model}'\n"
                print(line, end='')
        if self.mp_bool != None:
            sp.run(f'{self.CONNECT_PATH}/source/shell_scripts/run_montepython_iteration.sh {self.param.jobname} {MP_param_file} {self.mp_node}')
        else:
            sp.run(f'{self.CONNECT_PATH}/source/shell_scripts/run_montepython_iteration.sh {self.param.jobname} {MP_param_file}')

    def create_iterative_data(self):
        os.system(f"mkdir data/{self.param.jobname}/compare_iterations")
        MP_param_file = misc.create_montepython_param(self.param_file,self.param.montepython_path)
        misc.Gelman_Rubin(self.param_file,status='initial')
        if self.param.initial_model = None:
            misc.create_output_folders(self.param_file)
            call_calc_models(sampling='lhs')
            model = train_neural_network(sampling='lhs')
        else:
            model = self.param.initial_model

        while True:
            run_montepython_iteration(MP_param_file, model)
            
