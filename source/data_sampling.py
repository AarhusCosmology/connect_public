import numpy as np
import subprocess as sp
from source.lhc import LatinHypercubeSampler
from source.join_output import CreateSingleDataFile
from source.train_network import Training
import source.misc_functions as misc
from source.default_module import Parameters
import os
import fileinput
import shutil

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
        slurm_bool = int(sp.run('if [ -z $SLURM_NPROCS ]; then echo 0; else echo 1; fi', shell=True, stdout=sp.PIPE).stdout.decode('utf-8'))
        self.mp_node = None
        if slurm_bool:
            self.mp_node = misc.get_node_with_most_cpus()

    def copy_param_file(self):
        os.system(f"cp {self.CONNECT_PATH+'/'+self.param_file} {self.CONNECT_PATH+'/'+self.data_path+'/log_connect.param'}") 
        with open(self.CONNECT_PATH+'/'+self.data_path+'/log_connect.param', 'a+') as f:
            jobname_specified = False
            for line in f:
                if line.startswith('jobname'):
                    jobname_specified = True
            if not jobname_specified:
                f.write(f"\njobname = '{self.param.jobname}'")


    def create_lhc_data(self):
        self.latin_hypercube_sampling()
        misc.create_output_folders(self.param)
        self.copy_param_file()
        self.call_calc_models(sampling='lhc')

    def latin_hypercube_sampling(self):
        if not os.path.isfile(self.CONNECT_PATH+f'/data/lhc_samples/sample_models_{self.param.jobname}_{self.param.N}.txt'):
            lhc = LatinHypercubeSampler(self.param)
            lhc.run(self.CONNECT_PATH)

    def call_calc_models(self, sampling='lhc'):
        sp.run(f"export OMP_NUM_THREADS={self.N_cpus_per_task}", shell=True)
        sp.Popen(f"mpirun -np {self.N_tasks - 1} python {self.CONNECT_PATH}/source/calc_models_mpi.py {self.data_path}/log_connect.param {self.CONNECT_PATH} {sampling}".split()).wait()
        sp.run("export OMP_NUM_THREADS=1", shell=True)

    def train_neural_network(self, sampling='lhc'):
        if sampling == 'lhc':
            folder = ''
        elif sampling == 'iterative':
            i = max([int(f.split('number_')[-1]) for f in os.listdir(self.CONNECT_PATH+'/'+self.data_path) if f.startswith('number')])
            folder = f'/number_{i}'
        if not os.path.isfile(self.CONNECT_PATH+'/'+self.data_path+folder+'/model_params.txt'):
            CSDF = CreateSingleDataFile(self.param, self.CONNECT_PATH)
            CSDF.join()
        tr = Training(self.param, self.CONNECT_PATH)
        tr.train_model()
        tr.save_model()
        tr.save_history()
        tr.save_test_data()

        if self.param.save_name != None:
            model_name = self.param.save_name
        else:
            model_name = f'{self.param.jobname}_N{self.param.N}_bs{self.param.batchsize}_e{self.param.epochs}'
        if not self.param.overwrite_model:
            M = 1
            if os.path.isdir('trained_models/' + model_name):
                while os.path.isdir('trained_models/' + model_name + f'_{M}'):
                    M += 1
                if M-1 > 0:
                    model_name += f'_{M-1}'
        return model_name

    def run_montepython_iteration(self, MP_param_file, model):
        with fileinput.input(self.param.montepython_path+'/'+MP_param_file, inplace=True) as file:
            for line in file:
                if "data.cosmo_arguments['connect_model']" in line:
                    line = f"data.cosmo_arguments['connect_model'] = '{model}'\n"
                print(line, end='')
        
        if self.mp_node != None:
            sp.run(f'{self.CONNECT_PATH}/source/shell_scripts/run_montepython_iteration.sh {self.param.jobname} {MP_param_file} {self.CONNECT_PATH}/mp_plugin/connect.conf {self.param.mp_tol} {self.mp_node}', shell=True, cwd=self.param.montepython_path)
        else:
            sp.run(f'{self.CONNECT_PATH}/source/shell_scripts/run_montepython_iteration.sh {self.param.jobname} {MP_param_file} {self.CONNECT_PATH}/mp_plugin/connect.conf {self.param.mp_tol}', shell=True, cwd=self.param.montepython_path)

    def compare_iterations(self,i):
        chain1=f"data/{self.param.jobname}/compare_iterations/chain__{i-1}.txt"
        chain2=f"data/{self.param.jobname}/compare_iterations/chain__{i}.txt"
        output = sp.run(f'python2 {self.param.montepython_path}/montepython/MontePython.py info {chain1} {chain2} --noplot --minimal', shell=True, stdout=sp.PIPE).stdout.decode('utf-8')
        kill_iteration=misc.Gelman_Rubin_log(self.param,status=i,output=output)
        return kill_iteration

    def count_lines_in_dir(self, directory):
        txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        count = 0
        for f in txt_files:
            count += sum(1 for line in open(os.path.join(directory,f)) 
                         if line[0] != '#' and line.strip())
        return count

    def create_iterative_data(self):
        if not os.path.isdir(f"data/{self.param.jobname}"):
            os.system(f"mkdir data/{self.param.jobname}")
            os.system(f"mkdir data/{self.param.jobname}/compare_iterations")
        elif not os.path.isdir(f"data/{self.param.jobname}/compare_iterations"):
            os.system(f"mkdir data/{self.param.jobname}/compare_iterations")
        self.copy_param_file()
        if not self.param.resume_iterations:
            os.system(f"rm -rf {self.CONNECT_PATH}/data/{self.param.jobname}/number_*")
        MP_param_file = misc.create_montepython_param(self.param_file,self.param.montepython_path)
        misc.Gelman_Rubin_log(self.param,status='initial')
        if self.param.initial_model == None:
            print('No initial model given')
            print(f'Calculating {self.param.N} initial CLASS models')
            self.latin_hypercube_sampling()
            misc.create_output_folders(self.param)
            self.call_calc_models(sampling='lhc')
            print('Training neural network')
            model = self.train_neural_network(sampling='lhc')
        else:
            model = self.param.initial_model
        keep_idx = 0
        kill_iteration = False

        print(f'Initial model is {model}')

        i = 1
        while True:
            print(f'\n\nBeginning iteration no. {i}')
            print(f'Running MCMC sampling no. {i}...')
            self.run_montepython_iteration(MP_param_file, model)
            print(f'MCMC sampling stopped since R-1 less than 0.01 has been reached')

            N_acc = self.count_lines_in_dir(os.path.join(self.param.montepython_path,f'chains/connect_{self.param.jobname}_data'))
            print(f'Number of accepted steps: {N_acc}')

            shutil.copytree(self.param.montepython_path + f'/chains/connect_{self.param.jobname}_data', f'{self.CONNECT_PATH}/data/{self.param.jobname}/number_{i}')
            if i == 1 and not self.param.keep_first_iteration:
                N_keep = 5000
                keep_idx = 1
            else:
                N_keep = self.param.N_max_lines

            N_keep = misc.filter_chains(self.param,f'data/{self.param.jobname}/number_{i}',N_keep,i)

            print(f'Keeping only last {N_keep} of the accepted Markovian steps')

            print('Comparing latest iterations...')
            if i > 1:
                kill_iteration = self.compare_iterations(i)
                if kill_iteration:
                    print(f"iteration {i} will be the last since convergence in data has been reached")

            
            if i > keep_idx + 1:
                model_params=f"data/{self.param.jobname}/number_{i-1}/model_params.txt"
                N_accepted=misc.discard_oversampled_points(model_params,self.param,i)
                print(f"Accepted $N_accepted points out of {N_keep}")
            else:
                N_accepted=N_keep
            
            print(f'Calculating {N_accepted} CLASS models')
            misc.create_output_folders(self.param, iter_num=i, reset=False)
            self.call_calc_models(sampling='iterative')
            misc.join_data_files(self.param)
            if i > keep_idx + 1:
                misc.combine_iterations_data(self.param, i)
                print(f"Copied data from data/{self.param.jobname}/number_{i-1} into data/{self.param.jobname}/number_{i}")
            
            print("Training neural network")
            
            model_name = self.train_neural_network(sampling='iterative')
            
            if kill_iteration:
                print(f"Final model is {model_name}")
                break
            else:
                print(f"New model is {model_name}")
                i += 1
