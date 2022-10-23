import numpy as np
import os
import subprocess as sp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from source.lhc import LatinHypercubeSampler
from source.join_output import CreateSingleDataFile
from source.train_network import Training
import source.misc_functions as misc
from source.default_module import Parameters
import fileinput
import shutil
import pickle as pkl

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

        # The following three lines prevent an MPI warning (although harmless)
        #tmp_dir = sp.run('echo "${HOME}/tmp"', shell=True, stdout=sp.PIPE).stdout.decode('utf-8')
        #tmp_dir = os.path.join(self.CONNECT_PATH, 'tmp')
        #os.system(f'mkdir -p {tmp_dir}')
        #os.environ["TMPDIR"] = tmp_dir
        #slurm_bool = int(sp.run('if [ -z $SLURM_NPROCS ]; then echo 0; else echo 1; fi', shell=True, stdout=sp.PIPE).stdout.decode('utf-8'))
        #self.mp_node = None
        #if slurm_bool:
        #    self.mp_node = misc.get_node_with_most_cpus()
        #os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"

    def create_lhc_data(self):
        self.latin_hypercube_sampling()
        #misc.create_output_folders(self.param)
        self.copy_param_file()
        self.call_calc_models(sampling='lhc')

    def create_iterative_data(self):
        exec(f'from source.mcmc_samplers.{self.param.mcmc_sampler} import {self.param.mcmc_sampler}')
        _locals = {}
        exec(f'mcmc = {self.param.mcmc_sampler}(self.param, self.CONNECT_PATH)', locals(), _locals)
        mcmc = _locals['mcmc']

        mcmc.check_version()
        #misc.create_output_folders(self.param)
        if not os.path.isdir(f"data/{self.param.jobname}"):
            os.system(f"mkdir data/{self.param.jobname}")
        os.system(f"rm -rf data/{self.param.jobname}/compare_iterations")
        os.system(f"mkdir data/{self.param.jobname}/compare_iterations")
        self.copy_param_file()
        if not self.param.resume_iterations:
            os.system(f"rm -rf {self.CONNECT_PATH}/data/{self.param.jobname}/number_*")
        mcmc.Gelman_Rubin_log_ini()
        if self.param.initial_model == None:
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
        keep_idx = 0
        kill_iteration = False
        print(f'Initial model is {model}', flush=True)

        i = 1
        while True:
            print(f'\n\nBeginning iteration no. {i}', flush=True)
            print(f'Running MCMC sampling no. {i}...', flush=True)
            mcmc.run_mcmc_sampling(model, i)
            print(f'MCMC sampling stopped since R-1 less than {self.param.mcmc_tol} has been reached', flush=True)

            N_acc = mcmc.get_number_of_accepted_steps(i)
            print(f'Number of accepted steps: {N_acc}', flush=True)
            if i == 1 and not self.param.keep_first_iteration:
                N_keep = 5000
                keep_idx = 1
            else:
                N_keep = self.param.N_max_lines
            N_keep = mcmc.filter_chains(N_keep,i)
            print(f'Keeping only last {N_keep} of the accepted Markovian steps', flush=True)
            print('Comparing latest iterations...', flush=True)
            if i > 1:
                kill_iteration = mcmc.compare_iterations(i)
            if i > keep_idx + 1:
                model_params=f"data/{self.param.jobname}/number_{i-1}/model_params.txt"
                N_accepted=mcmc.discard_oversampled_points(model_params,i)
                print(f"Accepted {N_accepted} points out of {N_keep}", flush=True)
            else:
                N_accepted=N_keep

            if kill_iteration and N_accepted < 0.1*N_keep:
                print(f"Iteration {i} will be the last since convergence in data has been reached", flush=True)
            
            print(f'Calculating {N_accepted} CLASS models', flush=True)
            misc.create_output_folders(self.param, iter_num=i, reset=False)
            self.call_calc_models(sampling='iterative')
            misc.join_data_files(self.param)
            if i > keep_idx + 1:
                misc.combine_iterations_data(self.param, i)
                print(f"Copied data from data/{self.param.jobname}/number_{i-1} into data/{self.param.jobname}/number_{i}", flush=True)
            
            print("Training neural network", flush=True)
            model_name = self.train_neural_network(sampling='iterative',
                                                   output_file=os.path.join(self.data_path,
                                                                            f'number_{i}/training.log'))
            
            if kill_iteration and N_accepted < 0.1*N_keep:
                print(f"Final model is {model_name}", flush=True)
                break
            else:
                print(f"New model is {model_name}", flush=True)
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

    def latin_hypercube_sampling(self):
        if not os.path.isfile(self.CONNECT_PATH+f'/data/lhc_samples/sample_models_{self.param.jobname}_{self.param.N}.txt'):
            lhc = LatinHypercubeSampler(self.param)
            lhc.run(self.CONNECT_PATH)

    def call_calc_models(self, sampling='lhc'):
        os.environ["export OMP_NUM_THREADS"] = str({self.N_cpus_per_task})
        os.environ["PMIX_MCA_gds"] = "hash"
        #--mca shmem_mmap_enable_nfs_warning 0
        sp.Popen(f"mpirun --mca orte_base_help_aggregate 0 -np {self.N_tasks - 1} python {self.CONNECT_PATH}/source/calc_models_mpi.py {self.param.param_file} {self.CONNECT_PATH}".split()).wait()
        os.environ["export OMP_NUM_THREADS"] = "1"

    def train_neural_network(self, sampling='lhc', output_file=None):
        if sampling == 'lhc':
            folder = ''
        elif sampling == 'iterative':
            i = max([int(f.split('number_')[-1]) for f in os.listdir(self.CONNECT_PATH+'/'+self.data_path) if f.startswith('number')])
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
            model_name = f'{self.param.jobname}_N{self.param.N}_bs{self.param.batchsize}_e{self.param.epochs}'
        if not self.param.overwrite_model:
            M = 1
            if os.path.isdir(os.path.join('trained_models', model_name)):
                while os.path.isdir(os.path.join('trained_models', model_name, f'_{M}')):
                    M += 1
                if M-1 > 0:
                    model_name += f'_{M-1}'
        return model_name
"""

    def run_mcmc_sampling(self, model, iteration):
        if self.param.mcmc_sampler == 'montepython':
            self.run_montepython_iteration(model, iteration)
        elif self.param.mcmc_sampler == 'cobaya':
            self.run_cobaya_iteration(model, iteration)

    def run_montepython_iteration(self, model, iteration):
        if iteration == 1:
            MP_param_file = misc.create_montepython_param(self.param,self.param.montepython_path)
        else:
            MP_param_file = f'input/connect/{self.param.jobname}.param'
        with fileinput.input(self.param.montepython_path+'/'+MP_param_file, inplace=True) as file:
            for line in file:
                if "data.cosmo_arguments['connect_model']" in line:
                    line = f"data.cosmo_arguments['connect_model'] = '{model}'\n"
                print(line, end='')
        
        if self.mp_node != None:
            sp.run(f'{self.CONNECT_PATH}/source/shell_scripts/run_montepython_iteration.sh {self.param.jobname} {MP_param_file} {self.CONNECT_PATH}/mp_plugin/connect.conf {self.param.mcmc_tol} {self.mp_node}', shell=True, cwd=self.param.montepython_path)
        else:
            sp.run(f'{self.CONNECT_PATH}/source/shell_scripts/run_montepython_iteration.sh {self.param.jobname} {MP_param_file} {self.CONNECT_PATH}/mp_plugin/connect.conf {self.param.mcmc_tol}', shell=True, cwd=self.param.montepython_path)
        shutil.copytree(self.param.montepython_path + f'/chains/connect_{self.param.jobname}_data', f'{self.CONNECT_PATH}/data/{self.param.jobname}/number_{iteration}')

    def run_cobaya_iteration(self, model, iteration):
        os.environ["export OMP_NUM_THREADS"] = "1"
        sp.Popen(f"mpirun -np 4 python {self.CONNECT_PATH}/source/run_cobaya_iteration.py {self.data_path}/log_connect.param".split()).wait()

    def compare_iterations(self,i):
        chain1=f"data/{self.param.jobname}/compare_iterations/chain__{i-1}.txt"
        chain2=f"data/{self.param.jobname}/compare_iterations/chain__{i}.txt"
        kill_iteration=misc.Gelman_Rubin_log(self.param,status=i,all_chains=[chain1,chain2])
        return kill_iteration

    def get_number_of_accepted_steps(self, iteration):
        directory = os.path.join(self.CONNECT_PATH,f'data/{self.param.jobname}/number_{iteration}')
        if self.param.mcmc_sampler == 'montepython':
            txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
            N_acc = 0
            for f in txt_files:
                N_acc += sum(1 for line in open(os.path.join(directory,f)) 
                             if line[0] != '#' and line.strip())

        elif self.param.mcmc_sampler == 'cobaya':
            with open(os.path.join(directory,'cobaya_all_chains.pkl'),'rb') as f:
                all_chains = pkl.load(f)[1:]
            N_acc = 0
            for chain in all_chains:
                N_acc += chain.shape[0]

        return N_acc

    def check_montepython_version(self):
        with open(self.param.montepython_path + '/VERSION','r') as f:
            version = list(f)[0].strip()
        if int(version.split('.')[0]) < 3:
            err_msg = f'Your version of MontePython is {version}, which is not compatible with python 3. Your MontePython version must be at least 3.0'
            print(err_msg, flush=True)
            raise NotImplementedError(err_msg)
        else:
            print(f'Your version of MontePython is {version}', flush=True)

    def check_cobaya_version(self):
        from cobaya import __version__ as version
        print(f'Your version of Cobaya is {version}', flush=True)

"""
