import os
import subprocess as sp
import pickle as pkl

import numpy as np

from ..mcmc_base import MCMC_base_class


class cobaya(MCMC_base_class):

    def run_mcmc_sampling(self, model, iteration):
        os.mkdir(os.path.join(self.CONNECT_PATH, f'data/{self.param.jobname}/number_{iteration}'))
        os.environ["export OMP_NUM_THREADS"] = "1"
        if self.mcmc_node != None:
            mpi_flag = f" -np 4 --host {self.mcmc_node}"
        else:
            mpi_flag = " -np 4"
        sp.Popen(f"mpirun {mpi_flag} python {self.CONNECT_PATH}/source/mcmc_samplers/run_scripts/run_cobaya_iteration.py {model} {iteration} {self.param.param_file} {self.temperature}".split()).wait()
    
    def get_number_of_accepted_steps(self, iteration):
        directory = os.path.join(self.CONNECT_PATH,f'data/{self.param.jobname}/number_{iteration}')
        with open(os.path.join(directory,'cobaya_all_chains.pkl'),'rb') as f:
            all_chains = pkl.load(f)['chains']
        N_acc = 0
        for chain in all_chains:
            N_acc += chain.shape[0]

        return N_acc


    def filter_chains(self, N_max_points, iteration):

        path = f'data/{self.param.jobname}/number_{iteration}'
        with open(path+'/cobaya_all_chains.pkl','rb') as f:
            all_chains = pkl.load(f)
        N_left=[]
        for chain in all_chains['chains']:
            N_left.append(chain.shape[0])
        N_to_use = self.filter_steps(N_left, N_max_points)

        filtered_chains = []
        filtered_and_combined_chain = np.zeros(shape=(0,all_chains['chains'][0].shape[1]))
        for chain, N in zip(all_chains['chains'], N_to_use):
            filtered_chain = chain[-N:,:]
            filtered_chains.append(filtered_chain)
            filtered_and_combined_chain = np.concatenate((filtered_and_combined_chain, filtered_chain))
        out_all_chains = all_chains.copy()
        out_all_chains['chains'] = filtered_chains
        with open(path+'/cobaya_all_chains.pkl','wb') as f:
            pkl.dump(out_all_chains, f)
        out_all_chains['chains'] = [filtered_and_combined_chain]
        with open(f'data/{self.param.jobname}/compare_iterations/chain__{iteration}.pkl','wb') as f:
            pkl.dump(out_all_chains, f)

        return int(np.sum(N_to_use))


    def compare_iterations(self, iteration):
        with open(f'data/{self.param.jobname}/compare_iterations/chain__{iteration-1}.pkl','rb') as f:
            all_chains = pkl.load(f)
        with open(f'data/{self.param.jobname}/compare_iterations/chain__{iteration}.pkl','rb') as f:
            chain_add = pkl.load(f)['chains']
        all_chains['chains'] += chain_add
        kill_iteration=self.Gelman_Rubin_log(iteration, all_chains=all_chains)
        return kill_iteration


    def get_Rminus1_of_chains(self,
                              all_chains,    # Chains object
                              iteration      # Iteration number
                        ):

        param_names = self.param.parameters.keys()
        sampled_names = all_chains['names']['sampled']
        chains = all_chains['chains']
        indices = []
        for name in param_names:
            if name == 'ln10^{10}A_s':
                indices.append(sampled_names.index('logA'))
            elif name+'_log10_prior' in sampled_names:
                indices.append(sampled_names.index(name+'_log10_prior'))
            else:
                indices.append(sampled_names.index(name))

        N_params = len(param_names)
        data_chains = []
        for chain in chains:
            data_chains.append(chain[:,2:N_params+2])

        Rm1_line = ''
        print(f"Iteration {iteration} and {iteration-1} has the following R-1 values", flush=True)
        n_char_max = 0
        for par in param_names:
            if len(par) > n_char_max:
                n_char_max = len(par)
        for i, par in zip(indices, param_names):
            cs = []
            for chain in data_chains:
                cs.append(chain[:,i:1+i])
            Rm1_par = self.Rm1(cs)
            Rm1_line += '\t' + str(Rm1_par)
            n_char = len(par)
            print(par+' '*(n_char_max-n_char+2)+':    '+str(Rm1_par))
        Rm1_tot = self.Rm1(data_chains)
        Rm1_line += '\t' + str(Rm1_tot) + '\n'
        print('Combined'+' '*(n_char_max-6)+':    '+str(Rm1_tot))
        if Rm1_tot > self.param.iter_tol:
            kill_iteration = False
        else:
            kill_iteration = True

        return kill_iteration, Rm1_line


    def check_version(self):
        from cobaya import __version__ as version
        print(f'Your version of Cobaya is {version}', flush=True)

    def save_accepted_points(self,
                             indices_accepted,   # indices of accepted points
                             iteration           # Iteration Number
                         ):
        path = f'data/{self.param.jobname}/number_{iteration}'
        with open(os.path.join(path, 'cobaya_all_chains.pkl'),'rb') as f:
            all_chains = pkl.load(f)
        out_all_chains = all_chains.copy()
        out_all_chains['chains'] = []
        indices = indices_accepted
        for chain in all_chains['chains']:
            N = chain.shape[0]
            indices_chain=[i for i in indices if i < N]
            indices=[i-N for i in indices if i >= N]
            out_all_chains['chains'].append(chain[indices_chain,:])
        with open(os.path.join(path, 'cobaya_all_chains.pkl'),'wb') as f:
            pkl.dump(out_all_chains, f)



    def backup_full_chains(self, iteration):
        path = f'data/{self.param.jobname}/number_{iteration}'
        os.mkdir(os.path.join(path, 'full_chains'))
        os.system(f"cp {path}/cobaya_all_chains.pkl {path}/full_chains/cobaya_all_chains.pkl")

        with open(os.path.join(path, 'full_chains/cobaya_all_chains.pkl'), 'rb') as f:
            all_chains = pkl.load(f)
        names = all_chains['names']['all']
        for i, chain in enumerate(all_chains['chains']):
            with open(os.path.join(path, f'full_chains/cobaya_chain__{i+1}.txt'),'w') as f:
                header = '# '
                for name in names:
                    header += name+'\t'
                header = header[:-2]+'\n'
                f.write(header)
                for row in chain:
                    line = ''
                    for val in row:
                        line += str(val)+'\t'
                    line = line[:-2]+'\n'
                    f.write(line)

    def import_points_from_chains(self,
                                  iteration          # Current iteration number
                              ):

        path = f'data/{self.param.jobname}/number_{iteration}'
        paramnames = self.param.parameters.keys()

        with open(os.path.join(path, 'cobaya_all_chains.pkl'),'rb') as f:
            all_chains = pkl.load(f)
        combined_chains = np.zeros(shape=(0,all_chains['chains'][0].shape[1]))
        for chain in all_chains['chains']:
            combined_chains = np.concatenate((combined_chains, chain))

        sampled_names = all_chains['names']['sampled']
        indices = []
        log_indices = []
        for name in paramnames:
            if name == 'ln10^{10}A_s':
                indices.append(sampled_names.index('logA'))
            elif name+'_log10_prior' in sampled_names and name in self.param.log_priors:
                index = sampled_names.index(name+'_log10_prior')
                indices.append(index)
                log_indices.append(index)
            else:
                indices.append(sampled_names.index(name))
        data = np.zeros(shape=(combined_chains.shape[0], 0))
        for index in indices:
            if index in log_indices:
                data = np.concatenate((data, np.power(10.,combined_chains[:,2+index:3+index])), axis=1)
            else:
                data = np.concatenate((data, combined_chains[:,2+index:3+index]), axis=1)
        return data
