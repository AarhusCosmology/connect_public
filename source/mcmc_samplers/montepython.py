import os
import subprocess as sp
import re
import fileinput
import shutil

import numpy as np

from ..mcmc_base import MCMC_base_class


class montepython(MCMC_base_class):

    def __init__(self, param, CONNECT_PATH):
        super(montepython, self).__init__(param, CONNECT_PATH)
        path = {}
        with open(os.path.join(CONNECT_PATH, 'mcmc_plugin/connect.conf'), 'r') as f:
            for line in f:
                exec(line)
        self.montepython_path = path['montepython']

    def run_mcmc_sampling(self, model, iteration):
        MP_param_file = self.create_montepython_param(iteration)

        with fileinput.input(MP_param_file, inplace=True) as file:
            for line in file:
                if "data.cosmo_arguments['connect_model']" in line:
                    line = f"data.cosmo_arguments['connect_model'] = '{model}'\n"
                print(line, end='')

        output_dir = f'{self.CONNECT_PATH}/data/{self.param.jobname}/number_{iteration}'

        sp.run(f"{self.CONNECT_PATH}/source/mcmc_samplers/run_scripts/run_montepython_iteration.sh {output_dir} {MP_param_file} {os.path.join(self.CONNECT_PATH,'mcmc_plugin/connect.conf')} {self.param.mcmc_tol} {self.mcmc_node} {self.temperature}", shell=True, cwd=self.montepython_path)

    
    def get_number_of_accepted_steps(self, iteration):
        directory = os.path.join(self.CONNECT_PATH,f'data/{self.param.jobname}/number_{iteration}')
        txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        N_acc = 0
        for f in txt_files:
            N_acc += sum(1 for line in open(os.path.join(directory,f))
                         if line[0] != '#' and line.strip())
        return N_acc


    def filter_chains(self, N, iteration):
        path = f'data/{self.param.jobname}/number_{iteration}'
        N_max_points = N

        list_of_files=[]
        for filename in [f for f in os.listdir(path) if f.endswith('.txt')]:
            list_of_files.append(path + '/' + filename)
        lines_of_use_in_files = {}
        for filename in list_of_files:
            lines = self.useful_points_in_file(filename)
            lines_of_use_in_files[filename] = lines
        N_left = [lines_of_use_in_files[filename] for filename in list_of_files]
        N_to_use = self.filter_steps(N_left, N_max_points)

        output_lines = {}
        for filename, N_lines in zip(list_of_files, N_to_use):
            with open(filename, 'r') as f:
                f_list = list(f)
            output_lines[filename] = []
            for i, line in enumerate(reversed(f_list), 1):
                if i <= N_lines:
                    output_lines[filename].append(line)

        for filename in list_of_files:
            with open(filename, 'w') as f:
                for line in output_lines[filename]:
                    f.write(line)

        for filename in list_of_files:
            os.system(f"cat {filename} >> {f'data/{self.param.jobname}/compare_iterations/chain__{iteration}.txt'}")
        os.system(f"cp data/{self.param.jobname}/number_{iteration}/log.param data/{self.param.jobname}/compare_iterations/log.param")

        return int(np.sum(N_to_use))


    def compare_iterations(self, iteration):
        chain1=f"data/{self.param.jobname}/compare_iterations/chain__{iteration-1}.txt"
        chain2=f"data/{self.param.jobname}/compare_iterations/chain__{iteration}.txt"
        kill_iteration=self.Gelman_Rubin_log(iteration, all_chains=[chain1,chain2])
        return kill_iteration


    def create_montepython_param(self, iteration):
        path = os.path.join(self.CONNECT_PATH, 'data', self.param.jobname, 'montepython_input')
        os.system(f"mkdir -p {path}")
        with open('mcmc_plugin/mp_param_templates/connect_lite.param.template','r') as f:
            with open(os.path.join(path, f'number_{iteration}.param'),'w') as g:
                for line in f:
                    g.write(line)
                    if '#------Experiments to test (separated with commas)-----' in line:
                        g.write('')
                        experiments_line = 'data.experiments=['
                        for lkl in self.param.sampling_likelihoods:
                            if lkl == 'Planck_lite':
                                lkl = 'Planck_highl_TTTEEE_lite'
                            elif lkl == 'Planck_lowl_EE':
                                lkl = 'Planck_lowl_EE_connect'
                            experiments_line += f"'{lkl}', "
                        experiments_line = experiments_line[:-2] + ']'
                        g.write(experiments_line)
                    elif '# Cosmological parameters list' in line:
                        g.write('')
                        for par,interval in self.param.parameters.items():
                            if par in self.param.prior_ranges:
                                xmin  = self.param.prior_ranges[par][0]
                                xmax  = self.param.prior_ranges[par][1]
                            else:
                                xmin = 'None'
                                xmax = 'None'
                            if par in self.param.bestfit_guesses:
                                guess = self.param.bestfit_guesses[par]
                            else:
                                guess = (interval[0] + interval[1])/2
                            if par in self.param.sigma_guesses:
                                sig = self.param.sigma_guesses[par]
                            else:
                                sig = abs(((interval[0] + interval[1])/2)/100)
                            if par == 'omega_b' or par == 'Omega_b':
                                scale = 0.01
                                guess *= 1/scale
                                sig   *= 1/scale
                                if not isinstance(xmin, str):
                                    xmin  *= 1/scale
                                if not isinstance(xmax, str):
                                    xmax  *= 1/scale
                            else:
                                scale = 1
                            if par in self.param.log_priors:
                                if not isinstance(xmin, str):
                                    xmin = np.log10(xmin)
                                if not isinstance(xmax, str):
                                    xmax = np.log10(xmax)
                                guess = np.log10(guess)
                                if not isinstance(xmin, str) and not isinstance(xmax, str):
                                    sig = sig * (xmax-xmin)/(10**xmax-10**xmin)
                                else:
                                    sig = 0.01
                                par = par + '_log10_prior'

                            g.write(f"data.parameters['{par}'] = [{guess}, {xmin}, {xmax}, {sig}, {scale}, 'cosmo']\n")
                    elif '# Derived parameter list' in line:
                        g.write('')
                        for par in self.param.output_derived:
                            if par == 'A_s':
                                scale = 1e-9
                            else:
                                scale = 1
                            g.write(f"data.parameters['{par}'] = [1, None, None, 0, {scale}, 'derived']\n")
        return os.path.join(path, f'number_{iteration}.param')



    def get_Rminus1_of_chains(self,
                              all_chains,    # list of paths to chains
                              iteration      # Iteration number
                        ):

        output = sp.run(f"python {os.path.join(self.montepython_path, 'montepython/MontePython.py')} info {all_chains[0]} {all_chains[1]} --noplot --minimal", shell=True, stdout=sp.PIPE).stdout.decode('utf-8')
        output = list(iter(output.splitlines()))
        kill_iteration = False
        if any('Removed everything: chain not converged' in s for s in output):
            print(f"chains from iterations {iteration} and {iteration-1} did not have sufficient overlap")
            Rm1_line = '\t' + f'chains from iterations {iteration} and {iteration-1} did not have sufficient overlap\n'
        else:
            Rm1_line = ''
            try:
                index = [idx for idx, s in enumerate(output) if '-> R-1 is' in s][0]
            except:
                raise RuntimeError('chains are much too short for Monte Python to analyse. Increase N_max_points in the parameter file')
            kill_iteration = True
            largest_Rm1 = 0
            for par in self.param.parameters:
                for out in output[index:]:
                    if par == out[-len(par):] and out[-len(par)-1] == ' ':
                        Rm1 = np.float32(re.findall("\d+\.\d+", out)[0])
                        if Rm1 > largest_Rm1:
                            largest_Rm1 = Rm1
                        Rm1_line += '\t' + str(Rm1)
                        if Rm1 > self.param.iter_tol:
                            kill_iteration = False
                        break
            Rm1_line += '\t' + str(largest_Rm1) + '\n'
            print(f"Iteration {iteration} and {iteration-1} has the following R-1 values")
            for line in output[index:]:
                if any(par in line for par in self.param.parameters):
                    print(line)
        return kill_iteration, Rm1_line



    def check_version(self):
        with open(os.path.join(self.montepython_path, 'VERSION'),'r') as f:
            version = list(f)[0].strip()
        if int(version.split('.')[0]) < 3:
            err_msg = f'Your version of MontePython is {version}, which is not compatible with python 3. Your MontePython version must be at least 3.0'
            print(err_msg, flush=True)
            raise NotImplementedError(err_msg)
        else:
            print(f'Your version of MontePython is {version}', flush=True)


    def save_accepted_points(self,
                             indices_accepted,   # indices of accepted points
                             iteration           # Iteration Number
                         ):
        path = f'data/{self.param.jobname}/number_{iteration}'
        files = sorted([f for f in os.listdir(path) if f.endswith('.txt')])

        indices = indices_accepted
        for filename in files:
            N = self.useful_points_in_file(path + '/' + filename)
            indices_chain=[i for i in indices if i < N]
            indices=[i-N for i in indices if i >= N]
            with open(path + '/' + filename, 'r') as f:
                f_list = list(f)
            with open(path + '/' + filename, 'w') as f:
                line_number = 0
                for line in f_list:
                    if line_number in indices_chain: 
                        f.write(line)
                    line_number += 1


    def useful_points_in_file(self, filename):
        with open(filename, 'r') as f:
            lines = 0
            for i, line in enumerate(f, 1):
                if line[0] != '#':
                    lines += 1
                else:
                    lines = 0
        return lines

    def backup_full_chains(self, iteration):
        files = sorted([f for f in os.listdir(f'data/{self.param.jobname}/number_{iteration}') if f.endswith('.txt')])
        os.mkdir(f'data/{self.param.jobname}/number_{iteration}/full_chains')
        for filename in files:
            os.system(f"cp data/{self.param.jobname}/number_{iteration}/{filename} data/{self.param.jobname}/number_{iteration}/full_chains/{filename}")



    def import_points_from_chains(self,
                                  iteration          # Current iteration number
                              ):

        paramnames = self.param.parameters.keys()
        model_param_scales = []
        with open(f'data/{self.param.jobname}/number_{iteration}/log.param','r') as f:
            lines = list(f)
        for i, name in enumerate(paramnames):
            for line in lines:
                if name in self.param.log_priors:
                    if line.startswith(f"data.parameters['{name}_log10_prior']") and line.split("'")[-2] == 'cosmo':
                        model_param_scales.append(np.float32(line.split('=')[-1].replace(' ','').split(',')[4]))
                        break
                else:
                    if line.startswith(f"data.parameters['{name}']") and line.split("'")[-2] == 'cosmo':
                        model_param_scales.append(np.float32(line.split('=')[-1].replace(' ','').split(',')[4]))
                        break
        model_param_scales = np.array(model_param_scales)
        mp_names = []
        paramnames_file = [f for f in os.listdir(f'data/{self.param.jobname}/number_{iteration}') if f.endswith('.paramnames')][0]
        with open(f'data/{self.param.jobname}/number_{iteration}/' + paramnames_file, 'r') as f:
            list_f = list(f)
        for line in list_f[0:len(paramnames)]:
            name = line.replace(' ','').split('\t')[0]
            if name[-12:] == '_log10_prior' and name[:-12] in self.param.log_priors:
                name = name[:-12]
            mp_names.append(name)

        lines=[]
        i = 0
        data = []
        files = sorted([f for f in os.listdir(f'data/{self.param.jobname}/number_{iteration}') if f.endswith('.txt')])
        for filename in files:
            with open(f'data/{self.param.jobname}/number_{iteration}/' + filename, 'r') as f:
                list_f = list(f)
            lines += list_f
            for line in list_f:
                if line[0] != '#':
                    params = np.float32(line.replace('\n','').split('\t')[1:len(paramnames)+1])
                    for name in self.param.log_priors:
                        k = mp_names.index(name)
                        params[k] = np.power(10.,params[k])
                    params *= model_param_scales
                    data.append([params[j] for j in [mp_names.index(n) for n in paramnames]])
                    i += 1

        return np.array(data)
