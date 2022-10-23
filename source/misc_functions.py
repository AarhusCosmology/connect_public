import numpy as np
import os
import subprocess as sp
import re
import pickle as pkl

CONNECT_PATH  = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]

def get_computed_cls(cosmo       # A computed CLASS model
                 ):

    # get parameters from CLASS model
    BA                 = cosmo.get_background()
    conformal_age_     = BA['conf. time [Mpc]'][-1]
    der_pars           = cosmo.get_current_derived_parameters(['ra_rec','tau_rec'])
    cls                = cosmo.lensed_cl()
    l_max              = len(cls['ell']) - 1
    angular_rescaling_ = der_pars['ra_rec']/(conformal_age_ - der_pars['tau_rec'])
    l_linstep          = 40
    l_logstep          = 1.12

    # compute necessary ells like CLASS (see transfer module in CLASS)
    increment = max(int(2*(np.power(l_logstep,angular_rescaling_) - 1)), 1)
    l=[0,1,2]

    lin_increment = int(l_linstep*angular_rescaling_)
    while l[-1] + increment < l_max:
        if increment < lin_increment:
            l.append(l[-1] + increment)
            increment = max(int(l[-1]*(np.power(l_logstep,angular_rescaling_) - 1)), 1)
        else:
            l.append(l[-1] + lin_increment)

    # create reduced dict of cls
    ell = np.array(l)
    new_cls = {'ell':ell}
    for key, arr in cls.items():
        if key != 'ell':
            new_cls[key] = arr[ell]

    return new_cls


def create_output_folders(param,            # Parameter object
                          iter_num=None,    # Current iteration number
                          reset=True       # Resets output folders
                      ):
    path = os.path.join(CONNECT_PATH, f'data/{param.jobname}')
    if not os.path.isdir(path):
        os.mkdir(path)
    if iter_num == None:
        if (reset and param.initial_model == None) or param.sampling == 'lhc':
            for name in os.listdir(path):
                if not name.startswith('N-') or name == f'N-{param.N}':
                    os.system(f"rm -rf {os.path.join(path, name)}")
            os.mkdir(os.path.join(path,f'N-{param.N}'))
            os.mkdir(os.path.join(path, f'N-{param.N}/model_params_data'))
            for output in param.output_Cl:
                os.mkdir(os.path.join(path, f'N-{param.N}/Cl_{output}_data'))
            for output in param.output_Pk:
                os.mkdir(os.path.join(path, f'N-{param.N}/Pk_{output}_data'))
            for output in param.output_bg:
                os.mkdir(os.path.join(path, f'N-{param.N}/{output}_data'))
            for output in param.output_th:
                os.mkdir(os.path.join(path, f'N-{param.N}/{output}_data'))
            if len(param.output_derived) > 0:
                os.mkdir(os.path.join(path, f'N-{param.N}/derived_data'))

        elif reset:
            for name in os.listdir(path):
                if not name.startswith('N-'):
                    os.system(f"rm -rf {os.path.join(path, name)}")

    else:
        if reset:
            os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}')}")
            os.mkdir(os.path.join(path, f'number_{iter_num}'))
        os.mkdir(os.path.join(path, f'number_{iter_num}/model_params_data'))
        for output in param.output_Cl:
            os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/Cl_{output}_data')}")
            os.mkdir(os.path.join(path, f'number_{iter_num}/Cl_{output}_data'))
        for output in param.output_Pk:
            os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/Pk_{output}_data')}")
            os.mkdir(os.path.join(path, f'number_{iter_num}/Pk_{output}_data'))
        for output in param.output_bg:
            os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/{output}_data')}")
            os.mkdir(os.path.join(path, f'number_{iter_num}/{output}_data'))
        for output in param.output_th:
            os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/{output}_data')}")
            os.mkdir(os.path.join(path, f'number_{iter_num}/{output}_data'))
        if len(param.output_derived) > 0:
            os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/derived_data')}")
            os.mkdir(os.path.join(path, f'number_{iter_num}/derived_data'))

"""
def create_montepython_param(param,             # Parameter object
                             montepython_path   # Path to MontePython folder
                         ):
    
    if not os.path.isdir(montepython_path+'/input/connect'):
        os.mkdir(montepython_path+'/input/connect')

    with open(CONNECT_PATH+'/mp_plugin/param_templates/connect_lite.param.template','r') as f:
        with open(montepython_path+f'/input/connect/{param.jobname}.param','w') as g:
            for line in f:
                g.write(line)
                if '# Cosmological parameters list' in line:
                    g.write('')
                    for par,interval in param.parameters.items():
                        if par in param.prior_ranges:
                            xmin  = param.prior_ranges[par][0]
                            xmax  = param.prior_ranges[par][1]
                        else:
                            xmin = 'None'
                            xmax = 'None'
                        if par in param.bestfit_guesses:
                            guess = param.bestfit_guesses[par]
                        else:
                            guess = (interval[0] + interval[1])/2
                        if par in param.sigma_guesses:
                            sig = param.sigma_guesses[par]
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
                        if par in param.log_priors:
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
                    for par in param.output_derived:
                        if par == 'A_s':
                            scale = 1e-9
                        else:
                            scale = 1
                        g.write(f"data.parameters['{par}'] = [1, None, None, 0, {scale}, 'derived']\n")
    return f'input/connect/{param.jobname}.param'


def Rm1(chains):
    covs = []
    Ns = []
    means = []
    for c in chains:
        Np = c.shape[1]
        N = int(len(c[:,0]))
        Ns.append(N)
        means.append(np.sum(c,axis=0)/N)
        covs.append(np.cov(c,rowvar=False))
    covs=np.array(covs)
    means=np.array(means)
    Ns=np.array(Ns)
    mean_of_covs = np.average(covs, weights=Ns, axis=0)
    cov_of_means = np.atleast_2d(np.cov(means.T))
    d = np.sqrt(np.diag(cov_of_means))
    if Np > 1:
        corr_of_means = (cov_of_means / d).T / d
        norm_mean_of_covs = (mean_of_covs / d).T / d
        L = np.linalg.cholesky(norm_mean_of_covs)
        Linv = np.linalg.inv(L)
        eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
        Rminus1 = max(np.abs(eigvals))
    else:
        mean_of_means = np.mean(means)
        n = np.mean(ns)
        nc = len(means)
        B = 0
        for m in means:
            B += (m-mean_of_means)**2*n/(nc-1)
        W = 0
        for c, m in zip(chains, means):
            for p in c[:,0]:
                W += 1/(nc*(n-1))*(p-m)**2
        Rminus1 = ((n-1)/n*W + 1/n*B)/W - 1

    return Rminus1


def get_Rminus1_of_chains(all_chains,    # list of strings for montepython or data dict for cobaya
                          mcmc_sampler,  # mcmc sampler to use ('montepython' or 'cobaya')
                          convergence,   # target R-1 value for stopping the iterations
                          param_names,   # list of parameter names
                          iter_num       # Iteration number
                      ):

    if mcmc_sampler == 'cobaya':
        sampled_names = all_chains['names']['sampled']
        chains = all_chains['chains']
        indices = []
        for name in enumerate(param_names):
            if name == 'ln10^{10}A_s':
                indices.append(sampled_names.index('logA'))
            elif name+'_log_10_prior' in sampled_names:
                indices.append(sampled_names.index(name+'_log_10_prior'))
            else:
                indices.append(sampled_names.index(name))

        N_params = len(param_names)
        data_chains = []
        for chain in chains:
            data_chains.append(chain[:,2:N_params+2])

        Rm1_line = ''
        print(f"Iteration {iter_num} and {iter_num-1} has the following R-1 values")
        n_char_max = 0
        for par in param_names:
            if len(par) > n_char_max:
                n_char_max = len(par)
        for i, par in zip(indices, param_names):
            cs = []
            for chain in data_chains:
                cs.append(chain[:,i:1+i])
            Rm1_par = Rm1(cs)
            Rm1_line += '\t' + str(Rm1_par)
            n_char = len(par)
            print(par+' '*(n_char_max-n_char+2)+':    '+str(Rm1_par))
        Rm1_tot = Rm1(chains)
        Rm1_line += '\t' + str(Rm1_tot)
        if Rm1_tot > convergence:
            kill_iteration = False
        else:
            kill_iteration = True

    elif mcmc_sampler == 'montepython':
        output = sp.run(f'python2 {self.param.montepython_path}/montepython/MontePython.py info {all_chains[0]} {all_chains[1]} --noplot --minimal', shell=True, stdout=sp.PIPE).stdout.decode('utf-8')
        output = list(iter(output.splitlines()))
        kill_iteration = False
        if any('Removed everything: chain not converged' in s for s in output):
            print(f"chains from iterations {iter_num} and {iter_num-1} did not have sufficient overlap")
            Rm1_line = '\t' + f'chains from iterations {iter_num} and {iter_num-1} did not have sufficient overlap\n'
        else:
            Rm1_line = ''
            index = [idx for idx, s in enumerate(output) if '-> R-1 is' in s][0]
            kill_iteration = True
            for par in param_names:
                for out in output[index:]:
                    if par == out[-len(par):] and out[-len(par)-1] == ' ':
                        Rm1 = np.float32(re.findall("\d+\.\d+", out)[0])
                        Rm1_line += '\t' + str(Rm1)
                        if Rm1 > convergence:
                            kill_iteration = False
                        break
            Rm1_line += '\n'
            print(f"Iteration {iter_num} and {iter_num-1} has the following R-1 values")
            for line in output[index:]:
                if any(par in line for par in param.parameters):
                    print(line)
    return kill_iteration, Rm1_line


def Gelman_Rubin_log(param,           # Parameter object
                     status,          # 'initial' call or current iteration number 
                     all_chains=None, # list of strings for montepython or data structures for cobaya
                 ):

    if status == 'initial':
        with open(CONNECT_PATH+f'/data/{param.jobname}/Gelman-Rubin.txt', 'w') as f:
            f.write('# R-1 values for each parameter in every iteration\n')
            header = '# iterations'
            for par in param.parameters:
                header += '\t'+par
            if param.mcmc_sampler == 'cobaya':
                header += '\tCombined'
            header += '\n'
            f.write(header)
    else:
        try:
            status=int(status)
        except:
            raise TypeError("status must be either 'initial' or an integer.")

        Rm1_line = f'{status-1}-{status}'

        kill_iteration, Rm1_line_add = get_Rminus1_of_chains(all_chains,
                                                             param.mcmc_sampler,
                                                             param.iter_tol,
                                                             param.parameters.keys(),
                                                             status)
        Rm1_line += Rm1_line_add

        with open(CONNECT_PATH+f'/data/{param.jobname}/Gelman-Rubin.txt', 'a') as f:
            f.write(Rm1_line)

        return kill_iteration


def filter_steps(N_left, N_max):

    N_to_use = [0]*len(N_left)
    if np.sum(N_left) > N_max:
        N_per_chain = int(np.floor(N_max/len(N_left)))
        if not N_per_chain == 0:
            for i in range(len(N_left)):
                if N_left[i] >= N_per_chain:
                    N_left[i] = N_left[i]-N_per_chain
                    N_to_use[i] = N_per_chain
                else:
                    N_to_use[i] = N_left[i]
                    N_left[i] = 0
        N_missing = N_max - np.sum(N_to_use)
        if N_missing > 0:
            stop = False
        else:
            stop = True
        while not stop:
            for i in range(len(Ns)):
                if N_left[i] > 0:
                    N_to_use[i] += 1
                    N_left[i] -= 1
                    N_missing -= 1
                    if not N_missing > 0:
                        stop = True
                        break
    return N_to_use


def filter_chains(param,            # Parameter object
                  path,             # Path to folder containing chains
                  N,                # Maximal number of steps to keep
                  iter_num          # Current iteration number
              ):
    
    if param.mcmc_sampler == 'montepython':
        list_of_files=[]
        for filename in [f for f in os.listdir(path) if f.endswith('.txt')]:
            list_of_files.append(path + '/' + filename)
        lines_of_use_in_files = {}
        for filename in list_of_files:
            with open(filename, 'r') as f:
                lines = 0
                for i, line in enumerate(f, 1):
                    if line[0] != '#':
                        lines += 1
                    else:
                        lines = 0
            lines_of_use_in_files[filename] = lines
        N_left = [lines_of_use_in_files[filename] for filename in list_of_files]
        N_to_use = filter_steps(N_left, N)

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
            os.system(f"cat {filename} >> {f'data/{param.jobname}/compare_iterations/chain__{iter_num}.txt'}")
        os.system(f"cp data/{param.jobname}/number_{iter_num}/log.param data/{param.jobname}/compare_iterations/log.param")


    elif param.mcmc_sampler == 'cobaya':
        with open(path+'/cobaya_all_chains.pkl','rb') as f:
            all_chains = pkl.load(f)
        N_left=[]
        for chain in all_chains['chains']:
            N_left.append(chain.shape[0])
        N_to_use = filter_steps(N_left, N)

        filtered_chains = []
        filtered_and_combined_chain = np.zeros(shape=(0,len(all_chains['chains'][0].shape[1]))) 
        for chain, N in zip(chains, N_to_use):
            filtered_chains = chain[-N:,:]
            filtered_chains.append(filtered_chain)
            filtered_and_combined_chain = np.concatenate((filtered_and_combined_chain, filtered_chain))
        out_all_chains = all_chains.copy()
        out_all_chains['chains'] = filtered_chains
        with open(path+'/cobaya_all_chains.pkl','wb') as f:
            pkl.dump(out_all_chains, f)
        out_all_chains['chains'] = [filtered_and_combined_chain]
        with open(f'data/{param.jobname}/compare_iterations/chain__{iter_num}.pkl','wb') as f:
            pkl.dump(out_all_chains, f)

    return int(np.sum(N_to_use))


def import_points_from_chains(param,            # Parameter object
                              iter_num          # Current iteration number
                          ):

    if param.mcmc_sampler == 'montepython':
        model_param_scales = []
        with open(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/log.param','r') as f:
            lines = list(f)
        for i, name in enumerate(paramnames):
            for line in lines:
                if name in param.log_priors:
                    if line.startswith(f"data.parameters['{name}_log10_prior']") and line.split("'")[-2] == 'cosmo':
                        model_param_scales.append(np.float32(line.split('=')[-1].replace(' ','').split(',')[4]))
                        break
                else:
                    if line.startswith(f"data.parameters['{name}']") and line.split("'")[-2] == 'cosmo':
                        model_param_scales.append(np.float32(line.split('=')[-1].replace(' ','').split(',')[4]))
                        break
        model_param_scales = np.array(model_param_scales)
        mp_names = []
        paramnames_file = [f for f in os.listdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}') if f.endswith('.paramnames')][0]
        with open(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/' + paramnames_file, 'r') as f:
            list_f = list(f)
        for line in list_f[0:len(paramnames)]:
            name = line.replace(' ','').split('\t')[0]
            if name[-12:] == '_log10_prior' and name[:-12] in param.log_priors:
                name = name[:-12]
            mp_names.append(name)

        lines=[]
        i = 0
        data = []
        for filename in [f for f in os.listdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}') if f.endswith('.txt')]:
            with open(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/' + filename, 'r') as f:
                list_f = list(f)
            lines += list_f
            for line in list_f:
                if line[0] != '#':
                    params = np.float32(line.replace('\n','').split('\t')[1:len(paramnames)+1])
                    for name in param.log_priors:
                        k = mp_names.index(name)
                        params[k] = np.power(10.,params[k])
                    params *= model_param_scales
                    data.append([params[j] for j in [mp_names.index(n) for n in paramnames]])
                    i += 1

        data = np.array(data)

    elif param.mcmc_sampler == 'cobaya':
        with open(self.CONNECT_PATH+f'/data/{self.param.jobname}/number_{iteration}/cobaya_all_chains.pkl','rb') as f:
            all_chains = pkl.load(f)
        combined_chains = np.zeros(shape=(0,all_chains['chains'][0].shape[1]))
        for chain in all_chains['chains']:
            combined_chains = np.concatenate((combined_chains, chain))

        sampled_names = all_chains['names']['sampled']
        indices = []
        log_indices = []
        for name in enumerate(paramnames):
            if name == 'ln10^{10}A_s':
                indices.append(sampled_names.index('logA'))
            elif name+'_log_10_prior' in sampled_names and name in param.log_priors:
                index = sampled_names.index(name+'_log_10_prior')
                indices.append(index)
                log_indices.append(index)
            else:
                indices.append(sampled_names.index(name))
        data = np.zeros(shape=(all_chains['chains'][0].shape[0],0))
        for index in indices:
            if index in log_indices:
                data = np.concatenate((data, np.power(10.,combined_chains[:,2+index:3+index])))
            else:
                data = np.concatenate((data, combined_chains[:,2+index:3+index]))
    return data


def import_points_to_compare(file_old_data,    # model_params.txt file with all the previous data
                             param,            # Parameter object
                             iter_num          # Current iteration number
                         ):

    with open(file_old_data, 'r') as f:
        lines=list(f)
    paramnames = []
    for name in lines[0].replace('#', '').replace(' ', '').replace('\n', '').split('\t'):
        paramnames.append(name)
    points1 = []
    for line in lines[1:]:
        points1.append(np.float32(line.replace('\n', '').split('\t')))
    points1 = np.array(points1)

    os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/full_chains')
    points2 = import_points_to_compare(param, iter_num)

    if param.mcmc_sampler == 'montepython':
        file2point_dict = {}
        for filename in [f for f in os.listdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}') if f.endswith('.txt')]:
            with open(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/' + filename, 'r') as f:
                list_f = list(f)
            lines += list_f
            for line in list_f:
                if line[0] != '#':
                    file2point_dict[str(i)] = filename
            os.system(f"cp {CONNECT_PATH}/data/{param.jobname}/number_{iter_num}/{filename} {CONNECT_PATH}/data/{param.jobname}/number_{iter_num}/full_chains/{filename}")

    elif param.mcmc_sampler == 'cobaya':
        os.system(f"cp {CONNECT_PATH}/data/{param.jobname}/number_{iter_num}/cobaya_all_chains.pkl {CONNECT_PATH}/data/{param.jobname}/number_{iter_num}/full_chains/cobaya_all_chains.pkl")
        file2point_dict = None

    return points1, points2, file2point_dict


def discard_oversampled_points(file_old_data,    # model_params.txt file with all the previous data
                               param,            # Parameter object
                               iter_num          # Current iteration number
                           ):

    points1, points2, file2point_dict = import_points_from_chains(file_old_data, param, iter_num)

    num_points_in_vicinity = 10

    ranges = []
    for par1, par2 in zip(points1.T, points2.T):
        ranges.append([min(min(par1),min(par2)),max(max(par1),max(par2))])
    ranges = np.array(ranges)
    range_dim = np.diff(ranges).T[0]


    indices_accepted = []
    for i,p in enumerate(points2):
        distances1 = np.sqrt(np.sum(((points1-p)/range_dim)**2,axis=1))
        min_dist = min(distances1)
        points2_i = np.delete(points2,i,0)
        distances2 = np.sqrt(np.sum(((points2_i-p)/range_dim)**2,axis=1))
        j = np.argmin(distances1)
        p1_min = points1[j]
        points1_i = np.delete(points1,np.argmin(distances1),0)
        distances1_2 = np.sqrt(np.sum(((points1_i-p1_min)/range_dim)**2,axis=1))
        cp_list1 = distances1_2[np.argsort(distances1_2)[:num_points_in_vicinity]]
        av_dist1 = np.sum(cp_list1)/num_points_in_vicinity
        std_dist1 = np.std(cp_list1)
        cp_list2 = distances2[np.argsort(distances2)[:num_points_in_vicinity]]
        av_dist2 = np.sum(cp_list2)/num_points_in_vicinity
        std_dist2 = np.std(cp_list2)
        if min_dist > av_dist1 + 2*std_dist1:
            indices_accepted.append(i)
        elif av_dist2 + 0*std_dist2 < av_dist1 - 2*std_dist1:
            indices_accepted.append(i)

    if param.mcmc_sampler == 'montepython':
        for filename in [f for f in os.listdir(f'data/{param.jobname}/number_{iter_num}') if f.endswith('.txt')]:
            with open(f'data/{param.jobname}/number_{iter_num}/' + filename, 'w') as f:
                for idx in indices_accepted:
                    if file2point_dict[str(idx)] == filename:
                        f.write(lines[idx])
    elif param.mcmc_sampler == 'cobaya':
        with open(self.CONNECT_PATH+f'/data/self.param.jobname/number_{iteration}/cobaya_all_chains.pkl','rb') as f:
            all_chains = pkl.load(f)
        out_all_chains = all_chains.copy()
        out_all_chains['chains'] = []
        for chain in all_chains['chains']:
            N = chain.shape[0]
            i_chain=[i for i in indices if i < N]
            indices=[i-N for i in indices if i >= N]
            out_all_chains['chains'].append(chain[i_chain,:])
        with open(self.CONNECT_PATH+f'/data/self.param.jobname/number_{iteration}/cobaya_all_chains.pkl','wb') as f:
            pkl.dump(out_all_chains, f)


    return len(indices_accepted)

"""
def join_data_files(param     # Parameter object
                ):

    from source.join_output import CreateSingleDataFile
    CSDF = CreateSingleDataFile(param, CONNECT_PATH)
    CSDF.join()


def combine_sets_of_data_files(new_data,   # Data file for the new data (destination for combined data)
                               old_data    # Data file for the old data
                           ):

    with open(new_data,'a') as f:
        with open(old_data,'r') as g:
            for line in list(g)[1:]:
                f.write(line)


def combine_iterations_data(param,             # Parameter object
                            iter_num           # Current iteration number
                        ):

    path_i = os.path.join(CONNECT_PATH, f'data/{param.jobname}/number_{iter_num}')
    path_j = os.path.join(CONNECT_PATH, f'data/{param.jobname}/number_{iter_num-1}')
    combine_sets_of_data_files(os.path.join(path_i, 'model_params.txt'),
                               os.path.join(path_j, 'model_params.txt'))
    if len(param.output_derived) > 0:
        combine_sets_of_data_files(os.path.join(path_i, 'derived.txt'),
                                   os.path.join(path_j, 'derived.txt'))
    for output in param.output_Cl:
        combine_sets_of_data_files(os.path.join(path_i, f'Cl_{output}.txt'),
                                   os.path.join(path_j, f'Cl_{output}.txt'))
    for output in param.output_Pk:
        combine_sets_of_data_files(os.path.join(path_i, f'Pk_{output}.txt'),
                                   os.path.join(path_j, f'Pk_{output}.txt'))
    for output in param.output_bg:
        combine_sets_of_data_files(os.path.join(path_i, f'Cl_{output}.txt'),
                                   os.path.join(path_j, f'Cl_{output}.txt'))
    for output in param.output_th:
        combine_sets_of_data_files(os.path.join(path_i, f'{output}.txt'),
                                   os.path.join(path_j, f'{output}.txt'))
"""

def get_param_attributes(parameter_file,    # Parameter file from CONNECT
                         names              # Multiline string of attribute names
                     ):

    from source.default_module import Parameters
    param = Parameters(parameter_file)

    out=''
    for name in names.splitlines():
        attr = eval(f'param.{name}')
        out += f'"{attr}"'
        out += ' '
    out = out[:-1]
    return out
"""

def get_node_with_most_cpus():

    job_info_list = sp.run('scontrol show job -d $SLURM_JOB_ID', shell=True, stdout=sp.PIPE).stdout.decode('utf-8')

    node_info_list=[]
    for line in iter(job_info_list.splitlines()):
        if 'CPU_IDs' in line:
            node_info_list.append(line)

    num_cpu={}
    for i, line in enumerate(node_info_list):
        node_name = line.split('Nodes=')[-1].split(' CPU_IDs')[0]
        cpu_list  = line.split('CPU_IDs=')[-1].split(' Mem')[0].split(',')
        n=0
        for c in cpu_list:
            if '-' in c:
                cc=c.split('-')
                n+=int(cc[1])-int(cc[0])+1
            else:
                n+=1
        num_cpu[node_name] = n

    max_node = max(num_cpu, key=num_cpu.get)
    while '[' in max_node:
        _ = num_cpu.pop(max_node)
        max_node = max(num_cpu, key=num_cpu.get)

    return max_node
