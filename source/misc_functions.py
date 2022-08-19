import numpy as np
import os
import subprocess as sp

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


def create_lhc(parameter_file       # Parameter file from CONNECT
           ):
    
    from source.default_module import Parameters
    from source.lhc import LatinHypercubeSampler
    param = Parameters(parameter_file)
    
    if not os.path.isfile(CONNECT_PATH+f'/data/lhc_samples/sample_models_{param.jobname}_{param.N}.txt'):
        lhc = LatinHypercubeSampler(param)
        lhc.run(CONNECT_PATH)


def create_output_folders(param,#parameter_file,    # Parameter file from CONNECT
                          iter_num=None,    # Current iteration number
                          reset=True        # resets output folders
                      ):
    
    #from source.default_module import Parameters
    #param = Parameters(parameter_file)

    if not os.path.isdir(CONNECT_PATH+f'/data/{param.jobname}'):
        os.mkdir(CONNECT_PATH+f'/data/{param.jobname}')
    if iter_num == None:
        if reset:
            os.system(f"rm -rf {CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}'}")
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}')
        os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/model_params_data')
        for output in param.output_Cl:
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/Cl_{output}_data')
        for output in param.output_Pk:
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/Pk_{output}_data')
        for output in param.output_bg:
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/{output}_data')
        for output in param.output_th:
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/{output}_data')
        if len(param.output_derived) > 0:
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/derived_data')

    else:
        if reset:
            os.system(f"rm -rf {CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}'}")
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}')
        os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/model_params_data')
        for output in param.output_Cl:
            os.system(f"rm -rf {CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/Cl_{output}_data'}")
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/Cl_{output}_data')
        for output in param.output_Pk:
            os.system(f"rm -rf {CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/Pk_{output}_data'}")
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/Pk_{output}_data')
        for output in param.output_bg:
            os.system(f"rm -rf {CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/{output}_data'}")
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/{output}_data')
        for output in param.output_th:
            os.system(f"rm -rf {CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/{output}_data'}")
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/{output}_data')
        if len(param.output_derived) > 0:
            os.system(f"rm -rf {CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/derived_data'}")
            os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/derived_data')


def create_montepython_param(parameter_file,    # Parameter file from CONNECT
                             montepython_path   # Path to MontePython folder
                         ):
    
    from source.default_module import Parameters
    param = Parameters(parameter_file)
    
    if not os.path.isdir(montepython_path+'/montepython_public/input/connect'):
        os.mkdir(montepython_path+'/montepython_public/input/connect')

    with open(CONNECT_PATH+'/mp_plugin/param_templates/connect_lite.param.template','r') as f:
        with open(montepython_path+f'/montepython_public/input/connect/{param.jobname}.param','w') as g:
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
    return f'montepython_public/input/connect/{param.jobname}.param'


def Gelman_Rubin_log(param,#parameter_file,  # Parameter file from CONNECT
                     status,          # 'initial' call or current iteration number 
                     output=None,     # Output from montepython info <chains> --minimal
                 ):
    #from source.default_module import Parameters
    #param = Parameters(parameter_file)

    if status == 'initial':
        with open(CONNECT_PATH+f'/data/{param.jobname}/Gelman-Rubin.txt', 'w') as f:
            f.write('# R-1 values for each parameter in every iteration\n')
            header = '# iterations'
            for par in param.parameters:
                header += '\t'+par
            header += '\n'
            f.write(header)
    else:
        try:
            status=int(status)
        except:
            raise TypeError("status must be either 'initial' or an integer.")
        output = list(iter(output.splitlines()))
        kill_iteration = False
        Rm1_line = f'{status-1}-{status}'
        if any('Removed everything: chain not converged' in s for s in output):
            os.system(f"echo 'chains from iterations {status} and {status-1} did not have sufficient overlap'")
            Rm1_line += '\t' + f'chains from iterations {status} and {status-1} did not have sufficient overlap\n'
            with open(CONNECT_PATH+f'/data/{param.jobname}/Gelman-Rubin.txt', 'a') as f:
                f.write(Rm1_line)
        else:
            import re
            i=status
            index = [idx for idx, s in enumerate(output) if '-> R-1 is' in s][0]
            kill_iteration = True
            for par in param.parameters:
                for out in output[index:]:
                    if par == out[-len(par):] and out[-len(par)-1] == ' ':
                        Rm1 = np.float32(re.findall("\d+\.\d+", out)[0])
                        Rm1_line += '\t' + str(Rm1)
                        if Rm1 > param.iter_tol:
                            kill_iteration = False
                        break
            Rm1_line += '\n'
            with open(CONNECT_PATH+f'/data/{param.jobname}/Gelman-Rubin.txt', 'a') as f:
                f.write(Rm1_line)
            print(f"Iteration {status} and {status-1} has the following R-1 values")
            for line in output[index:]:
                if any(par in line for par in param.parameters):
                    print(line)
        return kill_iteration
            

def filter_chains(param,#parameter_file,   # Parameter file from CONNECT
                  path,             # Path to folder containing chains
                  N,                 # Maximal number of steps to keep
                  iter_num          # Current iteration number
              ):
    
    #from source.default_module import Parameters
    #param = Parameters(parameter_file)

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

    N_per_file = int(np.ceil(N/len(list_of_files)))

    if sum(lines_of_use_in_files.values()) <= N:
        lines_to_be_used = lines_of_use_in_files
    else:
        if any([lines_of_use_in_files[filename] < N_per_file for filename in list_of_files]):
            files_short = [filename for filename in list_of_files if lines_of_use_in_files[filename] < N_per_file]
            lines_missing = sum([N_per_file - lines_of_use_in_files[filename] for filename in files_short])
            lines_missing -= N_per_file*len(list_of_files) - N

            lines_to_be_used = {}
            for filename in list_of_files:
                if filename in files_short:
                    lines_to_be_used[filename] = lines_of_use_in_files[filename]
                else:
                    lines_to_be_used[filename] = N_per_file

            stop = False
            while not stop:
                for filename in list_of_files:
                    if not filename in files_short:
                        if lines_of_use_in_files[filename] > lines_to_be_used[filename]:
                            lines_to_be_used[filename] += 1
                            lines_missing -= 1
                            if not lines_missing > 0:
                                stop = True
                                break
                if stop:
                    break

        elif N < N_per_file*len(list_of_files):
            lines_excess = N_per_file*len(list_of_files) - N
            lines_to_be_used = {}
            for filename in list_of_files:
                lines_to_be_used[filename] = N_per_file
            stop = False
            while not stop:
                for filename in list_of_files:
                    lines_to_be_used[filename] -= 1
                    lines_excess -= 1
                    if not lines_excess > 0:
                        stop = True
                        break
                if stop:
                    break

        else:
            lines_to_be_used = {}
            for filename in list_of_files:
                lines_to_be_used[filename] = N_per_file

    output_lines = {}
    for filename, N_lines in lines_to_be_used.items():
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

    return sum([n for n in lines_to_be_used.values()])



def discard_oversampled_points(file_old_data,    # model_params.txt file with all the data from previous iterations
                               parameter_file,   # Parameter file from CONNECT
                               iter_num          # Current iteration number
                           ):

    num_points_in_vicinity = 10

    from source.default_module import Parameters
    param = Parameters(parameter_file)

    with open(file_old_data, 'r') as f:
        lines=list(f)
    paramnames = []
    for name in lines[0].replace('#', '').replace(' ', '').replace('\n', '').split('\t'):
        paramnames.append(name)
    points1 = []
    for line in lines[1:]:
        points1.append(np.float32(line.replace('\n', '').split('\t')))


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

    file2point_dict = {}
    os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/full_chains')
    lines=[]
    i = 0
    points2 = []
    for filename in [f for f in os.listdir(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}') if f.endswith('.txt')]:
        with open(CONNECT_PATH+f'/data/{param.jobname}/number_{iter_num}/' + filename, 'r') as f:
            list_f = list(f)
        lines += list_f
        for line in list_f:
            if line[0] != '#':
                file2point_dict[str(i)] = filename
                params = np.float32(line.replace('\n','').split('\t')[1:len(paramnames)+1])
                for name in param.log_priors:
                    k = mp_names.index(name)
                    params[k] = np.power(10.,params[k])
                params *= model_param_scales
                points2.append([params[j] for j in [mp_names.index(n) for n in paramnames]])
                i += 1
        os.system(f"cp {CONNECT_PATH}/data/{param.jobname}/number_{iter_num}/{filename} {CONNECT_PATH}/data/{param.jobname}/number_{iter_num}/full_chains/{filename}")

    points1 = np.array(points1)
    points2 = np.array(points2)
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


    for filename in [f for f in os.listdir(f'data/{param.jobname}/number_{iter_num}') if f.endswith('.txt')]:
        with open(f'data/{param.jobname}/number_{iter_num}/' + filename, 'w') as f:
            for idx in indices_accepted:
                if file2point_dict[str(idx)] == filename:
                    f.write(lines[idx])


    return len(indices_accepted)


def join_data_files(param#parameter_file  # Parameter file from CONNECT
                ):

    #from source.default_module import Parameters
    #param = Parameters(parameter_file)

    from source.join_output import CreateSingleDataFile
    CSDF = CreateSingleDataFile(param, CONNECT_PATH)
    CSDF.join()


def combine_sets_of_data_files(new_data,   # Data file for the new data (the destination for the combined data)
                               old_data    # Data file for the old data
                           ):

    with open(new_data,'a') as f:
        with open(old_data,'r') as g:
            for line in list(g)[1:]:
                f.write(line)


def combine_iterations_data(param,#parameter_file,    # Parameter file from CONNECT  
                            iter_num           # Current iteration number
                        ):

    #from source.default_module import Parameters
    #param = Parameters(parameter_file)

    path_i = CONNECT_PATH + f'/data/{param.jobname}/number_{iter_num}/'
    path_j = CONNECT_PATH + f'/data/{param.jobname}/number_{iter_num-1}/'
    combine_sets_of_data_files(path_i+'model_params.txt', path_j+'model_params.txt')
    if len(param.output_derived) > 0:
        combine_sets_of_data_files(path_i+'derived.txt', path_j+'derived.txt')
    for output in param.output_Cl:
        combine_sets_of_data_files(path_i+f'Cl_{output}.txt', path_j+f'Cl_{output}.txt')
    for output in param.output_Pk:
        combine_sets_of_data_files(path_i+f'Pk_{output}.txt', path_j+f'Pk_{output}.txt')
    for output in param.output_bg:
        combine_sets_of_data_files(path_i+f'Cl_{output}.txt', path_j+f'Cl_{output}.txt')
    for output in param.output_th:
        combine_sets_of_data_files(path_i+f'{output}.txt', path_j+f'{output}.txt')


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
