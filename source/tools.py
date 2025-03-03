import os
import subprocess as sp

import numpy as np

CONNECT_PATH  = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]

def get_computed_cls(cosmo,       # A computed CLASS model
                     ell_array=[]
                     ):

    cls = cosmo.lensed_cl()

    if len(ell_array) == 0:
        # get parameters from CLASS model
        BA                 = cosmo.get_background()
        conformal_age_     = BA['conf. time [Mpc]'][-1]
        der_pars           = cosmo.get_current_derived_parameters(['ra_rec','tau_rec'])
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
        ell = np.array(l)
    else:
        ell = np.array(ell_array)

    # create reduced dict of cls
    new_cls = {'ell':ell}
    for key, arr in cls.items():
        if key != 'ell':
            new_cls[key] = arr[ell]

    return new_cls


def get_z_idx(z):
    z=z.copy()
    if z[-1] > z[0]:
        grid = np.logspace(np.log10(z[1]), np.log10(z[-1]), 99)
    else:
        grid = np.logspace(np.log10(z[0]), np.log10(z[-2]), 99)
    z_idx = []
    for zg in grid:
        i = np.abs(z-zg).argmin()
        z[i] = np.inf
        z_idx.append(i)
    j = len(z) - 1
    while len(z_idx) < 100:
        if j in z_idx:
            j -= 1
        else:
            z_idx.append(j)
    return np.array(sorted(z_idx))



def create_output_folders(param,            # Parameter object
                          iter_num=None,    # Current iteration number
                          reset=True,       # Resets output folders
                          resume=False      # Resume from iteration
                      ):
    if not resume:
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
                    output = output.replace('/','\\')
                    os.mkdir(os.path.join(path, f'N-{param.N}/bg_{output}_data'))
                for output in param.output_th:
                    os.mkdir(os.path.join(path, f'N-{param.N}/th_{output}_data'))
                if len(param.output_derived) > 0:
                    os.mkdir(os.path.join(path, f'N-{param.N}/derived_data'))
                for output in param.extra_output:
                    os.mkdir(os.path.join(path, f'N-{param.N}/extra_{output}_data'))

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
                output = output.replace('/','\\')
                os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/bg_{output}_data')}")
                os.mkdir(os.path.join(path, f'number_{iter_num}/bg_{output}_data'))
            for output in param.output_th:
                os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/th_{output}_data')}")
                os.mkdir(os.path.join(path, f'number_{iter_num}/th_{output}_data'))
            if len(param.output_derived) > 0:
                os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/derived_data')}")
                os.mkdir(os.path.join(path, f'number_{iter_num}/derived_data'))
            for output in param.extra_output:
                os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}/extra_{output}_data')}")
                os.mkdir(os.path.join(path, f'number_{iter_num}/extra_{output}_data'))


def join_data_files(param     # Parameter object
                ):

    from source.join_output import CreateSingleDataFile
    CSDF = CreateSingleDataFile(param, CONNECT_PATH)
    CSDF.join()


def combine_sets_of_data_files(new_data,   # Data file for the new data (destination for combined data)
                               old_data,   # Data file for the old data
                               Pk=False,
                               no_header=False
                           ):
    if not Pk:
        with open(new_data,'a') as f:
            with open(old_data,'r') as g:
                if no_header:
                    start_idx = 0
                else:
                    start_idx = 1
                for line in list(g)[start_idx:]:
                    f.write(line)
    else:
        with open(new_data,'r') as f:
            new_lines = list(f)
        with open(old_data,'r') as f:
            old_lines = list(f)

        lines_separated = {}
        z_keys = []
        for line in new_lines[1:]+old_lines[1:]:
            if line[0] == "#":
                z_keys.append(line)
                if z_keys[-1] not in lines_separated:
                    lines_separated[z_keys[-1]] = []
            else:
                lines_separated[z_keys[-1]].append(line)
        
        N = int(len(z_keys)/2)
        with open(new_data,'w') as f:
            f.write(new_lines[0])
            for key in z_keys[:N]:
                f.write(key)
                for line in lines_separated[key]:
                    f.write(line)


def combine_iterations_data(param,             # Parameter object
                            iter_num           # Current iteration number
                        ):

    path_i = os.path.join(CONNECT_PATH, f'data/{param.jobname}/number_{iter_num}')
    if iter_num-1 == 0:
        path_j = os.path.join(CONNECT_PATH, f'data/{param.jobname}/N-{param.N}')
    else:
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
                                   os.path.join(path_j, f'Pk_{output}.txt'), Pk=True)
    for output in param.output_bg:
        output = output.replace('/','\\')
        combine_sets_of_data_files(os.path.join(path_i, f'bg_{output}.txt'),
                                   os.path.join(path_j, f'bg_{output}.txt'))
    for output in param.output_th:
        combine_sets_of_data_files(os.path.join(path_i, f'th_{output}.txt'),
                                   os.path.join(path_j, f'th_{output}.txt'))
    for output in param.extra_output:
        combine_sets_of_data_files(os.path.join(path_i, f'extra_{output}.txt'),
                                   os.path.join(path_j, f'extra_{output}.txt'), no_header=True)



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


def get_covmat(path, param):
    cov = np.loadtxt(path)
    with open(path, 'r') as f:
        header = f.readline().strip().replace('#','').replace(' ','').split(',')

    params = param.parameters.keys()
    cov_new = np.zeros((len(params), len(params)))
    indices = []
    for i, p in enumerate(params):
        try:
            indices.append(header.index(p))
        except ValueError:
            indices.append(None)
            cov_new[i,i] = ((param.parameters[p][1] - param.parameters[p][0])/10)**2

    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            if idx_i is not None and idx_j is not None:
                cov_new[i,j] = cov[idx_i,idx_j]

    for i, p in enumerate(params):
        if p in param.sigma_guesses:
            cov_new[i,i] = param.sigma_guesses[p]**2

    return cov_new
