import os
import subprocess as sp

import numpy as np

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
