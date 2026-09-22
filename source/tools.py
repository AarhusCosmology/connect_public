import os
import subprocess as sp

import numpy as np

from source import cosmic_solvers

CONNECT_PATH  = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]


def create_output_folders(param,            # Parameter object
                          iter_num=None,    # Current iteration number
                          reset=True,       # Resets output folders
                          resume=False      # Resume from iteration
                      ):
    cosmic_solver = getattr(cosmic_solvers, cosmic_solvers.name_dict[param.cosmic_solver])(param)
    output_structure = cosmic_solver.output_structures()
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
                for output in output_structure['1D']['list']:
                    output_name = output.replace('output_','').replace('_output','')
                    for array in getattr(param, output):
                        arr = array.replace('/','\\')
                        os.mkdir(os.path.join(path, f'N-{param.N}/{output_name}_{arr}_data'))
                for output in output_structure['2D']['list']:
                    output_name = output.replace('output_','').replace('_output','')
                    for array in getattr(param, output):
                        if type(array) == tuple:
                            array = '-'.join(map(str,array))
                        arr = array.replace('/','\\')
                        os.mkdir(os.path.join(path, f'N-{param.N}/{output_name}_{arr}_data'))
                for output in output_structure['Scalar']['list']:
                    output_name = output.replace('output_','').replace('_output','')
                    if len(getattr(param, output)) > 0:
                        os.mkdir(os.path.join(path, f'N-{param.N}/{output_name}_data'))


            elif reset:
                for name in os.listdir(path):
                    if not name.startswith('N-'):
                        os.system(f"rm -rf {os.path.join(path, name)}")

        else:
            if reset:
                os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}')}")
                os.mkdir(os.path.join(path, f'number_{iter_num}'))
            os.mkdir(os.path.join(path, f'number_{iter_num}/model_params_data'))
            for output in output_structure['1D']['list']:
                output_name = output.replace('output_','').replace('_output','')
                for array in getattr(param, output):
                    arr = array.replace('/','\\')
                    os.mkdir(os.path.join(path, f'number_{iter_num}/{output_name}_{arr}_data'))
            for output in output_structure['2D']['list']:
                output_name = output.replace('output_','').replace('_output','')
                for array in getattr(param, output):
                    if type(array) == tuple:
                        array = '-'.join(map(str,array))
                    arr = array.replace('/','\\')
                    os.mkdir(os.path.join(path, f'number_{iter_num}/{output_name}_{arr}_data'))
            for output in output_structure['Scalar']['list']:
                output_name = output.replace('output_','').replace('_output','')
                if len(getattr(param, output)) > 0:
                    os.mkdir(os.path.join(path, f'number_{iter_num}/{output_name}_data'))


def join_data_files(param     # Parameter object
                ):

    from source.join_output import CreateSingleDataFile
    CSDF = CreateSingleDataFile(param, CONNECT_PATH)
    CSDF.join()


def combine_sets_of_data_files(new_data,   # Data file for the new data (destination for combined data)
                               old_data,   # Data file for the old data
                               is2D=False,
                               no_header=False
                           ):
    if not is2D:
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
        y_keys = []
        for line in new_lines[1:]+old_lines[1:]:
            if line[0] == "#":
                y_keys.append(line)
                if y_keys[-1] not in lines_separated:
                    lines_separated[y_keys[-1]] = []
            else:
                lines_separated[y_keys[-1]].append(line)
        
        N = int(len(y_keys)/2)
        with open(new_data,'w') as f:
            f.write(new_lines[0])
            for key in y_keys[:N]:
                f.write(key)
                for line in lines_separated[key]:
                    f.write(line)


def combine_iterations_data(param,             # Parameter object
                            iter_num           # Current iteration number
                        ):

    cosmic_solver = getattr(cosmic_solvers, cosmic_solvers.name_dict[param.cosmic_solver])(param)
    output_structure = cosmic_solver.output_structures()
    path_i = os.path.join(CONNECT_PATH, f'data/{param.jobname}/number_{iter_num}')
    if iter_num-1 == 0:
        path_j = os.path.join(CONNECT_PATH, f'data/{param.jobname}/N-{param.N}')
    else:
        path_j = os.path.join(CONNECT_PATH, f'data/{param.jobname}/number_{iter_num-1}')
    combine_sets_of_data_files(os.path.join(path_i, 'model_params.txt'),
                               os.path.join(path_j, 'model_params.txt'))

    for idx, output in enumerate(output_structure['Scalar']['list']):
        output_name = output.replace('output_','').replace('_output','')
        if len(getattr(param, output)) > 0:
            combine_sets_of_data_files(os.path.join(path_i, f'{output_name}.txt'),
                                       os.path.join(path_j, f'{output_name}.txt'))
    for idx, output in enumerate(output_structure['1D']['list']):
        output_name = output.replace('output_','').replace('_output','')
        x_array = output_structure['1D']['x_arr'][idx]
        if x_array is not None:
            no_header = False
        else:
            no_header = True
        for array in getattr(param, output):
            arr = array.replace('/','\\')
            combine_sets_of_data_files(os.path.join(path_i, f'{output_name}_{arr}.txt'),
                                       os.path.join(path_j, f'{output_name}_{arr}.txt'), no_header=no_header)
    for idx, output in enumerate(output_structure['2D']['list']):
        output_name = output.replace('output_','').replace('_output','')
        for array in getattr(param, output):
            if type(array) == tuple:
                array = '-'.join(map(str,array))
            arr = array.replace('/','\\')
            combine_sets_of_data_files(os.path.join(path_i, f'{output_name}_{arr}.txt'),
                                       os.path.join(path_j, f'{output_name}_{arr}.txt'), is2D=True)




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
    if '[' in max_node:
        root = max_node.split('[')[0]
        max_node = root + max_node.replace(root, '').replace('[', '').replace(']', '').split(',')[0]

    return max_node, num_cpu[max_node]


def get_covmat(param):
    path = param.hypersphere_covmat

    params = param.parameters.keys()
    cov_new = np.zeros((len(params), len(params)))
    
    if path is not None:
        cov = np.loadtxt(path)
        with open(path, 'r') as f:
            header = f.readline().strip().replace('#','').replace(' ','').split(',')

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
            cov_new[i,:] = np.zeros(cov_new.shape[1])
            cov_new[:,i] = np.zeros(cov_new.shape[0])
            cov_new[i,i] = param.sigma_guesses[p]**2
        elif path is None:
            cov_new[i,i] = ((param.parameters[p][1] - param.parameters[p][0])/10)**2

    return cov_new
