# First command line argument is a keyword specifying what action you would like to perform.
#1;95;0c The implemented keywords are 'create', 'train', 'train-job', 'sample', and 'plot', which 
# corresponds to the actions creating data, training a network, training as a job, sampling
# of the posterior distribution, and plotting of the trained models and the inferred parameters.

# The code needs to have a parameter file specified as the second command line argument,
# which contains the parameters and hyperparameters of the model and the architechture of
# the network. 

# When choosing the plotting option, no parameter file should be given. Instead the following
# arguments should be paths to trained models that you wish to compare throgh plots.
# In the near future, a plot file containing information of which plots to create along with
# customizable parameters will be added as an optional argument. 


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

import sys
import os
import argparse

CONNECT_PATH  = os.path.realpath(os.path.dirname(__file__))
CURRENT_PATH  = os.path.abspath(os.getcwd())
sys.path.insert(1, CONNECT_PATH)

from source.default_module import Parameters

"""
parser = argparse.ArgumentParser(description="CONNECT - create training data and train neural networks",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("keyword", help="What connect should do, 'create' or 'train'")
subparsers = parser.add_subparsers(help='Types of keywords')

create_parser = subparsers.add_parser("create")
train_parser  = subparsers.add_parser("train")

create_parser.add_argument("create_param_file", help="Parameter file of data creation")
create_parser.add_argument("-N", "--number-of-points", help="Number of points in the latin hypercube or the maximum number of points from each iteration of sampling")
create_parser.add_argument("-j", "--jobname", help="Name of data folder and identifier")
create_parser.add_argument("-n", "--mpi-tasks", help="Number of MPI processes")
create_parser.add_argument("-s", "--sampling", help="Method of sampling", choices=["lhc","iterative"])
if vars(parser.parse_args())['sampling'] == 'iterative':
    create_parser.add_argument("-r", "--resume-iterations", action='store_true', help="Resume iterations of sampling")
    create_parser.add_argument("-i", "--initial-model", help="")

train_parser.add_argument("data_directory", help="Directory of data to train. Must contain a file called 'model_params.txt' and the corresponding output files")
train_parser.add_argument("-p", "--param-file", help="Parameter file of training")
train_parser.add_argument("-e", "--epochs", help="")
train_parser.add_argument("-b", "--batchsize", help="")
train_parser.add_argument("-N", "--N-nodes-", help="")
train_parser.add_argument("-h", "--N-hiddes-layers", help="")
train_parser.add_argument("-r", "--train-ratio", help="")
train_parser.add_argument("-v", "--val-ratio", help="")
train_parser.add_argument("-a", "--activation-function", help="")
train_parser.add_argument("-n", "--normalization-method", help="")
train_parser.add_argument("-l", "--loss-function", help="")
train_parser.add_argument("-s", "--save-name", help="")
train_parser.add_argument("-o", "--overwrite-model", action='store_true', help="")

args = parser.parse_args()
config = vars(args)
"""

keyword       = sys.argv[1]
param_file    = sys.argv[2]
param         = Parameters(param_file)
parameters    = param.parameters

path = CONNECT_PATH + f'/data/{param.jobname}/'

#####################################
# ____________ create _____________ #
#####################################
#if keyword == 'testing':
#    from source.data_sampling import Sampling
#    s=Sampling(param_file,CONNECT_PATH)

if keyword == 'create':
    from source.data_sampling import Sampling
    s = Sampling(param, CONNECT_PATH)

    print(s.N_tasks)
    print(s.N_cpus_per_task)
    if param.sampling == 'iterative':
        s.create_iterative_data()
        #passphrase = input('Enter passphrase to nodes: ')
        #job_out = f"{CONNECT_PATH}/iterative_sampling_{param.jobname}.out"
        #os.system(f"sbatch --ntasks={param.N_cpu+1} --output={job_out} source/iterative_sampling_method.js {CONNECT_PATH} {param_file} cpu")
    elif param.sampling == 'lhc':
        s.create_lhc_data()
        #from source.lhc import LatinHypercubeSampler
        #if not os.path.isfile(CONNECT_PATH+f'/data/lhc_samples/sample_models_{param.jobname}_{param.N}.txt'):
        #    lhc = LatinHypercubeSampler(param)
        #    lhc.run(CONNECT_PATH)

        # Check whether or not directories exist, and make them if they don't
        #if not os.path.isdir(CONNECT_PATH+f'/data/{param.jobname}'):
        #    os.mkdir(CONNECT_PATH+f'/data/{param.jobname}')
        #os.system(f"rm -rf {CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}'}")
        #os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}')
        #os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/model_params_data')
        #for output in param.output_Cl:
        #    os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/Cl_{output}_data')
        #for output in param.output_Pk:
        #    os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/Pk_{output}_data')
        #for output in param.output_bg:
        #    os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/{output}_data')
        #for output in param.output_th:
        #    os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/{output}_data')
        #if len(param.output_derived) > 0:
        #    os.mkdir(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/derived_data')

        # Copy parameter file to data directory
        #os.system(f"cp {CURRENT_PATH + f'/{param_file}'} {CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}'}/log_connect.param")
        #with open(CONNECT_PATH+f'/data/{param.jobname}/N-{param.N}/log_connect.param', 'a+') as f:
        #    jobname_specified = False
        #    for line in f:
        #        if line.startswith('jobname'):
        #            jobname_specified = True
        #    if not jobname_specified:
        #        f.write(f"\njobname = '{param.jobname}'")
        #
        #Nfolder = f'N-{param.N}'

        '''jobscript_string = f"""#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --partition=q24,q28,q36,q40,q48
#SBATCH --mem={param.mem_create}
#SBATCH --ntasks={param.N_cpu + 1}
#SBATCH --cpus-per-task=1
#SBATCH --time={param.walltime_create}
echo "========= Job started at `date` =========="
ml load anaconda3/2021.05
source activate mpienv
srun -n {param.N_cpu + 1} python {CONNECT_PATH}/source/calc_models_mpi.py {f'data/{param.jobname}/{Nfolder}'}/log_connect.param {CONNECT_PATH} lhc
echo "========= Job finished at `date` =========="
"""

        jobscript_file = CONNECT_PATH + f'/jobscripts/create_data_{param.jobname}.js'

        with open(jobscript_file,'w') as f:
            f.write(jobscript_string)

        # Send job to slurm queue
        os.system(f'sbatch {jobscript_file}')
        '''


#####################################
# _____________ train _____________ #
#####################################

def join_output_files():
    try:
        i = max([int(f.split('number_')[-1]) for f in os.listdir(path) if f.startswith('number')])
        if param.sampling == 'iterative' and not os.path.isfile(CONNECT_PATH + f'/data/{param.jobname}/number_{i}/model_params.txt'):
            from source.join_output import CreateSingleDataFile
            CSDF = CreateSingleDataFile(param, CONNECT_PATH)
            CSDF.join()
    except:
        if not os.path.isfile(CONNECT_PATH + f'/data/{param.jobname}/N-{param.N}/model_params.txt'):
            from source.join_output import CreateSingleDataFile
            CSDF = CreateSingleDataFile(param, CONNECT_PATH)
            CSDF.join()

if keyword == 'train':
    join_output_files()
    from source.train_network import Training
    tr = Training(param, CONNECT_PATH)
    try:
        tr.train_model(epochs=sys.argv[3])
    except:
        tr.train_model()
    tr.save_model()
    tr.save_history()
    tr.save_test_data()


#####################################
# ___________ train-job ___________ #
#####################################
if keyword == 'train-job':
    if param.sampling == 'iterative':
        i = max([int(f.split('number_')[-1]) for f in os.listdir(path) if f.startswith('number')])
        load_path = CONNECT_PATH + f'/data/{param.jobname}/number_{i}/'
        if os.path.isfile(load_path + 'model_params.txt'):
            param.N = sum(1 for line in open(load_path + 'model_params.txt')) - 1
        else:
            join_output_files()
            param.N = sum(1 for line in open(load_path + 'model_params.txt')) - 1
            
    # Create jobscript
    if not param.save_name == None:
        out_name = param.save_name
    else:
        out_name = f'{param.jobname}_N{param.N}_bs{param.batchsize}_e{param.epochs}'

    try:
        extra=f' {sys.argv[3]}'
    except:
        extra=''
    
    jobscript_string = f"""#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=qgpuo,qgpu
#SBATCH --exclude=s27n[01-02]
#SBATCH --mem={param.mem_train}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time={param.walltime_train}
#SBATCH --output={CONNECT_PATH}/trained_models/slurm_out/{out_name}.out
#SBATCH --error=error.err#/dev/null
echo "========= Job started at `date` =========="
ml load anaconda3/2021.05
ml load cuda/11.2.0
python {CONNECT_PATH}/connect.py train {CURRENT_PATH + f'/{param_file}'}{extra}
echo "========= Job finished at `date` =========="
"""
    jobscript_file = CONNECT_PATH + f'/jobscripts/train_model_{param.jobname}.js'
    with open(jobscript_file,'w') as f:
        f.write(jobscript_string)

    # Send job to slurm queue
    os.system(f'sbatch {jobscript_file}')





"""
    
#####################################
# ____________ sample _____________ #
#####################################
if keyword == 'sample':
    print('This feature will be implemented soon')


#####################################
# _____________ plot ______________ #
#####################################
if keyword == 'plot':
    from scipy.interpolate import CubicSpline
    from source.plot_tools import get_error
    import tensorflow as tf
    import pickle as pkl
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    SMALL_SIZE = 15
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    percentiles = [0.682,0.954]
    model_names = ['MSE', 'CV']
    alpha_list  = [0.9,    0.6]
    c_list      = ['crimson', 'navy']
    fc_array    = [['orange', 'red' ],
                   ['cyan',   'blue']]

    l = np.linspace(2,2500,2499)
    l_red = []
    for i,ll in enumerate(l):
        if i%10 == 0:
            l_red.append(ll)
    l_red.append(l[-1])
    l_red = np.array(l_red)

    model_paths = []
    for arg in sys.argv[2:]:
        model_paths.append(arg)

    y_max = 0
    plt.figure(figsize=(10,7))
    for i, path in enumerate(model_paths):
        model_name = model_names[i]
        errors =  get_error(path)

        err_lower_array = []
        err_upper_array = []
        err_mean = []
        for errs in errors:
            mean_est = np.mean(errs)
            err_mean.append(mean_est)

            err_lower_list = []
            err_upper_list = []
            for p in percentiles:

                """"""
                from source.plot_tools import percentiles_of_histogram as poh
                err_lim  = poh(errs,percentiles)
                err_min  = err_lim[0][0]
                err_max  = err_lim[0][1]
                err_min2 = err_lim[1][0]
                err_max2 = err_lim[1][1]
                err_upper.append(err_max)
                err_lower.append(err_min)
                err_upper2.append(err_max2)
                err_lower2.append(err_min2)
                """"""
                    
                err_lower_list.append(np.percentile(errs, 100*(1-p)/2))
                err_upper_list.append(np.percentile(errs, 100 - 100*(1-p)/2))

            err_lower_array.append(err_lower_list)
            err_upper_array.append(err_upper_list)

        if max(np.array(err_upper_array).flatten()) > y_max:
            y_max = max(np.array(err_upper_array).flatten())

        err_m = CubicSpline(l_red,err_mean)
        for j, p in reversed(list(enumerate(sorted(percentiles)))):
            err_u = CubicSpline(l_red,np.array(err_upper_array).T[j])
            err_l = CubicSpline(l_red,np.array(err_lower_array).T[j])

            plt.fill_between(l, err_l(l), err_u(l),
                             fc=fc_array[i][j], alpha=alpha_list[i],
                             label=f'{int(p*100)}% ('+model_name+')',
                             zorder=i)
       
        plt.plot(l, err_m(l),
                 '-',c=c_list[i], lw=1.5,
                 label='mean ('+model_name+')',
                 zorder=i)

    plt.legend(loc=2)
    plt.gcf().subplots_adjust(left=0.16,right=0.93)
    #plt.xscale('log')
    plt.ylim([0,y_max*1.1])
    plt.xlim([2,2500])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\frac{\left\vertC_{\ell,\mathrm{CONNECT}}^{\mathrm{TT}}-C_{\ell,\mathrm{CLASS}}^{\mathrm{TT}}\right\vert}{C_{\ell,\mathrm{CLASS}}^{\mathrm{TT}}}$')
    plt.title('Error of emulated model')
    if not os.path.isdir(model_paths[0]+'/plots'):
        os.mkdir(model_paths[0]+'/plots')
    plt.savefig(model_paths[0]+'/plots/error.pdf')

#    plt.figure(figsize=(10,7))
    
"""
