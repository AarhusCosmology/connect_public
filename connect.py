# First command line argument is a keyword specifying what action you would like to perform.
# The implemented keywords are 'create' and 'train', which corresponds to the actions creating
# data, training a network, training as a job, sampling of the posterior distribution, and 
# plotting of the trained models and the inferred parameters.

# The code needs to have a parameter file specified as the second command line argument,
# which contains the parameters and hyperparameters of the model and the architechture of
# the network. 


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

if keyword == 'create':
    os.system(f'rm -f {path}output.log')

    from source.misc_functions import create_output_folders
    create_output_folders(param) 

    from source.data_sampling import Sampling
    s = Sampling(param_file, CONNECT_PATH)

    if not os.path.isdir(path):
        os.mkdir(path)
    with open(os.path.join(CONNECT_PATH,'source/logo.txt'),'r') as f:
        log_string = '-'*62+'\n\n\n' +                        \
                     f.read()+'\n' +                          \
                     '-'*62+'\n\n' +                          \
                     'Running CONNECT\n' +                    \
                    f'Parameter file     :  {param_file}\n' + \
                     'Mode               :  Create'

    if param.sampling == 'iterative':
        with open(path+'output.log', 'w') as sys.stdout:
            print(log_string, flush=True)
            print('Sampling method    :  Iterative', flush=True)
            print('\n'+'-'*62+'\n', flush=True)
            s.create_iterative_data()
            
    elif param.sampling == 'lhc':
        with open(path+f'N-{param.N}/output.log', 'w') as sys.stdout:
            print(log_string, flush=True)
            print('Sampling method    :  Latin Hypercube', flush=True)
            print('\n'+'-'*62+'\n', flush=True)
            s.create_lhc_data()


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
