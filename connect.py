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

keyword       = sys.argv[1]
if keyword in ['create', 'train']:
    param_file    = sys.argv[2]
    param         = Parameters(param_file)
    parameters    = param.parameters

    path = CONNECT_PATH + f'/data/{param.jobname}/'



#####################################
# ____________ create _____________ #
#####################################

if keyword == 'create':
    if not param.resume_iterations:
        os.system(f'rm -f {path}output.log')
    else:
        os.system('echo "'+62*'#'+f'" >> {path}output.log')
        os.system('echo "'+7*'Resuming '+f'" >> {path}output.log')
        os.system('echo "'+62*'#'+f'" >> {path}output.log')

    from source.tools import create_output_folders
    create_output_folders(param, resume=param.resume_iterations) 

    from source.data_sampling import Sampling
    s = Sampling(param_file, CONNECT_PATH)

    if not os.path.isdir(path):
        os.mkdir(path)
    with open(os.path.join(CONNECT_PATH,'source/assets/logo_colour.txt'),'r') as f:
        log_string = '-'*62+'\n\n\n' +                        \
                     f.read()+'\n' +                          \
                     '-'*62+'\n\n' +                          \
                     'Running CONNECT\n' +                    \
                    f'Parameter file     :  {param_file}\n' + \
                     'Mode               :  Create'

    mode = param.resume_iterations*'a+' + (not param.resume_iterations)*'w'

    if param.sampling == 'iterative':
        with open(path+'output.log', mode) as sys.stdout:
            print(log_string, flush=True)
            print('Sampling method    :  Iterative', flush=True)
            print('\n'+'-'*62+'\n', flush=True)
            s.create_iterative_data()
            
    elif param.sampling == 'lhc':
        with open(path+f'N-{param.N}/output.log', mode) as sys.stdout:
            print(log_string, flush=True)
            print('Sampling method    :  Latin Hypercube', flush=True)
            print('\n'+'-'*62+'\n', flush=True)
            s.create_lhc_data()
            from source.tools import join_data_files
            join_data_files(param)

    elif param.sampling == 'hypersphere':
        with open(path+f'N-{param.N}/output.log', mode) as sys.stdout:
            print(log_string, flush=True)
            print('Sampling method    :  Hypersphere', flush=True)
            print('\n'+'-'*62+'\n', flush=True)
            s.create_hypersphere_data()
            from source.tools import join_data_files
            join_data_files(param)

    elif param.sampling == 'pickle':
        with open(path+f'N-{param.N}/output.log', mode) as sys.stdout:
            print(log_string, flush=True)
            print('Sampling method    :  From Pickle file', flush=True)
            print('\n'+'-'*62+'\n', flush=True)
            s.create_pickle_data()
            from source.tools import join_data_files
            join_data_files(param)


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
# ____________ animate ____________ #
#####################################

if keyword == 'animate':
    from source.assets.animate import play
    play()


#####################################
# _________ procrastinate _________ #
#####################################

if keyword == 'procrastinate':
    import base64
    with open('source/assets/surprise.txt','r') as f:
        obfuscated_code = f.readlines()[0]
    exec(base64.b85decode(obfuscated_code.encode('utf-8')))
