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



parser = argparse.ArgumentParser(description="CONNECT - create training data and train neural networks",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("keyword", help="What connect should do, 'create', 'train', 'profile', or 'animate'")
subparsers = parser.add_subparsers(help='Types of keywords', dest='keyword')

create_parser = subparsers.add_parser("create")
train_parser  = subparsers.add_parser("train")
profile_parser  = subparsers.add_parser("profile")
animate_parser  = subparsers.add_parser("animate")

create_parser.add_argument("param_file", help="Parameter file of data creation")
create_parser.add_argument("-N", "--number-of-points", help="Number of points in the latin hypercube or the maximum number of points from each iteration of sampling")
create_parser.add_argument("-j", "--jobname", help="Name of data folder and identifier")
create_parser.add_argument("-n", "--mpi-tasks", help="Number of MPI processes")
create_parser.add_argument("-s", "--sampling", help="Method of sampling", choices=["lhc","iterative"])
create_parser.add_argument("-r", "--resume-iterations", action='store_true', help="Resume iterations of sampling")
create_parser.add_argument("-i", "--initial-model", help="Initial model for iterative sampling")

#train_parser.add_argument("data_directory", help="Directory of data to train. Must contain a file called 'model_params.txt' and the corresponding output files")
#train_parser.add_argument("-p", "--param-file", help="Parameter file of training")
train_parser.add_argument("param_file", help="Parameter file of training")
train_parser.add_argument("-e", "--epochs", help="Number of epochs")
train_parser.add_argument("-b", "--batchsize", help="Batch size")
train_parser.add_argument("-N", "--N-nodes-", help="Number of nodes in hidden layers")
train_parser.add_argument("-H", "--N-hiddes-layers", help="Number of hidden layers")
train_parser.add_argument("-r", "--train-ratio", help="Ratio between training data and sum of training and test data")
train_parser.add_argument("-v", "--val-ratio", help="Ratio of training data to use for validation")
train_parser.add_argument("-a", "--activation-function", help="Activation function")
train_parser.add_argument("-n", "--normalisation-method", help="Normalisation method")
train_parser.add_argument("-l", "--loss-function", help="Loss function")
train_parser.add_argument("-s", "--save-name", help="Name to save trained model as")
train_parser.add_argument("-o", "--overwrite-model", action='store_true', help="Wheter or not to overwrite previous models with same name or increment a suffix")


profile_parser.add_argument("-m", "--model", type=str, help="Neural network model trained by CONNECT")
profile_parser.add_argument("-c", "--chain-folder", type=str, help="Folder with chains from an MCMC run with Monte Python")
profile_parser.add_argument("-o", "--output", type=str, help="Output folder for profiles")
profile_parser.add_argument("-a", "--add-points", type=str, help="Add points from file")
profile_parser.add_argument("-r", "--recalculate", type=str, help="Recalculate points specified in file")
profile_parser.add_argument("-f", "--fix-points", action="store_true", help="Fix failed points")
profile_parser.add_argument("-b", "--batch-size", default=1, type=int, help="Batch size of optimisation")
profile_parser.add_argument("-n", "--n-steps", default=10, type=int, help="Number of steps in each optimisation iteration")
profile_parser.add_argument("-t", "--convergence", default=0.05, type=float, help="Tolerance in chi squared for convergence")
profile_parser.add_argument("-d", "--only-1d", action="store_true", help="Compute only the 1d profiles")
profile_parser.add_argument("-g", "--only-global", action="store_true", help="Compute only the global best-fit")

args = parser.parse_args()
config = vars(args)
keyword = config['keyword']

if keyword in ['create', 'train']:
    param_file    = config['param_file']
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
    with open(os.path.join(CONNECT_PATH,'source/logo_colour.txt'),'r') as f:
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
        tr.train_model(epochs=config['epochs'])
    except:
        tr.train_model()
    tr.save_model()
    tr.save_history()
    tr.save_test_data()


#####################################
# ____________ profile ____________ #
#####################################

if keyword == 'profile':
    
    if config['output'] == None:
        raise RuntimeError("No output folder is specified. Specify one with the '-o' flag.")
    if config['model'] == None:
        raise RuntimeError("No neural network model specified. Specify one with the '-m' flag")
    if config['add_points'] == None and config['fix_points'] == None and config['recalculate'] == None and config['chain_folder'] == None and not config['only_global']:
        raise RuntimeError("No source for data points found. Please specifiy a chain folder from Monte Python histograms with '-c', a file with additional points with '-a', a file with points to recalculate with '-r', or use the '-f' flag to fix failed points from a previous run.")
    
    ############################
    
    import itertools
    import pickle as pkl
    import traceback
    from importlib.machinery import SourceFileLoader
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    import tensorflow as tf
    import numpy as np
    import tensorflow_probability as tfp
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    
    from source.profile_likelihoods.tf_planck_lite import tf_planck2018_lite as planck
    from source.profile_likelihoods.get_bins import Cut_histogram
    import source.profile_likelihoods.get_data_for_profiles as get_data

    
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    N_workers = comm.Get_size()
    
    
    output_file_base = config['output']
    chain_folder = config['chain_folder']
    
    
    calc_1d = True
    calc_2d = True
    calc_global = True
    if config['only_1d']:
        calc_global = False
        calc_2d = False
    if config['only_global']:
        calc_1d = False
        calc_2d = False
    if config['fix_points'] or config['recalculate'] != None or config['add_points'] != None:
        calc_global = False


    N_steps = config['n_steps']
    batch_size = config['batch_size']
    convergence_tolerance = config['convergence']
    connect_model = config['model']
    
    param_file = os.path.join(connect_model,'log.param')
    param = SourceFileLoader(param_file, param_file).load_module()
    parameters_and_priors = param.parameters
    for key, val in parameters_and_priors.items():
        parameters_and_priors[key].append('uniform')
    parameters_and_priors['A_planck'] = [ 1.0, 0.0025, 'gaussian' ]
    
    
    
    fixed_params = {}#{'parameter_name'   : value}
    fixed_indices = {}
    for i, par in enumerate(fixed_params.keys()):
        fixed_indices[par] = i
    
    param_names = list(parameters_and_priors.keys())
    param_indices = list(range(len(param_names)))
    
    for key in fixed_params:
        param_indices.pop([i for i, n in enumerate(param_names) if n == key][0])
        param_names.pop([i for i, n in enumerate(param_names) if n == key][0])
    
    
    if config['add_points'] == None and config['recalculate'] == None and not config['fix_points']:
        if calc_1d:
            vals_array_1d, idxs_array_1d, k_array_1d = get_data.get_data_from_mcmc_bins_1d(param_names, param_indices, batch_size, chain_folder)
        if calc_2d:
            vals_array_2d, idxs_array_2d, k_array_2d = get_data.get_data_from_mcmc_bins_2d(param_names, param_indices, batch_size, chain_folder)
    
    elif config['add_points'] != None:
        if calc_1d:
            vals_array_1d, idxs_array_1d, k_array_1d = get_data.get_additional_points_1d(param_names, param_indices, batch_size, config['add_points'])
        if calc_2d:
            vals_array_2d, idxs_array_2d, k_array_2d = get_data.get_additional_points_2d(param_names, param_indices, batch_size, config['add_points'])
    
    elif config['recalculate'] != None or config['fix_points']:
        if calc_1d:
            vals_array_1d, idxs_array_1d, k_array_1d = get_data.get_wrong_points_1d(param_names, param_indices, batch_size, output_file_base, config['recalculate'], fix_failed_points=config['fix_points'])
        if calc_2d:
            vals_array_2d, idxs_array_2d, k_array_2d = get_data.get_wrong_points_2d(param_names, param_indices, batch_size, output_file_base, config['recalculate'], fix_failed_points=config['fix_points'])
    
    
    
    
    vals_fix = []
    idxs_fix = []
    for key, val in fixed_params.items():
        vals_fix.append(val)
        idxs_fix.append(fixed_indices[key])
    
    if calc_global:
        vals_array_global = tf.constant([[[]]], dtype=tf.float32)
        idxs_array_global = tf.constant([[]], dtype=tf.int32)
        k_array_global = tf.constant([0])
    
        vals_array_global = tf.concat([vals_array_global, tf.repeat(tf.constant([[vals_fix]], dtype=tf.float32), vals_array_global.shape[0], axis=0)], 2)
        idxs_array_global = tf.concat([idxs_array_global, tf.repeat(tf.constant([idxs_fix], dtype=tf.int32), vals_array_global.shape[0], axis=0)], 1)
    
    if calc_1d:
        vals_array_1d = tf.concat([vals_array_1d, tf.repeat(tf.constant([[vals_fix]], dtype=tf.float32), vals_array_1d.shape[0], axis=0)], 2)
        idxs_array_1d = tf.concat([idxs_array_1d, tf.repeat(tf.constant([idxs_fix], dtype=tf.int32), vals_array_1d.shape[0], axis=0)], 1)
    
    if calc_2d:
        vals_array_2d = tf.concat([vals_array_2d, tf.repeat(tf.constant([[vals_fix]], dtype=tf.float32), vals_array_2d.shape[0], axis=0)], 2)
        idxs_array_2d = tf.concat([idxs_array_2d, tf.repeat(tf.constant([idxs_fix], dtype=tf.int32), vals_array_2d.shape[0], axis=0)], 1)
    
    
    
    tf_planck = planck(parameters=parameters_and_priors,
                       connect_ttteee_model=connect_model,
                       tf_planck2018_lite_path=os.path.join(CONNECT_PATH, 'source/profile_likelihoods/'))
    
    class Basinhopping():
        def __init__(self, N_steps, batch_size):
            self.N_steps = N_steps
            self.batch_size = batch_size
            self.batch_size = batch_size
            self.idxs = None
            print('initializing...')
            
            with open(f'{connect_model}/test_data.pkl', 'rb') as f:
                data = pkl.load(f)
            params = tf.constant(data[0])
            self.params = tf.pad(params,[[0,0],[0,1]],constant_values=1.0)
    
            data_mins = -tf_planck.get_loglkl(self.params)[:,0].numpy()
            loglike_start = np.min(data_mins)
            self.loc = self.params[np.argmin(data_mins)]
    
    
        def calc_point(self, zip_data):
            vals, idxs, file_number = zip_data
            print(rank,':',idxs, file_number)
            self.val = vals
            if not tf.reduce_all(idxs == self.idxs):
                self.idxs = idxs
                self.recalc_covmat()
            T=1.0
            self.prop_dist = tfp.distributions.MultivariateNormalTriL(loc=self.loc_reduced,scale_tril=self.tril)
            k=0
            while k < 10:
                print('Temperature is T =', T)
                loglike_mins = []
                positions = []
                for i in range(self.N_steps):
                    print(rank,': step',i)
                    X = self.take_step()
                    invH = self.make_inverse_hessian_estimate(X)
                    nan_bool = tf.reduce_any(tf.math.is_nan(invH))
                    nan_bool, X, invH = tf.while_loop(self.nan_cond, self.invH_is_nan_loop, loop_vars=(nan_bool, X, invH))
                    loglike, position = self.optimize_loglike(X, invH)
                    loglike_mins.append(loglike)
                    positions.append(position)
                loglike_mins = tf.stack(loglike_mins)
                positions = tf.stack(positions)
                T *= 0.5
                loglike_min, bestfit, self.prop_dist = self.update_proposal(T, loglike_mins, positions)
                k += 1
                if tf.where(loglike_mins - loglike_min < convergence_tolerance).shape[0] > tf.size(loglike_mins)/2:
                    break
            print(rank,':',file_number,'|',loglike_min)
            return loglike_min, bestfit, vals, file_number
    
    
        
        def nan_cond(self, nan_bool, X, invH):
            return nan_bool
    
        def invH_is_nan_loop(self, nan_bool, X, invH):
            print('invH was NaN - trying again')
            X = self.take_step()
            invH = self.make_inverse_hessian_estimate(X)
            nan_bool = tf.reduce_all(tf.math.is_finite(invH))
            return nan_bool, X, invH
    
            
        def take_step(self):
            y = self.prop_dist.sample(self.batch_size)
            return tf.clip_by_value(y, self.priors_reduced[0], self.priors_reduced[1])
        
        def add_smooth_wall_as_prior_bound(self, parameters, prior_param, prodpri):
            loglike = tf_planck.get_loglkl(parameters) - 100000*(tf.exp(tf.norm(parameters-prior_param, axis=1, keepdims=True)) - 1) * tf.cast(tf.equal(prodpri,0),tf.float32)
            return loglike
                                                                                                                
        def loglkl_smooth_prior(self,
                                parameters,
                                prior_param,
                                prodpri
                                ):
        
        
            loglike = self.add_smooth_wall_as_prior_bound(parameters, prior_param, prodpri)
            return loglike
    
        
        @tf.function
        def loglike(self, x, vals):
            y = tf.transpose(
                tf.reshape(
                    tf.dynamic_stitch([self.idx_reduced,
                                       self.idxs],
                                      [tf.transpose(x),
                                       tf.transpose(vals)]),
                    [self.N_param_tot,
                     self.batch_size]))
    
            pr = tf.reshape(tf_planck.priors.prob(y), [y.shape[0], 1])
            y_pr = tf.clip_by_value(y, self.priors[0], self.priors[1])
            return -self.loglkl_smooth_prior(y, y_pr, pr)[:,0] + 0.5*((y[:,-1]-1)/0.0025)**2
        
        @tf.function
        def loglike_grad(self, x, vals):
            return tfp.math.value_and_gradient(lambda x: self.loglike(x, vals), x)
        
        def make_inverse_hessian_estimate(self, X):
            dx = tf.sqrt(tf.linalg.tensor_diag_part(self.cov))/1000
            epsilon = 1e-10
            hess = []
            Y0 = self.loglike_grad(X, self.val)[1]
            X_copy = tf.identity(X)
            for i in range(dx.shape[0]):
                y1 = self.loglike_grad(tf.concat([X[:,0:0+1]+dx[0],X[:,0+1:]],1), self.val)[1][:,i]
                h_ii = (y1 - Y0[:,i]) / dx[i]
                hess.append([tf.abs(1/h_ii) + epsilon])
            return tf.linalg.diag(tf.transpose(tf.concat(hess,0)))
        
        
        def optimize_loglike(self, X, invH, recursive=False):
            facs = tf.constant([0.1, 0.01, 0.001, 1, 10, 100], dtype=tf.float32)
            bool_val = tf.constant(True)
            i = tf.constant(0)
            obj_val = tf.zeros((self.batch_size,), dtype=tf.float32)
            pos = X
        
            c = lambda i, bool_val, obj_val, pos: bool_val
            def body(i, bool_val, obj_val, pos):
                fac = tf.slice(facs, begin=tf.reshape(i, [1]), size=[1])[0]
                i += 1
                optim_results = tfp.optimizer.bfgs_minimize(lambda x: self.loglike_grad(x, self.val),
                                                            initial_position=X,
                                                            initial_inverse_hessian_estimate=fac*invH,
                                                            max_iterations=500,
                                                            f_absolute_tolerance=0.0000001,
                                                            parallel_iterations = X.shape[0])
            
                bool_val = optim_results.failed[0]
                obj_val  = optim_results.objective_value
                pos      = optim_results.position
                return i, bool_val, obj_val, pos
    
            try:
                r = tf.while_loop(c, body, (i, bool_val, obj_val, pos), maximum_iterations=facs.shape[0])
            except:
                if not recursive:
                    for i in range(3):
                        X = self.take_step()
                        invH = self.make_inverse_hessian_estimate(X)
                        nan_bool = tf.reduce_any(tf.math.is_nan(invH))
                        nan_bool, X, invH = tf.while_loop(self.nan_cond, self.invH_is_nan_loop, loop_vars=(nan_bool, X, invH))
                        loglike, position = self.optimize_loglike(X, invH, recursive=True)
                        if tf.where(loglike<100000.).shape[0]:
                            return loglike, position
                return 100000.*tf.ones([self.batch_size,]), 0.*pos
            return r[2], r[3]
        
        
        def update_proposal(self, temp, loglike_mins, positions):
            loglike_min = tf.reduce_min(loglike_mins)
            indices = tf.concat([tf.where(loglike_mins == loglike_min)[0,:], [0]], 0)
            position_min = tf.slice(tf.constant(positions), begin=indices, size=[1,1,self.N_param])[0,0,:]
            
            return loglike_min, position_min, tfp.distributions.MultivariateNormalTriL(loc=position_min,scale_tril=self.tril*tf.sqrt(temp))
            
    
        def recalc_covmat(self):
            idx_reduced = []
            for i in range(self.loc.shape[0]):
                if not tf.where(tf.equal(self.idxs, i)).shape[0]:
                    idx_reduced.append(i)
            self.idx_reduced = tf.constant(idx_reduced, dtype=tf.int32)
            self.loc_reduced = tf.gather(self.loc, idx_reduced)
            self.N_param = len(idx_reduced)
            self.N_param_tot = self.loc.shape[0]
            
            pr_min = []
            pr_max = []
            for pp in parameters_and_priors.values():
                if pp[2] == 'uniform':
                    pr_min.append(pp[0])
                    pr_max.append(pp[1])
                else:
                    pr_min.append(pp[0]-5*pp[1])
                    pr_max.append(pp[0]+5*pp[1])
            self.priors = (tf.constant(pr_min, dtype=tf.float32), tf.constant(pr_max, dtype=tf.float32))
            self.priors_reduced = (tf.gather(tf.constant(pr_min, dtype=tf.float32), idx_reduced), tf.gather(tf.constant(pr_max, dtype=tf.float32), idx_reduced))
            
            cov = tfp.stats.covariance(self.params)
            sig_sq_A_planck = list(parameters_and_priors.values())[-1][1]**2
            cov = tf.concat([cov[:,:-1], tf.reshape(tf.concat([cov[:-1,-1], [sig_sq_A_planck]],0), [self.N_param_tot,1])], 1)
            self.cov = tf.gather(tf.gather(cov, idx_reduced, axis=0), idx_reduced, axis=1)
            self.tril = tf.linalg.cholesky(self.cov)
    
    
    BH = Basinhopping(N_steps, batch_size)
    

    with MPICommExecutor(comm=comm, root=0) as executor:
        if executor is not None:
            os.system(f'mkdir -p {output_file_base}')
            
            file_global = os.path.join(output_file_base, 'global_bestfit.txt')
            header = 'loglike\t'
            for i, par in enumerate(param_names):
                header += par+'\t'
            header = header[:-1]+'\n'
            if calc_global:
                with open(file_global,'w') as f:
                    f.write(header)
    
            files_1d = []
            files_2d = []
            for i, par_i in enumerate(param_names):
                header = ''
                header += par_i+'\t'
                header += 'loglike\t'
                for ii, par in enumerate(param_names):
                    if i != ii:
                        header += par+'\t'
                header = header[:-1]+'\n'
    
                if calc_1d:
                    files_1d.append(output_file_base+f'/{par_i}.txt')
                    if config['add_points'] == None and config['recalculate'] == None and not config['fix_points']:
                        with open(files_1d[-1],'w') as f:
                            f.write(header)
    
                if calc_2d:
                    for j, par_j in enumerate(param_names):
                        if j > i:
                        
                            header = ''
                            var_par = []
                            fix_par = []
                            for par in param_names:
                                if par not in [par_i, par_j]:
                                    var_par.append(par)
                                else:
                                    fix_par.append(par)
                            for par in fix_par:
                                header += par+'\t'
                            header += 'loglike\t'
                            for par in var_par:
                                header += par+'\t'
                            header = header[:-1]+'\n'
    
                            files_2d.append(output_file_base+f'/{par_i}-{par_j}.txt')
                            if config['add_points'] == None and config['recalculate'] == None and not config['fix_points']:
                                with open(files_2d[-1],'w') as f:
                                    f.write(header)
    
    
            print('Running profile...\n')

            if calc_global:
                print('Computing global best-fit')
            if calc_1d:
                print('Computing',vals_array_1d.shape[0],'points for 1D profiles')
            if calc_2d:
                print('Computing',vals_array_2d.shape[0],'points for 2D profiles')
            
    
            if calc_global:
                answers_global = executor.map(BH.calc_point, zip(vals_array_global, idxs_array_global, k_array_global), unordered=False, chunksize=1)
            if calc_1d:
                answers_1d = executor.map(BH.calc_point, zip(vals_array_1d, idxs_array_1d, k_array_1d), unordered=False, chunksize=5)
            if calc_2d:
                answers_2d = executor.map(BH.calc_point, zip(vals_array_2d, idxs_array_2d, k_array_2d), unordered=False, chunksize=5)
    
            if calc_global:
                for loglike_min, bestfit, vals, file_number in answers_global:
                    line = ''
                    line += str(loglike_min.numpy())+'\t'
                    for par in bestfit.numpy():
                        line += str(par)+'\t'
                    line = line[:-1]+'\n'
                    with open(file_global,'a') as f:
                        f.write(line)
    
    
            if calc_1d:
                for loglike_min, bestfit, vals, file_number in answers_1d:
    
                    line = ''
                    for par in vals[0,:1].numpy():
                        line += str(par)+'\t'
                    line += str(loglike_min.numpy())+'\t'
                    for par in bestfit.numpy():
                        line += str(par)+'\t'
                    line = line[:-1]+'\n'
    
                    if config['recalculate'] != None or config['fix_points']:
                        with open(files_1d[file_number[0]],'r') as f:
                            f_list = list(f)
                        with open(files_1d[file_number[0]],'w') as f:
                            for n, old_line in enumerate(f_list):
                                if n == file_number[1]:
                                    f.write(line)
                                else:
                                    f.write(old_line)
                    else:
                        with open(files_1d[file_number],'a') as f:
                            f.write(line)
    
            if calc_2d:
                for loglike_min, bestfit, vals, file_number in answers_2d:
    
                    line = ''
                    for par in vals[0,:2].numpy():
                        line += str(par)+'\t'
                    line += str(loglike_min.numpy())+'\t'
                    for par in bestfit.numpy():
                        line += str(par)+'\t'
                    line = line[:-1]+'\n'
            
                    if config['recalculate'] != None or config['fix_points']:
                        with open(files_2d[file_number[0]],'r') as f:
                            f_list = list(f)
                        with open(files_2d[file_number[0]],'w') as f:
                            for n, old_line in enumerate(f_list):
                                if n == file_number[1]:
                                    f.write(line)
                                else:
                                    f.write(old_line)
                    else:
                        with open(files_2d[file_number],'a') as f:
                            f.write(line)


#####################################
# ____________ animate ____________ #
#####################################

if keyword == 'animate':
    from source.animate import play
    play()
