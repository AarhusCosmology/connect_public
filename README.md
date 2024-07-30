<div align="center">

![connect_logo](https://github.com/AarhusCosmology/connect_public/assets/61239752/bec6bf3e-5c44-4d4c-bf5b-a4c1e32f6391)

**CO**smological **N**eural **N**etwork **E**mulator of **C**LASS using **T**ensorFlow

![](https://img.shields.io/badge/Python-181717?style=plastic&logo=python)
![](https://img.shields.io/badge/Tensorflow-181717?style=plastic&logo=tensorflow)
![](https://img.shields.io/badge/Author-Andreas%20Nygaard-181717?style=plastic)

[Overview](#overview) •
[Installation and setup](#1-installation-and-setup) •
[Usage](#2-usage) •
[Example of workflow](#3-example-of-workflow---λcdm) • 
[Documentation](#4-documentation) • 
[Support](#5-support) • 
[Citation](#6-citation)

</div>

## Overview
CONNECT is a framework for emulating cosmological parameters using neural networks. This includes both sampling of training data and training of the actual networks using the [TensorFlow](https://www.tensorflow.org) library from Google. CONNECT is designed to aid in cosmological parameter inference by immensely speeding up the process. This is achieved by substituting the cosmological Einstein-Boltzmann solver codes, needed for every evaluation of the likelihood, with a neural network with a $10^2$ to $10^3$ times faster evaluation time. 

- [1. Installation and setup](#1-installation-and-setup)
  * [1.1 Manual setup](#11-manual-setup)
- [2. Usage](#2-usage)
  * [2.1 Creating a neural network](#21-creating-a-neural-network)
  * [2.2 Using a trained neural network for MCMC](#22-using-a-trained-neural-network-for-mcmc)
    + [2.2.1 Monte Python](#221-monte-python)
    + [2.2.2 Cobaya](#222-cobaya)
    + [2.2.3 Using a trained neural network on its own](#223-using-a-trained-neural-network-on-its-own)
- [3. Example of workflow - ΛCDM](#3-example-of-workflow---λcdm)
  * [3.1 Useful commands for monitoring the iterative sampling](#31-useful-commands-for-monitoring-the-iterative-sampling)
- [4. Documentation](#4-documentation)
- [5. Support](#5-support)
- [6. Citation](#6-citation)

## 1. Installation and setup
In order to use CONNECT, simply clone the repository into a folder on your local computer or a cluster
```
git clone https://github.com/AarhusCosmology/connect_public.git
```
The setup and installations are taken care of by the ```setup.sh``` script which ensures that compatible versions of all dependencies are installed in a conda environment. Simply run the following from within the repository:
```
./setup.sh
```
You will be presented with some yes/no questions and some requests for paths of other codes. If you do not have any previous codes you wish to link, you can have CONNECT install all of the dependencies by answering yes to all questions and leaving requests for paths blank (follow the instructions on the screen). This creates a conda environment called ```ConnectEnvironment```, where all dependencies are installed. This requires ```anaconda```, ```gcc```, ```openmpi``` and ```cmake``` and is tested for the following versions: 
```
anaconda = 4.10.1
gcc      = 12.2.0
openmpi  = 4.0.3
cmake    = 3.24.3
```
Other versions may work just as well but have not been tested. If you find that specific versions do not work, please inform me on the email address further down. 

Running the ```setup.sh``` script on a cluster with the [The Environment Modules package](https://modules.readthedocs.io) automatically loads ```gcc```, ```openmpi``` and ```cmake```, but ```anaconda``` needs to be loaded before running the script. If you are running locally or on a cluster without [The Environment Modules package](https://modules.readthedocs.io), all of the above need to be available from the start. You can check this with the following set of commands:
```
conda --version
gcc --version
mpirun --version
cmake --version
```

### 1.1 Manual setup
If one does not wish to use the ```setup.sh``` script, the setup can be performed manually. The code depends on [Class](https://github.com/lesgourg/class_public) and either [Monte Python](https://github.com/brinckmann/montepython_public) or [Cobaya](https://github.com/CobayaSampler/cobaya) (if iterative sampling is to be used - see [arXiv:2205.15726](https://arxiv.org/abs/2205.15726)), so one needs functioning installations of these. One also requires the Planck 2018 likelihood installed. The paths to ```connect_public/mcmc_plugin```, Monte Python, and the Planck likelihood should be set in ```mcmc_plugin/connect.conf``` in order for ```Monte Python``` to use a trained CONNECT model instead of Class. ```mcmc_plugin/connect.conf``` should look something like
```
path['cosmo'] = '<path to connect_public>/mcmc_plugin'
path['clik'] = '<path to planck2018>/code/plc_3.0/plc-3.01/'
path['montepython'] = '<path to montepython_public>'
```

The code is dependent on ```TensorFlow >= v2.0``` and ```mpi4py```, so these should be installed (pip or conda) within the environment to use with the code. If using an environment when running CONNECT, remember to build Class within this environment. Additional dependencies include
```
cython
matplotlib
scipy
numpy
```
along with all dependencies for Monte Python, Cobaya, Class, and the Planck 2018 likelihood.


## 2. Usage
CONNECT is currently meant for creating neural networks to be used with other codes, such as MCMC samplers. Some quick overviews of how to create a neural network and how to use it afterwards are presented below.

### 2.1 Creating a neural network
With CONNECT, one can either create training data or train a neural network model using specified training data. The syntax for creating training data is
```
python connect.py create input/<parameter_file>
```
and the parameter file specifies all details (see ```input/example.param```). The syntax for training is similarly
```
python connect.py train input/<parameter_file>
```
This is typically used if one wants to retrain on the same data with different training parameters.
Both of these can be called through a job script if on a cluster using SLURM (see the example job script ```jobscripts/example.js```).


There are three different kinds of parameters to give in the parameter files (all with default values), *training parameters*, *sampling parameters*, and *saving parameters*. 

The training parameters include the architecture of the network as well as normalisation method, loss function, activation function, etc. The list includes the following:
```
train_ratio          = 0.9                # how much data to use for training - the rest is stored as test data
val_ratio            = 0.01               # how much of the training data to use for validation during training
epochs               = 200                # number of epochs (cycles) to train for
batchsize            = 64                 # data is split into batches of this size during training
N_hidden_layers      = 4                  # number of hidden layers
N_nodes              = 512                # number of nodes (neurons) in each hidden layer
loss_function        = 'mse'              # loss function used during training for optimisation
activation_function  = 'relu'             # activation function used in all hidden layers 
normalization_method = 'standardization'  # normalisation method to use on output data
```

There are different ways of gathering training data, Latin hypercube sampling, hypersphere sampling described in [arXiv:2405.XXXXX], and the iterative approach described in [arXiv:2205.15726]. This is chosen in the parameter file as either ```sampling = 'lhc'```, ```sampling = 'hypersphere'```, or ```sampling = 'iterative'```. It is also possible to load an existing set of training data through a pickle file by setting ```sampling = 'pickle'```. Some additional parameters are available for the iterative sampling (see ```input/example.param```). The sampling parameters include the following:
```
parameters           = {'H0'        : [64,   76  ],   # parameters to sample in given as a dictionary with **min** and **max** values in a list
                       'omega_cdm'  : [0.10, 0.14]}
extra_input          = {'omega_b': 0.0223}            # extra input for CLASS given as normal CLASS input
output_Cl            = ['tt', 'te', 'ee']             # which Cl spectra to emulate. These must exist in the cosmo.lensed_cl() dictinary
output_Pk            = ['pk', 'pk_cb']                # which Pk spectra to emulate. These must exist as a method in the classy wrapper, e.g. cosmo.pk_cb(k,z)
k_grid               = [1e-2, 1e-1, 1, 10]            # k_grid of matter power spectra. The default grid of 100 values is optimal in most cases (optional)
z_Pk_list            = [0.0, 1.5, 13.6]               # list of z-values to compute matter power spectra for
output_bg            = ['ang.diam.dist']              # which background functions to emulate. These must exist in cosmo.get_background()
z_bg_list            = [0.0, 0.3, 0.7]                # list of z-values to compute background functions for. This defaults to 100 evenly spaced points in log space (optional)
output_th            = ['w_b', 'tau_d']               # which thermodynamics functions to emulate. These must exist in cosmo.get_thermodynamics()
z_th_list            = [0.0, 0.3, 0.7]                # list of z-values to compute thermodynamics functions for. This defaults to 100 evenly spaced points in log space (optional)
extra_output         = {'rs_drag': 'cosmo.rs_drag()'} # completely custom output. Name of the output is the key and the value is a string of code that outputs either a float, an int or a 1D array
output_derived       = ['YHe', 'sigma_8']             # which derived parameters to emulate
N                    = 1e+4                           # how many data points to sample using a Latin hypercube (initial step for iterative sampling)
sampling             = 'iterative'                    # sampling method - 'lhc', 'hypersphere', 'pickle', or 'iterative'

### Additional parameters for iterative sampling ###
N_max_points         = 2e+4                           # maximum number of points to sample from each iteration
mcmc_sampler         = 'montepython'                  # MCMC code to use in each iteration - 'montepython' or 'cobaya'
initial_model        = None                           # initial model to start from instead of using a Latin hypercube
initial_sampling     = 'hypersphere'                  # initial configuration of training data - 'lhc', 'hypersphere' or 'pickle'
mcmc_tol             = 0.01                           # convergence criterion for R-1 values in MCMC runs in each iteration
iter_tol             = 0.1                            # convergence criterion for R-1 values between data from two consecutive iterations
temperature          = 5.0                            # sampling temperature during MCMC - if a list is provided, additional iterations with annealing will be done
sampling_likelihoods = ['Planck_lite']                # likelihoods to use for sampling in the iterations
prior_ranges         = {'H0' : [50, 100]}             # prior ranges to be used by the MCMC sampler (optional)
bestfit_guesses      = {'H0' : 67.7}                  # bestfit guesses to be used for proposal distribution by the MCMC sampler (optional)
sigma_guesses        = {'H0' : 0.5}                   # sigma guesses to be used for proposal distribution by the MCMC sampler (optional)
log_priors           = ['omega_cdm']                  # which parameters to sample with logarithmic priors (optional)
keep_first_iteration = False                          # whether or not to keep data from the first iteration - usually bad
resume_iterations    = False                          # whether or not to resume a previous run if somthing failed or additional iterations are needed
extra_cobaya_lkls    = {}                             # additional likelihoods to sample with when using cobaya as MCMC samler

### Additional parameters for hypersphere sampling (when either 'sampling' or 'initial_sampling' is 'hypersphere') ###
hypersphere_surface  = False                          # Whether or not to just sample from the surface of the hypersphere
hypersphere_covmat   = None                           # Path to covariance matrix to align hypersphere along axes of correlation - same format as MontePython

### Additional parameters for reading pickled data (when either 'sampling' or 'initial_sampling' is 'pickle') ###
pickle_data_file     = None                           # Path to pickle file containing an (N,M)-dimensional array where M is the number of parameters and N is the number og points
```

The saving parameters are used for naming the outputted neural network models along with the folder for training data. The parameters include the following:
```
jobname         = 'example'            # identifier for the output and created training data stored under data/<jobname>
save_name       = 'example_network'    # name of trained models
overwrite_model = False                # whether or not to overwrite names of trained models or to append a suffix
```


### 2.2 Using a trained neural network for MCMC

All trained models are stored in ```trained_models/```, and these can be loaded using native ```TensorFlow``` commands or the plugin module located in ```mcmc_plugin/python/build/lib.connect_disguised_as_classy/``` which functions like the ```classy``` wrapper for Class. To use the plugin instead of Class, one needs to add the path to ```sys.path``` in order for this module to be loaded when importing ```classy```. One needs to specify the name of the model with ```cosmo.set()``` in the same way as any other Class parameter is set. This can be done in the following way:
```
import sys
path_to_connect_classy = '<path to connect_public>/mcmc_plugin/python/build/lib.connect_disguised_as_classy'
sys.path.insert(1, path_to_connect_classy)
from classy import Class
cosmo = Class()
cosmo.set({'connect_model': <name of neural network model>,
           'H0'           : 67.7,
           ...
          })
cosmo.compute()
cls = cosmo.lensed_cl()
```

This wrapper is used by both Monte Python and Cobaya when using the neural network for MCMC runs. 
#### 2.2.1 Monte Python
When running an MCMC with Monte Python, one has to set the configuration file to ```<path to connect_public>/mcmc_plugin/connect.conf``` using the Monte Python flag ```--conf```. The important thing here is that the configuration file points to the CONNECT wrapper instead of Class for the cosmological module. It should look something like this
```
path['cosmo'] = '<path to connect_public>/mcmc_plugin'
path['clik'] = '<path to planck2018>/code/plc_3.0/plc-3.01/'
```
Additionally, one has to specify the CONNECT neural network model in the Monte Python parameter file as an extra cosmological argument:
```
data.cosmo_arguments['connect_model'] = '<name of connect model>'
```
If the model is located under ```trained_models``` the name is sufficient, but otherwise, the absolute path should be put instead.
Now one can just start Monte Python as usual. Use only a single CPU core per chain, since the evaluation of the network is not paralleliasable.

##### \# Bug in Monte Python >= 3.6
There is a [bug](https://github.com/brinckmann/montepython_public/issues/333) introduced in the newest version of Monte Python which ignores the path to the cosmological module set in the ```.conf``` file. The easiest fix is to switch to the ```3.5``` branch using ```git checkout 3.5``` from within the ```montepython_public``` repository. Another fix is to run the following piece of bash code from your ```connect_public``` repository:
```
mkdir "mcmc_plugin/python/build/lib.$(python -c 'import sys; print(sys.version)')"
ln mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy.py "mcmc_plugin/python/build/lib.$(python -c 'import sys; print(sys.version)')/classy.py"
```
This creates a hard link to the wrapper with a name that is accepted by the Monte Python bug.

The ```setup.sh``` script now automatically switches to the ```3.5``` branch by default. Later branches are also supported now by automatically running the code snippet above when newer versions of Monte Python are used.

#### 2.2.2 Cobaya
In Cobaya one must specify the CONNECT wrapper as the theory code in the input dictionary/yaml file. It should be specified in the following way:
```
info = {'likelihood': {...},
        'params': {...},
        'sampler': {'mcmc': {...}},
        'theory': {'CosmoConnect': {'ignore_obsolete': True,
                                    'path':            '<path to connect_public>/mcmc_plugin/cobaya',
                                    'python_path':     '<path to connect_public>mcmc_plugin/cobaya',
                                    'extra_args':      {'connect_model': <name of connect model>}
                                   }}}
```
Again, if the model is located under ```trained_models``` the name is sufficient, but otherwise, the absolute path should be put instead.
Use only a single CPU core for each chain here as well

#### 2.2.3 Using a trained neural network on its own
If one is loading a model without the wrapper, it is important to know about the info dictionary that is now stored within the model object. This dictionary contains information on the parameter names, the output dimensions, etc. The following code snippet loads a model and computes the output:
```
import os
import numpy as np
import tensorflow as tf

model_name = <path to connect model>
model = tf.keras.models.load_model(model_name, compile=False)
info_dict = model.info_dict

v = tf.constant([[..., ...]])    # input for neural network with dimension (N, M)
                                 # where N is the number of input vectors and M is
                                 # the number of cosmological parameters

emulated_output = model(v)
```
The indices for the different types of output is stored in the dictionary ```info_dict['interval']```. For each kind of output (each type of Cl spectrum, matter power spectrum, derived parameter, etc.) an index or a list of two indices (start and end) is an item with the output name as the key.

The ```info_dict``` in the above code snippet is a ```DictWrapper``` object that functions like a dictionary. All values are TensorFlow variables and all strings (except keys) are byte strings. The byte strings can be converted using ```<byte string>.decode('utf-8')```. If one wants a pure python dictionary with regular types (```list```, ```float```, ```str```), then the same info dictionary can be loaded from a raw string version:
```
info_dict = eval(model.get_raw_info().numpy().decode('utf-8'))
```
This ```info_dict``` is now useable without having to change any types.

## 3. Example of workflow - ΛCDM
Start by cloning CONNECT
```
git clone https://github.com/AarhusCosmology/connect_public.git
```
Then run the setup script from within the repository
```
cd connect_public
./setup.sh
```
Answer yes to all questions and leave paths as blank

Your CONNECT installation is now ready to create neural networks.

The first thing you want to do is to create a parameter file in the ```input/``` folder. It is a good idea for the first run to use the ```input/example.param``` file (with iterative sampling), and this is also helpful for creating new parameter files. Open the parameter file in your favourite text editor and make sure that the parameter ```mcmc_sampler``` is set to the MCMC sampler you want to use. 

If using a cluster with SLURM, you can use the jobscript ```jobscripts/example.js```. Open this in a text editor and adjust the SLURM parameters to fit your cluster. Now submit the job:
```
sbatch jobscripts/example.js
```

Once the job starts, you can monitor the progress in the ```data/<jobname>/output.log``` file. This tells you how far the iterative sampling has come, and what the code is currently doing. The first thing the code does is to create an initial model from a Latin hypercube sampling. The output from this will look like:
```
No initial model given
Calculating 10000 initial CLASS models
Training neural network
1/1 - 0s - loss: 228.5294 - 58ms/epoch - 58ms/step

Test loss: 228.5294189453125
Initial model is example
```
From here it will begin the iterative process and each iteration will look something like
```
Beginning iteration no. 1
Temperature is now 5.0
Running MCMC sampling no. 1...
MCMC sampling stopped since R-1 less than 0.05 has been reached.
Number of accepted steps: 12340
Keeping only last 5000 of the accepted Markovian steps
Comparing latest iterations...
Calculating 5000 CLASS models
Training neural network
1/1 - 0s - loss: 7.6460 - 34ms/epoch - 34ms/step

Test loss: 7.645951747894287
New model is example_1
```
This should not take more than 3-5 iterations with the setup in ```input/example.param```, so using 100 CPU cores with a walltime of 8-10 hours should be sufficient. The computational bottleneck is the ```Calculating N CLASS models``` step, but this is very parallelisable, so given enough CPU cores, this will be fast. The more time consuming bottleneck is the MCMC samplings which can (as of now) only utilise a few cores at a time, given that it is not very parallelisable.

If the walltime was set too low or the iterative sampling did not halt for some reason, it is possible to resume the sampling from the last iteration. This is done by adding this line to your parameter file and submitting the job again:
```
resume_iterations = True
```
This can also be used if you want to continue a job with new settings (different loss function, architecture, etc.).

When your job has halted, you can look in the ```data/<jobname>/output.log``` file for the name of the last model. This will generally be a good model that you can use for MCMC and similar, but if you want to train a new model for more epochs or with another architecture, you can do so on the same data collected by the iterative process. This done by changing the training parameters in the parameter file (```example.param``` used here) and running
```python connect.py train input/example.param```
either in a jobscript similar to ```jobscripts/example.js``` or locally with CPUs or GPUs (remember to load ```cuda``` if using GPUs).

Once a neural network has been trained, this can be used as described in section [2.2](#22-using-a-trained-neural-network-for-mcmc).

### 3.1 Useful commands for monitoring the iterative sampling
While the iterative process is running each individual step can be monitored with different ```.log``` files. 

All errors can be seen in the SLURM output file defined in the job script

When calculating Class models, the computed amount can be monitored by the command
```
cat data/<jobname>/number_<iteration>/model_params_data/*.txt | wc -l
```
When an MCMC is running, the output from either Monte Python or Cobaya can be seen in
```
cat data/<jobname>/number_<iteration>/montepython.log
```
or
```
cat data/<jobname>/number_<iteration>/cobaya.log
```
When training the neural network, the progress can be monitored in
```
cat data/<jobname>/number_<iteration>/training.log
```

## 4. Documentation
The documentation is available [here](https://aarhuscosmology.github.io/connect_public) along with a web-based application for emulating a CMB power spectrum.

## 5. Support
CONNECT is a work in progress and will be updated continuously. Please feel free to write me at andreas@phys.au.dk regarding any problems you might encounter (or just to get started properly). 

You can also create an issue if you encounter a bug or have ideas for new features.

## 6. Citation
Please cite the paper [arXiv:2205.15726](https://arxiv.org/abs/2205.15726) if using CONNECT for publications. 