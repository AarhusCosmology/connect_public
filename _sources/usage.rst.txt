=====
Usage
=====

CONNECT is currently meant for creating neural networks to be used with other codes, such as MCMC samplers. Some quick overviews of how to create a neural network and how to use it afterwards are presented below.

Creating a neural network
=========================

With CONNECT, one can either create training data or train a neural network model using specified training data. The syntax for creating training data is:

.. code-block:: shell

    python connect.py create input/<parameter_file>

The parameter file specifies all details (see ``input/example.param``). The syntax for training is similarly:

.. code-block:: shell

    python connect.py train input/<parameter_file>

This is typically used if one wants to retrain on the same data with different training parameters. Both of these can be called through a job script if on a cluster using SLURM (see the example job script ``jobscripts/example.js``).

There are three different kinds of parameters to give in the parameter files (all with default values): *training parameters*, *sampling parameters*, and *saving parameters*.

The training parameters include the architecture of the network as well as normalisation method, loss function, activation function, etc. The list includes the following:

.. code-block:: python

    train_ratio          = 0.9                # how much data to use for training - the rest is stored as test data
    val_ratio            = 0.01               # how much of the training data to use for validation during training
    epochs               = 200                # number of epochs (cycles) to train for
    batchsize            = 64                 # data is split into batches of this size during training
    N_hidden_layers      = 4                  # number of hidden layers
    N_nodes              = 512                # number of nodes (neurons) in each hidden layer
    loss_function        = 'mse'              # loss function used during training for optimisation
    activation_function  = 'relu'             # activation function used in all hidden layers 
    normalization_method = 'standardization'  # normalisation method to use on output data

There are different ways of gathering training data: Latin hypercube sampling, hypersphere sampling described in `[arXiv:2405.01396] <https://arxiv.org/abs/2405.01396>`_, and the iterative approach described in `[arXiv:2205.15726] <https://arxiv.org/abs/2205.15726>`_. This is chosen in the parameter file as either ``sampling = 'lhc'``, ``sampling = 'hypersphere'``, or ``sampling = 'iterative'``. It is also possible to load an existing set of training data through a pickle file by setting ``sampling = 'pickle'``. Some additional parameters are available for the iterative sampling (see ``input/example.param``). The sampling parameters include the following:

.. code-block:: python

    parameters           = {'H0'        : [64,   76  ],   # parameters to sample in given as a dictionary with **min** and **max** values in a list
                           'omega_cdm'  : [0.10, 0.14]}
    extra_input          = {'omega_b': 0.0223}            # extra input for CLASS given as normal CLASS input
    output_Cl            = ['tt', 'te', 'ee']             # which Cl spectra to emulate. These must exist in the cosmo.lensed_cl() dictionary
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
    resume_iterations    = False                          # whether or not to resume a previous run if something failed or additional iterations are needed
    extra_cobaya_lkls    = {}                             # additional likelihoods to sample with when using cobaya as MCMC sampler

    ### Additional parameters for hypersphere sampling (when either 'sampling' or 'initial_sampling' is 'hypersphere') ###
    hypersphere_surface  = False                          # Whether or not to just sample from the surface of the hypersphere
    hypersphere_covmat   = None                           # Path to covariance matrix to align hypersphere along axes of correlation - same format as MontePython

    ### Additional parameters for reading pickled data (when either 'sampling' or 'initial_sampling' is 'pickle') ###
    pickle_data_file     = None                           # Path to pickle file containing an (N,M)-dimensional array where M is the number of parameters and N is the number of points

The saving parameters are used for naming the outputted neural network models along with the folder for training data. The parameters include the following:

.. code-block:: python

    jobname         = 'example'            # identifier for the output and created training data stored under data/<jobname>
    save_name       = 'example_network'    # name of trained models
    overwrite_model = False                # whether or not to overwrite names of trained models or to append a suffix

.. _usage-using-trained-nn-for-mcmc:
Using a trained neural network for MCMC
=======================================

All trained models are stored in ``trained_models/``, and these can be loaded using native ``TensorFlow`` commands or the plugin module located in ``mcmc_plugin/python/build/lib.connect_disguised_as_classy/`` which functions like the ``classy`` wrapper for Class. To use the plugin instead of Class, one needs to add the path to ``sys.path`` in order for this module to be loaded when importing ``classy``. One needs to specify the name of the model with ``cosmo.set()`` in the same way as any other Class parameter is set. This can be done in the following way:

.. code-block:: python

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

This wrapper is used by both Monte Python and Cobaya when using the neural network for MCMC runs.

Monte Python
------------

When running an MCMC with Monte Python, one has to set the configuration file to ``<path to connect_public>/mcmc_plugin/connect.conf`` using the Monte Python flag ``--conf``. The important thing here is that the configuration file points to the CONNECT wrapper instead of Class for the cosmological module. It should look something like this:

.. code-block:: python

    path['cosmo'] = '<path to connect_public>/mcmc_plugin'
    path['clik'] = '<path to planck2018>/code/plc_3.0/plc-3.01/'

Additionally, one has to specify the CONNECT neural network model in the Monte Python parameter file as an extra cosmological argument:

.. code-block:: python

    data.cosmo_arguments['connect_model'] = '<name of connect model>'

If the model is located under ``trained_models`` the name is sufficient, but otherwise, the absolute path should be put instead. Now one can just start Monte Python as usual. Use only a single CPU core per chain, since the evaluation of the network is not parallelisable.

Bug in Monte Python >= 3.6
^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a `bug <https://github.com/brinckmann/montepython_public/issues/333>`_ introduced in the newest version of Monte Python which ignores the path to the cosmological module set in the ``.conf`` file. The easiest fix is to switch to the ``3.5`` branch using ``git checkout 3.5`` from within the ``montepython_public`` repository. Another fix is to run the following piece of bash code from your ``connect_public`` repository:

.. code-block:: bash

    mkdir "mcmc_plugin/python/build/lib.$(python -c 'import sys; print(sys.version)')"
    ln mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy.py "mcmc_plugin/python/build/lib.$(python -c 'import sys; print(sys.version)')/classy.py"

This creates a hard link to the wrapper with a name that is accepted by the Monte Python bug.

The ``setup.sh`` script now automatically switches to the ``3.5`` branch by default. Later branches are also supported now by automatically running the code snippet above when newer versions of Monte Python are used.

Cobaya
------

In Cobaya one must specify the CONNECT wrapper as the theory code in the input dictionary/yaml file. It should be specified in the following way:

.. code-block:: python

    info = {'likelihood': {...},
            'params': {...},
            'sampler': {'mcmc': {...}},
            'theory': {'CosmoConnect': {'ignore_obsolete': True,
                                        'path':            '<path to connect_public>/mcmc_plugin/cobaya',
                                        'python_path':     '<path to connect_public>mcmc_plugin/cobaya',
                                        'extra_args':      {'connect_model': <name of connect model>}
                                       }}}

Again, if the model is located under ``trained_models`` the name is sufficient, but otherwise, the absolute path should be put instead. Use only a single CPU core for each chain here as well.

Using a trained neural network on its own
-----------------------------------------

If one is loading a model without the wrapper, it is important to know about the info dictionary that is now stored within the model object. This dictionary contains information on the parameter names, the output dimensions, etc. The following code snippet loads a model and computes the output:

.. code-block:: python

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

The indices for the different types of output is stored in the dictionary ``info_dict['interval']``. For each kind of output (each type of Cl spectrum, matter power spectrum, derived parameter, etc.) an index or a list of two indices (start and end) is an item with the output name as the key.

The ``info_dict`` in the above code snippet is a ``DictWrapper`` object that functions like a dictionary. All values are TensorFlow variables and all strings (except keys) are byte strings. The byte strings can be converted using ``<byte string>.decode('utf-8')``. If one wants a pure python dictionary with regular types (``list``, ``float``, ``str``), then the same info dictionary can be loaded from a raw string version:

.. code-block:: python

    info_dict = eval(model.get_raw_info().numpy().decode('utf-8'))

This ``info_dict`` is now usable without having to change any types.
