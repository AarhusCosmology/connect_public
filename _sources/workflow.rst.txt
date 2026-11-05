==========================
Example of workflow - ΛCDM
==========================

Start by cloning CONNECT

.. code-block::	shell

    git clone https://github.com/AarhusCosmology/connect_public.git

Then run the setup script from within the repository

.. code-block::	shell

    cd connect_public
    ./setup.sh

Answer yes to all questions and leave paths as blank.

Your CONNECT installation is now ready to create neural networks.

The first thing you want to do is to create a parameter file in the ``input/`` folder. It is a good idea for the first run to use the ``input/example.param`` file (with iterative sampling), and this is also helpful for creating new parameter files. Open the parameter file in your favourite text editor and make sure that the parameter ``mcmc_sampler`` is set to the MCMC sampler you want to use.

If using a cluster with SLURM, you can use the jobscript ``jobscripts/example.js``. Open this in a text editor and adjust the SLURM parameters to fit your cluster. Now submit the job

.. code-block:: shell

    sbatch jobscripts/example.js

Once the job starts, you can monitor the progress in the ``data/<jobname>/output.log`` file. This tells you how far the iterative sampling has come, and what the code is currently doing. The first thing the code does is to create an initial model from a Latin hypercube sampling. The output from this will look like

.. code-block::	text

    No initial model given
    Calculating 10000 initial CLASS models
    Training neural network
    1/1 - 0s - loss: 228.5294 - 58ms/epoch - 58ms/step
    
    Test loss: 228.5294189453125
    Initial model is example

From here it will begin the iterative process and each iteration will look something like

.. code-block::	text

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

This should not take more than 3-5 iterations with the setup in ``input/example.param``, so using 100 CPU cores with a walltime of 8-10 hours should be sufficient. The computational bottleneck is the ``Calculating N CLASS models`` step, but this is very parallelisable, so given enough CPU cores, this will be fast. The more time consuming bottleneck is the MCMC samplings which can (as of now) only utilise a few cores at a time, given that it is not very parallelisable.

If the walltime was set too low or the iterative sampling did not halt for some reason, it is possible to resume the sampling from the last iteration. This is done by adding this line to your parameter file and submitting the job again

.. code-block::	python

    resume_iterations = True

This can also be used if you want to continue a job with new settings (different loss function, architecture, etc.).

When your job has halted, you can look in the ``data/<jobname>/output.log`` file for the name of the last model. This will generally be a good model that you can use for MCMC and similar, but if you want to train a new model for more epochs or with another architecture, you can do so on the same data collected by the iterative process. This done by changing the training parameters in the parameter file (``example.param`` used here) and running

.. code-block::	shell

    python connect.py train input/example.param

either in a jobscript similar to ``jobscripts/example.js`` or locally with CPUs or GPUs (remember to load ``cuda`` if using GPUs).

Once a neural network has been trained, this can be used as described in :ref:`this section <usage-using-trained-nn-for-mcmc>` (Using a trained neural network for MCMC).

Useful commands for monitoring the iterative sampling
=====================================================

While the iterative process is running each individual step can be monitored with different ``.log`` files. 

All errors can be seen in the SLURM output file defined in the job script.

When calculating Class models, the computed amount can be monitored by the command

.. code-block::	shell

    cat data/<jobname>/number_<iteration>/model_params_data/*.txt | wc -l

When an MCMC is running, the output from either Monte Python or Cobaya can be seen in

.. code-block::	shell

    cat data/<jobname>/number_<iteration>/montepython.log

or

.. code-block::	shell

    cat data/<jobname>/number_<iteration>/cobaya.log

When training the neural network, the progress can be monitored in

.. code-block::	shell

    cat data/<jobname>/number_<iteration>/training.log
