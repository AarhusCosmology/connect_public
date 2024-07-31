======================
Installation and setup
======================

In order to use CONNECT, simply clone the repository into a folder on your local computer or a cluster:

.. code-block:: shell

    git clone https://github.com/AarhusCosmology/connect_public.git

The setup and installations are taken care of by the ``setup.sh`` script which ensures that compatible versions of all dependencies are installed in a conda environment. Simply run the following from within the repository:

.. code-block:: shell

    ./setup.sh

You will be presented with some yes/no questions and some requests for paths of other codes. If you do not have any previous codes you wish to link, you can have CONNECT install all of the dependencies by answering yes to all questions and leaving requests for paths blank (follow the instructions on the screen). This creates a conda environment called ``ConnectEnvironment``, where all dependencies are installed. This requires ``anaconda``, ``gcc``, ``openmpi`` and ``cmake`` and is tested for the following versions:

.. code-block:: text

    anaconda = 4.10.1
    gcc      = 12.2.0
    openmpi  = 4.0.3
    cmake    = 3.24.3

Other versions may work just as well but have not been tested. If you find that specific versions do not work, please inform me on the email address further down.

Running the ``setup.sh`` script on a cluster with the `The Environment Modules package <https://modules.readthedocs.io>`_ automatically loads ``gcc``, ``openmpi`` and ``cmake``, but ``anaconda`` needs to be loaded before running the script. If you are running locally or on a cluster without `The Environment Modules package <https://modules.readthedocs.io>`_, all of the above need to be available from the start. You can check this with the following set of commands:

.. code-block:: shell

    conda --version
    gcc --version
    mpirun --version
    cmake --version

Manual setup
============

If one does not wish to use the ``setup.sh`` script, the setup can be performed manually. The code depends on `Class <https://github.com/lesgourg/class_public>`_ and either `Monte Python <https://github.com/brinckmann/montepython_public>`_ or `Cobaya <https://github.com/CobayaSampler/cobaya>`_ (if iterative sampling is to be used - see `arXiv:2205.15726 <https://arxiv.org/abs/2205.15726>`_), so one needs functioning installations of these. One also requires the Planck 2018 likelihood installed. The paths to ``connect_public/mcmc_plugin``, Monte Python, and the Planck likelihood should be set in ``mcmc_plugin/connect.conf`` in order for ``Monte Python`` to use a trained CONNECT model instead of Class. ``mcmc_plugin/connect.conf`` should look something like:

.. code-block:: python

    path['cosmo'] = '<path to connect_public>/mcmc_plugin'
    path['clik'] = '<path to planck2018>/code/plc_3.0/plc-3.01/'
    path['montepython'] = '<path to montepython_public>'

The code is dependent on ``TensorFlow >= v2.0`` and ``mpi4py``, so these should be installed (pip or conda) within the environment to use with the code. If using an environment when running CONNECT, remember to build Class within this environment. Additional dependencies include:

.. code-block:: text

    cython
    matplotlib
    scipy
    numpy

along with all dependencies for Monte Python, Cobaya, Class, and the Planck 2018 likelihood.
