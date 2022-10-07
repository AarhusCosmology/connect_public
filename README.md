<div align="center">

![connect_logo.pdf](https://github.com/AarhusCosmology/connect/files/9734781/connect_logo.pdf)


**CO**smological **N**eural **N**etwork **E**mulator of **C**LASS using **T**ensorFlow

![](https://img.shields.io/badge/Python-181717?style=plastic&logo=python)
![](https://img.shields.io/badge/Tensorflow-181717?style=plastic&logo=tensorflow)
![](https://img.shields.io/badge/Author-Andreas%20Nygaard-181717?style=plastic)

[Overview](#overview) •
[Installation and setup](#installation-and-setup) •
[Usage](#usage) •
[Support](#support) • 
[Citation](#citation)

</div>

## Overview
```connect``` is a framework for emulating cosmological parameters using neural networks. This includes both sampling of training data and training of the actual networks using the [TensorFlow](https://www.tensorflow.org) library from Google. ```connect```is designed to aid in cosmological parameter inference by immensely speeding up the process. This is achieved by substituting the cosmological Einstein-Boltzmann solver codes, needed for every evaluation of the likelihood, with a neural network with a $10^2$ to $10^3$ times faster evaluation time. 

## Installation and setup
In order to use ```connect```, simply clone the repository into a folder on your local computer or a cluster
```
git clone https://github.com/AarhusCosmology/connect_public.git
```
The code depends on [Class](https://github.com/lesgourg/class_public) and [Monte Python](https://github.com/brinckmann/montepython_public) (if iterative sampling is to be used - see [arXiv:2205.15726](https://arxiv.org/abs/2205.15726)), so one needs functioning installations of these. One also requires the Planck 2018 likelihood installed. The path (absolute) to ```Monte Python``` should be given as an input in the parameter file, if one uses iterative sampling. Alternatively the path can be set as default in ```source/default_module.py```. The paths to ```connect_public/mp_plugin``` and the Planck likelihood should be set in ```mp_plugin/connect.conf``` in order for ```Monte Python``` to use ```connect```instead of ```class```. 

The code is dependent on ```TensorFlow >= v2.0``` and ```mpi4py```, so these should be installed (pip or conda) within the environment to use with the code. If using an environment when running ```connect```, remember to build ```class``` within this environment.

## Usage
With ```connect``` one can either create training data or train a model using specified training data. The syntax for creating training data is
```
python connect.py create input/<parameter_file>
```
and the parameter file specifies all details (see ```input/example.param```). The syntax for training is similarly
```
python connect.py train input/<parameter_file>
```
Both of these can be called through a job script if on a cluster using SLURM (see the example job script ```jobscripts/example.js```).

All trained models are stored in ```trained_models/```, and these can be loaded using native ```TensorFlow``` commands or the plugin module located in ```mp_plugin/python/build/lib.connect_disguised_as_classy/``` which functions like the ```classy``` wrapper for ```class```.

## Support
```connect``` is a work in progress and will be updated continuously. Please feel free to write me at andreas@phys.au.dk regarding any problems you might encounter (or just to get started properly). 

You can also create an issue if you encounter a bug or have ideas for new features.

## Citation
Please cite the paper [arXiv:2205.15726](https://arxiv.org/abs/2205.15726) if using ```connect``` for publications. 