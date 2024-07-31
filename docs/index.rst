.. CONNECT documentation master file, created by
   sphinx-quickstart on Wed Jul 10 13:29:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
Welcome to CONNECT's documentation!
===================================

**CO**\ smological **N**\ eural **N**\ etwork **E**\ mulator of **C**\ LASS using **T**\ ensorFlow

.. image:: https://github.com/AarhusCosmology/connect_public/assets/61239752/bec6bf3e-5c44-4d4c-bf5b-a4c1e32f6391
   :align: center
   :alt: connect_logo
	 
.. image:: https://img.shields.io/badge/Python-181717?style=plastic&logo=python
   :alt: Python

.. image:: https://img.shields.io/badge/Tensorflow-181717?style=plastic&logo=tensorflow
   :alt: Tensorflow

.. image:: https://img.shields.io/badge/Author-Andreas%20Nygaard-181717?style=plastic
   :alt: Author Andreas Nygaard

Overview
========

CONNECT is a framework for emulating cosmological parameters using neural networks. This includes both sampling of training data and training of the actual networks using the `TensorFlow <https://www.tensorflow.org>`_ library from Google. CONNECT is designed to aid in cosmological parameter inference by immensely speeding up the process. This is achieved by substituting the cosmological Einstein-Boltzmann solver codes, needed for every evaluation of the likelihood, with a neural network with a :math:`10^2` to :math:`10^3` times faster evaluation time.

Citation
--------

Please cite the paper `[arXiv:2205.15726] <https://arxiv.org/abs/2205.15726>`_ if using CONNECT for publications. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   workflow
   support
   cosmoslider
