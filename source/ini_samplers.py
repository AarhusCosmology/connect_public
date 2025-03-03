import os
import pickle as pkl
import itertools

import numpy as np
from scipy.stats import qmc

from source.tools import get_covmat

class LatinHypercubeSampler():
    def __init__(self, param):

        self.N = param.N
        self.parameters = param.parameters
        self.param_names = self.parameters.keys()
        self.d = len(self.parameters)
        self.log_priors = param.log_priors        

    def run(self):
        sampler = qmc.LatinHypercube(d=self.d)
        sample = sampler.random(n=self.N)
        data = sample.T
        for i, name in enumerate(self.param_names):
            if name in self.log_priors:
                data[i] *= np.log10(self.parameters[name][1]) - np.log10(self.parameters[name][0])
                data[i] += np.log10(self.parameters[name][0])
                data[i] = np.power(10.,data[i])
            else:
                data[i] *= self.parameters[name][1] - self.parameters[name][0]
                data[i] += self.parameters[name][0]
        return data.T


class HypersphereSampler():
    """
    Authors: Andreas Nygaard and Thomas Tram (2024)
    """
    def __init__(self, param):

        self.N = param.N
        self.parameters = param.parameters
        self.d = len(self.parameters)
        self.surface = param.hypersphere_surface
        self.buffer_size = 100000
        
        bestfit_guesses = []
        for name in self.parameters:
            try:
                bestfit_guesses.append(param.bestfit_guesses[name])
            except:
                bestfit_guesses.append((param.parameters[name][0]+param.parameters[name][1])/2)
        if type(param.temperature) in [int, float]:
            fac = (param.temperature * 2)**2 # This factor is multiplied on the covmat and ensures the boundary of the
        else:                                # hyperellipsoid to be at T*sigma, where T is the sampling temperature.
            fac = (param.temperature[0] * 2)**2

        seed = None
        self.rng = np.random.default_rng(seed=seed)
        if param.hypersphere_covmat is not None:
            cov_file = param.hypersphere_covmat
            cov = get_covmat(cov_file, param) * fac
            self.L = np.linalg.cholesky(cov)

        self.bounds = np.array(list(self.parameters.values()))
        self.bounds = np.insert(self.bounds, 1, bestfit_guesses, axis=1)
        # Compute bounding box around 0
        self.bbox = np.empty((self.bounds.shape[0], 2))
        self.bbox[:,0] = self.bounds[:,0] - self.bounds[:,1]
        self.bbox[:,1] = self.bounds[:,2] - self.bounds[:,1]
        
    def points_in_bbox(self, A, B):
        # Ensure that A and B have the same number of dimensions
        assert A.shape[0] == B.shape[0], "Dimensions of A and B must match"
        # Use boolean indexing for each dimension
        mask = np.all((A.T >= B[:, 0]) & (A.T <= B[:, 1]), axis=1)
        # Select points that satisfy the condition
        return A[:, mask]
    
    def gaussian_hypersphere(self, M):
        # Sample D vectors of N Gaussian coordinates
        samples = self.rng.normal(loc=0.0, scale=1.0, size=M*self.d).reshape((self.d, M))
        # Normalise all distances (radii) to 1
        radii = np.sqrt(np.sum(samples*samples, axis=0))
        samples = samples/radii
        # Sample N radii with exponential distribution
        # (unless points are to be on the surface)
        if self.surface:
            return samples
        new_radii = (self.rng.uniform(low=0.0, high=1.0, size=M)**(1./self.d))
        return (samples*new_radii)
    
    def get_transformed_hypersphere_vector_with_bbox(self):
        while True:
            buffer = self.gaussian_hypersphere(self.buffer_size)
            if hasattr(self,'L'):
                buffer = self.L@buffer
            selected_points = self.points_in_bbox(buffer, self.bbox).T
            for c in selected_points:
                yield c

    def run(self):
        data = np.array(list(itertools.islice(self.get_transformed_hypersphere_vector_with_bbox(),self.N)))
        data += self.bounds[:,1]
        return data

class PickleSampler():
    def __init__(self, param):
        self.pickle_data_file = param.pickle_data_file

    def run(self):
        with open(self.pickle_data_file, 'rb') as f:
            data = pkl.load(f)
        return data
