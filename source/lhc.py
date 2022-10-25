from scipy.stats import qmc
import pickle as pkl
import os

class LatinHypercubeSampler():
    def __init__(self, param):

        self.jobname = param.jobname
        self.N = param.N
        self.parameters = param.parameters
        self.param_names = self.parameters.keys()
        self.d = len(self.param_names)

    def run(self, CONNECT_PATH):
        sampler = qmc.LatinHypercube(d=self.d)
        sample = sampler.random(n=self.N)

        with open(os.path.join(CONNECT_PATH, f'data/lhc_samples/{self.d}_{self.N}.sample'),'wb') as f:
            pkl.dump(sample,f)

