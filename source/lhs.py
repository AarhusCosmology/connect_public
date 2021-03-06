from scipy.stats import qmc
import pickle as pkl

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

        with open(CONNECT_PATH+f'/data/lhs_samples/sample_models_{self.jobname}_{self.N}.txt','wb') as f:
            pkl.dump(sample,f)

