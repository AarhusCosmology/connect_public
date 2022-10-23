from cobaya.likelihoods.base_classes import Planck2018Clik


class TTTEEE_lite(Planck2018Clik):
    def log_likelihood(self, cl, **params_values):
        loglkl = super(TTTEEE_lite,self).log_likelihood(cl, **params_values)
        T = 5.0
        return loglkl/T
    
