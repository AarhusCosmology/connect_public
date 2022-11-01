from cobaya.likelihoods.base_classes import Planck2018Clik


class TT(Planck2018Clik):
    def log_likelihood(self, cl, **params_values):
        loglkl = super(TT,self).log_likelihood(cl, **params_values)
        T = 5.0
        return loglkl/T
