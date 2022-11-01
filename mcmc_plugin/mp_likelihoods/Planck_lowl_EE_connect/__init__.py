"""
This was put here by the CONNECT setup script. 
Apparently, the low-l EE likelihood raises an error which is not
caught by the code, and this crashes the mcmc run. This likelihood
wraps the old low-l EE likelihood and catches the error if necessary.
"""

from montepython.likelihood_class import Likelihood_clik
from montepython.likelihoods.Planck_lowl_EE import Planck_lowl_EE
import numpy as np
import io_mp

print(Likelihood_clik)
print(Planck_lowl_EE)

class Planck_lowl_EE_connect(Planck_lowl_EE):
    
    def loglkl(self, cosmo, data):

        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])
        cl = self.get_cl(cosmo)

        length = len(self.clik.get_has_cl())
        tot = np.zeros(
            np.sum(self.clik.get_lmax()) + length +
            len(self.clik.get_extra_parameter_names()))
        index = 0

        for i in range(length):
            if (self.clik.get_lmax()[i] > -1):
                for j in range(self.clik.get_lmax()[i]+1):
                    if (i == 0):
                        tot[index+j] = cl['tt'][j]
                    if (i == 1):
                        tot[index+j] = cl['ee'][j]
                    if (i == 2):
                        tot[index+j] = cl['bb'][j]
                    if (i == 3):
                        tot[index+j] = cl['te'][j]
                    if (i == 4):
                        tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                    if (i == 5):
                        tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                index += self.clik.get_lmax()[i]+1


        for nuisance in self.clik.get_extra_parameter_names():

            if nuisance in nuisance_parameter_names:
                nuisance_value = data.mcmc_parameters[nuisance]['current'] *\
                    data.mcmc_parameters[nuisance]['scale']
            else:
                raise io_mp.LikelihoodError(
                    "the likelihood needs a parameter %s. " % nuisance +
                    "You must pass it through the input file " +
                    "(as a free nuisance parameter or a fixed parameter)")

            tot[index] = nuisance_value
            index += 1

        try:
            lkl = self.clik(tot)[0]
        except:
            #print(f"Planck_lowl_EE failed for tau_reio = {data.cosmo_arguments['tau_reio']}")
            return -1e+30

        lkl = self.add_nuisance_prior(lkl, data)

        if getattr(self, 'joint_sz_prior', False):

            if not ('A_sz' in self.clik.get_extra_parameter_names() and 'ksz_norm' in self.clik.get_extra_parameter_names()):
                 raise io_mp.LikelihoodError(
                    "You requested a gaussian prior on ksz_norm + 1.6 * A_sz," +
                    "however A_sz or ksz_norm are not present in your param file.")

            A_sz =  data.mcmc_parameters['A_sz']['current'] * data.mcmc_parameters['A_sz']['scale']
            ksz_norm = data.mcmc_parameters['ksz_norm']['current'] * data.mcmc_parameters['ksz_norm']['scale']

            joint_sz = ksz_norm + 1.6 * A_sz

            if not (hasattr(self, 'joint_sz_prior_center') and hasattr(self, 'joint_sz_prior_variance')):
                raise io_mp.LikelihoodError(
                    " You requested a gaussian prior on ksz_norm + 1.6 * A_sz," +
                    " however you did not pass the center and variance." +
                    " You can pass this in the .data file.")

            if not self.joint_sz_prior_variance == 0:
                lkl += -0.5*((joint_sz-self.joint_sz_prior_center)/self.joint_sz_prior_variance)**2

        return lkl
