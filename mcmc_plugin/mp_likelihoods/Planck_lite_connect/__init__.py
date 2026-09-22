import numpy as np
from scipy.io import FortranFile
import scipy.linalg
from montepython.likelihood_class import Likelihood
import os
import tensorflow as tf
import sys
from pathlib import Path
abspath = Path(__file__).parent.absolute()
sys.path.insert(1, str(abspath))
from tf_planck_lite.tf_planck_lite import tf_planck2018_lite

class Planck_lite_connect(Likelihood):

    def __init__(self, path, data, command_line):
        super(Planck_lite_connect, self).__init__(path, data, command_line)
        self.need_cosmo_arguments(data,
            {'wrapper_does_nothing': True})
        self.nuisance = ['A_planck']
        use_lowl_TT = False
        use_lowl_EE = False
        if 'Planck_lowl_tf' in data.experiments or 'Planck_lowl_TT_tf' in data.experiments:
            use_lowl_TT = True
        if 'Planck_lowl_tf' in data.experiments or 'Planck_lowl_EE_tf' in data.experiments:
            use_lowl_EE = True
        connect_model = data.cosmo_arguments['connect_model']
        dirname = os.path.dirname(os.path.abspath(path))
        self.planck_tf = tf_planck2018_lite(tf_planck2018_lite_path=os.path.join(dirname,'tf_planck_lite'),
                                            connect_ttteee_model=connect_model,
                                            use_highl=True,
                                            use_lowl_TT=use_lowl_TT,
                                            use_lowl_EE=use_lowl_EE)
        self.cosmo_names = self.planck_tf.cosmo_params

    def loglkl(self, cosmo, data):
        params = []
        for name in self.cosmo_names:
            params.append(data.mcmc_parameters[name]['current']*data.mcmc_parameters[name]['scale'])        
        params.append(data.mcmc_parameters['A_planck']['current']*data.mcmc_parameters['A_planck']['scale'])
        lkl = self.planck_tf.get_loglkl(tf.constant([params], dtype=tf.float32))[0].numpy()
        lkl += -0.5*((params[-1]-1)/0.0025)**2
        return lkl


	
