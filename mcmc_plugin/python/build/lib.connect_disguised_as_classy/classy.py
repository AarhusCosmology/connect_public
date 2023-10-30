import os
import sys
import pickle as pkl
import warnings
from pathlib import Path

from scipy.interpolate import CubicSpline
import tensorflow as tf
import numpy as np

FILE_PATH = os.path.realpath(os.path.dirname(__file__))
CONNECT_PATH = Path(FILE_PATH).parents[3]

p_backup = sys.path.pop(0)
m_backup = sys.modules.pop('classy')

import classy as real_classy
from classy import CosmoSevereError, CosmoComputationError


class Class(real_classy.Class):
    def __init__(self, input_parameters={}, model_name=None):
        self.model = None
        try:
            self.model_name = input_parameters.pop('connect_model')
        except:
            pass
        super(Class, self).__init__()
        super(Class, self).set(input_parameters)
        if not model_name == None:
            self.model_name = model_name
        self._model_name = model_name

            
    @property
    def model_name(self):
        return self._model_name


    @model_name.setter
    def model_name(self, new_name):
        if self.model == None:
            self._model_name = new_name
            self.load_model(model_name=new_name)


    def load_model(self, model_name=None):
        if not model_name == None:
            name = model_name
        else:
            if self.model_name == None:
                raise NameError('No model was specified - Set the attribute model_name to the name of a trained CONNECT model')
            else:
                name = self.model_name
        try:
            try:
                self.model = tf.keras.models.load_model(os.path.join(CONNECT_PATH,'trained_models',name), compile=False)
            except:
                self.model = tf.keras.models.load_model(name, compile=False)
        except:
            raise NameError(f"No trained model by the name of '{name}'")

        try:
            self.info = eval(self.model.get_raw_info().numpy().decode('utf-8'))
        except:
            with open(os.path.join(CONNECT_PATH,'trained_models',name,'output_info.pkl'), 'rb') as f:
                self.info = pkl.load(f)
            warnings.warn("Loading info dictionary from output_info.pkl is deprecated and will not be supported in the next update.")

        self.param_names = self.info['input_names']
        self.output_Cl = self.info['output_Cl']
        self.output_Pk = self.info['output_Pk']
        self.output_bg = self.info['output_bg']
        self.output_th = self.info['output_th']
        self.output_derived = self.info['output_derived']
        if len(self.output_Cl) > 0:
            self.ell_computed = self.info['ell']
        if len(self.output_Pk) > 0:
            self.k_grid = self.info['k_grid']
        if len(self.output_bg) > 0:
            self.z_bg = self.info['z_bg']
        if len(self.output_th) > 0:
            self.z_th = self.info['z_th']
        self.output_interval = self.info['interval']
        if 'normalize' in self.info:
            raise RuntimeError('Unnormalising the output from models is deprecated. Models now output the correct values instead of normalised values.')

        self.default = {'omega_b': 0.0223,
                        'omega_cdm': 0.1193,
                        'omega_g': 0.0001,
                        'H0': 67.4,
                        'h': 0.674,
                        'tau_reio': 0.0541,
                        'n_s': 0.965,
                        'ln10^{10}A_s': 3.04,
                        'omega_ini_dcdm': 0,
                        'Gamma_dcdm': 0,
                        'm_ncdm': 0.03,
                        'deg_ncdm': 0.8}

        # Initialize calculations (first one is always slower due to internal graph execution) 
        _params = []
        for par_name in self.param_names:
            _params.append(self.default[par_name])
        _v = tf.constant([_params])
        _output_predict = self.model(_v).numpy()[0]
        del _params
        del _v
        del _output_predict


    def set(self, *args, **kwargs):
        if len(args) == 1 and not bool(kwargs):
            input_parameters = dict(args[0])
        elif len(args) == 0 and bool(kwargs):
            input_parameters = kwargs
        else:
            raise ValueError('Bad call!')
        if bool(input_parameters):
            ip_keys = list(input_parameters.keys())
            for par in ip_keys:
                if par[-12:] == '_log10_prior':
                    log10_par = input_parameters.pop(par)
                    input_parameters[par[:-12]]=10**log10_par
            try:
                self.model_name = input_parameters.pop('connect_model')
            except:
                pass
            super(Class, self).set(input_parameters)


    def compute(self, level=None):
        try:
            params = []
            for par_name in self.param_names:
                if par_name in self.pars:
                    params.append(self.pars[par_name])
                elif 'A_s' in self.pars and par_name == 'ln10^{10}A_s':
                    params.append(np.log(self.pars['A_s']*1e+10))
                elif 'ln10^{10}A_s' in self.pars and par_name == 'A_s':
                    params.append(np.exp(self.pars['ln10^{10}A_s'])*1e-10)
                else:
                    try:
                        params.append(self.default[par_name])
                    except:
                        raise KeyError(f'The parameter {par_name} is not listed with a default value. You can add one in the load_model method in file: {os.path.join(CONNECT_PATH,__file__)}')
            v = tf.constant([params])
        
            self.output_predict = self.model(v).numpy()[0]
        except:
            raise SystemError('No model has been loaded - Set the attribute model_name to the name of a trained CONNECT model')


    def lensed_cl(self, lmax=2500):
        if not hasattr(self, 'output_predict'):
            self.compute()

        if lmax == -1:
            try:
                lmax = self.pars['l_max_scalars']
            except:
                lmax = 2508

        spectra = ['tt','ee','bb','te','pp','tp']
        out_dict = {}
        ell = np.array(list(range(lmax+1)))
        for output in self.output_Cl:
            lim0 = self.output_interval['Cl'][output][0]
            lim1 = self.output_interval['Cl'][output][1]
            Cl_computed = self.output_predict[lim0:lim1]
            if output[0] == output[1]:
                Cl_computed = Cl_computed.clip(min=0)
                Cl_spline = CubicSpline(self.ell_computed, Cl_computed, bc_type='natural', extrapolate=True)
                Cl = Cl_spline(ell).clip(min=0)
            else:
                Cl_computed = Cl_computed
                Cl_spline = CubicSpline(self.ell_computed, Cl_computed, bc_type='natural', extrapolate=True)
                Cl = Cl_spline(ell)
            out_dict[output] = np.insert(Cl[1:]*2*np.pi/(ell[1:]*(ell[1:]+1)), 0, 0.0) # Convert back to Cl form

        out_dict['ell'] = ell
        null_out = np.ones(len(ell), dtype=np.float32) * 1e-15
        for spec in np.setdiff1d(spectra, self.output_Cl):
            out_dict[spec] = null_out

        return out_dict


    def raw_cl(self, lmax=2500):
        warnings.warn("Warning: Cls are lensed even though raw Cls were requested.")
        return self.lensed_cl(lmax)


    def T_cmb(self):
        if hasattr(self, 'Tcmb'):
            return self.Tcmb

        kb = 1.3806504e-23
        hp = 6.62606896e-34
        c  = 2.99792458e8
        G  = 6.67428e-11
        _Mpc_over_m_ = 3.085677581282e22
        sigma_B      = 2*np.pi**5 * kb**4 / (15. * hp**3 * c**2)
        store_value = False

        if 'omega_g' in self.pars:
            omega_g = self.pars['omega_g']
            if 'omega_g' not in self.param_names:
                store_value = True

        elif 'Omega_g' in self.pars:
            if 'H0' in self.pars:
                h = H0/100
            else:
                super(Class,self).compute(level=['background'])
                if 'Omega_g' not in self.param_names:
                    self.Tcmb = super(Class,self).T_cmb()
                return super(Class,self).T_cmb()
            omega_g = self.pars['Omega_g']*h**2
            if 'Omega_g' not in self.param_names and 'H0' not in self.param_names:
                store_value = True

        elif 'T_cmb' in self.pars:
            if 'T_cmb' not in self.param_names:
                self.Tcmb = self.pars['T_cmb']
            return self.pars['T_cmb']

        else:
            self.Tcmb = 2.7255 # default in CLASS
            return self.Tcmb

        if store_value:
            self.Tcmb = pow( 3*omega_g*c**3*1.e10 / (sigma_B*_Mpc_over_m_**2*32*np.pi*G), 1/4)
        return pow( 3*omega_g*c**3*1.e10 / (sigma_B*_Mpc_over_m_**2*32*np.pi*G), 1/4)
        

    def get_current_derived_parameters(self, names):
        if not hasattr(self, 'output_predict'):
            self.compute()

        if 'theta_s_100' in names:
            names = [name if name != 'theta_s_100' else '100*theta_s' for name in names]

        names = set(names)
        class_names = names - set(self.output_derived)
        names -= class_names

        if len(class_names) > 0:
            warnings.warn("Warning: Derived parameter not emulated by CONNECT. CLASS is used instead.")
            pars_update={}
            level='thermodynamics'

            if 'output' in self.pars:
                if not 'mPk' in self.pars['output']:
                    pars_update['output'] = self.pars['output']
                    if not self.pars['output'] == '':
                        pars_update['output'] += ', mPk'
                    else:
                        pars_update['output'] = 'mPk'
            else:
                pars_update['output'] = 'mPk'
            if any('sigma' in name for name in class_names):
                pars_update['P_k_max_h/Mpc'] = 1.
                level='lensing'

            if not pars_update == {}:
                super(Class,self).set(pars_update)
                super(Class,self).compute(level=[level])

        out_dict = super(Class,self).get_current_derived_parameters(list(class_names))
        for name in names:
            idx = self.output_interval['derived'][name]
            value = self.output_predict[idx]
            out_dict[name] = value 

        return out_dict


    def get_background(self):
        if not hasattr(self, 'output_predict'):
            self.compute()

        z_bg = self.info['z_bg']
        z_out = np.linspace(z_bg[-1],z_bg[0],1000)
        bg_dict={'z':z_out}
        for output in self.output_bg:
            out = self.output_predict[self.output_interval['bg'][output][0]:
                                      self.output_interval['bg'][output][1]]
            out_s = CubicSpline(z_bg, out, bc_type='natural')(z_out)
            bg_dict[output] = out_s

        return bg_dict


    def get_thermodynamics(self):
        if not hasattr(self, 'output_predict'):
            self.compute()

        z_th = self.info['z_th']
        z_out = np.linspace(z_th[0],z_th[-1],10000)
        th_dict={'z':z_out}
        for output in self.output_th:
            out = self.output_predict[self.output_interval['th'][output][0]:
                                      self.output_interval['th'][output][1]]
            out_s = CubicSpline(z_th, out, bc_type='natural')(z_out)
            th_dict[output] = out_s

        return th_dict
    


sys.path.insert(0, p_backup)
sys.modules['classy'] = m_backup
