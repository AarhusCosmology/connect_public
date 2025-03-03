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

p_backup = [(i,sys.path.pop(i)) for i, p in enumerate(sys.path) if 'connect_disguised_as_classy' in p]
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
        self.compute_class_background = False

            
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
        self.output_Cl = []
        self.ell_computed = []
        self.output_Pk = []
        self.k_grid = []
        self.output_bg = []
        self.z_bg = []
        self.output_th = []
        self.z_th = []
        self.output_derived = []
        self.extra_output = []
        self.cached_splines = {}
        self.param_names = self.info['input_names']
        if 'output_Cl' in self.info:
            self.output_Cl = self.info['output_Cl']
            self.ell_computed = self.info['ell']
        if 'output_Pk' in self.info:
            self.output_Pk = self.info['output_Pk']
            self.k_grid = self.info['k_grid']
            self.create_pk_methods()
        if 'output_bg' in self.info:
            self.output_bg = self.info['output_bg']
            self.z_bg = self.info['z_bg']
        if 'output_th' in self.info:
            self.output_th = self.info['output_th']
            self.z_th = self.info['z_th']
        if 'output_derived' in self.info:
            self.output_derived = self.info['output_derived']
            self.create_derived_methods()
        if 'extra_output' in self.info:
            self.extra_output = self.info['extra_output']
        self.output_interval = self.info['interval']
        if 'normalize' in self.info:
            raise RuntimeError('Unnormalising the output from models is deprecated. Models now output the correct values instead of normalised values.')

        self.default = {'omega_b': 0.0223,
                        'omega_cdm': 0.1193,
                        'omega_g': 0.0001,
                        'H0': 67.7,
                        'h': 0.677,
                        'tau_reio': 0.0541,
                        'n_s': 0.965,
                        'ln10^{10}A_s': 3.04
                        }


        # Initialize calculations (first one is always slower due to internal graph execution) 
        _params = []
        for par_name in self.param_names:
            if par_name in self.default:
                _params.append(self.default[par_name])
            else:
                _params.append(0.0)
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
            try:
                self.compute_class_background = input_parameters.pop('compute_class_background')
                values_not_matching = {}
                parameters_not_set = {}
                for key in self.info['extra_input']:
                    if key in input_parameters:
                        if self.info['extra_input'][key] != input_parameters[key]:
                            values_not_matching[key] = (self.info['extra_input'][key], input_parameters[key])
                    else:
                        parameters_not_set[key] = self.info['extra_input'][key]
                if parameters_not_set:
                    warnings.warn("The following parameters have been used to train the network, but have not been given to CLASS for background computations. The values used by the network will be set:\n" +
                                  "\n".join([f"        - {key} = {val}" for key, val in parameters_not_set.items()]))
                if values_not_matching:
                    warnings.warn("The following parameters have values that differ from what the network has been trained on. The values will be overwritten by the ones used to train the network:\n" +
                                  "\n".join([f"        - {key} = {val[1]}\t->\t{key} = {val[0]}" for key, val in values_not_matching.items()]))
                input_parameters.update(self.info['extra_input'])
                input_parameters.update({'output': 'tCl,lCl,pCl,mPk', 'lensing': 'yes'})
            except:
                pass
            super(Class, self).set(input_parameters)


    def compute(self, level=None):
        if self.compute_class_background:
            try:
                super(Class, self).compute(level=['thermodynamics'])
            except CosmoSevereError as e:
                print('CLASS computation failed with these parameters:')
                print(self.pars)
                raise e
        try:
            params = []
            for par_name in self.param_names:
                if par_name in self.pars:
                    params.append(self.pars[par_name])
                elif 'A_s' in self.pars and par_name == 'ln10^{10}A_s':
                    params.append(np.log(self.pars['A_s']*1e+10))
                elif 'ln10^{10}A_s' in self.pars and par_name == 'A_s':
                    params.append(np.exp(self.pars['ln10^{10}A_s'])*1e-10)
                elif par_name in self.default:
                    params.append(self.default[par_name])
                else:
                    params.append(0.0)
                    warnings.warn(f'The parameter {par_name} is not listed with a default value, so a value of 0.0 is used instead. You can add a default value to the load_model method in the file: {os.path.join(CONNECT_PATH,__file__)}')
            v = tf.constant([params])
        
            self.output_predict = self.model(v).numpy()[0]
        except:
            raise SystemError('No model has been loaded - Set the attribute model_name to the name of a trained CONNECT model')
        self.cached_splines = {}


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
        if self.output_derived:
            class_names = names - set(self.output_derived)
        else:
            class_names = names
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


    def create_pk_methods(self):
        for pk_name in self.output_Pk:
            def pk_method(k,z, pk_name=pk_name):
                if not hasattr(self, 'output_predict'):
                    self.compute()
                intervals = self.output_interval['Pk'][pk_name]
                for i, z_key in enumerate([np.float64(x) for x in intervals.keys()]):
                    if z == z_key:
                        lim0 = intervals[list(intervals.keys())[i]][0]
                        lim1 = intervals[list(intervals.keys())[i]][1]
                        break
                    elif i == len(intervals)-1:
                        raise ValueError(f'The requested redshift of {z} has not been emulated.')
                Pk_computed = self.output_predict[lim0:lim1]
                Pk_spline = CubicSpline(self.k_grid, Pk_computed, bc_type='natural', extrapolate=True)
                return np.float64(Pk_spline(k))
            setattr(self, pk_name, pk_method)


    def create_derived_methods(self):
        for derived_name in self.output_derived:
            if derived_name.isidentifier():
                def derived_method(derived_name=derived_name):
                    if not hasattr(self, 'output_predict'):
                        self.compute()
                    index = self.output_interval['derived'][derived_name]
                    return self.output_predict[index]
                setattr(self, derived_name, derived_method)


    # These methods enable BAO and SN type likelihoods
    # (bao_boss_dr12, bao_smallz_2014, Pantheon, etc.)
    # to use emulated backround quantities

    def Hubble(self, z):
        if 'H [1/Mpc]' in self.output_bg:
            if not hasattr(self, 'output_predict'):
                self.compute()
            if z in self.z_bg:
                index = self.output_interval['bg']['H [1/Mpc]'][0]+self.z_bg.index(z)
                return self.output_predict[index]
            elif len(self.z_bg) > 2:
                if 'Hubble' in self.cached_splines:
                    return float(self.cached_splines['Hubble'](z))
                else:
                    out = self.output_predict[self.output_interval['bg']['H [1/Mpc]'][0]:
                                              self.output_interval['bg']['H [1/Mpc]'][1]]
                    spline = CubicSpline(self.z_bg, out, bc_type='natural')
                    self.cached_splines['Hubble'] = spline
                    return float(spline(z))
            elif self.compute_class_background:
                return super(Class, self).Hubble(z)
            else:
                raise ValueError(f"The requested redshift of {z} was not emulated and there are too few values for interpolation. You can use CLASS for all background computations by setting 'compute_class_background' to True.")
        elif self.compute_class_background:
            return super(Class, self).Hubble(z)
        else:
            raise ValueError("The Hubble parameter has not been emulated. You can use CLASS for all background computations by setting 'compute_class_background' to True.")

    def angular_distance(self, z):
        if 'ang.diam.dist.' in self.output_bg:
            if not hasattr(self, 'output_predict'):
                self.compute()
            if z in self.z_bg:
                index = self.output_interval['bg']['ang.diam.dist.'][0]+self.z_bg.index(z)
                return self.output_predict[index]
            elif len(self.z_bg) > 2:
                if 'angular_distance' in self.cached_splines:
                    return float(self.cached_splines['angular_distance'](z))
                else:
                    out = self.output_predict[self.output_interval['bg']['ang.diam.dist.'][0]:
                                              self.output_interval['bg']['ang.diam.dist.'][1]]
                    spline = CubicSpline(self.z_bg, out, bc_type='natural')
                    self.cached_splines['anugular_distance'] = spline
                    return float(spline(z))
            elif self.compute_class_background:
                return super(Class, self).Hubble(z)
            else:
                raise ValueError(f"The requested redshift of {z} was not emulated and there are too few values for interpolation. You can use CLASS for all background computations by setting 'compute_class_background' to True.")
        elif self.compute_class_background:
            return super(Class, self).Hubble(z)
        else:
            raise ValueError("The angular diameter distance has not been emulated. You can use CLASS for all background computations by setting 'compute_class_background' to True.")

    def rs_drag(self):
        if 'rs_drag' in self.extra_output:
            if not hasattr(self, 'output_predict'):
                self.compute()
            index = self.output_interval['extra']['rs_drag'][0]
            return self.output_predict[index]
        elif self.compute_class_background:
            return super(Class, self).rs_drag()
        else:
            raise ValueError("The rs_drag has not been emulated. You can use CLASS for all background computations by setting 'compute_class_background' to True.")


    def struct_cleanup(self):
        super(Class,self).struct_cleanup()

for i, p in p_backup:
    sys.path.insert(i,p)
sys.modules['classy'] = m_backup
