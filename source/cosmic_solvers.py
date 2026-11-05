import numpy as np
from scipy.interpolate import CubicSpline


name_dict = {'class': 'ClassModule',
             'camb':  'CambModule'}


class ClassModule():
    def __init__(self, param, get_defaults=False):
        import classy
        self.cosmo = classy.Class()
        self.param = param
        if get_defaults:
            return

        self.param_names = list(self.param.parameters.keys())
        self.initialise_params()

        # exeptions:
        self.severe_exception = classy.CosmoSevereError
        self.computation_exception = classy.CosmoComputationError


    def initialise_params(self):
        self.params = {}

        if len(self.param.output_Cl) > 0:
            self.params['output']            = 'tCl,lCl'
            self.params['lensing']           = 'yes'
            self.params['l_max_scalars']     = 2508
            if any("b" in s or "e" in s for s in self.param.output_Cl):
                self.params['output']       += ',pCl'

        self.params.update(self.param.extra_input)

        if len(self.param.output_Pk) > 0:
            if 'output' in self.params:
                self.params['output']       += ',mPk'
            else:
                self.params['output']        = 'mPk'
            self.params['P_k_max_1/Mpc']     = 2.5*max(self.param.k_grid)
            self.params['z_max_pk']          = max(self.param.z_Pk_list)

        if 'sigma8' in self.param.output_derived:
            if not 'mPk' in self.params['output']:
                if len(self.params['output']) > 0:
                    self.params['output']   += ',mPk'
                else:
                    self.params['output']    = 'mPk'
            if not 'P_k_max_1/Mpc' in self.params:
                self.params['P_k_max_1/Mpc'] = 1.


    def default_parameters(self):
        ell_grid, z_bg, z_th = self.compute_global_grids()
        default = {
                   'output_Cl'            :  ( ['tt']                  ,  list         ),
                   'ell_grid'             :  ( ell_grid                ,  list         ),
                   'output_Pk'            :  ( []                      ,  list         ),
                   'k_grid'               :  ( self.get_k_grid()       ,  list         ),
                   'z_Pk_list'            :  ( [0.0]                   ,  list         ),
                   'output_bg'            :  ( []                      ,  list         ),
                   'z_bg_list'            :  ( z_bg                    ,  list         ),
                   'output_th'            :  ( []                      ,  list         ),
                   'z_th_list'            :  ( z_th                    ,  list         ),
                   'extra_output'         :  ( {}                      ,  dict         ),
                   'output_derived'       :  ( []                      ,  list         ),
                   }
        return default

    def output_structures(self):
        structure_dict = {
                          'Scalar': {'list':  ['output_derived'],
                                     'weight':[lambda x: np.ones(x.shape[0])*1e+4]
                                     },
                          '1D':     {'list':  ['output_Cl', 'output_bg', 'output_th', 'extra_output'],
                                     'x_arr': ['ell_grid', 'z_bg_list', 'z_th_list', None],
                                     'x_keys':['ell',      'z_bg',      'z_th',      None],
                                     'weight':[lambda l: l,
                                               lambda z: np.clip(1e+3*np.power(np.clip(z, a_min=1e-15, a_max=None), -0.3),
                                                                 a_min=0.5, a_max=3e+3),
                                               lambda z: np.clip(1e+3*np.power(np.clip(z, a_min=1e-15, a_max=None), -0.3),
                                                                 a_min=0.5, a_max=3e+3),
                                               lambda x: np.ones(x.shape[0])*1e+4]
                                     },
                          '2D':     {'list':  ['output_Pk'],
                                     'x_arr': ['k_grid'],
                                     'x_keys':['k_grid'],
                                     'y_arr': ['z_Pk_list'],
                                     'y_keys':['z'],
                                     'weight':[lambda k, z: k**(1/3) * 1500]
                                     }
                          }
        return structure_dict

    def get_k_grid(self):
        k_grid = np.sort(np.concatenate([np.logspace(np.log10(5e-6),  np.log10(5e-5),  15),
                                         np.logspace(np.log10(1e-4),  np.log10(0.008), 8),
                                         np.logspace(np.log10(0.76),  np.log10(5),     17),
                                         np.logspace(np.log10(0.009), np.log10(0.75),  60)]))
        k_grid *= 0.67556 # value of h in LCDM
        return k_grid

    def compute_global_grids(self):
        try:
            lmax = self.param.extra_input['l_max_scalars']
        except:
            lmax = 2508
        self.cosmo.set({'l_max_scalars': lmax,
                        'output': 'tCl lCl pCl',
                        'lensing': 'yes',
                        'H0': 67.8,
                        'omega_b': 0.0223,
                        'omega_cdm': 0.119,
                        'A_s': 2.1e-9,
                        'n_s': 0.965,
                        'tau_reio': 0.0543})
        self.cosmo.compute()
        cls = self.get_computed_cls()
        bg = self.cosmo.get_background()
        th = self.cosmo.get_thermodynamics()
        bg_idx = self.get_z_idx(bg['z'])
        z_bg = bg['z'][bg_idx]
        th_idx = self.get_z_idx(th['z'])
        z_th = th['z'][th_idx]
        ell_grid = cls['ell']
        self.cosmo.struct_cleanup()
        return ell_grid, z_bg, z_th
        
    def set_parameters(self, model):
        for i, par_name in enumerate(self.param_names):
            self.params[par_name] = model[i]
        self.cosmo.set(self.params)

    def compute_outputs(self):
        self.cosmo.compute()

    def get_outputs(self):
        outputs = {}
        
        if len(self.param.output_bg) > 0:
            bg = self.cosmo.get_background()
            z_bg = self.param.z_bg_list
            outputs['bg'] = {}
            for output in self.param.output_bg:
                outputs['bg'][output] = CubicSpline(np.flip(bg['z']), np.flip(bg[output]), bc_type='natural')(self.param.z_bg_list)
            outputs['bg']['z'] = z_bg

        if len(self.param.output_th) > 0:
            th = self.cosmo.get_thermodynamics()
            z_th = self.param.z_th_list
            outputs['th'] = {}
            for output in self.param.output_th:
                outputs['th'][output] = CubicSpline(th['z'], th[output], bc_type='natural')(self.param.z_th_list)
            outputs['th']['z'] = z_th

        if len(self.param.output_derived) > 0:
            der = self.cosmo.get_current_derived_parameters(self.param.output_derived)
            outputs['derived'] = {}
            for output in self.param.output_derived:
                outputs['derived'][output] = der[output]

        if len(self.param.output_Cl) > 0:
            cls = self.get_computed_cls(ell_array=self.param.ell_grid)
            if any(np.isnan(cls[key]).any() for key in cls):
                raise self.computation_exception(
                    'Class computation completed with NaN values in CMB power spectra.')
            ell = cls['ell']
            outputs['Cl'] = {}
            for output in self.param.output_Cl:
                outputs['Cl'][output] = cls[output]*ell*(ell+1)/(2*np.pi)
                if output == 'pp':
                    outputs['Cl'][output] *= ell*(ell+1)
            outputs['Cl']['ell'] = ell

        if len(self.param.output_Pk) > 0:
            outputs['Pk'] = {}
            for pk in self.param.output_Pk:
                outputs['Pk'][pk] = {}
                for z in self.param.z_Pk_list:
                    pk_array = []
                    for k in self.param.k_grid:
                        pk_array.append(eval(f'self.cosmo.{pk}(k,z)'))
                    outputs['Pk'][pk][z] = pk_array
            outputs['Pk']['k'] = self.param.k_grid    


        if len(self.param.extra_output) > 0:
            outputs['extra'] = {}
            for output in self.param.extra_output.keys():
                res = eval(self.param.extra_output[output].replace('cosmo', 'self.cosmo'))
                if type(res) in [np.ndarray, list, tuple]:
                    outputs['extra'][output] = res
                else:
                    outputs['extra'][output] = np.array([res])

        self.cosmo.struct_cleanup()
        return outputs

    def get_computed_cls(self, ell_array=[]):
        cls = self.cosmo.lensed_cl()
    
        if len(ell_array) == 0:
            # get parameters from CLASS model
            BA                 = self.cosmo.get_background()
            conformal_age_     = BA['conf. time [Mpc]'][-1]
            der_pars           = self.cosmo.get_current_derived_parameters(['ra_rec','tau_rec'])
            l_max              = len(cls['ell']) - 1
            angular_rescaling_ = der_pars['ra_rec']/(conformal_age_ - der_pars['tau_rec'])
            l_linstep          = 40
            l_logstep          = 1.12
    
            # compute necessary ells like CLASS (see transfer module in CLASS)
            increment = max(int(2*(np.power(l_logstep,angular_rescaling_) - 1)), 1)
            l=[2]
    
            lin_increment = int(l_linstep*angular_rescaling_)
            while l[-1] + increment < l_max:
                if increment < lin_increment:
                    l.append(l[-1] + increment)
                    increment = max(int(l[-1]*(np.power(l_logstep,angular_rescaling_) - 1)), 1)
                else:
                    l.append(l[-1] + lin_increment)
            ell = np.array(l)
        else:
            ell = np.array(ell_array)
    
        # create reduced dict of cls
        new_cls = {'ell':ell}
        for key, arr in cls.items():
            if key != 'ell':
                new_cls[key] = arr[ell]
    
        return new_cls

    def get_z_idx(self, z):
        z=z.copy()
        if z[-1] > z[0]:
            grid = np.logspace(np.log10(z[1]), np.log10(z[-1]), 99)
        else:
            grid = np.logspace(np.log10(z[0]), np.log10(z[-2]), 99)
        z_idx = []
        for zg in grid:
            i = np.abs(z-zg).argmin()
            z[i] = np.inf
            z_idx.append(i)
        j = len(z) - 1
        while len(z_idx) < 100:
            if j in z_idx:
                j -= 1
            else:
                z_idx.append(j)
        return np.array(sorted(z_idx))




class CambModule():
    def __init__(self, param, get_defaults=False):
        import camb
        self.camb = camb
        self.param = param
        if get_defaults:
            return
        self.param_names = list(self.param.parameters.keys())
        self.initialise_params()

        # exeptions:
        self.severe_exception = camb.CAMBError
        self.computation_exception = camb.CAMBFortranError

    def initialise_params(self):
        self.params = {}

        try:
            self.lmax = self.param.extra_input['lmax']
        except:
            self.lmax = 2508

        if len(self.param.output_Cl) > 0:
            self.params['WantCls']           = True
            self.params['DoLensing']         = True
            self.params['lmax']              = 2508

        self.set_matter_power = False
        self.z_Pk_list = np.array(self.param.z_Pk_list)
        if len(self.param.output_Pk) > 0:
            self.params['WantTransfer']      = True
            self.set_matter_power = True

        if 'sigma8' in self.param.output_derived:
            self.params['WantTransfer']      = True
            self.params['WantScalars']       = True
            self.set_matter_power = True
            if 0 not in self.z_Pk_list: # sigma8 needs z=0 in order to be computed at the correct redshift
                self.z_Pk_list = np.insert(self.z_Pk_list, 0, 0.0)

        self.params.update(self.param.extra_input)

    def default_parameters(self):
        ell_grid, z_bg, z_th = self.compute_global_grids()
        default = {
                   'output_Cl'            :  ( ['tt']                  ,  list         ),
                   'ell_grid'             :  ( ell_grid                ,  list         ),
                   'output_Pk'            :  ( []                      ,  list         ),
                   'k_grid'               :  ( self.get_k_grid()       ,  list         ),
                   'z_Pk_list'            :  ( [0.0]                   ,  list         ),
                   'output_bg'            :  ( []                      ,  list         ),
                   'z_bg_list'            :  ( z_bg                    ,  list         ),
                   'output_th'            :  ( []                      ,  list         ),
                   'z_th_list'            :  ( z_th                    ,  list         ),
                   'extra_output'         :  ( {}                      ,  dict         ),
                   'output_derived'       :  ( []                      ,  list         ),
                   }
        return default

    def output_structures(self):
        structure_dict = {
                          'Scalar': {'list':  ['output_derived'],
                                     'weight':[lambda x: np.ones(x.shape[0])*1e+4]
                                     },
                          '1D':     {'list':  ['output_Cl', 'output_bg', 'output_th', 'extra_output'],
                                     'x_arr': ['ell_grid', 'z_bg_list', 'z_th_list', None],
                                     'x_keys':['ell',      'z_bg',      'z_th',      None],
                                     'weight':[lambda l: l,
                                               lambda z: np.clip(1e+3*np.power(np.clip(z, a_min=1e-15, a_max=None), -0.3),
                                                                 a_min=0.5, a_max=3e+3),
                                               lambda z: np.clip(1e+3*np.power(np.clip(z, a_min=1e-15, a_max=None), -0.3),
                                                                 a_min=0.5, a_max=3e+3),
                                               lambda x: np.ones(x.shape[0])*1e+4]
                                     },
                          '2D':     {'list':  ['output_Pk'],
                                     'x_arr': ['k_grid'],
                                     'x_keys':['k_grid'],
                                     'y_arr': ['z_Pk_list'],
                                     'y_keys':['z'],
                                     'weight':[lambda k, z: k**(1/3) * 1500]
                                     }
                          }
        return structure_dict

    def get_k_grid(self):
        k_grid = np.sort(np.concatenate([np.logspace(np.log10(5e-6),  np.log10(5e-5),  15),
                                         np.logspace(np.log10(1e-4),  np.log10(0.008), 8),
                                         np.logspace(np.log10(0.76),  np.log10(5),     17),
                                         np.logspace(np.log10(0.009), np.log10(0.75),  60)]))
        k_grid *= 0.67556 # value of h in LCDM
        return k_grid

    def compute_global_grids(self):
        try:
            self.lmax = self.param.extra_input['lmax']
        except:
            self.lmax = 2508
        pars = self.camb.set_params(ombh2=0.022,
                                    omch2=0.12,
                                    H0=67.5,
                                    As=2e-9,
                                    ns=0.965,
                                    tau=0.06,
                                    lmax = self.lmax,
                                    WantCls = True,
                                    Want_CMB = True,
                                    DoLensing = True)
        self.results = self.camb.get_results(pars)
        cls = self.get_computed_cls(grid_only=True)
        ell_grid = cls['ell']
        z_bg = np.logspace(-3, 14, 99)
        z_bg = np.insert(z_bg, 0, 0.0)
        z_th = np.logspace(-2, 6.7, 99)
        z_th = np.insert(z_th, 0, 0.0)
        return ell_grid, z_bg, z_th
        
    def set_parameters(self, model):
        for i, par_name in enumerate(self.param_names):
            self.params[par_name] = model[i]
        self.pars = self.camb.set_params(**self.params)
        if self.set_matter_power:
            self.pars.set_matter_power(redshifts=self.z_Pk_list, kmax=max(self.param.k_grid), silent=True)

    def compute_outputs(self):
        self.results = self.camb.get_results(self.pars)

    def get_outputs(self):
        outputs = {}
        
        if len(self.param.output_bg) > 0:
            z_bg = self.param.z_bg_list
            outputs['bg'] = {}
            for output in self.param.output_bg:
                try:
                    if hasattr(self.results, output):
                        outputs['bg'][output] = eval(f'self.results.{output}(z_bg)')
                    elif hasattr(self.results, f'get_{output}'):
                        outputs['bg'][output] = eval(f'self.results.get_{output}(z_bg)')
                    else:
                        raise ValueError(f"Background parameter '{output}' not computed or recognized by CAMB.")
                except:
                    raise ValueError(f"Background parameter '{output}' not callable with an array of redshifts in CAMB. Should this be a derived parameter instead?")
            outputs['bg']['z'] = z_bg

        if len(self.param.output_th) > 0:
            z_th = self.param.z_th_list
            outputs['th'] = {}
            for output in self.param.output_th:
                try:
                    if hasattr(self.results, output):
                        outputs['th'][output] = eval(f'self.results.{output}(z_th)')
                    elif hasattr(self.results, f'get_{output}'):
                        outputs['th'][output] = eval(f'self.results.get_{output}(z_th)')
                    else:
                        raise ValueError(f"Thermodynamics parameter '{output}' not computed or recognized by CAMB.")
                except:
                    raise ValueError(f"Thermodynamics parameter '{output}' not callable with an array of redshifts in CAMB. Should this be a derived parameter instead?")
            outputs['th']['z'] = z_th

        if len(self.param.output_derived) > 0:
            der = self.results.get_derived_params()
            outputs['derived'] = {}
            for output in self.param.output_derived:
                if output not in der:
                    if hasattr(self.results, output):
                        res = eval(f'self.results.{output}')
                    elif hasattr(self.results, f'get_{output}'):
                        res = eval(f'self.results.get_{output}()')
                    else:
                        raise ValueError(f"Derived parameter '{output}' not computed or recognized by CAMB.")
                    if type(res) in [np.ndarray, list, tuple]:
                        if len(res) > 1:
                            if output == 'sigma8': # only sigma8 is expected to return multiple values and we take only the one for z=0
                                res = res[-1]
                            else:
                                raise ValueError(f"Derived parameter '{output}' returned multiple values. Should this be a background or thermodynamics quantity instead?")
                        else:
                            res = res[0]
                    outputs['derived'][output] = res
                else:
                    outputs['derived'][output] = der[output]

        if len(self.param.output_Cl) > 0:
            cls = self.get_computed_cls(ell_array=self.param.ell_grid)
            if any(np.isnan(cls[key]).any() for key in cls):
                raise self.computation_exception(
                    'CAMB computation completed with NaN values in CMB power spectra.')
            ell = cls['ell']
            outputs['Cl'] = {}
            for output in self.param.output_Cl:
                outputs['Cl'][output] = cls[output]
            outputs['Cl']['ell'] = ell

        if len(self.param.output_Pk) > 0:
            outputs['Pk'] = {}
            for pk in self.param.output_Pk:
                if type(pk) == tuple:
                    if len(pk) == 3 and pk[2] == 'linear':
                        nonlinear = False
                        var1, var2 = pk[0], pk[1]
                    elif len(pk) == 2:
                        nonlinear = True
                        var1, var2 = pk[0], pk[1]
                    elif len(pk) == 3:
                        raise ValueError("The third element of the tuple must be 'linear' to indicate linear matter power spectrum.")
                    else:
                        raise ValueError("Tuple for matter power spectrum must be of the form (pk_type_1, pk_type_2) or (pk_type_1, pk_type_2, 'linear').")
                    pk = '-'.join(map(str,pk))
                elif type(pk) == str:
                    if pk.endswith('_linear'):
                        nonlinear = False
                        var1 = pk.replace('_linear', '')
                        var2 = pk.replace('_linear', '')
                    else:
                        nonlinear = True
                        var1 = pk
                        var2 = pk
                Pk_interp = self.results.get_matter_power_interpolator(nonlinear=nonlinear, var1=var1, var2=var2, k_hunit=False)
                outputs['Pk'][pk] = {}
                for z in self.param.z_Pk_list:
                    pk_array = []
                    for k in self.param.k_grid:
                        pk_array.append(Pk_interp.P(z, k))
                    outputs['Pk'][pk][z] = pk_array
            outputs['Pk']['k'] = self.param.k_grid    


        if len(self.param.extra_output) > 0:
            outputs['extra'] = {}
            for output in self.param.extra_output.keys():
                res = eval(self.param.extra_output[output].replace('results', 'self.results'))
                if type(res) in [np.ndarray, list, tuple]:
                    outputs['extra'][output] = res
                else:
                    outputs['extra'][output] = np.array([res])

        return outputs

    def get_computed_cls(self, ell_array=[], grid_only=False):
        cls = self.results.get_cmb_power_spectra(lmax=self.lmax)['total']
        l_max = cls.shape[0] - 1
    
        if len(ell_array) == 0:
            conformal_age_ = self.results.conformal_time(0)   # [Mpc]
            z_rec = self.results.get_derived_params()['zstar']   # last scattering redshift
            chi_rec = self.results.comoving_radial_distance(z_rec)
            ra_rec = chi_rec
            tau_rec = 0.0   # not exposed directly; assume 0 baseline
            angular_rescaling_ = ra_rec/(conformal_age_ - tau_rec)
            l_linstep = 40
            l_logstep = 1.12
    
            increment = max(int(2*(np.power(l_logstep,angular_rescaling_) - 1)), 1)
            l = [2]
    
            lin_increment = int(l_linstep*angular_rescaling_)
            while l[-1] + increment < l_max:
                if increment < lin_increment:
                    l.append(l[-1] + increment)
                    increment = max(int(l[-1]*(np.power(l_logstep,angular_rescaling_) - 1)), 1)
                else:
                    l.append(l[-1] + lin_increment)
            ell = np.array(l)
        else:
            ell = np.array(ell_array)
    
        new_cls = {'ell': ell}
        if grid_only:
            return new_cls
        keys = ['tt','ee','bb','te']
        for i, key in enumerate(keys):
            new_cls[key] = cls[ell, i]
        if 'pp' in self.param.output_Cl or 'tp' in self.param.output_Cl or 'ep' in self.param.output_Cl:
            cls_lens = self.results.get_lens_potential_cls(lmax=self.lmax)
            new_cls['pp'] = cls_lens[ell, 0]
            new_cls['tp'] = cls_lens[ell, 1]
            new_cls['ep'] = cls_lens[ell, 2]
            
        return new_cls
