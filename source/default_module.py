from importlib.machinery import SourceFileLoader
import difflib as dl

import numpy as np


class Parameters():
    def __init__(self, param_file):
        param = SourceFileLoader(param_file, param_file).load_module()
        jobname = param_file.split('/')[-1].split('.')[0]

        self.param_file = param_file

        default = {
                   'parameters'           :  ( {}                      ,  dict  ),
                   'jobname'              :  ( jobname                 ,  str   ),
                   'N'                    :  ( 10000                   ,  int   ),
                   'train_ratio'          :  ( 0.9                     ,  float ),
                   'val_ratio'            :  ( 0.01                    ,  float ),
                   'save_name'            :  ( None                    ,  str   ),
                   'overwrite_model'      :  ( False                   ,  bool  ),
                   'batchsize'            :  ( 64                      ,  int   ),
                   'epochs'               :  ( 200                     ,  int   ),
                   'loss_function'        :  ( 'cosmic_variance'       ,  str   ),
                   'activation_function'  :  ( 'alsing'                ,  str   ),
                   'N_nodes'              :  ( 512                     ,  int   ),
                   'N_hidden_layers'      :  ( 4                       ,  int   ),
                   'output_activation'    :  ( False                   ,  bool  ),
                   'extra_input'          :  ( {}                      ,  dict  ),
                   'output_Cl'            :  ( ['tt']                  ,  list  ),
                   'output_Pk'            :  ( []                      ,  list  ),
                   'k_grid'               :  ( self.get_k_grid(param)  ,  list  ),
                   'z_Pk_list'            :  ( [0.0]                   ,  list  ),
                   'output_bg'            :  ( []                      ,  list  ),
                   'z_bg_list'            :  ( []                      ,  list  ),
                   'output_th'            :  ( []                      ,  list  ),
                   'z_th_list'            :  ( []                      ,  list  ),
                   'output_derived'       :  ( []                      ,  list  ),
                   'extra_output'         :  ( {}                      ,  dict  ),
                   'N_max_points'         :  ( 10000                   ,  int   ),
                   'normalisation_method' :  ( 'standardisation'       ,  str   ),
                   'sampling'             :  ( 'lhc'                   ,  str   ),
                   'mcmc_sampler'         :  ( 'cobaya'                ,  str   ),
                   'initial_model'        :  ( None                    ,  str   ),
                   'resume_iterations'    :  ( False                   ,  bool  ),
                   'prior_ranges'         :  ( {}                      ,  dict  ),
                   'log_priors'           :  ( []                      ,  list  ),
                   'bestfit_guesses'      :  ( {}                      ,  dict  ),
                   'sigma_guesses'        :  ( {}                      ,  dict  ),
                   'keep_first_iteration' :  ( False                   ,  bool  ),
                   'mcmc_tol'             :  ( 0.01                    ,  float ),
                   'iter_tol'             :  ( 0.1                     ,  float ),
                   'temperature'          :  ( 5.0                     ,  float ),
                   'sampling_likelihoods' :  ( ['Planck_lite']         ,  list  ),
                   'extra_cobaya_lkls'    :  ( {}                      ,  dict  ),
                   }
        
        for key, val in default.items():
            setattr(self, key, getattr(param, key, val[0]))
            print(eval(f'self.{key}'))
        self.error_handling(param, default)



    def get_k_grid(self, param):
        if hasattr(param, 'output_Pk') and len(param.output_Pk) > 0:
            k_grid = np.sort(np.concatenate([np.logspace(np.log10(5e-6),  np.log10(5e-5),  15),
                                             np.logspace(np.log10(1e-4),  np.log10(0.008), 8),
                                             np.logspace(np.log10(0.76),  np.log10(5),     17),
                                             np.logspace(np.log10(0.009), np.log10(0.75),  60)]))
            k_grid *= 0.67556 # value of h in LCDM
            return k_grid
        return None

    def cast_to_native_type(self, value, target_type, boolean=False):
        if boolean:
            try:
                cast_to_native_type(value, target_type)
                return True
            except:
                return False
        return eval(f"{target_type.__name__}(value)")
    
    def error_handling(self, param, default):
        name_errors = self.get_name_errors(param, default)
        type_errors = self.get_type_errors(param, default)
        input_error = 'Error with inputs in parameter file'
        if len(type_errors) > 11 and len(name_errors) > 11:
            input_errors = input_error + '\n' + name_errors + '\n' + type_errors
            raise Exception(input_errors)
        elif len(name_errors) > 11:
            name_errors = input_error + '\n' + name_errors
            raise Exception(name_errors)
        elif len(type_errors) > 11:
            type_errors = input_error + '\n' + type_errors
            raise Exception(type_errors)
        if not self.parameters:
            raise ValueError('No set of parameters were given. Please provide a dictionary with lower and upper bounds.')

    def get_name_errors(self, param, default):
        name_errors = 'NameErrors:'
        for name in [n for n in dir(param) if not n.startswith('__') and not n.endswith('__')]:
            if not name in default:
                matches = dl.get_close_matches(name, default.keys())
                if len(matches) == 0:
                    name_errors += '\n' + '    '*2 + '- '
                    name_errors += f"'{name}' is not a recognised parameter and no close matches exist."
                else:
                    name_errors += '\n' + '    '*2 + '- '
                    name_errors += f"'{name}' is not a recognised parameter. Did you mean '{matches[0]}'?"
                    input_val = getattr(param, name)
                    input_type = type(input_val)
                    target_type = default[matches[0]][1]
                    if input_type != target_type:
                        if target_type == bool:
                            name_errors += f" The type should then be 'bool'."
                        elif (not self.cast_to_native_type(input_val, target_type, boolean=True)
                              or (target_type in [list, tuple, set] and input_type == str)):
                            name_errors += f" The type should then be '{target_type.__name__}'."
        return name_errors

    def get_type_errors(self, param, default):
        type_errors = 'TypeErrors:'
        for key, val in default.items():
            input_val = getattr(self, key)
            input_type = type(input_val)
            target_type = val[1]
            if key in dir(param) and type(input_val) != val[1]:
                if target_type == bool:
                    type_errors += '\n' + '    '*2 + '- '
                    type_errors += f"'{key}' is of type '{input_type.__name__}' and should be of type 'bool'."
                elif (not self.cast_to_native_type(input_val, target_type, boolean=True)
                      or (target_type in [list, tuple, set] and input_type == str)):
                    type_errors += '\n' + '    '*2 + '- '
                    type_errors += f"'{key}' is of type '{input_type.__name__}' and could not be converted to type '{target_type.__name__}'."
                else:
                    setattr(self, key, cast_to_native_type(input_val, target_type))
        return type_errors
