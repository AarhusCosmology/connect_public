from importlib.machinery import SourceFileLoader
import numpy as np


class Parameters():
    def __init__(self, param_file):
        param = SourceFileLoader(param_file, param_file).load_module()
        jobname = param_file.split('/')[-1].split('.')[0]

        self.param_file = param_file
        self.parameters = param.parameters

        self.jobname              = getattr(param, 'jobname',              jobname            )
        self.N                    = getattr(param, 'N',                    10000              )
        self.N_initial            = getattr(param, 'N_initial',            10000              )
        self.N_cpu                = getattr(param, 'N_cpu',                100                )
        self.mem_create           = getattr(param, 'mem_create',           '50G'              )
        self.walltime_create      = getattr(param, 'walltime_create',      '20:00:00'         )
        self.mem_train            = getattr(param, 'mem_train',            '20G'              )
        self.walltime_train       = getattr(param, 'walltime_train',       '10:00:00'         )
        self.train_ratio          = getattr(param, 'train_ratio',          0.9                )
        self.val_ratio            = getattr(param, 'val_ratio',            0.01               )
        self.save_name            = getattr(param, 'save_name',            None               )
        self.overwrite_model      = getattr(param, 'overwrite_model',      False              )
        self.batchsize            = getattr(param, 'batchsize',            64                 )
        self.epochs               = getattr(param, 'epochs',               200                )
        self.loss_function        = getattr(param, 'loss_function',        'cosmic_variance'  )
        self.activation_function  = getattr(param, 'activation_function',  'alsing'           )
        self.N_nodes              = getattr(param, 'N_nodes',              512                )
        self.N_hidden_layers      = getattr(param, 'N_hidden_layers',      4                  )
        self.output_activation    = getattr(param, 'output_activation',    False              )
        self.extra_input          = getattr(param, 'extra_input',          {}                 )
        self.output_Cl            = getattr(param, 'output_Cl',            ['tt']             )
        self.output_Pk            = getattr(param, 'output_Pk',            []                 )
        self.output_Pk_grid       = getattr(param, 'output_Pk_grid',       self.get_kz_array())
        self.output_bg            = getattr(param, 'output_bg',            []                 )
        self.output_th            = getattr(param, 'output_th',            []                 )
        self.output_derived       = getattr(param, 'output_derived',       []                 )
        self.N_max_lines          = getattr(param, 'N_max_lines',          10000              )
        self.normalization_method = getattr(param, 'normalization_method', 'standardization'  )
        self.sampling             = getattr(param, 'sampling',             'lhs'              )
        self.montepython_path     = getattr(param, 'montepython_path',     None               )
        self.initial_model        = getattr(param, 'initial_model',        None               )
        self.resume_iterations    = getattr(param, 'resume_iterations',    False              )
        self.prior_ranges         = getattr(param, 'prior_ranges',         {}                 )
        self.log_priors           = getattr(param, 'log_priors',           []                 )
        self.bestfit_guesses      = getattr(param, 'bestfit_guesses',      {}                 )
        self.sigma_guesses        = getattr(param, 'sigma_guesses',        {}                 )
        self.keep_first_iteration = getattr(param, 'keep_first_iteration', False              )


    def get_kz_array(self):
        if len(self.output_Pk) > 0:
            z_vec = np.logspace(np.log(1e-2),
                                np.log(20),
                                10)
            z_vec = np.insert(z_vec,
                              0, 
                              0.0)
            k_vec = np.array(
                sorted(
                    list(np.logspace(np.log(0.04),
                                     np.log(0.5),
                                     30,base=np.exp(1.0)))
                    +
                    [x for x in np.array(
                        [x if x < 0.04 or x > 0.5 else None
                         for x in np.logspace(np.log(1e-3),
                                              np.log(1.0),
                                              20,
                                              base=np.exp(1.0))]) if x is not None]))
            return [k_vec, z_vec]
        else:
            return None
