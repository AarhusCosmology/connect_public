from importlib.machinery import SourceFileLoader

import numpy as np


class Parameters():
    def __init__(self, param_file):
        param = SourceFileLoader(param_file, param_file).load_module()
        jobname = param_file.split('/')[-1].split('.')[0]

        self.param_file = param_file
        self.parameters = param.parameters

        self.jobname              =     getattr(param, 'jobname',              jobname             )
        self.N                    = int(getattr(param, 'N',                    10000               ))
        self.train_ratio          =     getattr(param, 'train_ratio',          0.9                 )
        self.val_ratio            =     getattr(param, 'val_ratio',            0.01                )
        self.save_name            =     getattr(param, 'save_name',            None                )
        self.overwrite_model      =     getattr(param, 'overwrite_model',      False               )
        self.batchsize            = int(getattr(param, 'batchsize',            64                  ))
        self.epochs               = int(getattr(param, 'epochs',               200                 ))
        self.loss_function        =     getattr(param, 'loss_function',        'cosmic_variance'   )
        self.activation_function  =     getattr(param, 'activation_function',  'alsing'            )
        self.N_nodes              = int(getattr(param, 'N_nodes',              512                 ))
        self.N_hidden_layers      = int(getattr(param, 'N_hidden_layers',      4                   ))
        self.output_activation    =     getattr(param, 'output_activation',    False               )
        self.extra_input          =     getattr(param, 'extra_input',          {}                  )
        self.output_Cl            =     getattr(param, 'output_Cl',            ['tt']              )
        self.output_Pk            =     getattr(param, 'output_Pk',            []                  )
        self.k_grid               =     getattr(param, 'k_grid',               self.get_k_grid()   )
        self.z_Pk_list            =     getattr(param, 'z_Pk_list',            [0.0]               )
        self.output_bg            =     getattr(param, 'output_bg',            []                  )
        self.z_bg_list            =     getattr(param, 'z_bg_list',            []                  )
        self.output_th            =     getattr(param, 'output_th',            []                  )
        self.z_th_list            =     getattr(param, 'z_th_list',            []                  )
        self.output_derived       =     getattr(param, 'output_derived',       []                  )
        self.extra_output         =     getattr(param, 'extra_output',         {}                  )
        self.N_max_points         = int(getattr(param, 'N_max_points',         10000               ))
        self.normalisation_method =     getattr(param, 'normalisation_method', 'standardisation'   )
        self.sampling             =     getattr(param, 'sampling',             'lhc'               )
        self.mcmc_sampler         =     getattr(param, 'mcmc_sampler',         'cobaya'            )
        self.initial_model        =     getattr(param, 'initial_model',        None                )
        self.resume_iterations    =     getattr(param, 'resume_iterations',    False               )
        self.prior_ranges         =     getattr(param, 'prior_ranges',         {}                  )
        self.log_priors           =     getattr(param, 'log_priors',           []                  )
        self.bestfit_guesses      =     getattr(param, 'bestfit_guesses',      {}                  )
        self.sigma_guesses        =     getattr(param, 'sigma_guesses',        {}                  )
        self.keep_first_iteration =     getattr(param, 'keep_first_iteration', False               )
        self.mcmc_tol             =     getattr(param, 'mcmc_tol',             0.01                )
        self.iter_tol             =     getattr(param, 'iter_tol',             0.1                 )
        self.temperature          =     getattr(param, 'temperature',          5.0                 )
        self.sampling_likelihoods =     getattr(param, 'sampling_likelihoods', ['Planck_lite']     )
        self.extra_cobaya_lkls    =     getattr(param, 'extra_cobaya_lkls',    {}                  )

    def get_k_grid(self):
        if len(self.output_Pk) > 0:
            k_grid = np.sort(np.concatenate([np.logspace(np.log10(5e-6),  np.log10(5e-5),  15),
                                             np.logspace(np.log10(1e-4),  np.log10(0.008), 8),
                                             np.logspace(np.log10(0.76),  np.log10(5),     17),
                                             np.logspace(np.log10(0.009), np.log10(0.75),  60)]))
            k_grid *= 0.67556 # value of h in LCDM
            return k_grid

        else:
            return None
