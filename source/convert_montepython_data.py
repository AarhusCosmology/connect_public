import numpy as np
from scipy.interpolate import CubicSpline
import os

class CreateSingleDataFile():
    def __init__(self, param, CONNECT_PATH):
        self.param = param
        self.path  = CONNECT_PATH + f'/data/{self.param.jobname}/montepython_data/'
        self.output_files = ['model_params.txt','derived.txt']
        for output in self.param.output_Cl:
            self.output_files.append(f'Cl_{output}.txt')

        if len(self.param.output_Cl) > 0:
            self.ell_common = False
            counts = []
            Cl_out_1 = self.param.output_Cl[0]
            for filename in sorted(os.listdir(self.path + 'Cl_data')):
                if filename.endswith(f'{self.param.output_Cl[0]}.txt'):
                    with open(self.path + 'Cl_data/'+filename,'r') as f:
                        f_list = list(f)
                        for j, f_line in enumerate(f_list):
                            if j%2 == 0:
                                counts.append(f_line.count('\t'))
            self.Cl_len = max(set(counts), key = counts.count) + 1
            files = sorted(os.listdir(self.path + f'Cl_data'))
            i = 0
            while not self.ell_common:
                if files[i].endswith(f'{self.param.output_Cl[0]}.txt'):
                    with open(self.path + 'Cl_data/'+files[i],'r') as f:
                        f_list = list(f)
                        for j, f_line in enumerate(f_list):
                            if f_line.count('\t')+1 == self.Cl_len and j%2 == 0:
                                self.ell_common = f_list[0]
                                break
                i += 1


    def join(self):

        for output in self.param.output_Cl:
            Cl_header = True
            with open(self.path + f'Cl_{output}.txt','w') as f:
                for filename in sorted(os.listdir(self.path + 'Cl_data')):
                    if filename.endswith(f'{output}.txt'):
                        with open(self.path + 'Cl_data/'+filename,'r') as g:
                            g_list = list(g)
                            if Cl_header:
                                f.write(self.ell_common)
                                Cl_header = False
                            for i, line in enumerate(g_list):
                                if i%2 == 1:
                                    if line.count('\t') + 1 == self.Cl_len:
                                        f.write(line)
                                    else:
                                        ell = np.int32(g_list[i-1].replace('\n','').split('\t'))
                                        Cl_old  = np.float32(line.replace('\n','').split('\t'))
                                        Cl_sp_fun = CubicSpline(ell, Cl_old, bc_type='natural', extrapolate=True)
                                        Cl_sp = Cl_sp_fun(np.int32(self.ell_common.replace('\n','').split('\t')))
                                        for i, c in enumerate(Cl_sp):
                                            if i != len(Cl_sp)-1:
                                                f.write(str(c)+'\t')
                                            else:
                                                f.write(str(c)+'\n')

        paramnames_files = [f for f in os.listdir(self.path) if f.endswith('.paramnames')]
        if len(paramnames_files) != 1:
            raise ValueError('There should be only one .paramnames file in the current directory')
        with open(self.path + paramnames_files[0], 'r') as f:
            paramnames = []
            for line in f:
                paramnames.append(line.split('\t')[0].replace(' ',''))

        model_names   = [] 
        derived_names = []


        for name in self.param.parameters:
            if name.replace('*','') in paramnames:
                model_names.append(name)
        for name in self.param.output_derived:
            if name.replace('*','') in paramnames:
                derived_names.append(name)



        model_param_scales = {}
        with open(self.path + 'log.param','r') as f:
            lines = list(f)
        for name in model_names:
            for line in lines:
                if line.startswith(f"data.parameters['{name}']") and line.split("'")[-2] == 'cosmo':
                    model_param_scales[name] = np.float32(line.split('=')[-1].replace(' ','').split(',')[4])
                    break

        with open(self.path + 'model_params.txt','w') as f:
            f.write('# ')
            for name in model_names:
                f.write(name)
                if name == model_names[-1]:
                    f.write('\n')
                else:
                    f.write('\t')
            for filename in sorted(os.listdir(self.path)):
                if filename.endswith('.txt') and filename not in self.output_files:
                    with open(self.path + filename, 'r') as g:
                        for line in g:
                            if line[0] != '#':
                                line = line.replace('\n','').split('\t')
                                for name in model_names:
                                    idx = paramnames.index(name.replace('*',''))
                                    f.write(f'{np.float32(line[idx+1])*model_param_scales[name]:.6e}')
                                    if name == model_names[-1]:
                                        f.write('\n')
                                    else:
                                        f.write('\t')


        derived_scales = {}
        with open(self.path + 'log.param','r') as f:
            lines = list(f)
        for name in derived_names:
            for line in lines:
                if line.startswith(f"data.parameters['{name}']") and line.split("'")[-2] == 'derived':
                    derived_scales[name] = np.float32(line.split('=')[-1].replace(' ','').split(',')[4])
                    break

        with open(self.path + 'derived.txt','w') as f:
            f.write('# ')
            for name in derived_names:
                f.write(name)
                if name == derived_names[-1]:
                    f.write('\n')
                else:
                    f.write('\t')
            for filename in sorted(os.listdir(self.path)):
                if filename.endswith('.txt') and filename not in self.output_files:
                    with open(self.path + filename, 'r') as g:
                        for line in g:
                            if line[0] != '#':
                                line = line.replace('\n','').split('\t')
                                for name in derived_names:
                                    idx = paramnames.index(name.replace('*',''))
                                    f.write(f'{np.float32(line[idx+1])*derived_scales[name]:.6e}')
                                    if name == derived_names[-1]:
                                        f.write('\n')
                                    else:
                                        f.write('\t')
