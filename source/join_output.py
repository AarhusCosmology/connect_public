import os
import shutil

import numpy as np
from scipy.interpolate import CubicSpline

class CreateSingleDataFile():
    def __init__(self, param, CONNECT_PATH):
        self.param = param
        path = os.path.join(CONNECT_PATH, f'data/{self.param.jobname}')
        if param.sampling == 'iterative':
            try:
                i = max([int(f.split('number_')[-1]) for f in os.listdir(path) if f.startswith('number')])
                self.path  = os.path.join(path, f'number_{i}')
            except:
                self.path  = os.path.join(path, f'N-{self.param.N}')
        else:
            self.path  = os.path.join(path, f'N-{self.param.N}')
        
        if len(self.param.output_Cl) > 0:
            self.ell_common = False
            counts = []
            Cl_out_1 = self.param.output_Cl[0]
            for filename in sorted(os.listdir(os.path.join(self.path, f'Cl_{Cl_out_1}_data'))):
                if filename.endswith('.txt'):
                    with open(os.path.join(self.path, f'Cl_{Cl_out_1}_data', filename),'r') as f:
                        try:
                            counts.append(list(f)[0].count('\t'))
                        except:
                            pass
            self.Cl_len = max(set(counts), key = counts.count) + 1

            files = sorted(os.listdir(os.path.join(self.path, f'Cl_{Cl_out_1}_data')))
            i = 0
            while not self.ell_common:
                if files[i].endswith('.txt'):
                    with open(os.path.join(self.path, f'Cl_{Cl_out_1}_data', files[i]),'r') as f:
                        f_list = list(f)
                        for j, f_line in enumerate(f_list):
                            if f_line.count('\t')+1 == self.Cl_len and j%2 == 0:
                                self.ell_common = f_list[0]
                                break
                i += 1
        

    def join(self):
        
        for output in self.param.output_Cl:
            Cl_header = True
            with open(os.path.join(self.path, f'Cl_{output}.txt'),'w') as f:
                for filename in sorted(os.listdir(os.path.join(self.path, f'Cl_{output}_data'))):
                    if filename.endswith('.txt'):
                        with open(os.path.join(self.path, f'Cl_{output}_data', filename),'r') as g:
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

        for output in self.param.output_Pk:
            output_dict = {'k': None, 'Pk': {}}
            for z in self.param.z_Pk_list:
                output_dict['Pk'][z]=[]
            N = len(self.param.z_Pk_list) + 1
            for filename in sorted(os.listdir(os.path.join(self.path, f'Pk_{output}_data'))):
                if filename.endswith('.txt'):
                    with open(os.path.join(self.path, f'Pk_{output}_data', filename),'r') as g:
                        for i, line in enumerate(g):
                            if i%N != 0:
                                output_dict['Pk'][self.param.z_Pk_list[i%N-1]].append(line)
                            elif output_dict['k'] == None and i == 0:
                                output_dict['k'] = line
            with open(os.path.join(self.path, f'Pk_{output}.txt'),'w') as f:
                f.write(output_dict['k'])
                for z in self.param.z_Pk_list:
                    f.write(f'# z = {z}\n')
                    for line in output_dict['Pk'][z]:
                        f.write(line)

        for output in self.param.output_bg:
            output = output.replace('/','\\')
            i = 0
            with open(os.path.join(self.path, f'bg_{output}.txt'),'w') as f:
                for filename in sorted(os.listdir(os.path.join(self.path, f'bg_{output}_data'))):
                    if filename.endswith('.txt'):
                        with open(os.path.join(self.path, f'bg_{output}_data', filename),'r') as g:
                            for j, line in enumerate(g):
                                if j == 0 and i == 0:
                                    f.write(line)
                                    i = 1
                                elif j%2 == 1:
                                    f.write(line)

        for output in self.param.output_th:
            i = 0
            with open(os.path.join(self.path, f'th_{output}.txt'),'w') as f:
                for filename in sorted(os.listdir(os.path.join(self.path, f'th_{output}_data'))):
                    if filename.endswith('.txt'):
                        with open(os.path.join(self.path, f'th_{output}_data', filename),'r') as g:
                            for j, line in enumerate(g):
                                if j == 0 and i == 0:
                                    f.write(line)
                                    i = 1
                                elif j%2 == 1:
                                    f.write(line)
                                    
        if len(self.param.output_derived) > 0:
            i = 0
            with open(os.path.join(self.path, 'derived.txt'),'w') as f:
                for filename in sorted(os.listdir(os.path.join(self.path, 'derived_data'))):
                    if filename.endswith('.txt'):
                        with open(os.path.join(self.path, 'derived_data', filename),'r') as g:
                            for line in g:
                                if line[0] == '#' and i == 0:
                                    f.write(line)
                                    i = 1
                                elif line[0] != '#':
                                    f.write(line)

        for output in self.param.extra_output:
            with open(os.path.join(self.path, f'extra_{output}.txt'),'w') as f:
                for filename in sorted(os.listdir(os.path.join(self.path, f'extra_{output}_data'))):
                    if filename.endswith('.txt'):
                        with open(os.path.join(self.path, f'extra_{output}_data', filename),'r') as g:
                            for line in g:
                                f.write(line)

        i = 0
        with open(os.path.join(self.path, 'model_params.txt'),'w') as f:
            for filename in sorted(os.listdir(os.path.join(self.path, 'model_params_data'))):
                if filename.endswith('.txt'):
                    with open(os.path.join(self.path, 'model_params_data', filename),'r') as g:
                        for line in g:
                            if line[0] == '#' and i == 0:
                                f.write(line)
                                i = 1
                            elif line[0] != '#':
                                f.write(line)

        for folder in [f for f in os.listdir(self.path) if f.endswith('_data')]:
            shutil.rmtree(f'{self.path}/{folder}')
