import os
import shutil

import numpy as np
from scipy.interpolate import CubicSpline

from source import cosmic_solvers

class CreateSingleDataFile():
    def __init__(self, param, CONNECT_PATH):
        self.param = param
        self.cosmic_solver = getattr(cosmic_solvers, cosmic_solvers.name_dict[self.param.cosmic_solver])(self.param)
        self.output_structure = self.cosmic_solver.output_structures()
        path = os.path.join(CONNECT_PATH, f'data/{self.param.jobname}')
        if param.sampling == 'iterative':
            try:
                i = max([int(f.split('number_')[-1]) for f in os.listdir(path) if f.startswith('number')])
                self.path  = os.path.join(path, f'number_{i}')
            except:
                self.path  = os.path.join(path, f'N-{self.param.N}')
        else:
            self.path  = os.path.join(path, f'N-{self.param.N}')


    def join(self):

        for idx, output in enumerate(self.output_structure['1D']['list']):
            output_name = output.replace('output_','').replace('_output','')
            x_array = self.output_structure['1D']['x_arr'][idx]
            for array in getattr(self.param, output):
                arr = array.replace('/','\\')
                i = 0
                with open(os.path.join(self.path, f'{output_name}_{arr}.txt'),'w') as f:
                    for filename in sorted(os.listdir(os.path.join(self.path, f'{output_name}_{arr}_data'))):
                        if filename.endswith('.txt'):
                            with open(os.path.join(self.path, f'{output_name}_{arr}_data', filename),'r') as g:
                                for j, line in enumerate(g):
                                    if j == 0 and i == 0:
                                        f.write(line)
                                        i = 1
                                    elif (j+1)%(1+int(x_array is not None)) == 0:
                                        f.write(line)


        for idx, output in enumerate(self.output_structure['2D']['list']):
            output_name = output.replace('output_','').replace('_output','')
            x_array = self.output_structure['2D']['x_arr'][idx]
            y_array = self.output_structure['2D']['y_arr'][idx]
            y_key = self.output_structure['2D']['y_keys'][idx]
            for array in getattr(self.param, output):
                if type(array) == tuple:
                    array = '-'.join(map(str,array))
                arr = array.replace('/','\\')
                output_dict = {'x': None, 'f(x,y)': {}}
                for y in getattr(self.param, y_array):
                    output_dict['f(x,y)'][y]=[]
                N = len(getattr(self.param, y_array)) + 1
                for filename in sorted(os.listdir(os.path.join(self.path, f'{output_name}_{arr}_data'))):
                    if filename.endswith('.txt'):
                        with open(os.path.join(self.path, f'{output_name}_{arr}_data', filename),'r') as g:
                            for i, line in enumerate(g):
                                if i%N != 0:
                                    output_dict['f(x,y)'][getattr(self.param, y_array)[i%N-1]].append(line)
                                elif output_dict['x'] == None and i == 0:
                                    output_dict['x'] = line
                with open(os.path.join(self.path, f'{output_name}_{arr}.txt'),'w') as f:
                    f.write(output_dict['x'])
                    for y in getattr(self.param, y_array):
                        f.write(f'# {y_key} = {y}\n')
                        for line in output_dict['f(x,y)'][y]:
                            f.write(line)

        for idx, output in enumerate(self.output_structure['Scalar']['list']):
            output_name = output.replace('output_','').replace('_output','')
            i = 0
            with open(os.path.join(self.path, f'{output_name}.txt'),'w') as f:
                for filename in sorted(os.listdir(os.path.join(self.path, f'{output_name}_data'))):
                    if filename.endswith('.txt'):
                        with open(os.path.join(self.path, f'{output_name}_data', filename),'r') as g:
                            for line in g:
                                if line[0] == '#' and i == 0:
                                    f.write(line)
                                    i = 1
                                elif line[0] != '#':
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
