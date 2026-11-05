import requests
import xml.etree.ElementTree as ET
from importlib.machinery import SourceFileLoader
import shutil
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tf_keras
import numpy as np
from scipy.optimize import minimize

try:
    from resources.planck_lite_py.planck_lite_py import PlanckLitePy
except ImportError:
    #prompt the user to install planck_lite_py from github. ask if they want to install it now automatically
    print("The python version of the planck lite likelihood is required to find the best-fit parameters.")
    install = input("Do you want to install planck_lite_py now? (y/n): ")
    if install.lower() == 'y':
        # clone the repository from github
        os.system("git clone https://github.com/heatherprince/planck-lite-py.git resources/planck_lite_py")
        from resources.planck_lite_py.planck_lite_py import PlanckLitePy
    else:
        print("Please install planck_lite_py from GitHub and put it in the 'resources' folder.")
        raise SystemExit



class Convert_to_TFLite:
    def __init__(self, tf_model_path):
        if tf_model_path.endswith('/'):
            tf_model_path = tf_model_path[:-1]
        self.tf_model_path = tf_model_path
        self.output_dir = tf_model_path + '_tflite_tmp'

        self.overwrite_tex_names = {'Neff_ini_dncdm': 'N_{\\rm eff,dncdm}^{\\rm ini}', 'log10lifetime_ncdm_decay_dr': '{\\rm log}_{10}\\tau_{\\rm ncdm}^{\\rm decay,dr}', 'm_ncdm_decay_dr': 'm_{\\rm ncdm}^{\\rm decay,dr}'}

        os.makedirs(self.output_dir, exist_ok=True)

        self.load_and_rescale_model()
        self.generate_svg_files()
        self.write_meta_data()
        self.get_bestfit()

        shutil.make_archive(self.output_dir, format='zip', root_dir=self.output_dir)
        # change extension of the created zip file to .tflite
        os.rename(f'{self.output_dir}.zip', f'{tf_model_path}.tflite')
        shutil.rmtree(self.output_dir)
        
        print(f'tflite model and metadata written to {tf_model_path}.tflite')


    def load_and_rescale_model(self):
        print('converting model to tflite...')
        self.model = tf_keras.models.load_model(self.tf_model_path, compile=False)
        self.info_dict = eval(self.model.get_raw_info().numpy().decode('utf-8'))

        self.param = SourceFileLoader(self.tf_model_path+'/log.param', self.tf_model_path+'/log.param').load_module()


        ranges = np.array(list(self.param.parameters.values()))
        self.input_ranges = np.zeros((len(ranges), 3))

        for i, r in enumerate(ranges):
            stepsize = min(10**np.floor(np.log10((r[1] - r[0])/100)),1)
            self.input_ranges[i] = [r[0], r[1], stepsize]

        self.rescale_idx = np.where(self.input_ranges[:,0]<1e-4)[0]
        self.rescale_factor = 10**(np.floor(np.log10(self.input_ranges[self.rescale_idx,0])))

        input_ranges_rescaled = np.copy(self.input_ranges)
        self.rescale_factor_array = np.ones(len(self.input_ranges))
        for i, f in zip(self.rescale_idx, self.rescale_factor):
            self.rescale_factor_array[i] = f
            input_ranges_rescaled[i,0] /= f
            input_ranges_rescaled[i,1] /= f
            input_ranges_rescaled[i,2] /= f

        self.input_ranges_rescaled = np.around(input_ranges_rescaled, decimals=8)



        if len(self.rescale_idx) > 0:
            # create tf model that rescales inputs and calls original model
            print(f'Rescaling input parameter {list(self.param.parameters.keys())[i]} by factor {1/self.rescale_factor_array[i]:.1e} for improved numerical stability.')
            new_model = tf.keras.Sequential()
            class RescaleLayer(tf.keras.layers.Layer):
                def __init__(self, factor_array):
                    super(RescaleLayer, self).__init__()
                    self.factor_array = tf.constant(factor_array, dtype=tf.float32)
            
                def call(self, inputs):
                    return inputs * self.factor_array
            new_model.add(RescaleLayer(self.rescale_factor_array))
            new_model.add(self.model)
            new_model.build((None, len(self.input_ranges)))
            
            converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
        else:
            converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_path)
        tfmodel = converter.convert()
        open(self.output_dir+'/model.tflite','wb').write(tfmodel)


    def find_largest_height(self, svg_dir):
        max_height = 0.0
        max_height_svg = None
        
        # Iterate through all SVG files in the directory
        for filename in os.listdir(svg_dir):
            if filename.endswith(".svg"):
                svg_path = os.path.join(svg_dir, filename)
                
                # Parse SVG file
                tree = ET.parse(svg_path)
                root = tree.getroot()
                
                # Find height attribute
                height_attr = root.attrib.get('height', None)
                if height_attr:
                    height_value = float(height_attr.replace('px', '').replace('pt', ''))
                    if height_value > max_height:
                        max_height = height_value
                        max_height_svg = svg_path
        
        return max_height, max_height_svg
    
    def adjust_svg_height(self, svg_file, new_height):
        # Parse SVG file
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Update height attribute
        root.set('height', f'{new_height}pt')
        
        # Update viewBox attribute (if it exists)
        viewBox_attr = root.attrib.get('viewBox', None)
        if viewBox_attr:
            parts = viewBox_attr.split()
            if len(parts) == 4:
                parts[3] = f'{new_height}'
                root.set('viewBox', ' '.join(parts))
        
        # Write back to the SVG file
        tree.write(svg_file)
    
    
    def render_latex_to_svg(self, latex_string, output_file, fontsize=r'\Large'):
        # Define the endpoint URL of CodeCogs API
        url = 'https://latex.codecogs.com/svg.image?'
        latex_string = f"{fontsize} {latex_string}"
    
        latex_string = latex_string.replace('\\', '%5C')
        # Set up the payload with the LaTeX string
    
        url += latex_string
    
        try:
            # Make a POST request to CodeCogs API
            response = requests.post(url)
    
            # Check if the request was successful
            if response.status_code == 200:
                # Save the response content (SVG) to a file
                with open(output_file, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Error: HTTP status code {response.status_code}")
                print(response.text)
    
        except requests.RequestException as e:
            print(f"Error: {e}")


    def generate_svg_files(self):
        greek_letters = ['alpha', 'Alpha', 'beta', 'Beta', 'gamma', 'Gamma', 'delta', 'Delta', 'epsilon', 'Epsilon', 'zeta', 'Zeta', 'eta', 'Eta', 'theta', 'Theta', 'iota', 'Iota', 'kappa', 'Kappa', 'lambda', 'Lambda', 'mu', 'Mu', 'nu', 'Nu', 'xi', 'Xi', 'omicron', 'Omicron', 'pi', 'Pi', 'rho', 'Rho', 'sigma', 'Sigma', 'tau', 'Tau', 'upsilon', 'Upsilon', 'phi', 'Phi', 'chi', 'Chi', 'psi', 'Psi', 'omega', 'Omega']
        
        function_names = ['ln', 'log', 'exp', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh', 'Ln', 'Log', 'Exp', 'Sin', 'Cos', 'Tan', 'Arcsin', 'Arccos', 'Arctan', 'Sinh', 'Cosh', 'Tanh', 'Arcsinh', 'Arccosh', 'Arctanh']
        
        variable_names = {'lifetime':'tau', 'decayconstant':'Gamma', 'mass':'m', 'velocity':'v', 'temperature':'T', 'density':'rho', 'energy':'E', 'pressure':'P', 'entropy':'S', 'momentum':'p'}
        
        tex_names = {}
        
        print('generating latex names for input parameters...')
        
        for name in self.info_dict['input_names']:
            original_name = name
            for key, value in variable_names.items():
                if key in name:
                    name = name.replace(key, value)
            for greek_letter in greek_letters:
                if greek_letter in name:
                    name = name.replace(greek_letter, "\\" + greek_letter)
            for function_name in function_names:
                if function_name in name:
                    name = name.replace(function_name, "\\mathrm{" + function_name + "}")
            if '_' in name:
                name_parts = name.split('_')
                for i, part in enumerate(name_parts[1:]):
                    no_greek = True
                    for greek_letter in greek_letters:
                        if greek_letter in part:
                            no_greek = False
                            break
                    if no_greek:
                        name_parts[i+1] = '\\mathrm{ ' + name_parts[i+1] + r'}'
        
                new_name = name_parts[0]
                upper = ''
                for i, part in enumerate(name_parts[1:]):
                    if i == 0:
                        new_name += '_{' + part + '}'
                    else:
                        upper += part + "\\;"
                if upper:
                    new_name += '^{' + upper[:-2] + '}'
        
                name = new_name
        
        
            # check if a number is at the end of the name, and then add an underscore before it. It could be multidigit
            if name[-1].isdigit():
                for i in range(len(name)-1, -1, -1):
                    if name[i].isdigit():
                        continue
                    else:
                        name = name[:i+1] + '_' + name[i+1:]
                        break
        
            if 'log}10' in name:
                name = name.replace('log}10', 'log}_{10}')
        
            tex_names[original_name] = name
        
        
        for i, key in enumerate(tex_names):
            if key in self.overwrite_tex_names:
                tex_names[key] = overwrite_tex_names[key].replace(",","\\;")
            if i in self.rescale_idx:
                tex_names[key] = '10^{'+str(int(np.log10(1/self.rescale_factor_array[i])))+'}' + tex_names[key]
            self.render_latex_to_svg(tex_names[key], f"{self.output_dir}/{key}.svg")
            print(key, ' '*max(1, (10-len(key))),'-> ', tex_names[key])
        
        
        svg_dir = self.output_dir
        
        print('adjusting SVG heights...')
        # Find the largest height among all SVG files
        max_height, max_height_svg = self.find_largest_height(svg_dir)
        
        if max_height_svg:
            print(f"Largest height found: {max_height} in {max_height_svg}")
            
            # Adjust all SVG files to match the largest height
            for filename in os.listdir(svg_dir):
                if filename.endswith(".svg"):
                    svg_path = os.path.join(svg_dir, filename)
                    self.adjust_svg_height(svg_path, max_height)
                    
            print("SVG files adjusted successfully.")
        else:
            print("No SVG files found in the directory.")

    def write_meta_data(self):
        print('writing input/output metadata files...')
        with open(f'{self.output_dir}/input_names.txt', 'w') as f:
            for name, min_max_step in zip(self.info_dict['input_names'], self.input_ranges_rescaled):
                f.write(f"{name}, {min_max_step[0]}, {min_max_step[1]}, {min_max_step[2]}\n")
        
        output_spectra = {}
        for key, value in self.info_dict['interval'].items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if isinstance(value2, int):
                        value2 = [value2, value2]
                    output_spectra[f"{key}_{key2}"] = value2
            else:
                if isinstance(value, int):
                    value = [value, value]
                output_spectra[key] = value
        
        with open(f'{self.output_dir}/output_indices.txt', 'w') as f:
            for name, min_max in output_spectra.items():
                if type(min_max) is dict:
                    for new_name, new_min_max in min_max.items():
                        f.write(f"{name}_{new_name}, {new_min_max[0]}, {new_min_max[1]}\n")
                else:
                    f.write(f"{name}, {min_max[0]}, {min_max[1]}\n")
        
        with open(f'{self.output_dir}/x_values.txt', 'w') as f:
            if 'ell' in self.info_dict:
                f.write("Cl:\n")
                for ell in self.info_dict['ell']:
                    f.write(f"{ell}\n")
            if 'k_grid' in self.info_dict:
                f.write("Pk:\n")
                for k in self.info_dict['k_grid']:
                    f.write(f"{k}\n")
            if 'z_bg' in self.info_dict:
                f.write("Background:\n")
                for z in self.info_dict['z_bg']:
                    f.write(f"{z}\n")
            if 'z_th' in self.info_dict:
                f.write("Thermodynamics:\n")
                for z in self.info_dict['z_th']:
                    f.write(f"{z}\n")



    def get_bestfit(self):
        print('finding Planck best-fit parameters...')
        # Find best-fit
        Planck_highl_TTTEEE_lite = PlanckLitePy(data_directory='resources/planck_lite_py/data', year=2018, spectra='TTTEEE', use_low_ell_bins=False)
        def log_prob(theta):
            A_planck = 1.0
            cls = self.model.get_cls_interp(tf.constant([theta], dtype=tf.float32), 2508)
            T = 2.7255
            for key in cls.keys():
                if key not in ['pp', 'ell', 'tp', 'ep']:
                    cls[key] *= (T*1.e6)**2
                elif key in ['tp', 'ep']:
                    cls[key] *= (T*1.e6)
        
            loglike_highl = Planck_highl_TTTEEE_lite.loglike(cls['tt'].numpy()[0], cls['te'].numpy()[0], cls['ee'].numpy()[0], 2)
            loglike_highl += -0.5*((A_planck-1)/0.0025)**2
        
            return (loglike_highl)
        
        
        
        params = np.array(self.input_ranges)[:,:2]
        
        def rescaled_log_prob(x):
            x_min = params[:,0]
            x_max = params[:,1]
        
            theta = x*(x_max - x_min) + x_min
            return -log_prob(theta)
            
        
        
        midpoint = np.ones(len(params))/2
        rescaled_bounds = [(0,1)]*len(params)
        if 'tau_reio' in self.info_dict['input_names']:
            # get index of tau_reio
            tau_reio_idx = self.info_dict['input_names'].index('tau_reio')
            midpoint[tau_reio_idx] = 0.1
        
        res = minimize(rescaled_log_prob, midpoint, options={'xatol':1e-9, 'fatol':0.001}, bounds=rescaled_bounds, method='Nelder-Mead')
        unscaled_params = res.x*(params[:,1] - params[:,0]) + params[:,0]
        print('Planck lite best-fit likelihood:',res.fun)
        print('best-fit point:', unscaled_params)
        
        for i, fac in zip(self.rescale_idx, self.rescale_factor):
            unscaled_params[i] /= fac
        
        # export best-fit (unscaled) parameters to file
        with open(f'{self.output_dir}/best_fit.txt', 'w') as f:
            for name, value in zip(self.info_dict['input_names'], unscaled_params):
                f.write(f"{name}, {value}\n")

