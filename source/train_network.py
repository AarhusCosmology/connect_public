from source.custom_functions import LossFunctions
from source.architecture.sequential_models import Dense_model
import source.callbacks as cb
from scipy.interpolate import CubicSpline
import tensorflow as tf
import numpy as np
import pickle
import os
import sys

class Training():

    def __init__(self, param, CONNECT_PATH):
        self.param = param
        self.CONNECT_PATH = CONNECT_PATH
        
        path = os.path.join(CONNECT_PATH, f'data/{param.jobname}')
        if param.sampling == 'iterative':
            try:
                i = max([int(f.split('number_')[-1]) for f in os.listdir(path) if f.startswith('number')])
                self.load_path = os.path.join(path, f'number_{i}')
                self.param.N = sum(1 for line in open(os.path.join(self.load_path, 'model_params.txt'))) - 1
            except:
                self.load_path = os.path.join(path, f'N-{self.param.N}')
                self.param.N = sum(1 for line in open(os.path.join(self.load_path, 'model_params.txt'))) - 1
        else:
            self.load_path = os.path.join(path, f'N-{self.param.N}')
            self.param.N = sum(1 for line in open(os.path.join(self.load_path, 'model_params.txt'))) - 1

        self.N_train = int(np.floor(self.param.N*self.param.train_ratio))
        input_dim = len(self.param.parameters.keys())

        model_params          = []
        outputs               = []
        ell_weights           = []
        self.output_size      = {}
        self.output_normalize = {}
        self.output_interval  = {}
        output_size_iter      = 0
        self.output_normalize['method'] = self.param.normalization_method

        ### Load input ###
        with open(os.path.join(self.load_path, 'model_params.txt'), 'r') as f:
            for line in f:
                if line[0] == '#':
                    paramnames = line[2:].replace('\n','').split('\t')
                else:
                    line_vals = []
                    for name in list(self.param.parameters.keys()):
                        idx = paramnames.index(name)
                        line_vals.append(line.replace('\n','').split('\t')[idx])
                    line_vals = np.float32(line_vals)
                    model_params.append(line_vals)

        
        ### Load output data ###
        if len(self.param.output_Cl) > 0:
            if self.output_normalize['method'] in ['factor','log']:
                self.output_normalize['Cl'] = {}
            self.output_interval['Cl']  = {}
        for output in self.param.output_Cl:
            out = []
            with open(os.path.join(self.load_path, f'Cl_{output}.txt'), 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        ell = np.int32(line.replace('\n','').split('\t'))
                        if not hasattr(self, 'output_ell'):
                            self.output_ell = ell
                    else:
                        if i == 1:
                            if self.output_normalize['method'] == 'factor':
                                self.output_normalize['Cl'][output] = 1./max(
                                    np.float32(line.replace('\n','').split('\t')))
                            ell_weights.append(ell)
                        if self.output_normalize['method'] == 'factor':
                            out.append(self.output_normalize['Cl'][output]*np.float32(
                                line.replace('\n','').split('\t')))
                        else:
                            out.append(np.float32(line.replace('\n','').split('\t')))
            if self.output_normalize['method'] == 'log':
                self.output_normalize['Cl'][output] = []
                if np.min(out) <= 0:
                    offset = -(np.min(out)*1.01)
                    self.output_normalize['Cl'][output].append(offset)
                    out = np.array(out) + offset
                else:
                    self.output_normalize['Cl'][output].append(0)
                out = np.log(out)
                if np.min(out) <= 0:
                    offset = -(np.min(out)*1.01)
                    self.output_normalize['Cl'][output].append(offset)
                    out = np.array(out) + offset
                else:
                    self.output_normalize['Cl'][output].append(0)
                out = np.log(out)
            outputs.append(out)
            self.output_interval['Cl'][output] = [output_size_iter, output_size_iter + len(out[0])]
            output_size_iter += len(out[0])


        if len(self.param.output_bg) > 0:
            if self.output_normalize['method'] in ['factor','log']:
                self.output_normalize['bg'] = {}
            self.output_interval['bg']  = {}
        for output in self.param.output_bg:
            out = []
            with open(os.path.join(self.load_path, f'bg_{output}.txt'), 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        if self.output_normalize['method'] == 'factor':
                            self.output_normalize['bg'][output] = 1./max(
                                np.float32(line.replace('\n','').split('\t')))
                        ell_weights.append(10**4*np.ones(line.count('\t')+1))
                    if self.output_normalize['method'] == 'factor':
                        out.append(self.output_normalize['bg'][output]*np.float32(
                            line.replace('\n','').split('\t')))
                    else:
                        out.append(np.float32(line.replace('\n','').split('\t')))
            if self.output_normalize['method'] == 'log':
                self.output_normalize['bg'][output] = []
                if np.min(out) <= 0:
                    offset = -(np.min(out)*1.01)
                    self.output_normalize['bg'][output].append(offset)
                    out = np.array(out) + offset
                else:
                    self.output_normalize['bg'][output].append(0)
                out = np.log(out)
                if np.min(out) <= 0:
                    offset = -(np.min(out)*1.01)
                    self.output_normalize['bg'][output].append(offset)
                    out = np.array(out) + offset
                else:
                    self.output_normalize['bg'][output].append(0)
                out = np.log(out)
            outputs.append(np.array(out))
            self.output_interval['bg'][output] = [output_size_iter, output_size_iter + len(out[0])]
            output_size_iter += len(out[0])


        if len(self.param.output_th) > 0:
            if self.output_normalize['method'] in ['factor','log']:
                self.output_normalize['th'] = {}
            self.output_interval['th']  = {}
        for output in self.param.output_th:
            out = []
            with open(os.path.join(self.load_path, f'th_{output}.txt'), 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        if self.output_normalize['method'] == 'factor':
                            self.output_normalize['th'][output] = 1./max(
                                np.float32(line.replace('\n','').split('\t')))
                        ell_weights.append(10**4*np.ones(line.count('\t')+1))
                    if self.output_normalize['method'] == 'factor':
                        out.append(self.output_normalize['th'][output]*np.float32(
                            line.replace('\n','').split('\t')))
                    else:
                        out.append(np.float32(line.replace('\n','').split('\t')))
            if self.output_normalize['method'] == 'log':
                self.output_normalize['th'][output] = []
                if np.min(out) <= 0:
                    offset = -(np.min(out)*1.01)
                    self.output_normalize['th'][output].append(offset)
                    out = np.array(out) + offset
                else:
                    self.output_normalize['th'][output].append(0)
                out = np.log(out)
                if np.min(out) <= 0:
                    offset = -(np.min(out)*1.01)
                    self.output_normalize['th'][output].append(offset)
                    out = np.array(out) + offset
                else:
                    self.output_normalize['th'][output].append(0)
                out = np.log(out)
            outputs.append(np.array(out))
            self.output_interval['th'][output] = [output_size_iter, output_size_iter + len(out[0])]
            output_size_iter += len(out[0])


        if len(self.param.output_derived) > 0:
            self.output_interval['derived']  = {}
            if self.output_normalize['method'] == 'log':
                self.output_normalize['derived'] = {}
            output_derived = []
            with open(os.path.join(self.load_path, 'derived.txt'), 'r') as f:
                lines = list(f)
                derived_names = lines[0].replace('# ','').replace('\n','').split('\t')
                for i, line in enumerate(lines[1:]):
                    out = []
                    line_vals = []
                    for name in self.param.output_derived:
                        idx = derived_names.index(name)
                        line_vals.append(line.replace('\n','').split('\t')[idx])
                    line_vals = np.float32(line_vals)
                    if i == 0:
                        if self.output_normalize['method'] == 'factor':
                            self.output_normalize['derived'] = dict(zip(self.param.output_derived, 
                                                                        1./line_vals))
                        ell_weights.append(10**4*np.ones(line.count('\t')+1))
                    for j, output in enumerate(self.param.output_derived):
                        if self.output_normalize['method'] == 'factor':
                            out.append(np.array(self.output_normalize['derived'][output]*line_vals[j]))
                        else:
                            out.append(np.array(line_vals[j]))
                        if i == 0:
                            self.output_interval['derived'][output] = output_size_iter
                            output_size_iter += 1
                    output_derived.append(out)
            output_derived = np.array(output_derived)
            if self.output_normalize['method'] == 'log':
                for i, (output, out) in enumerate(zip(self.param.output_derived, output_derived.T)):
                    self.output_normalize['derived'][output] = []
                    if np.min(out) <= 0:
                        offset = -(np.min(out)*1.01)
                        self.output_normalize['derived'][output].append(offset)
                        out = np.array(out) + offset
                    else:
                        self.output_normalize['derived'][output].append(0)
                    out = np.log(out)
                    if np.min(out) <= 0:
                        offset = -(np.min(out)*1.01)
                        self.output_normalize['derived'][output].append(offset)
                        out = np.array(out) + offset
                    else:
                        self.output_normalize['derived'][output].append(0)
                    output_derived.T[i] = np.log(out)
            outputs.append(output_derived)
        
        self.output_dim = output_size_iter


        outputs = np.concatenate(outputs, axis=1)


        if self.output_normalize['method'] == 'standardization':
            def normalize_with_moments(x, epsilon=1e-20):
                delta = 1e-3
                mean = np.mean(x)
                var = np.var(x)
                variance = var
                x_normed = (x - mean) / np.sqrt(var + epsilon) # epsilon to avoid dividing by zero
                var = np.var(x_normed)
                while var > 1+delta or var < 1-delta:
                    x_normed = x_normed/np.sqrt(var + epsilon)
                    variance = (variance + epsilon)*(var + epsilon)
                    var = np.var(x_normed)
                return x_normed, mean, variance


            out_nodes = outputs.T
            out_mean = []
            out_var = []
            outputs = []

            for n in out_nodes:
                data, mean, var = normalize_with_moments(n)
                outputs.append(data)
                out_mean.append(mean)
                out_var.append(var)
            
            outputs = np.array(outputs).T
            self.output_normalize['mean'] = out_mean
            self.output_normalize['variance'] = out_var

            out_nodes =[]
            out_mean = []
            out_var = []
            del out_nodes
            del out_mean
            del out_var

        elif self.output_normalize['method'] == 'min-max':
            def normalize_with_minmax(x, epsilon=1e-20):
                x_min = min(x)
                x_max = max(x) - epsilon
                x_normed = (x - x_min) / (x_max - x_min) # epsilon to avoid dividing by zero
                return x_normed, x_min, x_max


            out_nodes = outputs.T
            out_x_min = []
            out_x_max = []
            outputs = []

            for n in out_nodes:
                data, x_min, x_max = normalize_with_minmax(n)
                outputs.append(data)
                out_x_min.append(x_min)
                out_x_max.append(x_max)

            outputs = np.array(outputs).T
            self.output_normalize['x_min'] = out_x_min
            self.output_normalize['x_max'] = out_x_max

            out_nodes =[]
            out_x_min = []
            out_x_max = []
            del out_nodes
            del out_x_min
            del out_x_max


        ### Prepare output info dictionary for model ###
        self.output_info = {'input_names':    paramnames,
                            'normalize':      self.output_normalize,
                            'sizes':          self.output_size,
                            'interval':       self.output_interval}
        if len(self.param.output_Cl) > 0:
            self.output_info['ell']       = self.output_ell
            self.output_info['output_Cl'] = self.param.output_Cl
        if len(self.param.output_Pk) > 0:
            self.output_info['output_Pk'] = self.param.output_Pk
        if len(self.param.output_bg) > 0:
            self.output_info['output_bg'] = self.param.output_bg
        if len(self.param.output_th) > 0:
            self.output_info['output_th'] = self.param.output_th
        if len(self.param.output_derived) > 0:
            self.output_info['output_derived'] = self.param.output_derived


        ### Convert data to tensors ###
        model_params_tf = tf.constant(model_params)
        model_params = []
        del model_params
        output_tf = tf.constant(outputs)
        outputs = []
        del outputs
        dataset = tf.data.Dataset.from_tensor_slices((model_params_tf, output_tf))
        model_params_tf = []
        output_tf = []
        del model_params_tf
        del output_tf


        ### Shuffle dataset and split in training, test and validation ###
        dataset = dataset.shuffle(buffer_size = 10 * self.param.batchsize)
        self.train_dataset = dataset.take(self.N_train)
        self.test_dataset  = dataset.skip(self.N_train).batch(self.param.batchsize)
        dataset = []
        del dataset
        self.val_dataset = self.train_dataset.take(int(
            self.param.val_ratio*self.param.N)).batch(self.param.batchsize)
        self.train_dataset = self.train_dataset.skip(int(
            self.param.val_ratio*self.param.N)).batch(self.param.batchsize)


        ### Define strategy for training on multiple GPUs ###
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.training_strategy = tf.distribute.MirroredStrategy(
                devices=["/gpu:0","/gpu:1"],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()).scope()
        else:
            class NoneStrategy(object):
                def __init__(self):
                    pass
                def __enter__(self):
                    pass
                def __exit__(self,a,b,c):
                    pass
            self.training_strategy = NoneStrategy()


        ### Set loss function ###
        ell_weights = np.concatenate(ell_weights)
        Loss = LossFunctions(ell_weights) 
        if callable(getattr(Loss, self.param.loss_function, None)):
            _locals = {}
            exec('loss_fun = Loss.'+self.param.loss_function, locals(), _locals)
            self.loss_fun = _locals['loss_fun']
        else:
            self.loss_fun = self.param.loss_function


        ### Normalize input data ###
        self.normalizer = tf.keras.layers.experimental.preprocessing.Normalization(input_dim=input_dim)
        self.normalizer.adapt(np.concatenate([x for x, y in self.train_dataset], axis=0))


    def train_model(self, epochs=None, output_file=None):
        if self.param.output_activation:
            out_act = self.output_info
        else:
            out_act = 0

        if epochs != None:
            self.param.epochs = epochs

        adam = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-4,
                                        amsgrad=True,
                                        name='Adam')

        self.training_success = False
        training_try_number = 0

        if output_file != None:
            stdout_backup = sys.stdout
            sys.stdout = open(output_file, 'w')

        while not self.training_success:
            # beginning of while loop
            training_try_number += 1

            with self.training_strategy:
                self.model = Dense_model(self.param.N_nodes, 
                                         len(self.param.parameters.keys()),
                                         self.output_dim,
                                         normalizer = self.normalizer,
                                         activation = self.param.activation_function,
                                         num_hidden_layers = self.param.N_hidden_layers,
                                         output_info = out_act)

                self.model.compile(optimizer=adam,
                                   loss=self.loss_fun,
                                   metrics=['accuracy'])

            self.history = self.model.fit(self.train_dataset,
                                          epochs=self.param.epochs,
                                          validation_data=self.val_dataset,
                                          callbacks=[cb.CheckNaN(self, success_param_name='training_success')])

            if training_try_number == 3:
                raise RuntimeError('Training failed 3 times due to NaN loss. Try adjusting the hyperparameters of the optimizer')
            # end of while loop

        if output_file != None:
            sys.stdout.close()
            sys.stdout = stdout_backup
            
        
        test_loss, test_acc = self.model.evaluate(self.test_dataset, verbose=2)
        print('\nTest accuracy:', test_acc)

    def save_model(self):
        if not getattr(self, 'model', None):
            errmsg = "the class has no attribute 'model'. You should run the 'train_model' method before using the 'save_model' method"
            raise AttributeError(errmsg)
        if self.param.save_name:
            self.save_path = os.path.join(self.CONNECT_PATH,
                                          'trained_models',
                                          self.param.save_name)
        else:
            save_name  = f'{self.param.jobname}'
            save_name += f'_N{self.param.N}'
            save_name += f'_bs{self.param.batchsize}'
            save_name += f'_e{self.param.epochs}'
            self.save_path = os.path.join(self.CONNECT_PATH,
                                          f'trained_models',
                                          save_name)

        if not self.param.overwrite_model:
            M = 1
            if os.path.isdir(self.save_path):
                while os.path.isdir(self.save_path + f'_{M}'):
                    M += 1
                self.save_path += f'_{M}'
        self.model.save(self.save_path)

        with open(os.path.join(self.save_path, 'output_info.pkl'),'wb') as f:
            pickle.dump(self.output_info, f)

        os.system(f"cp {self.param.param_file} {os.path.join(self.save_path, 'log.param')}")
        with open(os.path.join(self.save_path, 'log.param'), 'a+') as f:
            jobname_specified = False
            for line in f:
                if line.startswith('jobname'):
                    jobname_specified = True
            if not jobname_specified:
                f.write(f"\njobname = '{self.param.jobname}'")

    def save_history(self):
        if not getattr(self, 'save_path', None):
            errmsg = "the class has no attribute 'save_path'. You should run the 'save_model' method before using the 'save_history' method"
            raise AttributeError(errmsg)
        acc  = self.history.history['val_accuracy']
        loss = self.history.history['val_loss']

        with open(os.path.join(self.save_path, 'val_accuracy.pkl'),'wb') as f:
            pickle.dump(acc, f)

        with open(os.path.join(self.save_path, 'val_loss.pkl'),'wb') as f:
            pickle.dump(loss, f)

    def save_test_data(self):
        if not getattr(self, 'save_path', None):
            errmsg = "the class has no attribute 'save_path'. You should run the 'save_model' method before using the 'save_history' method"
            raise AttributeError(errmsg)
        
        test_mp, test_out = tuple(zip(*self.test_dataset))

        try:
            dim1 = self.param.N - self.N_train
            mp_dim2 = len(test_mp[0][0])
            cl_dim2 = len(test_out[0][0])
            test_model_params = tf.reshape(test_mp, (dim1, mp_dim2)).numpy()
            test_output = tf.reshape(test_out, (dim1, cl_dim2)).numpy()
        except:
            test_model_params = []
            test_output      = []
            for tmp, tout in zip(test_mp, test_out):
                for tmp_i, tout_i in zip(tmp, tout):
                    test_model_params.append(tmp_i.numpy())
                    test_output.append(tout_i.numpy())
            test_model_params = np.array(test_model_params)
            test_output      = np.array(test_output)

        test_data = [test_model_params, test_output]
        
        with open(os.path.join(self.save_path, 'test_data.pkl'),'wb') as f:
            pickle.dump(test_data, f)
