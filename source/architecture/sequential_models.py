import tensorflow as tf
import numpy as np
import tf_keras

from ..custom_functions import ActivationFunctions
from ..Splines import Spline

class Dense_model(tf_keras.Model):
    def __init__(self,
                 N_nodes,
                 num_out,
                 activation='alsing',
                 input_normaliser=lambda x: x,
                 output_unnormaliser=lambda x: x,
                 num_hidden_layers=4,
                 dropout=False,
                 batch_norm=False,
                 ell_computed=[2,3,4,5]):

        # Inherit from tf_keras.Model
        super(Dense_model, self).__init__()

        # Define attributes
        self.N_nodes = N_nodes
        self.num_out = num_out
        self.activation_name = activation

        self.dropout_bool = dropout
        self.batch_norm_bool = batch_norm
        self._train_normalise = True
        self.raw_info = '{}'
        self.info_dict = {}
        self.cached_input = []
        self.cached_output = []

        self.input_normaliser = input_normaliser
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = []
        self.batch_norm_layers = []
        self.dropout_layers = []

        self.ell_computed = tf.constant(ell_computed, dtype=tf.float32)
        self.ell = tf.linspace(2.,10000.,9999)
        self.Cl_spline = Spline(self.ell_computed, self.ell)


        ### Define architecture ###

        # Create hidden layers
        self.activations = []
        for i in range(self.num_hidden_layers):
            # Set the chosen activation function
            Act = ActivationFunctions()
            act_fun_custom = getattr(Act, self.activation_name, None)
            if callable(act_fun_custom):
                self.activations.append(act_fun_custom)
                act_fun = 'linear'
            else:
                self.activations.append(lambda x: x)
                act_fun = self.activation_name
            # Create hidden layer
            self.hidden_layers.append(tf_keras.layers.Dense(self.N_nodes, activation=act_fun, kernel_initializer='glorot_normal', bias_initializer='glorot_normal', kernel_constraint=tf_keras.constraints.MaxNorm(5)))
            if self.dropout_bool:
                self.dropout_layers.append(tf_keras.layers.Dropout(0.2))
            if i%5 == 4 and self.batch_norm_bool:
                self.batch_norm_layers.append(tf_keras.layers.BatchNormalization())
        # Create output layer
        self.output_layer = tf_keras.layers.Dense(self.num_out, activation='linear', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')
        self.unnormalise_layer = UnnormaliseOutputLayer(output_unnormaliser)


    def call(self, x, **kwargs):
        x = self.input_normaliser(x)
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
            x = self.activations[i](x)
            if self.dropout_bool:
                x = self.dropout_layers[i](x)
            if i%5 == 4 and self.batch_norm_bool:
                x = self.batch_norm_layers[int(i/5)](x)
        x = self.output_layer(x)
        x = self.unnormalise_layer(x)
        return x

    def build(self, input_shape):
        super(Dense_model, self).build(input_shape)
        self.built = True

    @tf.function(input_signature=[])
    def get_raw_info(self):
        return self.get_config()['raw_info']

    @property
    def train_normalise(self):
        return self._train_normalise

    @train_normalise.setter
    def train_normalise(self, val):
        for layer in self.layers:
            layer.train_normalise = val
        self._train_normalise = val

    def convert_types_to_tf(self, d):
        out_dict = {}
        for key, val in d.items():
            if type(val) == dict:
                out_dict[key] = self.convert_types_to_tf(val)
            else:
                out_dict[key] = tf.Variable(val, trainable=False)
        return out_dict

    def get_config(self):
        config = super().get_config()

        config.update({
            "raw_info": self.raw_info,
        })

        return config

    @tf.function(input_signature=[tf.TensorSpec([None,None], dtype=tf.float32)])
    def get_cls(self, params):
        if 'Cl' in self.info_dict['interval']:
            output = self(params)
            cls = {'ell': self.info_dict['ell']}
            for key, lims in self.info_dict['interval']['Cl'].items():
                cls[key] = output[:,lims[0]:lims[1]]
            return cls
        else:
            print('No Cls were emulated')

    @tf.function(input_signature=[tf.TensorSpec([None,None], dtype=tf.float32),
                                  tf.TensorSpec(None, dtype=tf.int32)])
    def get_cls_interp(self, params, l_max):
        if 'Cl' in self.info_dict['interval']:
            if l_max > 10000:
                print("l_max can maximally be 10,000.")
                l_max = 10000
            output = self(params)
            cls = {'ell': self.ell[:l_max-1]}
            for key, lims in self.info_dict['interval']['Cl'].items():
                cls[key] = self.Cl_spline.do_spline(output[:,lims[0]:lims[1]])[:,:l_max-1]
            return cls
        else:
            print('No Cls were emulated')

    @tf.function(input_signature=[tf.TensorSpec([None,None], dtype=tf.float32)])
    def get_Pks(self, params):
        if 'Pk' in self.info_dict['interval']:
            output = self(params)
            pks = {'k_grid': self.info_dict['k_grid']}
            for key, z_dict in self.info_dict['interval']['Pk'].items():
                pks[key] = {}
                for key2, lims in z_dict.items():
                    pks[key][key2] = output[:,lims[0]:lims[1]]
            return pks
        else:
            print('No Pks were emulated')

    @tf.function(input_signature=[tf.TensorSpec([None,None], dtype=tf.float32)])
    def get_derived(self, params=None):
        if 'derived' in self.info_dict['interval']:
            output = self(params)
            derived = {}
            for key, lims in self.info_dict['interval']['derived'].items():
                derived[key] = output[:,lims]
            return derived
        else:
            print('No derived parameters were emulated')


class UnnormaliseOutputLayer(tf_keras.layers.Layer):

    def __init__(self, output_unnormalise, **kwargs):
        super(UnnormaliseOutputLayer, self).__init__(**kwargs)
        self.unnormalise = output_unnormalise
        self.train_normalise = True

    def call(self, inputs, training=None):
        if self.train_normalise:
            return self.call_train(inputs)
        else:
            return self.call_eval(inputs)

    def call_train(self, inputs):
        return inputs

    def call_eval(self, inputs):
        return self.unnormalise(inputs)
