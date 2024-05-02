import tensorflow as tf
import numpy as np

from ..custom_functions import ActivationFunctions

class Dense_model(tf.keras.Model):
    def __init__(self,
                 N_nodes,
                 num_parameters,
                 num_out,
                 activation='alsing',
                 input_normaliser=lambda x: x,
                 output_unnormaliser=lambda x: x,
                 num_hidden_layers=4,
                 dropout=False,
                 batch_norm=False):

        # Inherit from tf.keras.Model
        super(Dense_model, self).__init__()

        # Define attributes
        self.dropout_bool = dropout
        self.batch_norm_bool = batch_norm
        self._train_normalise = True
        self.raw_info = '{}'
        self.info_dict = {}

        self.input_normaliser = input_normaliser
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = []
        self.batch_norm_layers = []
        self.dropout_layers = []


        ### Define architecture ###

        # Create input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_parameters,))
        # Create hidden layers
        self.act_params = []
        for i in range(self.num_hidden_layers):
            # Set the chosen activation function
            Act = ActivationFunctions(N_nodes)
            if callable(getattr(Act, activation, None)):
                _locals = {}
                exec('act_fun = Act.' + activation, locals(), _locals)
                act_fun = _locals['act_fun']
                for name in (Act.params[activation]):
                    self.act_params.append(eval(f'Act.{name}'))
            else:
                act_fun = activation
            # Create hidden layer
            self.hidden_layers.append(tf.keras.layers.Dense(N_nodes, activation=act_fun, kernel_initializer='glorot_normal', bias_initializer='glorot_normal', kernel_constraint=tf.keras.constraints.MaxNorm(5)))
            if self.dropout_bool:
                self.dropout_layers.append(tf.keras.layers.Dropout(0.2))
            if i%5 == 4 and self.batch_norm_bool:
                self.batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        # Create output layer
        self.output_layer = tf.keras.layers.Dense(num_out, activation='linear', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')
        self.unnormalise_layer = UnnormaliseOutputLayer(output_unnormaliser)


    def call(self, x, **kwargs):
        
        x = self.input_normaliser(x)
        x = self.input_layer(x)
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
            if self.dropout_bool:
                x = self.dropout_layers[i](x)
            if i%5 == 4 and self.batch_norm_bool:
                x = self.batch_norm_layers[int(i/5)](x)
        x = self.output_layer(x)
        x = self.unnormalise_layer(x)
        return x


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
        config['raw_info'] = self.raw_info
        return config


class UnnormaliseOutputLayer(tf.keras.layers.Layer):

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
