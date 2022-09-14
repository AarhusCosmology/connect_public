import tensorflow as tf
import numpy as np
from source.custom_functions import ActivationFunctions

class Dense_model(tf.keras.Model):
    def __init__(self,
                 N_nodes,
                 num_parameters,
                 num_out,
                 activation='alsing',
                 normalizer = 0,
                 num_hidden_layers = 4,
                 output_info = 0):

        # Inherit from tf.keras.Model
        super(Dense_model, self).__init__()

        # Define attributes
        self.normalizer = normalizer
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = []


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
            self.hidden_layers.append(tf.keras.layers.Dense(N_nodes, activation=act_fun))
        # Create output layer
        self.output_layer = tf.keras.layers.Dense(num_out, activation='linear')

    def call(self, x, **kwargs):
        if self.normalizer:
            x = self.normalizer(x)
        x = self.input_layer(x)
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
        x = self.output_layer(x)
        return x
