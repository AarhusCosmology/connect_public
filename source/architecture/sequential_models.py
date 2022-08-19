import tensorflow as tf
import numpy as np
from source.custom_functions import ActivationFunctions

class Dense_model(tf.keras.Model, ActivationFunctions):
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
        ActivationFunctions.__init__(self, N_nodes)
        
        # Define attributes
        self.normalizer = normalizer
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = []

        # Set the chosen activation function
        if callable(getattr(self, activation, None)):
            _locals = {}
            exec('act_fun = self.' + activation, locals(), _locals)
            act_fun = _locals['act_fun']
        else:
            act_fun = activation

        if not isinstance(output_info, int):
            if any("Cl" in s for s in output_info['names']):
                relu     = []
                size_tot = 0
                for output in output_info['names']:
                    size_tot += output_info['sizes'][output]
                    interval = output_info['interval'][output]
                    indices = list(range(interval[0],interval[1]))
                    if output[:2] == 'Cl':
                        if output[4] == output[3]:
                            relu += indices
                mask = np.ones(size_tot)
                mask[relu] = 0
                mask = tf.constant(mask, dtype=tf.float32)
                if len(relu) > 0:
                    def relu_linear(x):
                        relu_idx = tf.cast(tf.less_equal(0.0, x), dtype=tf.float32)
                        x = tf.multiply(x,mask) + tf.multiply(tf.abs(relu_idx*x),(1-mask))
                        return x
                    act_fun_out = relu_linear
                else:
                    act_fun_out = 'linear'
            else:
                act_fun_out = 'linear'
        else:
            act_fun_out = 'linear'

        # Define architecture
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_parameters,))
        for i in range(self.num_hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(N_nodes, activation=act_fun))
        self.output_layer = tf.keras.layers.Dense(num_out, activation=act_fun_out)

    def call(self, x, **kwargs):
        if self.normalizer:
            x = self.normalizer(x)
        x = self.input_layer(x)
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
        x = self.output_layer(x)
        return x
