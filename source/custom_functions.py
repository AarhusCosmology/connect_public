# Custom loss and activation functions
import tensorflow as tf
import numpy as np

class LossFunctions():

    def __init__(self, ell):
        self.ell = ell

    def cosmic_variance(self, y_actual, y_pred):
        Var  = tf.constant(2/(2 * self.ell + 1), dtype=tf.float32)
        loss = tf.divide(tf.square(y_actual-y_pred),Var)
        return loss

    def minkowski(self, y_actual, y_pred):
        Var  = tf.constant(2/(2 * self.ell + 1), dtype=tf.float32)
        loss = tf.divide(tf.pow(tf.abs(y_actual-y_pred), 1.5),Var)
        return loss



class ActivationFunctions():

    def __init__(self, N_nodes):
        # Initialise trainable parameters for custom activation function
        alpha_ini = tf.initializers.GlorotNormal()
        beta_ini  = tf.initializers.GlorotNormal()
        gamma_ini = tf.initializers.GlorotNormal()

        self.alpha = tf.Variable(
            initial_value=alpha_ini(shape=(N_nodes,), dtype="float32"),
            trainable=True,
        )
        self.beta  = tf.Variable(
            initial_value=beta_ini(shape=(N_nodes,), dtype="float32"),
            trainable=True,
        )
        self.gamma = tf.Variable(
            initial_value=gamma_ini(shape=(N_nodes,), dtype="float32"),
            trainable=True,
        )

        self.params = {'alsing' : ['beta', 'gamma'],
                       'nygaard': ['alpha','beta','gamma']}

    def alsing(self, x):
        # Custom activation function - see arXiv:1911.11778
        exp_term = tf.math.exp(-tf.math.multiply(self.beta,x))
        inverse_one_plus_exp = tf.math.divide(1,1+exp_term)
        mult_elementwise = tf.math.multiply(inverse_one_plus_exp,1-self.gamma)
        return tf.math.multiply(self.gamma + mult_elementwise, x)

    def nygaard(self, x):
        # Custom activation function - see arXiv:???????
        exp_term = tf.math.exp(-tf.math.multiply(self.beta,x))
        inverse_one_plus_exp = tf.math.divide(1,1+exp_term)
        mult_elementwise = tf.math.multiply(inverse_one_plus_exp,self.gamma-self.alpha)
        return tf.math.multiply(self.alpha + mult_elementwise, x)
