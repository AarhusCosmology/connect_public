# Custom loss and activation functions
import tensorflow as tf
import tf_keras
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
    def __init__(self):
        self.alsing = Alsing()
        self.nolu = NoLU()

class Alsing(tf_keras.layers.Layer):
    """
    Custom activation function based on the Alsing et al. (2019) paper.
    This activation function is defined as:
        Alsing(x) = (gamma + (1 - gamma) / (1 + exp(-beta * x))) * x
    where beta and gamma are trainable parameters.
    """
    def build(self, input_shape):
        self.beta = self.add_weight(
            name="beta",
            shape=(input_shape[-1],),
            initializer=tf_keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=self.dtype
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer=tf_keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=self.dtype
        )

    def call(self, x):
        exp_term = tf.keras.ops.exp(-tf.keras.ops.multiply(self.beta, x))
        inverse_one_plus_exp = tf.keras.ops.divide(1, 1 + exp_term)
        mult_elementwise = tf.keras.ops.multiply(inverse_one_plus_exp, 1 - self.gamma)
        return tf.keras.ops.multiply(self.gamma + mult_elementwise, x)


class NoLU(tf_keras.layers.Layer):
    """
    Normalised Linear Unit (NoLU) activation function.
    (see arXiv:???????)

    This activation function is defined as:
        NoLU(x) = slope * x

        where slope is a function of the input x and two trainable parameters beta and gamma:
        slope = (1 - S) * gamma + S,
        S = 3 * t^2 - 2 * t^3, t = eps + ((1 - 2 * eps) / 2) * (r + 1), r = x / sqrt(x^2 + beta^2)

    This is essentially a smoothed generalization of the ReLU activation function, where the slope of the function is controlled by the parameters beta and gamma. The parameter beta controls the "sharpness" of the transition from negative to positive values, while gamma controls the slope of the function for positive values. 
    """
    def build(self, input_shape):
        self.beta = self.add_weight(
            name="beta",
            shape=(input_shape[-1],),
            initializer=HalfNormalInitializer(stddev=1.0),
            trainable=True,
            constraint=ClipConstraint(min_value=1e-4, max_value=None),
            dtype=self.dtype
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer=HalfNormalInitializer(stddev=0.1),
            trainable=True,
            constraint=ClipConstraint(min_value=0, max_value=None),
            dtype=self.dtype
        )

    def call(self, x):
        eps = 1.0e-3
        r = tf.keras.ops.divide(x, tf.keras.ops.sqrt(tf.keras.ops.square(x) + tf.keras.ops.square(self.beta)))
        t = eps + ((1.0 - 2.0 * eps) * 0.5) * (r + 1.0)
        S = 3.0 * tf.keras.ops.square(t) - 2.0 * tf.keras.ops.power(t, 3)
        slope = (1.0 - S) * self.gamma + S
        return slope * x

class ClipConstraint(tf_keras.constraints.Constraint):
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(
            w,
            clip_value_min=self.min_value if self.min_value is not None else -float("inf"),
            clip_value_max=self.max_value if self.max_value is not None else float("inf")
        )

    def get_config(self):
        return {
            "min_value": self.min_value,
            "max_value": self.max_value
        }

class HalfNormalInitializer(tf_keras.initializers.Initializer):
    def __init__(self, stddev=1.0):
        self.stddev = stddev

    def __call__(self, shape, dtype=None):
        return tf.abs(
            tf.random.normal(
                shape,
                mean=0.0,
                stddev=self.stddev,
                dtype=dtype
            )
        )

    def get_config(self):
        return {
            "stddev": self.stddev
        }
