import tensorflow as tf
import numpy as np

class CheckNaN(tf.keras.callbacks.Callback):
    def __init__(self, Training_instance, success_param_name='training_success'):
        self.TI = Training_instance
        self.success_param_name= success_param_name
        exec(f'self.TI.{self.success_param_name} = True')
    def on_epoch_end(self, epoch, logs=None):
        if not np.isfinite(logs['loss']):
            exec(f'self.TI.{self.success_param_name} = False')
            self.model.stop_training = True
    def on_train_end(self, logs=None):
        if not eval(f'self.TI.{self.success_param_name}'):
            print('\n\nTraining has been stopped due to NaN encounter and will restart now\n\n')
