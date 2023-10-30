import tensorflow as tf
import numpy as np
from copy import deepcopy

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


class KeepBestEpoch(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_weights = None
        self.best_loss = None
        self.best_epoch = 1
    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        if not np.isfinite(loss):
            self.model.stop_training = True
        elif self.best_loss == None or self.best_loss > loss:
            self.best_loss = loss
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        print(f'The epoch with lowest validation loss was epoch {self.best_epoch}')
        print(f'with a loss of {self.best_loss} and this has been saved.')


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=10):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialise the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
