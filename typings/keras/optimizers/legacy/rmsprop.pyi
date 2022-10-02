"""
This type stub file was generated by pyright.
"""

from keras.optimizers.optimizer_v2 import rmsprop
from tensorflow.python.util.tf_export import keras_export

"""Legacy RMSprop optimizer implementation."""
@keras_export("keras.optimizers.legacy.RMSprop")
class RMSprop(rmsprop.RMSprop):
    ...


