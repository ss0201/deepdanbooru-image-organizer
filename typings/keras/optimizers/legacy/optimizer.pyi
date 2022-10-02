"""
This type stub file was generated by pyright.
"""

from keras.optimizers.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export

"""Legacy Adam optimizer implementation."""
@keras_export("keras.optimizers.legacy.Optimizer")
class Optimizer(optimizer_v2.OptimizerV2):
    ...


