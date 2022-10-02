"""
This type stub file was generated by pyright.
"""

import sys as _sys
from keras.initializers import deserialize, get, serialize
from keras.initializers.initializers_v1 import HeNormal as he_normal, HeUniform as he_uniform, LecunNormal as lecun_normal, LecunUniform as lecun_uniform, RandomNormal, RandomUniform, TruncatedNormal, _v1_constant_initializer as Constant, _v1_glorot_normal_initializer as glorot_normal, _v1_glorot_uniform_initializer as glorot_uniform, _v1_identity as Identity, _v1_ones_initializer as Ones, _v1_orthogonal_initializer as Orthogonal, _v1_variance_scaling_initializer as VarianceScaling, _v1_zeros_initializer as Zeros
from keras.initializers.initializers_v2 import Initializer
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Keras initializer serialization / deserialization.
"""
if notisinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...