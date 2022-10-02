"""
This type stub file was generated by pyright.
"""

import sys as _sys
from keras.engine.sequential import Sequential
from keras.engine.training import Model
from keras.models.cloning import clone_model
from keras.premade_models.linear import LinearModel
from keras.premade_models.wide_deep import WideDeepModel
from keras.saving.model_config import model_from_config, model_from_json, model_from_yaml
from keras.saving.save import load_model, save_model
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Keras models API.
"""
if notisinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
