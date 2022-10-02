"""
This type stub file was generated by pyright.
"""

import sys as _sys
from keras.applications.resnet_v2 import ResNet101V2, ResNet152V2, ResNet50V2, decode_predictions, preprocess_input
from tensorflow.python.util import module_wrapper as _module_wrapper

"""ResNet v2 models for Keras.

Reference:
  - [Identity Mappings in Deep Residual Networks]
    (https://arxiv.org/abs/1603.05027) (CVPR 2016)

"""
if notisinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...