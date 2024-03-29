"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import keras_export

"""ResNet v2 models for Keras.

Reference:
  - [Identity Mappings in Deep Residual Networks]
    (https://arxiv.org/abs/1603.05027) (CVPR 2016)
"""
@keras_export("keras.applications.resnet_v2.ResNet50V2", "keras.applications.ResNet50V2")
def ResNet50V2(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    """Instantiates the ResNet50V2 architecture."""
    ...

@keras_export("keras.applications.resnet_v2.ResNet101V2", "keras.applications.ResNet101V2")
def ResNet101V2(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    """Instantiates the ResNet101V2 architecture."""
    ...

@keras_export("keras.applications.resnet_v2.ResNet152V2", "keras.applications.ResNet152V2")
def ResNet152V2(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    """Instantiates the ResNet152V2 architecture."""
    ...

@keras_export("keras.applications.resnet_v2.preprocess_input")
def preprocess_input(x, data_format=...):
    ...

@keras_export("keras.applications.resnet_v2.decode_predictions")
def decode_predictions(preds, top=...): # -> list[Unknown]:
    ...

DOC = ...
