"""
This type stub file was generated by pyright.
"""

"""Utilities for Keras classes with v1 and v2 versions."""
training = ...
training_v1 = ...
base_layer = ...
base_layer_v1 = ...
callbacks = ...
callbacks_v1 = ...
class ModelVersionSelector:
    """Chooses between Keras v1 and v2 Model class."""
    def __new__(cls, *args, **kwargs):
        ...
    


class LayerVersionSelector:
    """Chooses between Keras v1 and v2 Layer class."""
    def __new__(cls, *args, **kwargs):
        ...
    


class TensorBoardVersionSelector:
    """Chooses between Keras v1 and v2 TensorBoard callback class."""
    def __new__(cls, *args, **kwargs):
        ...
    


def should_use_v2(): # -> bool:
    """Determine if v1 or v2 version should be used."""
    ...

def swap_class(cls, v2_cls, v1_cls, use_v2):
    """Swaps in v2_cls or v1_cls depending on graph mode."""
    ...

def disallow_legacy_graph(cls_name, method_name): # -> None:
    ...

def is_v1_layer_or_model(obj): # -> bool:
    ...

