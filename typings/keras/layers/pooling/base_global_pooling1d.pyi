"""
This type stub file was generated by pyright.
"""

from keras.engine.base_layer import Layer

"""Private base class for global pooling 1D layers."""
class GlobalPooling1D(Layer):
    """Abstract class for different global pooling 1D layers."""
    def __init__(self, data_format=..., keepdims=..., **kwargs) -> None:
        ...
    
    def compute_output_shape(self, input_shape):
        ...
    
    def call(self, inputs):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


