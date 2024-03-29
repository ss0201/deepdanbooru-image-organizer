"""
This type stub file was generated by pyright.
"""

from keras.engine.base_layer import Layer
from keras.utils import tf_utils

"""Private base class for layers that can merge several inputs into one."""
class _Merge(Layer):
    """Generic merge layer for elementwise merge functions.

    Used to implement `Sum`, `Average`, etc.
    """
    def __init__(self, **kwargs) -> None:
        """Initializes a Merge layer.

        Args:
          **kwargs: standard layer keyword arguments.
        """
        ...
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs):
        ...
    
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        ...
    
    def compute_mask(self, inputs, mask=...): # -> None:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


