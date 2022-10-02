"""
This type stub file was generated by pyright.
"""

from keras.engine.base_layer import Layer

"""Private base class for pooling 2D layers."""
class Pooling2D(Layer):
    """Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).

    This class only exists for code reuse. It will never be an exposed API.

    Args:
      pool_function: The pooling function to apply, e.g. `tf.nn.max_pool2d`.
      pool_size: An integer or tuple/list of 2 integers:
        (pool_height, pool_width)
        specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.
      name: A string, the name of the layer.
    """
    def __init__(self, pool_function, pool_size, strides, padding=..., data_format=..., name=..., **kwargs) -> None:
        ...
    
    def call(self, inputs):
        ...
    
    def compute_output_shape(self, input_shape):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


