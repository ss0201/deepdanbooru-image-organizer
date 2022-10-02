"""
This type stub file was generated by pyright.
"""

from keras.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export

"""Keras upsampling layer for 1D inputs."""
@keras_export("keras.layers.UpSampling1D")
class UpSampling1D(Layer):
    """Upsampling layer for 1D inputs.

    Repeats each temporal step `size` times along the time axis.

    Examples:

    >>> input_shape = (2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]
    >>> y = tf.keras.layers.UpSampling1D(size=2)(x)
    >>> print(y)
    tf.Tensor(
      [[[ 0  1  2]
        [ 0  1  2]
        [ 3  4  5]
        [ 3  4  5]]
       [[ 6  7  8]
        [ 6  7  8]
        [ 9 10 11]
        [ 9 10 11]]], shape=(2, 4, 3), dtype=int64)

    Args:
      size: Integer. Upsampling factor.

    Input shape:
      3D tensor with shape: `(batch_size, steps, features)`.

    Output shape:
      3D tensor with shape: `(batch_size, upsampled_steps, features)`.
    """
    def __init__(self, size=..., **kwargs) -> None:
        ...
    
    def compute_output_shape(self, input_shape):
        ...
    
    def call(self, inputs):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


