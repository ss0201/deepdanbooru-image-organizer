"""
This type stub file was generated by pyright.
"""

from keras.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export

"""Keras zero-padding layer for 2D input."""
@keras_export("keras.layers.ZeroPadding2D")
class ZeroPadding2D(Layer):
    """Zero-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns of zeros
    at the top, bottom, left and right side of an image tensor.

    Examples:

    >>> input_shape = (1, 1, 2, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[[0 1]
       [2 3]]]]
    >>> y = tf.keras.layers.ZeroPadding2D(padding=1)(x)
    >>> print(y)
    tf.Tensor(
      [[[[0 0]
         [0 0]
         [0 0]
         [0 0]]
        [[0 0]
         [0 1]
         [2 3]
         [0 0]]
        [[0 0]
         [0 0]
         [0 0]
         [0 0]]]], shape=(1, 3, 4, 2), dtype=int64)

    Args:
      padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        - If int: the same symmetric padding
          is applied to height and width.
        - If tuple of 2 ints:
          interpreted as two different
          symmetric padding values for height and width:
          `(symmetric_height_pad, symmetric_width_pad)`.
        - If tuple of 2 tuples of 2 ints:
          interpreted as
          `((top_pad, bottom_pad), (left_pad, right_pad))`
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, rows, cols)`

    Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, padded_rows, padded_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, padded_rows, padded_cols)`
    """
    def __init__(self, padding=..., data_format=..., **kwargs) -> None:
        ...
    
    def compute_output_shape(self, input_shape): # -> None:
        ...
    
    def call(self, inputs):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    

