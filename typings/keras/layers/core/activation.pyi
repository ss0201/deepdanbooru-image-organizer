"""
This type stub file was generated by pyright.
"""

from keras.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export

"""Contains the Activation layer."""
@keras_export("keras.layers.Activation")
class Activation(Layer):
    """Applies an activation function to an output.

    Args:
      activation: Activation function, such as `tf.nn.relu`, or string name of
        built-in activation function, such as "relu".

    Usage:

    >>> layer = tf.keras.layers.Activation('relu')
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [0.0, 0.0, 0.0, 2.0]
    >>> layer = tf.keras.layers.Activation(tf.nn.relu)
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [0.0, 0.0, 0.0, 2.0]

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the batch axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """
    def __init__(self, activation, **kwargs) -> None:
        ...
    
    def call(self, inputs): # -> Any:
        ...
    
    def compute_output_shape(self, input_shape):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    

