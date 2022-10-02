"""
This type stub file was generated by pyright.
"""

from keras.layers.rnn.base_cudnn_rnn import _CuDNNRNN
from tensorflow.python.util.tf_export import keras_export

"""Fast GRU layer backed by cuDNN."""
@keras_export(v1=["keras.layers.CuDNNGRU"])
class CuDNNGRU(_CuDNNRNN):
    """Fast GRU implementation backed by cuDNN.

    More information about cuDNN can be found on the [NVIDIA
    developer website](https://developer.nvidia.com/cudnn).
    Can only be run on GPU.

    Args:
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the `kernel` weights matrix, used
          for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights
          matrix, used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
          matrix.
        recurrent_regularizer: Regularizer function applied to the
          `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the
          layer (its "activation").
        kernel_constraint: Constraint function applied to the `kernel` weights
          matrix.
        recurrent_constraint: Constraint function applied to the
          `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        return_sequences: Boolean. Whether to return the last output in the
          output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state in addition to
          the output.
        go_backwards: Boolean (default False). If True, process the input
          sequence backwards and return the reversed sequence.
        stateful: Boolean (default False). If True, the last state for each
          sample at index i in a batch will be used as initial state for the
          sample of index i in the following batch.
    """
    def __init__(self, units, kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., activity_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., return_sequences=..., return_state=..., go_backwards=..., stateful=..., **kwargs) -> None:
        ...
    
    @property
    def cell(self): # -> cell:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


