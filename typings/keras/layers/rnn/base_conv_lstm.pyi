"""
This type stub file was generated by pyright.
"""

from keras.engine import base_layer
from keras.layers.rnn.base_conv_rnn import ConvRNN
from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin

"""Base class for N-D convolutional LSTM layers."""
class ConvLSTMCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
    """Cell class for the ConvLSTM layer.

    Args:
      rank: Integer, rank of the convolution, e.g. "2" for 2D convolutions.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
      strides: An integer or tuple/list of n integers, specifying the strides of
        the convolution. Specifying any stride value != 1 is incompatible with
        specifying any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive). `"valid"` means
        no padding. `"same"` results in padding evenly to the left/right or
        up/down of the input such that output has the same height/width
        dimension as the input.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.  It defaults to the `image_data_format` value found in
        your Keras config file at `~/.keras/keras.json`. If you never set it,
        then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying the
        dilation rate to use for dilated convolution. Currently, specifying any
        `dilation_rate` value != 1 is incompatible with specifying any `strides`
        value != 1.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate
      at initialization. Use in combination with `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et al., 2015](
      http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state.
    Call arguments:
      inputs: A (2+ `rank`)D tensor.
      states:  List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """
    def __init__(self, rank, filters, kernel_size, strides=..., padding=..., data_format=..., dilation_rate=..., activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., unit_forget_bias=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs, states, training=...): # -> tuple[Unknown | Any, list[Unknown]]:
        ...
    
    def input_conv(self, x, w, b=..., padding=...):
        ...
    
    def recurrent_conv(self, x, w):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


class ConvLSTM(ConvRNN):
    """Abstract N-D Convolutional LSTM layer (used as implementation base).

    Similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Args:
      rank: Integer, rank of the convolution, e.g. "2" for 2D convolutions.
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, time, ..., channels)`
        while `channels_first` corresponds to
        inputs with shape `(batch, time, channels, ...)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
        By default hyperbolic tangent activation function is applied
        (`tanh(x)`).
      recurrent_activation: Activation function to use
        for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Use in combination with `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et al., 2015](
          http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence. (default False)
      return_state: Boolean Whether to return the last state
        in addition to the output. (default False)
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
    """
    def __init__(self, rank, filters, kernel_size, strides=..., padding=..., data_format=..., dilation_rate=..., activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., unit_forget_bias=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., activity_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., return_sequences=..., return_state=..., go_backwards=..., stateful=..., dropout=..., recurrent_dropout=..., **kwargs) -> None:
        ...
    
    def call(self, inputs, mask=..., training=..., initial_state=...): # -> list[Unknown]:
        ...
    
    @property
    def filters(self):
        ...
    
    @property
    def kernel_size(self):
        ...
    
    @property
    def strides(self):
        ...
    
    @property
    def padding(self):
        ...
    
    @property
    def data_format(self):
        ...
    
    @property
    def dilation_rate(self):
        ...
    
    @property
    def activation(self):
        ...
    
    @property
    def recurrent_activation(self):
        ...
    
    @property
    def use_bias(self):
        ...
    
    @property
    def kernel_initializer(self):
        ...
    
    @property
    def recurrent_initializer(self):
        ...
    
    @property
    def bias_initializer(self):
        ...
    
    @property
    def unit_forget_bias(self):
        ...
    
    @property
    def kernel_regularizer(self):
        ...
    
    @property
    def recurrent_regularizer(self):
        ...
    
    @property
    def bias_regularizer(self):
        ...
    
    @property
    def kernel_constraint(self):
        ...
    
    @property
    def recurrent_constraint(self):
        ...
    
    @property
    def bias_constraint(self):
        ...
    
    @property
    def dropout(self):
        ...
    
    @property
    def recurrent_dropout(self):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    
    @classmethod
    def from_config(cls, config): # -> Self@ConvLSTM:
        ...
    

