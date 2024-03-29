"""
This type stub file was generated by pyright.
"""

from keras.layers.rnn import lstm
from keras.layers.rnn.base_rnn import RNN
from tensorflow.python.util.tf_export import keras_export

"""Long Short-Term Memory V1 layer."""
@keras_export(v1=["keras.layers.LSTMCell"])
class LSTMCell(lstm.LSTMCell):
    """Cell class for the LSTM layer.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Setting it to true will also force `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et al., 2015](
          http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """
    def __init__(self, units, activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., unit_forget_bias=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., **kwargs) -> None:
        ...
    


@keras_export(v1=["keras.layers.LSTM"])
class LSTM(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.

     Note that this cell is not optimized for performance on GPU. Please use
    `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Setting it to true will also force `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et al., 2015](
          http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
      time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `(timesteps, batch, ...)`, whereas in the False case, it will be
        `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form.

    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked. An individual `True` entry indicates
        that the corresponding timestep should be utilized, while a `False`
        entry indicates that the corresponding timestep should be ignored.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    """
    def __init__(self, units, activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., unit_forget_bias=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., activity_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., return_sequences=..., return_state=..., go_backwards=..., stateful=..., unroll=..., **kwargs) -> None:
        ...
    
    def call(self, inputs, mask=..., training=..., initial_state=...): # -> list[Unknown]:
        ...
    
    @property
    def units(self):
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
    
    @property
    def implementation(self):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    
    @classmethod
    def from_config(cls, config): # -> Self@LSTM:
        ...
    


