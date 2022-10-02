"""
This type stub file was generated by pyright.
"""

from keras.layers.rnn import gru
from keras.layers.rnn.base_rnn import RNN
from tensorflow.python.util.tf_export import keras_export

"""Gated Recurrent Unit V1 layer."""
@keras_export(v1=["keras.layers.GRUCell"])
class GRUCell(gru.GRUCell):
    """Cell class for the GRU layer.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
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
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before" (default),
        True = "after" (cuDNN compatible).

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """
    def __init__(self, units, activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., reset_after=..., **kwargs) -> None:
        ...
    


@keras_export(v1=["keras.layers.GRU"])
class GRU(RNN):
    """Gated Recurrent Unit - Cho et al. 2014.

    There are two variants. The default one is based on 1406.1078v3 and
    has reset gate applied to hidden state before matrix multiplication. The
    other one is based on original 1406.1078v1 and has the order reversed.

    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. Use `'reset_after'=True` and
    `recurrent_activation='sigmoid'`.

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
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
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
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before" (default),
        True = "after" (cuDNN compatible).

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
    def __init__(self, units, activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., activity_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., return_sequences=..., return_state=..., go_backwards=..., stateful=..., unroll=..., reset_after=..., **kwargs) -> None:
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
    
    @property
    def reset_after(self):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    
    @classmethod
    def from_config(cls, config): # -> Self@GRU:
        ...
    


