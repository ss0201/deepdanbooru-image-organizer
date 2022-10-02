"""
This type stub file was generated by pyright.
"""

from keras.engine import base_layer
from keras.layers.rnn.base_rnn import RNN
from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export

"""Fully connected RNN layer."""
@keras_export("keras.layers.SimpleRNNCell")
class SimpleRNNCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
    """Cell class for SimpleRNN.

    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.

    This class processes one step within the whole time sequence input, whereas
    `tf.keras.layer.SimpleRNN` processes the whole sequence.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state.  Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector. Default: `zeros`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector.
        Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_constraint: Constraint function applied to the bias vector. Default:
        `None`.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state. Default: 0.

    Call arguments:
      inputs: A 2D tensor, with shape of `[batch, feature]`.
      states: A 2D tensor with shape of `[batch, units]`, which is the state
        from the previous time step. For timestep 0, the initial state provided
        by user will be feed to cell.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.

    Examples:

    ```python
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    rnn = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(4))

    output = rnn(inputs)  # The output has shape `[32, 4]`.

    rnn = tf.keras.layers.RNN(
        tf.keras.layers.SimpleRNNCell(4),
        return_sequences=True,
        return_state=True)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = rnn(inputs)
    ```
    """
    def __init__(self, units, activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., **kwargs) -> None:
        ...
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs, states, training=...): # -> tuple[Unknown | Any, list[Unknown | Any] | Unknown | Any]:
        ...
    
    def get_initial_state(self, inputs=..., batch_size=..., dtype=...):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


@keras_export("keras.layers.SimpleRNN")
class SimpleRNN(RNN):
    """Fully-connected RNN where the output is to be fed back to input.

    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state.  Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector. Default: `zeros`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector.
        Default: `None`.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation"). Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.  Default: `None`.
      bias_constraint: Constraint function applied to the bias vector. Default:
        `None`.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for the linear transformation of the
        inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for the linear transformation of the
        recurrent state. Default: 0.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence. Default: `False`.
      return_state: Boolean. Whether to return the last state
        in addition to the output. Default: `False`
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

    Call arguments:
      inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
      mask: Binary tensor of shape `[batch, timesteps]` indicating whether
        a given timestep should be masked. An individual `True` entry indicates
        that the corresponding timestep should be utilized, while a `False`
        entry indicates that the corresponding timestep should be ignored.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.

    Examples:

    ```python
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    simple_rnn = tf.keras.layers.SimpleRNN(4)

    output = simple_rnn(inputs)  # The output has shape `[32, 4]`.

    simple_rnn = tf.keras.layers.SimpleRNN(
        4, return_sequences=True, return_state=True)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = simple_rnn(inputs)
    ```
    """
    def __init__(self, units, activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., activity_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., return_sequences=..., return_state=..., go_backwards=..., stateful=..., unroll=..., **kwargs) -> None:
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
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    
    @classmethod
    def from_config(cls, config): # -> Self@SimpleRNN:
        ...
    

