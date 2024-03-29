"""
This type stub file was generated by pyright.
"""

from tensorflow.python.keras.layers import recurrent
from tensorflow.python.util.tf_export import keras_export

"""Recurrent layers for TF 2."""
_FUNCTION_API_NAME_ATTRIBUTE = ...
_FUNCTION_DEVICE_ATTRIBUTE = ...
_CPU_DEVICE_NAME = ...
_GPU_DEVICE_NAME = ...
_RUNTIME_UNKNOWN = ...
_RUNTIME_CPU = ...
_RUNTIME_GPU = ...
_CUDNN_AVAILABLE_MSG = ...
_CUDNN_NOT_AVAILABLE_MSG = ...
class _DefunWrapper:
  """A wrapper with no deep copy of the Defun in LSTM/GRU layer."""
  def __init__(self, time_major, go_backwards, layer_name) -> None:
    ...
  
  def __deepcopy__(self, memo): # -> Self@_DefunWrapper:
    ...
  


@keras_export('keras.layers.GRUCell', v1=[])
class GRUCell(recurrent.GRUCell):
  """Cell class for the GRU layer.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.GRU` processes the whole sequence.

  For example:

  >>> inputs = tf.random.normal([32, 10, 8])
  >>> rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(4))
  >>> output = rnn(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> rnn = tf.keras.layers.RNN(
  ...    tf.keras.layers.GRUCell(4),
  ...    return_sequences=True,
  ...    return_state=True)
  >>> whole_sequence_output, final_state = rnn(inputs)
  >>> print(whole_sequence_output.shape)
  (32, 10, 4)
  >>> print(final_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. Default: hyperbolic tangent
      (`tanh`). If you pass None, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
      linear transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before",
      True = "after" (default and CuDNN compatible).

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: A 2D tensor with shape of `[batch, units]`, which is the state from
      the previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """
  def __init__(self, units, activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., reset_after=..., **kwargs) -> None:
    ...
  


@keras_export('keras.layers.GRU', v1=[])
class GRU(recurrent.DropoutRNNCellMixin, recurrent.GRU):
  """Gated Recurrent Unit - Cho et al. 2014.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (cuDNN-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the CuDNN kernel
  (see below for details), the layer will use a fast cuDNN implementation.

  The requirements to use the cuDNN implementation are:

  1. `activation` == `tanh`
  2. `recurrent_activation` == `sigmoid`
  3. `recurrent_dropout` == 0
  4. `unroll` is `False`
  5. `use_bias` is `True`
  6. `reset_after` is `True`
  7. Inputs, if use masking, are strictly right-padded.
  8. Eager execution is enabled in the outermost context.

  There are two variants of the GRU implementation. The default one is based on
  [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to hidden
  state before matrix multiplication. The other one is based on
  [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

  The second variant is compatible with CuDNNGRU (GPU-only) and allows
  inference on CPU. Thus it has separate biases for `kernel` and
  `recurrent_kernel`. To use this variant, set `'reset_after'=True` and
  `recurrent_activation='sigmoid'`.

  For example:

  >>> inputs = tf.random.normal([32, 10, 8])
  >>> gru = tf.keras.layers.GRU(4)
  >>> output = gru(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
  >>> whole_sequence_output, final_state = gru(inputs)
  >>> print(whole_sequence_output.shape)
  (32, 10, 4)
  >>> print(final_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: sigmoid (`sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
       weights matrix, used for the linear transformation of the recurrent
       state. Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state in addition to the
      output. Default: `False`.
    go_backwards: Boolean (default `False`).
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
      `[timesteps, batch, feature]`, whereas in the False case, it will be
      `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before",
      True = "after" (default and CuDNN compatible).

  Call arguments:
    inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[samples, timesteps]` indicating whether
      a given timestep should be masked  (optional, defaults to `None`).
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the
      corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used  (optional, defaults to `None`).
    initial_state: List of initial state tensors to be passed to the first
      call of the cell  (optional, defaults to `None` which causes creation
      of zero-filled initial state tensors).
  """
  def __init__(self, units, activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., activity_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., return_sequences=..., return_state=..., go_backwards=..., stateful=..., unroll=..., time_major=..., reset_after=..., **kwargs) -> None:
    ...
  
  def call(self, inputs, mask=..., training=..., initial_state=...): # -> list[Unknown | _dispatcher_for_reverse_v2 | object | RaggedTensor] | tuple[Unknown | _dispatcher_for_reverse_v2 | object | RaggedTensor, Unknown | Any] | _dispatcher_for_reverse_v2 | object | RaggedTensor:
    ...
  


def standard_gru(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, zero_output_for_mask): # -> tuple[Unknown, Unknown, Unknown, Unknown]:
  """GRU with standard kernel implementation.

  This implementation can be run on all types of hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Args:
    inputs: Input tensor of GRU layer.
    init_h: Initial state tensor for the cell output.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. The bias contains the
      combined input_bias and recurrent_bias.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False` entry
      indicates that the corresponding timestep should be ignored.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  ...

def gpu_gru(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths): # -> tuple[Unknown | _dispatcher_for_reverse_v2, Unknown | _dispatcher_for_reverse_v2 | object, Unknown, Unknown]:
  """GRU with CuDNN implementation which is only available for GPU."""
  ...

def gru_with_backend_selection(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, zero_output_for_mask): # -> tuple[Unknown | Any, Unknown | Any, Unknown | Any, Unknown | Any]:
  """Call the GRU with optimized backend kernel selection.

  Under the hood, this function will create two TF function, one with the most
  generic kernel and can run on all device condition, and the second one with
  CuDNN specific kernel, which can only run on GPU.

  The first function will be called with normal_lstm_params, while the second
  function is not called, but only registered in the graph. The Grappler will
  do the proper graph rewrite and swap the optimized TF function based on the
  device placement.

  Args:
    inputs: Input tensor of GRU layer.
    init_h: Initial state tensor for the cell output.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    List of output tensors, same as standard_gru.
  """
  ...

@keras_export('keras.layers.LSTMCell', v1=[])
class LSTMCell(recurrent.LSTMCell):
  """Cell class for the LSTM layer.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.LSTM` processes the whole sequence.

  For example:

  >>> inputs = tf.random.normal([32, 10, 8])
  >>> rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4))
  >>> output = rnn(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> rnn = tf.keras.layers.RNN(
  ...    tf.keras.layers.LSTMCell(4),
  ...    return_sequences=True,
  ...    return_state=True)
  >>> whole_seq_output, final_memory_state, final_carry_state = rnn(inputs)
  >>> print(whole_seq_output.shape)
  (32, 10, 4)
  >>> print(final_memory_state.shape)
  (32, 4)
  >>> print(final_carry_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. Default: hyperbolic tangent
      (`tanh`). If you pass `None`, no activation is applied (ie. "linear"
      activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs. Default: `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
      the forget gate at initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: List of 2 tensors that corresponding to the cell's units. Both of
      them have shape `[batch, units]`, the first tensor is the memory state
      from previous time step, the second tensor is the carry state from
      previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """
  def __init__(self, units, activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., unit_forget_bias=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., **kwargs) -> None:
    ...
  


@keras_export('keras.layers.LSTM', v1=[])
class LSTM(recurrent.DropoutRNNCellMixin, recurrent.LSTM):
  """Long Short-Term Memory layer - Hochreiter 1997.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (cuDNN-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the CuDNN kernel
  (see below for details), the layer will use a fast cuDNN implementation.

  The requirements to use the cuDNN implementation are:

  1. `activation` == `tanh`
  2. `recurrent_activation` == `sigmoid`
  3. `recurrent_dropout` == 0
  4. `unroll` is `False`
  5. `use_bias` is `True`
  6. Inputs, if use masking, are strictly right-padded.
  7. Eager execution is enabled in the outermost context.

  For example:

  >>> inputs = tf.random.normal([32, 10, 8])
  >>> lstm = tf.keras.layers.LSTM(4)
  >>> output = lstm(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
  >>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
  >>> print(whole_seq_output.shape)
  (32, 10, 4)
  >>> print(final_memory_state.shape)
  (32, 4)
  >>> print(final_carry_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
      is applied (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs. Default: `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
      the forget gate at initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    return_sequences: Boolean. Whether to return the last output. in the output
      sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state in addition to the
      output. Default: `False`.
    go_backwards: Boolean (default `False`). If True, process the input sequence
      backwards and return the reversed sequence.
    stateful: Boolean (default `False`). If True, the last state for each sample
      at index i in a batch will be used as initial state for the sample of
      index i in the following batch.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `[timesteps, batch, feature]`, whereas in the False case, it will be
      `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    unroll: Boolean (default `False`). If True, the network will be unrolled,
      else a symbolic loop will be used. Unrolling can speed-up a RNN, although
      it tends to be more memory-intensive. Unrolling is only suitable for short
      sequences.

  Call arguments:
    inputs: A 3D tensor with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[batch, timesteps]` indicating whether
      a given timestep should be masked (optional, defaults to `None`).
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used (optional, defaults to `None`).
    initial_state: List of initial state tensors to be passed to the first
      call of the cell (optional, defaults to `None` which causes creation
      of zero-filled initial state tensors).
  """
  def __init__(self, units, activation=..., recurrent_activation=..., use_bias=..., kernel_initializer=..., recurrent_initializer=..., bias_initializer=..., unit_forget_bias=..., kernel_regularizer=..., recurrent_regularizer=..., bias_regularizer=..., activity_regularizer=..., kernel_constraint=..., recurrent_constraint=..., bias_constraint=..., dropout=..., recurrent_dropout=..., return_sequences=..., return_state=..., go_backwards=..., stateful=..., time_major=..., unroll=..., **kwargs) -> None:
    ...
  
  def call(self, inputs, mask=..., training=..., initial_state=...): # -> list[Unknown | _dispatcher_for_reverse_v2 | object | RaggedTensor] | tuple[Unknown | _dispatcher_for_reverse_v2 | object | RaggedTensor, Unknown | Any] | _dispatcher_for_reverse_v2 | object | RaggedTensor:
    ...
  


def standard_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, zero_output_for_mask): # -> tuple[Unknown, Unknown, Unknown, Unknown, Unknown]:
  """LSTM with standard kernel implementation.

  This implementation can be run on all types for hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Note that the first half of the bias tensor should be ignored by this impl.
  The CuDNN impl need an extra set of input gate bias. In order to make the both
  function take same shape of parameter, that extra set of bias is also feed
  here.

  Args:
    inputs: input tensor of LSTM layer.
    init_h: initial state tensor for the cell output.
    init_c: initial state tensor for the cell hidden state.
    kernel: weights for cell kernel.
    recurrent_kernel: weights for cell recurrent kernel.
    bias: weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    time_major: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    state_1: the cell hidden state, which has same shape as init_c.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  ...

def gpu_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths): # -> tuple[Unknown | _dispatcher_for_reverse_v2, Unknown | _dispatcher_for_reverse_v2 | object, Unknown, Unknown, Unknown]:
  """LSTM with either CuDNN or ROCm implementation which is only available for GPU.

  Note that currently only right padded data is supported, or the result will be
  polluted by the unmasked data which should be filtered.

  Args:
    inputs: Input tensor of LSTM layer.
    init_h: Initial state tensor for the cell output.
    init_c: Initial state tensor for the cell hidden state.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    time_major: Boolean, whether the inputs are in the format of [time, batch,
      feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.

  Returns:
    last_output: Output tensor for the last timestep, which has shape
      [batch, units].
    outputs: Output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: The cell output, which has same shape as init_h.
    state_1: The cell hidden state, which has same shape as init_c.
    runtime: Constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should not be used by user.
  """
  ...

def lstm_with_backend_selection(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, zero_output_for_mask): # -> tuple[Unknown | Any, Unknown | Any, Unknown | Any, Unknown | Any, Unknown | Any]:
  """Call the LSTM with optimized backend kernel selection.

  Under the hood, this function will create two TF function, one with the most
  generic kernel and can run on all device condition, and the second one with
  CuDNN specific kernel, which can only run on GPU.

  The first function will be called with normal_lstm_params, while the second
  function is not called, but only registered in the graph. The Grappler will
  do the proper graph rewrite and swap the optimized TF function based on the
  device placement.

  Args:
    inputs: Input tensor of LSTM layer.
    init_h: Initial state tensor for the cell output.
    init_c: Initial state tensor for the cell hidden state.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    List of output tensors, same as standard_lstm.
  """
  ...

def is_sequence_right_padded(mask):
  """Check the mask tensor and see if it right padded.

  For CuDNN kernel, it uses the sequence length param to skip the tailing
  timestep. If the data is left padded, or not a strict right padding (has
  masked value in the middle of the sequence), then CuDNN kernel won't be work
  properly in those cases.

  Left padded data: [[False, False, True, True, True]].
  Right padded data: [[True, True, True, False, False]].
  Mixture of mask/unmasked data: [[True, False, True, False, False]].

  Note that for the mixed data example above, the actually data RNN should see
  are those 2 Trues (index 0 and 2), the index 1 False should be ignored and not
  pollute the internal states.

  Args:
    mask: the Boolean tensor with shape [batch, timestep]

  Returns:
    boolean scalar tensor, whether the mask is strictly right padded.
  """
  ...

def has_fully_masked_sequence(mask):
  ...

def is_cudnn_supported_inputs(mask, time_major): # -> _dispatcher_for_logical_and | object:
  ...

def calculate_sequence_by_mask(mask, time_major):
  """Calculate the sequence length tensor (1-D) based on the masking tensor.

  The masking tensor is a 2D boolean tensor with shape [batch, timestep]. For
  any timestep that should be masked, the corresponding field will be False.
  Consider the following example:
    a = [[True, True, False, False],
         [True, True, True, False]]
  It is a (2, 4) tensor, and the corresponding sequence length result should be
  1D tensor with value [2, 3]. Note that the masking tensor must be right
  padded that could be checked by, e.g., `is_sequence_right_padded()`.

  Args:
    mask: Boolean tensor with shape [batch, timestep] or [timestep, batch] if
      time_major=True.
    time_major: Boolean, which indicates whether the mask is time major or batch
      major.
  Returns:
    sequence_length: 1D int32 tensor.
  """
  ...

