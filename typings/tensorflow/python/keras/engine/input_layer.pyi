"""
This type stub file was generated by pyright.
"""

from tensorflow.python.keras.engine import base_layer
from tensorflow.python.util.tf_export import keras_export

"""Input layer code (`Input` and `InputLayer`)."""
@keras_export('keras.layers.InputLayer')
class InputLayer(base_layer.Layer):
  """Layer to be used as an entry point into a Network (a graph of layers).

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create a placeholder tensor (pass arguments `input_shape`, and
  optionally, `dtype`).

  It is generally recommend to use the functional layer API via `Input`,
  (which creates an `InputLayer`) without directly using `InputLayer`.

  When using InputLayer with Keras Sequential model, it can be skipped by
  moving the input_shape parameter to the first layer after the InputLayer.

  This class can create placeholders for tf.Tensors, tf.SparseTensors, and
  tf.RaggedTensors by choosing 'sparse=True' or 'ragged=True'. Note that
  'sparse' and 'ragged' can't be configured to True at same time.
  Usage:

  ```python
  # With explicit InputLayer.
  model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    tf.keras.layers.Dense(8)])
  model.compile(tf.optimizers.RMSprop(0.001), loss='mse')
  model.fit(np.zeros((10, 4)),
            np.ones((10, 8)))

  # Without InputLayer and let the first layer to have the input_shape.
  # Keras will add a input for the model behind the scene.
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(4,))])
  model.compile(tf.optimizers.RMSprop(0.001), loss='mse')
  model.fit(np.zeros((10, 4)),
            np.ones((10, 8)))
  ```

  Args:
      input_shape: Shape tuple (not including the batch axis), or `TensorShape`
        instance (not including the batch axis).
      batch_size: Optional input batch size (integer or None).
      dtype: Optional datatype of the input. When not provided, the Keras
          default float type will be used.
      input_tensor: Optional tensor to use as layer input. If set, the layer
          will use the `tf.TypeSpec` of this tensor rather
          than creating a new placeholder tensor.
      sparse: Boolean, whether the placeholder created is meant to be sparse.
          Default to False.
      ragged: Boolean, whether the placeholder created is meant to be ragged.
          In this case, values of 'None' in the 'shape' argument represent
          ragged dimensions. For more information about RaggedTensors, see
          [this guide](https://www.tensorflow.org/guide/ragged_tensors).
          Default to False.
      type_spec: A `tf.TypeSpec` object to create Input from. This `tf.TypeSpec`
          represents the entire batch. When provided, all other args except
          name must be None.
      name: Optional name of the layer (string).
  """
  def __init__(self, input_shape=..., batch_size=..., dtype=..., input_tensor=..., sparse=..., name=..., ragged=..., type_spec=..., **kwargs) -> None:
    ...
  
  def get_config(self): # -> dict[str, Unknown]:
    ...
  


@keras_export('keras.Input', 'keras.layers.Input')
def Input(shape=..., batch_size=..., name=..., dtype=..., sparse=..., tensor=..., ragged=..., type_spec=..., **kwargs): # -> list[Unknown]:
  """`Input()` is used to instantiate a Keras tensor.

  A Keras tensor is a symbolic tensor-like object,
  which we augment with certain attributes that allow us to build a Keras model
  just by knowing the inputs and outputs of the model.

  For instance, if `a`, `b` and `c` are Keras tensors,
  it becomes possible to do:
  `model = Model(input=[a, b], output=c)`

  Args:
      shape: A shape tuple (integers), not including the batch size.
          For instance, `shape=(32,)` indicates that the expected input
          will be batches of 32-dimensional vectors. Elements of this tuple
          can be None; 'None' elements represent dimensions where the shape is
          not known.
      batch_size: optional static batch size (integer).
      name: An optional name string for the layer.
          Should be unique in a model (do not reuse the same name twice).
          It will be autogenerated if it isn't provided.
      dtype: The data type expected by the input, as a string
          (`float32`, `float64`, `int32`...)
      sparse: A boolean specifying whether the placeholder to be created is
          sparse. Only one of 'ragged' and 'sparse' can be True. Note that,
          if `sparse` is False, sparse tensors can still be passed into the
          input - they will be densified with a default value of 0.
      tensor: Optional existing tensor to wrap into the `Input` layer.
          If set, the layer will use the `tf.TypeSpec` of this tensor rather
          than creating a new placeholder tensor.
      ragged: A boolean specifying whether the placeholder to be created is
          ragged. Only one of 'ragged' and 'sparse' can be True. In this case,
          values of 'None' in the 'shape' argument represent ragged dimensions.
          For more information about RaggedTensors, see
          [this guide](https://www.tensorflow.org/guide/ragged_tensors).
      type_spec: A `tf.TypeSpec` object to create the input placeholder from.
          When provided, all other args except name must be None.
      **kwargs: deprecated arguments support. Supports `batch_shape` and
          `batch_input_shape`.

  Returns:
    A `tensor`.

  Example:

  ```python
  # this is a logistic regression in Keras
  x = Input(shape=(32,))
  y = Dense(16, activation='softmax')(x)
  model = Model(x, y)
  ```

  Note that even if eager execution is enabled,
  `Input` produces a symbolic tensor-like object (i.e. a placeholder).
  This symbolic tensor-like object can be used with lower-level
  TensorFlow ops that take tensors as inputs, as such:

  ```python
  x = Input(shape=(32,))
  y = tf.square(x)  # This op will be treated like a layer
  model = Model(x, y)
  ```

  (This behavior does not work for higher-order TensorFlow APIs such as
  control flow and being directly watched by a `tf.GradientTape`).

  However, the resulting model will not track any variables that were
  used as inputs to TensorFlow ops. All variable usages must happen within
  Keras layers to make sure they will be tracked by the model's weights.

  The Keras Input can also create a placeholder from an arbitrary `tf.TypeSpec`,
  e.g:

  ```python
  x = Input(type_spec=tf.RaggedTensorSpec(shape=[None, None],
                                          dtype=tf.float32, ragged_rank=1))
  y = x.values
  model = Model(x, y)
  ```
  When passing an arbitrary `tf.TypeSpec`, it must represent the signature of an
  entire batch instead of just one example.

  Raises:
    ValueError: If both `sparse` and `ragged` are provided.
    ValueError: If both `shape` and (`batch_input_shape` or `batch_shape`) are
      provided.
    ValueError: If `shape`, `tensor` and `type_spec` are None.
    ValueError: If arguments besides `type_spec` are non-None while `type_spec`
                is passed.
    ValueError: if any unrecognized parameters are provided.
  """
  ...

