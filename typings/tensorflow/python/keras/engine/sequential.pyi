"""
This type stub file was generated by pyright.
"""

from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util.tf_export import keras_export

"""Home of the `Sequential` model."""
SINGLE_LAYER_OUTPUT_ERROR_MSG = ...
@keras_export('keras.Sequential', 'keras.models.Sequential')
class Sequential(functional.Functional):
  """`Sequential` groups a linear stack of layers into a `tf.keras.Model`.

  `Sequential` provides training and inference features on this model.

  Examples:

  >>> # Optionally, the first layer can receive an `input_shape` argument:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
  >>> # Afterwards, we do automatic shape inference:
  >>> model.add(tf.keras.layers.Dense(4))

  >>> # This is identical to the following:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.Input(shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(8))

  >>> # Note that you can also omit the `input_shape` argument.
  >>> # In that case the model doesn't have any weights until the first call
  >>> # to a training/evaluation method (since it isn't yet built):
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> # model.weights not created yet

  >>> # Whereas if you specify the input shape, the model gets built
  >>> # continuously as you are adding layers:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> len(model.weights)
  4

  >>> # When using the delayed-build pattern (no input shape specified), you can
  >>> # choose to manually build your model by calling
  >>> # `build(batch_input_shape)`:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> model.build((None, 16))
  >>> len(model.weights)
  4

  ```python
  # Note that when using the delayed-build pattern (no input shape specified),
  # the model gets built the first time you call `fit`, `eval`, or `predict`,
  # or the first time you call the model on some input data.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(8))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='sgd', loss='mse')
  # This builds the model for the first time:
  model.fit(x, y, batch_size=32, epochs=10)
  ```
  """
  @trackable.no_automatic_dependency_tracking
  def __init__(self, layers=..., name=...) -> None:
    """Creates a `Sequential` model instance.

    Args:
      layers: Optional list of layers to add to the model.
      name: Optional name for the model.
    """
    ...
  
  @property
  def layers(self): # -> list[Layer]:
    ...
  
  @trackable.no_automatic_dependency_tracking
  def add(self, layer): # -> None:
    """Adds a layer instance on top of the layer stack.

    Args:
        layer: layer instance.

    Raises:
        TypeError: If `layer` is not a layer instance.
        ValueError: In case the `layer` argument does not
            know its input shape.
        ValueError: In case the `layer` argument has
            multiple output tensors, or is already connected
            somewhere else (forbidden in `Sequential` models).
    """
    ...
  
  @trackable.no_automatic_dependency_tracking
  def pop(self): # -> None:
    """Removes the last layer in the model.

    Raises:
        TypeError: if there are no layers in the model.
    """
    ...
  
  @generic_utils.default
  def build(self, input_shape=...): # -> None:
    ...
  
  def call(self, inputs, training=..., mask=...): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy | ndarray | ((func: Unknown | None = None) -> (... | MethodType | ((*args: Unknown, **kwargs: Unknown) -> Unknown))) | MethodType | ((*args: Unknown, **kwargs: Unknown) -> Unknown):
    ...
  
  def compute_output_shape(self, input_shape): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
    ...
  
  def compute_mask(self, inputs, mask): # -> Any | None:
    ...
  
  def predict_proba(self, x, batch_size=..., verbose=...): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
    """Generates class probability predictions for the input samples.

    The input samples are processed batch by batch.

    Args:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A Numpy array of probability predictions.
    """
    ...
  
  def predict_classes(self, x, batch_size=..., verbose=...): # -> Any:
    """Generate class predictions for the input samples.

    The input samples are processed batch by batch.

    Args:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A numpy array of class predictions.
    """
    ...
  
  def get_config(self): # -> dict[str, Unknown]:
    ...
  
  @classmethod
  def from_config(cls, config, custom_objects=...): # -> Self@Sequential:
    ...
  
  @property
  def input_spec(self): # -> None:
    ...
  
  @input_spec.setter
  def input_spec(self, value): # -> None:
    ...
  


def relax_input_shape(shape_1, shape_2): # -> tuple[Unknown | None, ...] | None:
  ...

def clear_previously_created_nodes(layer, created_nodes): # -> None:
  """Remove nodes from `created_nodes` from the layer's inbound_nodes."""
  ...

def track_nodes_created_by_last_call(layer, created_nodes): # -> None:
  """Adds to `created_nodes` the nodes created by the last call to `layer`."""
  ...

