"""
This type stub file was generated by pyright.
"""

import threading
from tensorflow.python.keras.utils import tf_contextlib

"""Keras SavedModel serialization.

TODO (kathywu): Move to layer_serialization.py. Some model-specific logic should
go to model_serialization.py.
"""
base_layer = ...
metrics = ...
input_layer = ...
training_lib = ...
sequential_lib = ...
def should_skip_serialization(layer): # -> bool:
  """Skip serializing extra objects and functions if layer inputs aren't set."""
  ...

def wrap_layer_objects(layer, serialization_cache): # -> dict[str, Unknown | Trackable | _DictWrapper | ListWrapper | _TupleWrapper | tuple[Unknown, ...]]:
  """Returns extra trackable objects to attach to the serialized layer.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all checkpointable objects from a
    SerializedAttributes object. See LayerAttributes and ModelAttributes for
    entire list of objects
  """
  ...

def wrap_layer_functions(layer, serialization_cache): # -> dict[Unknown, Any | None] | dict[str, LayerCall]:
  """Returns dict of wrapped layer call function and losses in tf.functions.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all keras tf.functions to serialize. See
    LayerAttributes and ModelAttributes for the list of all attributes.
  """
  ...

def default_save_signature(layer): # -> None:
  ...

class LayerTracingContext(threading.local):
  def __init__(self) -> None:
    ...
  


_thread_local_data = ...
@tf_contextlib.contextmanager
def tracing_scope(): # -> Generator[None, None, None]:
  """Enables tracing scope."""
  ...

def add_trace_to_queue(fn, args, kwargs, training=...): # -> None:
  ...

def tracing_enabled(): # -> bool:
  """Whether to add extra traces to the queue."""
  ...

class LayerCallCollection:
  """Groups wrapped layer call functions.

  This is used to ensure that all layer call functions are traced with the same
  inputs-
    - call
    - call_and_return_conditional_losses
    - call_and_return_all_conditional_losses
  """
  def __init__(self, layer) -> None:
    ...
  
  def add_trace(self, *args, **kwargs): # -> None:
    """Traces all functions with the same args and kwargs.

    Args:
      *args: Positional args passed to the original function.
      **kwargs: Keyword args passed to the original function.
    """
    ...
  
  @property
  def fn_input_signature(self): # -> tuple[Unknown, ...] | defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy | list[Unknown | defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy] | None:
    """Returns input signature for the wrapped layer call function."""
    ...
  
  def training_arg_was_passed(self, args, kwargs): # -> bool:
    ...
  
  def get_training_arg_value(self, args, kwargs):
    ...
  
  def get_input_arg_value(self, args, kwargs):
    ...
  
  def add_function(self, call_fn, name, match_layer_training_arg): # -> LayerCall:
    """Adds a layer call function to the collection.

    Args:
      call_fn: a python function
      name: Name of call function
      match_layer_training_arg: If True, removes the `training` from the
        function arguments when calling `call_fn`.

    Returns:
      LayerCall (tf.function)
    """
    ...
  
  def trace_with_input_signature(self): # -> None:
    """Trace with the layer/models inferred input signature if possible."""
    ...
  


def layer_call_wrapper(call_collection, method, name):
  """Ensures layer losses are kept the same, and runs method in call context."""
  ...

class LayerCall:
  """Function that triggers traces of other functions in the same collection."""
  def __init__(self, call_collection, call_fn, name, input_signature) -> None:
    """Initializes a LayerCall object.

    Args:
      call_collection: a LayerCallCollection, which contains the other layer
        call functions (e.g. call_with_conditional_losses, call). These
        functions should be traced with the same arguments.
      call_fn: A call function.
      name: Name of the call function.
      input_signature: Input signature of call_fn (can be None).
    """
    ...
  
  def __call__(self, *args, **kwargs): # -> None:
    ...
  
  def get_concrete_function(self, *args, **kwargs): # -> ConcreteFunction:
    ...
  


