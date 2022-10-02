"""
This type stub file was generated by pyright.
"""

import threading
from tensorflow.python.keras.utils import tf_contextlib

"""Utility functions shared between SavedModel saving/loading implementations."""
training_lib = ...
def use_wrapped_call(layer, call_fn, default_training_value=..., return_method=...): # -> MethodType:
  """Creates fn that adds the losses returned by call_fn & returns the outputs.

  Args:
    layer: A Keras layer object
    call_fn: tf.function that takes layer inputs (and possibly a training arg),
      and returns a tuple of (outputs, list of losses).
    default_training_value: Default value of the training kwarg. If `None`, the
      default is `K.learning_phase()`.
    return_method: Whether to return a method bound to the layer.

  Returns:
    function that calls call_fn and returns the outputs. Losses returned by
    call_fn are added to the layer losses.
  """
  ...

def layer_uses_training_bool(layer): # -> bool:
  """Returns whether this layer or any of its children uses the training arg."""
  ...

def list_all_layers(obj): # -> list[Unknown]:
  ...

def list_all_layers_and_sublayers(obj): # -> set[Unknown]:
  ...

def maybe_add_training_arg(original_call, wrapped_call, expects_training_arg, default_training_value): # -> tuple[Unknown, None] | tuple[(*args: Unknown, **kwargs: Unknown) -> (Any | Unknown | list[Unknown] | _basetuple | defaultdict[Unknown, Unknown] | ObjectProxy), FullArgSpec]:
  """Decorate call and optionally adds training argument.

  If a layer expects a training argument, this function ensures that 'training'
  is present in the layer args or kwonly args, with the default training value.

  Args:
    original_call: Original call function.
    wrapped_call: Wrapped call function.
    expects_training_arg: Whether to include 'training' argument.
    default_training_value: Default value of the training kwarg to include in
      the arg spec. If `None`, the default is `K.learning_phase()`.

  Returns:
    Tuple of (
      function that calls `wrapped_call` and sets the training arg,
      Argspec of returned function or `None` if the argspec is unchanged)
  """
  ...

def get_training_arg_index(call_fn): # -> int | None:
  """Returns the index of 'training' in the layer call function arguments.

  Args:
    call_fn: Call function.

  Returns:
    - n: index of 'training' in the call function arguments.
    - -1: if 'training' is not found in the arguments, but layer.call accepts
          variable keyword arguments
    - None: if layer doesn't expect a training argument.
  """
  ...

def set_training_arg(training, index, args, kwargs): # -> tuple[Unknown, Unknown]:
  ...

def get_training_arg(index, args, kwargs):
  ...

def remove_training_arg(index, args, kwargs): # -> None:
  ...

class SaveOptionsContext(threading.local):
  def __init__(self) -> None:
    ...
  


_save_options_context = ...
@tf_contextlib.contextmanager
def keras_option_scope(save_traces): # -> Generator[None, None, None]:
  ...

def should_save_traces(): # -> bool:
  """Whether to trace layer functions-can be disabled in the save_traces arg."""
  ...

@tf_contextlib.contextmanager
def no_automatic_dependency_tracking_scope(obj): # -> Generator[None, None, None]:
  """A context that disables automatic dependency tracking when assigning attrs.

  Objects that inherit from Autotrackable automatically creates dependencies
  to trackable objects through attribute assignments, and wraps data structures
  (lists or dicts) with trackable classes. This scope may be used to temporarily
  disable this behavior. This works similar to the decorator
  `no_automatic_dependency_tracking`.

  Example usage:
  ```
  model = tf.keras.Model()
  model.arr1 = []  # Creates a ListWrapper object
  with no_automatic_dependency_tracking_scope(model):
    model.arr2 = []  # Creates a regular, untracked python list
  ```

  Args:
    obj: A trackable object.

  Yields:
    a scope in which the object doesn't track dependencies.
  """
  ...

