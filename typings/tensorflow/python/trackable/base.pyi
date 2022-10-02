"""
This type stub file was generated by pyright.
"""

from tensorflow.python.framework import ops
from tensorflow.python.trackable import constants
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

"""An object-local variable management scheme."""
OBJECT_GRAPH_PROTO_KEY = ...
VARIABLE_VALUE_KEY = ...
OBJECT_CONFIG_JSON_KEY = ...
SaveType = constants.SaveType
@tf_export("__internal__.tracking.TrackableReference", v1=[])
class TrackableReference:
  """A named reference to a trackable object for use with the `Trackable` class.

  These references mark named `Trackable` dependencies of a `Trackable` object
  and should be created when overriding `Trackable._checkpoint_dependencies`.

  Attributes:
    name: The local name for this dependency.
    ref: The `Trackable` object being referenced.
  """
  __slots__ = ...
  def __init__(self, name, ref) -> None:
    ...
  
  @property
  def name(self): # -> Unknown:
    ...
  
  @property
  def ref(self): # -> Unknown:
    ...
  
  def __iter__(self): # -> Generator[Unknown, None, None]:
    ...
  
  def __repr__(self): # -> str:
    ...
  
  def __eq__(self, o) -> bool:
    ...
  


class WeakTrackableReference(TrackableReference):
  """TrackableReference that stores weak references."""
  __slots__ = ...
  def __init__(self, name, ref) -> None:
    ...
  
  @property
  def ref(self):
    ...
  


ShardInfo = ...
@tf_export("__internal__.tracking.CheckpointInitialValueCallable", v1=[])
class CheckpointInitialValueCallable:
  """A callable object that returns a CheckpointInitialValue.

  See CheckpointInitialValue for more information.
  """
  def __init__(self, checkpoint_position) -> None:
    ...
  
  @property
  def checkpoint_position(self): # -> Unknown:
    ...
  
  def __call__(self, shape=..., dtype=..., shard_info=...): # -> CheckpointInitialValue:
    ...
  
  @property
  def restore_uid(self):
    ...
  


@tf_export("__internal__.tracking.CheckpointInitialValue", v1=[])
class CheckpointInitialValue(ops.Tensor):
  """Tensor wrapper for managing update UIDs in `Variables`.

  When supplied as an initial value, objects of this type let a `Variable`
  (`Variable`, `ResourceVariable`, etc.) know the UID of the restore the initial
  value came from. This allows deferred restorations to be sequenced in the
  order the user specified them, and lets us fall back on assignment if an
  initial value is not set (e.g. due to a custom getter interfering).

  See comments in _add_variable_with_custom_getter for more information about
  how `CheckpointInitialValue` is used.
  """
  def __init__(self, checkpoint_position, shape=..., shard_info=...) -> None:
    ...
  
  def __getattr__(self, attr): # -> Any:
    ...
  
  @property
  def checkpoint_position(self): # -> Unknown:
    ...
  


class NoRestoreSaveable(saveable_object.SaveableObject):
  """Embeds a tensor in a checkpoint with no restore ops."""
  def __init__(self, tensor, name, dtype=..., device=...) -> None:
    ...
  
  def restore(self, restored_tensors, restored_shapes): # -> _dispatcher_for_no_op | object | None:
    ...
  


_SlotVariableRestoration = ...
@tf_export("__internal__.tracking.no_automatic_dependency_tracking", v1=[])
def no_automatic_dependency_tracking(method):
  """Disables automatic dependency tracking on attribute assignment.

  Use to decorate any method of a Trackable object. Attribute assignment in
  that method will not add dependencies (also respected in Model). Harmless if
  used in a class which does not do automatic dependency tracking (which means
  it's safe to use in base classes which may have subclasses which also inherit
  from Trackable).

  Args:
    method: The method to decorate.

  Returns:
    A decorated method which sets and un-sets automatic dependency tracking for
    the object the method is called on (not thread safe).
  """
  ...

@tf_contextlib.contextmanager
def no_manual_dependency_tracking_scope(obj): # -> Generator[None, None, None]:
  """A context that disables manual dependency tracking for the given `obj`.

  Sometimes library methods might track objects on their own and we might want
  to disable that and do the tracking on our own. One can then use this context
  manager to disable the tracking the library method does and do your own
  tracking.

  For example:

  class TestLayer(tf.keras.Layer):
    def build():
      with no_manual_dependency_tracking_scope(self):
        var = self.add_variable("name1")  # Creates a var and doesn't track it
      self._track_trackable("name2", var)  # We track variable with name `name2`

  Args:
    obj: A trackable object.

  Yields:
    a scope in which the object doesn't track dependencies manually.
  """
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

@tf_export("__internal__.tracking.Trackable", v1=[])
class Trackable:
  """Base class for `Trackable` objects without automatic dependencies.

  This class has no __setattr__ override for performance reasons. Dependencies
  must be added explicitly. Unless attribute assignment is performance-critical,
  use `AutoTrackable` instead. Use `Trackable` for `isinstance`
  checks.
  """
  ...

