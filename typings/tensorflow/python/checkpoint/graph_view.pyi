"""
This type stub file was generated by pyright.
"""

from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.util.tf_export import tf_export

"""Manages a graph of Trackable objects."""
@tf_export("__internal__.tracking.ObjectGraphView", v1=[])
class ObjectGraphView(trackable_view.TrackableView):
  """Gathers and serializes an object graph."""
  def __init__(self, root, attached_dependencies=...) -> None:
    """Configure the graph view.

    Args:
      root: A `Trackable` object whose variables (including the variables of
        dependencies, recursively) should be saved. May be a weak reference.
      attached_dependencies: List of dependencies to attach to the root object.
        Used when saving a Checkpoint with a defined root object. To avoid
        reference cycles, this should use the WeakTrackableReference class.
    """
    ...
  
  def __deepcopy__(self, memo): # -> Self@ObjectGraphView:
    ...
  
  def list_children(self, obj, save_type=..., **kwargs): # -> list[Unknown]:
    """Returns list of all child trackables attached to obj.

    Args:
      obj: A `Trackable` object.
      save_type: A string, can be 'savedmodel' or 'checkpoint'.
      **kwargs: kwargs to use when retrieving the object's children.

    Returns:
      List of all children attached to the object.
    """
    ...
  
  def children(self, obj, save_type=..., **kwargs): # -> dict[Unknown, Unknown]:
    """Returns all child trackables attached to obj.

    Args:
      obj: A `Trackable` object.
      save_type: A string, can be 'savedmodel' or 'checkpoint'.
      **kwargs: kwargs to use when retrieving the object's children.

    Returns:
      Dictionary of all children attached to the object with name to trackable.
    """
    ...
  
  @property
  def attached_dependencies(self): # -> None:
    """Returns list of dependencies that should be saved in the checkpoint.

    These dependencies are not tracked by root, but are in the checkpoint.
    This is defined when the user creates a Checkpoint with both root and kwargs
    set.

    Returns:
      A list of TrackableReferences.
    """
    ...
  
  @property
  def root(self):
    ...
  
  def breadth_first_traversal(self): # -> tuple[list[Unknown], ObjectIdentityDictionary]:
    ...
  
  def serialize_object_graph(self, saveables_cache=...): # -> tuple[list[Unknown], TrackableObjectGraph, dict[Unknown, Unknown] | None]:
    """Determine checkpoint keys for variables and build a serialized graph.

    Non-slot variables are keyed based on a shortest path from the root saveable
    to the object which owns the variable (i.e. the one which called
    `Trackable._add_variable` to create it).

    Slot variables are keyed based on a shortest path to the variable being
    slotted for, a shortest path to their optimizer, and the slot name.

    Args:
      saveables_cache: An optional cache storing previously created
        SaveableObjects created for each Trackable. Maps Trackables to a
        dictionary of attribute names to Trackable.

    Returns:
      A tuple of (named_variables, object_graph_proto, feed_additions):
        named_variables: A dictionary mapping names to variable objects.
        object_graph_proto: A TrackableObjectGraph protocol buffer
          containing the serialized object graph and variable references.
        feed_additions: A dictionary mapping from Tensors to values which should
          be fed when saving.

    Raises:
      ValueError: If there are invalid characters in an optimizer's slot names.
    """
    ...
  
  def frozen_saveable_objects(self, object_map=..., to_graph=..., call_with_mapped_captures=...): # -> list[Unknown]:
    """Creates SaveableObjects with the current object graph frozen."""
    ...
  


