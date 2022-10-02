"""
This type stub file was generated by pyright.
"""

"""Utilities for extracting checkpoint info`."""
_CheckpointFactoryData = ...
def get_checkpoint_factories_and_keys(object_names, object_map=...): # -> tuple[ObjectIdentityDictionary, defaultdict[Unknown, dict[Unknown, Unknown]]]:
  """Gets a map of saveable factories and corresponding checkpoint keys.

  Args:
    object_names: a dictionary that maps `Trackable` objects to auto-generated
      string names.
    object_map: a dictionary mapping `Trackable` to copied `Trackable` objects.
      The copied objects are generated from `Trackable._map_resources()` which
      copies the object into another graph. Generally only resource objects
      (e.g. Variables, Tables) will be in this map.

  Returns:
    A tuple of (
      Dictionary mapping trackable -> list of _CheckpointFactoryData,
      Dictionary mapping registered saver name -> {object name -> trackable})
  """
  ...

def serialize_gathered_objects(graph_view, object_map=..., call_with_mapped_captures=..., saveables_cache=...): # -> tuple[list[Unknown], TrackableObjectGraph, dict[Unknown, Unknown] | None, defaultdict[Unknown, dict[Unknown, Unknown]]]:
  """Create SaveableObjects and protos for gathered objects."""
  ...

def serialize_object_graph_with_registered_savers(graph_view, saveables_cache): # -> tuple[list[Unknown], TrackableObjectGraph, dict[Unknown, Unknown] | None, defaultdict[Unknown, dict[Unknown, Unknown]]]:
  """Determine checkpoint keys for variables and build a serialized graph."""
  ...

def frozen_saveables_and_savers(graph_view, object_map=..., to_graph=..., call_with_mapped_captures=..., saveables_cache=...): # -> tuple[list[Unknown], defaultdict[Unknown, dict[Unknown, Unknown]]]:
  """Generates SaveableObjects and registered savers in the frozen graph."""
  ...

def objects_ids_and_slot_variables_and_paths(graph_view): # -> tuple[Unknown, Unknown, ObjectIdentityDictionary, ObjectIdentityDictionary, ObjectIdentityDictionary]:
  """Traverse the object graph and list all accessible objects.

  Looks for `Trackable` objects which are dependencies of
  `root_trackable`. Includes slot variables only if the variable they are
  slotting for and the optimizer are dependencies of `root_trackable`
  (i.e. if they would be saved with a checkpoint).

  Args:
    graph_view: A GraphView object.

  Returns:
    A tuple of (trackable objects, paths from root for each object,
                object -> node id, slot variables, object_names)
  """
  ...

def list_objects(graph_view):
  """Traverse the object graph and list all accessible objects."""
  ...
