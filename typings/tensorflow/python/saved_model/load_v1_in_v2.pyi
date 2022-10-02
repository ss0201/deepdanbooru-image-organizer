"""
This type stub file was generated by pyright.
"""

from tensorflow.python.saved_model import loader_impl
from tensorflow.python.trackable import resource

"""Import a TF v1-style SavedModel when executing eagerly."""
_LOAD_V1_V2_LABEL = ...
class _Initializer(resource.CapturableResource):
  """Represents an initialization operation restored from a SavedModel.

  Without this object re-export of imported 1.x SavedModels would omit the
  original SavedModel's initialization procedure.

  Created when `tf.saved_model.load` loads a TF 1.x-style SavedModel with an
  initialization op. This object holds a function that runs the
  initialization. It does not require any manual user intervention;
  `tf.saved_model.save` will see this object and automatically add it to the
  exported SavedModel, and `tf.saved_model.load` runs the initialization
  function automatically.
  """
  def __init__(self, init_fn, asset_paths) -> None:
    ...
  


class _EagerSavedModelLoader(loader_impl.SavedModelLoader):
  """Loads a SavedModel without using Sessions."""
  def get_meta_graph_def_from_tags(self, tags):
    """Override to support implicit one-MetaGraph loading with tags=None."""
    ...
  
  def load_graph(self, returns, meta_graph_def): # -> None:
    """Called from wrap_function to import `meta_graph_def`."""
    ...
  
  def restore_variables(self, wrapped, restore_from_saver): # -> None:
    """Restores variables from the checkpoint."""
    ...
  
  def load(self, tags): # -> AutoTrackable:
    """Creates an object from the MetaGraph identified by `tags`."""
    ...
  


def load(export_dir, tags): # -> AutoTrackable:
  """Load a v1-style SavedModel as an object."""
  ...

