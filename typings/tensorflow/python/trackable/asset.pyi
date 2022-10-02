"""
This type stub file was generated by pyright.
"""

from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export

"""Asset-type Trackable object."""
saved_model_utils = ...
@tf_export("saved_model.Asset")
class Asset(base.Trackable):
  """Represents a file asset to hermetically include in a SavedModel.

  A SavedModel can include arbitrary files, called assets, that are needed
  for its use. For example a vocabulary file used initialize a lookup table.

  When a trackable object is exported via `tf.saved_model.save()`, all the
  `Asset`s reachable from it are copied into the SavedModel assets directory.
  Upon loading, the assets and the serialized functions that depend on them
  will refer to the correct filepaths inside the SavedModel directory.

  Example:

  ```
  filename = tf.saved_model.Asset("file.txt")

  @tf.function(input_signature=[])
  def func():
    return tf.io.read_file(filename)

  trackable_obj = tf.train.Checkpoint()
  trackable_obj.func = func
  trackable_obj.filename = filename
  tf.saved_model.save(trackable_obj, "/tmp/saved_model")

  # The created SavedModel is hermetic, it does not depend on
  # the original file and can be moved to another path.
  tf.io.gfile.remove("file.txt")
  tf.io.gfile.rename("/tmp/saved_model", "/tmp/new_location")

  reloaded_obj = tf.saved_model.load("/tmp/new_location")
  print(reloaded_obj.func())
  ```

  Attributes:
    asset_path: A path, or a 0-D `tf.string` tensor with path to the asset.
  """
  def __init__(self, path) -> None:
    """Record the full path to the asset."""
    ...
  
  @property
  def asset_path(self): # -> Tensor | Any:
    """Fetch the current asset path."""
    ...
  


