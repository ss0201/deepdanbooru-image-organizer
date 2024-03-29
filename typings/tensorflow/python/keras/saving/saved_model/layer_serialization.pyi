"""
This type stub file was generated by pyright.
"""

from tensorflow.python.keras.saving.saved_model import base_serialization

"""Classes and functions implementing Layer SavedModel serialization."""
class LayerSavedModelSaver(base_serialization.SavedModelSaver):
  """Implements Layer SavedModel serialization."""
  @property
  def object_identifier(self): # -> Literal['_tf_keras_layer']:
    ...
  
  @property
  def python_properties(self): # -> dict[str, Any | None]:
    ...
  
  def objects_to_serialize(self, serialization_cache):
    ...
  
  def functions_to_serialize(self, serialization_cache):
    ...
  


def get_serialized(obj): # -> Any | dict[str, Unknown] | None:
  ...

class InputLayerSavedModelSaver(base_serialization.SavedModelSaver):
  """InputLayer serialization."""
  @property
  def object_identifier(self): # -> Literal['_tf_keras_input_layer']:
    ...
  
  @property
  def python_properties(self): # -> dict[str, Unknown]:
    ...
  
  def objects_to_serialize(self, serialization_cache): # -> dict[Unknown, Unknown]:
    ...
  
  def functions_to_serialize(self, serialization_cache): # -> dict[Unknown, Unknown]:
    ...
  


class RNNSavedModelSaver(LayerSavedModelSaver):
  """RNN layer serialization."""
  @property
  def object_identifier(self): # -> Literal['_tf_keras_rnn_layer']:
    ...
  


class IndexLookupLayerSavedModelSaver(LayerSavedModelSaver):
  """Index lookup layer serialization."""
  @property
  def python_properties(self): # -> dict[str, Any | None]:
    ...
  


