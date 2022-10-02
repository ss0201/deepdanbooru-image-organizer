"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import keras_export

"""Layer serialization/deserialization functions.
"""
ALL_MODULES = ...
ALL_V2_MODULES = ...
LOCAL = ...
def populate_deserializable_objects(): # -> None:
  """Populates dict ALL_OBJECTS with every built-in layer.
  """
  ...

@keras_export('keras.layers.serialize')
def serialize(layer): # -> Any | dict[str, Unknown] | None:
  ...

@keras_export('keras.layers.deserialize')
def deserialize(config, custom_objects=...): # -> Any | None:
  """Instantiates a layer from a config dictionary.

  Args:
      config: dict of the form {'class_name': str, 'config': dict}
      custom_objects: dict mapping class names (or function names)
          of custom (non-Keras) objects to class/functions

  Returns:
      Layer instance (may be Model, Sequential, Network, Layer...)
  """
  ...
