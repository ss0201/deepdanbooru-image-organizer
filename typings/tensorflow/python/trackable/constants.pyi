"""
This type stub file was generated by pyright.
"""

import enum

"""Constants used in Trackable for checkpointing and serialization."""
OBJECT_GRAPH_PROTO_KEY = ...
VARIABLE_VALUE_KEY = ...
OBJECT_CONFIG_JSON_KEY = ...
@enum.unique
class SaveType(str, enum.Enum):
  SAVEDMODEL = ...
  CHECKPOINT = ...


