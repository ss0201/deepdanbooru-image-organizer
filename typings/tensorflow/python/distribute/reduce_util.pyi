"""
This type stub file was generated by pyright.
"""

import enum
from tensorflow.python.util.tf_export import tf_export

"""Utilities for reduce operations."""
@tf_export("distribute.ReduceOp")
class ReduceOp(enum.Enum):
  """Indicates how a set of values should be reduced.

  * `SUM`: Add all the values.
  * `MEAN`: Take the arithmetic mean ("average") of the values.
  """
  SUM = ...
  MEAN = ...
  @staticmethod
  def from_variable_aggregation(aggregation): # -> ReduceOp:
    ...
  

