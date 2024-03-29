"""
This type stub file was generated by pyright.
"""

from enum import Enum

"""Annotations used by the static analyzer."""
class NoValue(Enum):
  def __repr__(self): # -> str:
    ...
  


class NodeAnno(NoValue):
  """Additional annotations used by the static analyzer.

  These are in addition to the basic annotations declared in anno.py.
  """
  IS_LOCAL = ...
  IS_PARAM = ...
  IS_MODIFIED_SINCE_ENTRY = ...
  ARGS_SCOPE = ...
  COND_SCOPE = ...
  ITERATE_SCOPE = ...
  ARGS_AND_BODY_SCOPE = ...
  BODY_SCOPE = ...
  ORELSE_SCOPE = ...


