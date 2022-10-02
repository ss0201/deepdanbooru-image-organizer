"""
This type stub file was generated by pyright.
"""

import collections

"""Code transformation exceptions."""
class FrameInfo(collections.namedtuple('FrameInfo', ('filename', 'lineno', 'function_name', 'code', 'is_converted', 'is_allowlisted'))):
  __slots__ = ...


KNOWN_STRING_CONSTRUCTOR_ERRORS = ...
class MultilineMessageKeyError(KeyError):
  def __init__(self, message, original_key) -> None:
    ...
  
  def __str__(self) -> str:
    ...
  


class ErrorMetadataBase:
  """Container objects attached to exceptions raised in user code.

  This metadata allows re-raising exceptions that occur in generated code, with
  a custom error message that includes a stack trace relative to user-readable
  code from which the generated code originated.
  """
  __slots__ = ...
  def __init__(self, callsite_tb, cause_metadata, cause_message, source_map, converter_filename) -> None:
    ...
  
  def get_message(self): # -> LiteralString:
    """Returns the message for the underlying exception."""
    ...
  
  def create_exception(self, source_error): # -> Any | MultilineMessageKeyError | None:
    """Creates exception from source_error."""
    ...
  
  def to_exception(self, source_error): # -> Any | MultilineMessageKeyError | None:
    ...
  


