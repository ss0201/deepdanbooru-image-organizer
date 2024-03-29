"""
This type stub file was generated by pyright.
"""

from absl.flags import *

"""Import router for absl.flags. See https://github.com/abseil/abseil-py."""
_RENAMED_ARGUMENTS = ...
class _FlagValuesWrapper:
  """Wrapper class for absl.flags.FLAGS.

  The difference is that tf.flags.FLAGS implicitly parses flags with sys.argv
  when accessing the FLAGS values before it's explicitly parsed,
  while absl.flags.FLAGS raises an exception.
  """
  def __init__(self, flags_object) -> None:
    ...
  
  def __getattribute__(self, name): # -> Any:
    ...
  
  def __getattr__(self, name): # -> Any:
    ...
  
  def __setattr__(self, name, value): # -> Any:
    ...
  
  def __delattr__(self, name): # -> Any:
    ...
  
  def __dir__(self): # -> Any:
    ...
  
  def __getitem__(self, name): # -> Any:
    ...
  
  def __setitem__(self, name, flag): # -> Any:
    ...
  
  def __len__(self): # -> Any:
    ...
  
  def __iter__(self): # -> Any:
    ...
  
  def __str__(self) -> str:
    ...
  
  def __call__(self, *args, **kwargs): # -> Any:
    ...
  


DEFINE_string = ...
DEFINE_boolean = ...
DEFINE_bool = ...
DEFINE_float = ...
DEFINE_integer = ...
FLAGS = ...
