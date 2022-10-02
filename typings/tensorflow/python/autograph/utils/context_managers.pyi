"""
This type stub file was generated by pyright.
"""

"""Various context managers."""
def control_dependency_on_returns(return_value): # -> _GeneratorContextManager[Unknown] | NullContextmanager | _ControlDependenciesController:
  """Create a TF control dependency on the return values of a function.

  If the function had no return value, a no-op context is returned.

  Args:
    return_value: The return value to set as control dependency.

  Returns:
    A context manager.
  """
  ...
