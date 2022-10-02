"""
This type stub file was generated by pyright.
"""

"""Miscellaneous utilities that don't fit anywhere else."""
def alias_tensors(*args): # -> Generator[Unknown | defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy, None, None] | defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
  """Wraps any Tensor arguments with an identity op.

  Any other argument, including Variables, is returned unchanged.

  Args:
    *args: Any arguments. Must contain at least one element.

  Returns:
    Same as *args, with Tensor instances replaced as described.

  Raises:
    ValueError: If args doesn't meet the requirements.
  """
  ...

def get_range_len(start, limit, delta): # -> _dispatcher_for_maximum | object:
  ...

