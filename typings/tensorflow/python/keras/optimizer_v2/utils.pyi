"""
This type stub file was generated by pyright.
"""

"""Optimizer utilities."""
def all_reduce_sum_gradients(grads_and_vars): # -> list[Unknown]:
  """Returns all-reduced gradients aggregated via summation.

  Args:
    grads_and_vars: List of (gradient, variable) pairs.

  Returns:
    List of (gradient, variable) pairs where gradients have been all-reduced.
  """
  ...

def filter_empty_gradients(grads_and_vars): # -> tuple[Unknown, ...]:
  """Filter out `(grad, var)` pairs that have a gradient equal to `None`."""
  ...

def make_gradient_clipnorm_fn(clipnorm): # -> ((grads_and_vars: Unknown) -> Unknown) | ((grads_and_vars: Unknown) -> list[tuple[IndexedSlices | Unknown | defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy, Unknown]]):
  """Creates a gradient transformation function for clipping by norm."""
  ...

def make_global_gradient_clipnorm_fn(clipnorm): # -> ((grads_and_vars: Unknown) -> Unknown) | ((grads_and_vars: Unknown) -> list[tuple[IndexedSlices | Unknown, Unknown]]):
  """Creates a gradient transformation function for clipping by norm."""
  ...

def make_gradient_clipvalue_fn(clipvalue): # -> ((grads_and_vars: Unknown) -> Unknown) | ((grads_and_vars: Unknown) -> list[tuple[IndexedSlices | Unknown | _dispatcher_for_maximum | object, Unknown]]):
  """Creates a gradient transformation function for clipping by value."""
  ...

def strategy_supports_no_merge_call(): # -> bool:
  """Returns if the current Strategy can operate in pure replica context."""
  ...

