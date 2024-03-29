"""
This type stub file was generated by pyright.
"""

"""A library of common shape functions."""
def is_broadcast_compatible(shape_x, shape_y): # -> bool:
  """Returns True if `shape_x` and `shape_y` are broadcast compatible.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    True if a shape exists that both `shape_x` and `shape_y` can be broadcasted
    to.  False otherwise.
  """
  ...

def broadcast_shape(shape_x, shape_y): # -> TensorShape:
  """Returns the broadcasted shape between `shape_x` and `shape_y`.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    A `TensorShape` representing the broadcasted shape.

  Raises:
    ValueError: If the two shapes can not be broadcasted.
  """
  ...

