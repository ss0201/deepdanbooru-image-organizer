"""
This type stub file was generated by pyright.
"""

from tensorflow.python.ops import variables
from tensorflow.python.types import core

"""Contains AutoCastVariable, a variable which automatically casts itself."""
_autocast_dtype = ...
def numpy_text(tensor, is_repr=...): # -> str:
  """Human readable representation of a tensor's numpy value."""
  ...

class AutoCastVariable(variables.Variable, core.Tensor):
  """Variable that will cast itself to a different dtype in applicable contexts.

  This class wraps a floating-point `tf.Variable`. It emulates the variable
  interface and delegates to the wrapped variable, but it additionally will cast
  the wrapped variable under an `enable_auto_cast_variables(dtype)` context
  manager.

  For example:

  >>> v = tf.Variable(1.0, dtype=tf.float32)
  >>> v = AutoCastVariable(v)
  >>> tf.identity(v).dtype
  tf.float32
  >>> with enable_auto_cast_variables(tf.float16):
  ...   tf.identity(v).dtype
  tf.float16

  The purpose of this class is to allow Keras layers to create variables in
  float32, and automatically cast them to float16 or bfloat16 when the layer is
  called.
  """
  def __init__(self, variable) -> None:
    """Creates an AutoCastVariable instance.

    Args:
      variable: A floating-point resource variable to wrap.

    Raises:
      ValueError: If `variable` is not a floating-point resource variable
    """
    ...
  
  @property
  def dtype(self):
    """The dtype of the underlying variable, before any casts are done."""
    ...
  
  @property
  def true_dtype(self):
    """Deprecated alias of `dtype`."""
    ...
  
  def value(self): # -> SparseTensor | IndexedSlices | Tensor | Any:
    ...
  
  def read_value(self): # -> SparseTensor | IndexedSlices | Tensor | Any:
    ...
  
  def sparse_read(self, indices, name=...):
    """Reads the value of this variable sparsely, using `gather`."""
    ...
  
  def gather_nd(self, indices, name=...):
    """Gather slices of the variable into a Tensor."""
    ...
  
  def __getattr__(self, name): # -> Any:
    ...
  
  def __repr__(self): # -> str:
    ...
  
  def set_shape(self, shape):
    ...
  
  @property
  def trainable(self):
    ...
  
  @property
  def synchronization(self):
    ...
  
  @property
  def aggregation(self):
    ...
  
  def eval(self, session=...):
    ...
  
  def initialized_value(self): # -> Any | list[Unknown] | _basetuple | defaultdict[Unknown, Unknown] | ObjectProxy:
    ...
  
  @property
  def initial_value(self):
    ...
  
  @property
  def constraint(self):
    ...
  
  def assign(self, value, use_locking=..., name=..., read_value=...): # -> AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def assign_add(self, delta, use_locking=..., name=..., read_value=...): # -> AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def assign_sub(self, delta, use_locking=..., name=..., read_value=...): # -> AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_sub(self, sparse_delta, use_locking=..., name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_add(self, sparse_delta, use_locking=..., name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_max(self, sparse_delta, use_locking=..., name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_min(self, sparse_delta, use_locking=..., name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_mul(self, sparse_delta, use_locking=..., name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_div(self, sparse_delta, use_locking=..., name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_update(self, sparse_delta, use_locking=..., name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def batch_scatter_update(self, sparse_delta, use_locking=..., name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_nd_sub(self, indices, updates, name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_nd_add(self, indices, updates, name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def scatter_nd_update(self, indices, updates, name=...): # -> Self@AutoCastVariable | AutoCastVariable | AutoCastDistributedVariable:
    ...
  
  def load(self, value, session=...): # -> None:
    ...
  
  @property
  def name(self):
    ...
  
  @property
  def initializer(self):
    ...
  
  @property
  def device(self):
    ...
  
  @property
  def op(self): # -> str:
    ...
  
  @property
  def graph(self):
    ...
  
  @property
  def shape(self):
    ...
  
  def get_shape(self):
    ...
  
  def to_proto(self, export_scope=...):
    ...
  
  def from_proto(self, variable_def, import_scope=...): # -> RefVariable:
    ...
  
  def __add__(self, o):
    ...
  
  def __radd__(self, o):
    ...
  
  def __sub__(self, o):
    ...
  
  def __rsub__(self, o):
    ...
  
  def __mul__(self, o):
    ...
  
  def __rmul__(self, o):
    ...
  
  def __truediv__(self, o):
    ...
  
  def __rtruediv__(self, o):
    ...
  
  def __floordiv__(self, o):
    ...
  
  def __rfloordiv__(self, o):
    ...
  
  def __mod__(self, o):
    ...
  
  def __rmod__(self, o):
    ...
  
  def __lt__(self, o) -> bool:
    ...
  
  def __le__(self, o) -> bool:
    ...
  
  def __gt__(self, o) -> bool:
    ...
  
  def __ge__(self, o) -> bool:
    ...
  
  def __getitem__(self, o): # -> Any:
    ...
  
  def __pow__(self, o, modulo=...):
    ...
  
  def __rpow__(self, o):
    ...
  
  def __neg__(self):
    ...
  
  def __abs__(self):
    ...
  
  def __div__(self, o): # -> Any | _NotImplementedType:
    ...
  
  def __rdiv__(self, o): # -> Any | _NotImplementedType:
    ...
  
  def __matmul__(self, o): # -> Any | _NotImplementedType:
    ...
  
  def __rmatmul__(self, o): # -> Any | _NotImplementedType:
    ...
  


def create_autocast_variable(variable): # -> AutoCastVariable | AutoCastDistributedVariable:
  """Creates an AutoCastVariable that wraps another variable.

  This typically just returns `AutoCastVariable(variable)`. But, if the variable
  is a DistributedVariable or one of its subclasses, we instead dynamically
  create a class that subclasses from both AutoCastVariable and
  variable.__class__. This is so the returned variable will still pass
  `isinstance(variable, variable.__class__)`, which is required for
  DistributedVariables and its subclasses to work properly.

  Args:
    variable: A floating-point resource variable to wrap.

  Returns:
    An AutoCastVariable that wraps the variable.
  """
  class AutoCastDistributedVariable(AutoCastVariable, variable.__class__):
    """An AutoCastVariable that also subclasses from variable.__class__.

    variable.__class__ is either a DistributedVariable or an
    AggregatingVariable.
    """
    ...
  
  

class enable_auto_cast_variables:
  """Context manager which enables the autocasting of `AutoCastVariable`s.

  Under this context manager, `AutoCastVariable`s will be cast to `dtype` if
  `dtype` is floating-point. Otherwise, `AutoCastVariable`s will not be cast.
  """
  __slots__ = ...
  def __init__(self, dtype) -> None:
    ...
  
  def __enter__(self): # -> None:
    ...
  
  def __exit__(self, type_arg, value_arg, traceback_arg): # -> None:
    ...
  


