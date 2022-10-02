"""
This type stub file was generated by pyright.
"""

"""Functions called by the generated code to execute an eager-mode op."""
def quick_execute(op_name, num_outputs, inputs, attrs, ctx, name=...):
  """Execute a TensorFlow operation.

  Args:
    op_name: Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
      execute.
    num_outputs: The number of outputs of the operation to fetch. (Explicitly
      provided instead of being inferred for performance reasons).
    inputs: A list of inputs to the operation. Each entry should be a Tensor, or
      a value which can be passed to the Tensor constructor to create one.
    attrs: A tuple with alternating string attr names and attr values for this
      operation.
    ctx: The value of context.context().
    name: Customized name for the operation.

  Returns:
    List of output Tensor objects. The list is empty if there are no outputs

  Raises:
    An exception on error.
  """
  ...

def execute_with_cancellation(op_name, num_outputs, inputs, attrs, ctx, cancellation_manager, name=...):
  """Execute a TensorFlow operation.

  Args:
    op_name: Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
      execute.
    num_outputs: The number of outputs of the operation to fetch. (Explicitly
      provided instead of being inferred for performance reasons).
    inputs: A list of inputs to the operation. Each entry should be a Tensor, or
      a value which can be passed to the Tensor constructor to create one.
    attrs: A tuple with alternating string attr names and attr values for this
      operation.
    ctx: The value of context.context().
    cancellation_manager: a `CancellationManager` object that can be used to
      cancel the operation.
    name: Customized name for the operation.

  Returns:
    List of output Tensor objects. The list is empty if there are no outputs

  Raises:
    An exception on error.
  """
  ...

def execute_with_callbacks(op_name, num_outputs, inputs, attrs, ctx, name=...):
  """Monkey-patch to execute to enable execution callbacks."""
  ...

execute = ...
def must_record_gradient(): # -> Literal[False]:
  """Import backprop if you want gradients recorded."""
  ...

def record_gradient(unused_op_name, unused_inputs, unused_attrs, unused_outputs): # -> None:
  """Import backprop if you want gradients recorded."""
  ...

def make_float(v, arg_name): # -> float:
  ...

def make_int(v, arg_name): # -> int:
  ...

def make_str(v, arg_name): # -> bytes:
  ...

def make_bool(v, arg_name): # -> bool:
  ...

def make_type(v, arg_name):
  ...

def make_shape(v, arg_name): # -> list[int | None] | None:
  """Convert v into a list."""
  ...

def make_tensor(v, arg_name): # -> TensorProto:
  """Ensure v is a TensorProto."""
  ...

def args_to_matching_eager(l, ctx, allowed_dtypes, default_dtype=...): # -> tuple[Unknown, list[Unknown]] | tuple[Unknown, Unknown] | tuple[Unknown | Any, list[Unknown | Tensor | Any]]:
  """Convert sequence `l` to eager same-type Tensors."""
  ...

def convert_to_mixed_eager_tensors(values, ctx): # -> tuple[list[Unknown | Any], list[Unknown | Tensor | Any]]:
  ...

def args_to_mixed_eager_tensors(lists, ctx): # -> tuple[list[Unknown], list[list[Unknown]]]:
  """Converts a list of same-length lists of values to eager tensors."""
  ...

