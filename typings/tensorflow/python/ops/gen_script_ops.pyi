"""
This type stub file was generated by pyright.
"""

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: script_ops.cc
"""
def eager_py_func(input, token, Tout, is_async=..., name=...):
  r"""Eagerly executes a python function to compute func(input)->output. The

  semantics of the input, output, and attributes are the same as those for
  PyFunc.

  Args:
    input: A list of `Tensor` objects.
    token: A `string`.
    Tout: A list of `tf.DTypes`.
    is_async: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  ...

EagerPyFunc = ...
def eager_py_func_eager_fallback(input, token, Tout, is_async, name, ctx):
  ...

def py_func(input, token, Tout, name=...):
  r"""Invokes a python function to compute func(input)->output.

  This operation is considered stateful. For a stateless version, see
  PyFuncStateless.

  Args:
    input: A list of `Tensor` objects.
      List of Tensors that will provide input to the Op.
    token: A `string`.
      A token representing a registered python function in this address space.
    Tout: A list of `tf.DTypes`. Data types of the outputs from the op.
      The length of the list specifies the number of outputs.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  ...

PyFunc = ...
def py_func_eager_fallback(input, token, Tout, name, ctx):
  ...

def py_func_stateless(input, token, Tout, name=...):
  r"""A stateless version of PyFunc.

  Args:
    input: A list of `Tensor` objects.
    token: A `string`.
    Tout: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  ...

PyFuncStateless = ...
def py_func_stateless_eager_fallback(input, token, Tout, name, ctx):
  ...
