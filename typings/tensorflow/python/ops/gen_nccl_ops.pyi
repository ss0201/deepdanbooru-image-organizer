"""
This type stub file was generated by pyright.
"""

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: nccl_ops.cc
"""
def nccl_all_reduce(input, reduction, num_devices, shared_name, name=...):
  r"""Outputs a tensor containing the reduction across all input tensors.

  Outputs a tensor containing the reduction across all input tensors passed to ops
  within the same `shared_name.

  The graph should be constructed so if one op runs with shared_name value `c`,
  then `num_devices` ops will run with shared_name value `c`.  Failure to do so
  will cause the graph execution to fail to complete.

  input: the input to the reduction
  data: the value of the reduction across all `num_devices` devices.
  reduction: the reduction operation to perform.
  num_devices: The number of devices participating in this reduction.
  shared_name: Identifier that shared between ops of the same reduction.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    reduction: A `string` from: `"min", "max", "prod", "sum"`.
    num_devices: An `int`.
    shared_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

NcclAllReduce = ...
def nccl_all_reduce_eager_fallback(input, reduction, num_devices, shared_name, name, ctx):
  ...

def nccl_broadcast(input, shape, name=...):
  r"""Sends `input` to all devices that are connected to the output.

  Sends `input` to all devices that are connected to the output.

  The graph should be constructed so that all ops connected to the output have a
  valid device assignment, and the op itself is assigned one of these devices.

  input: The input to the broadcast.
  output: The same as input.
  shape: The shape of the input tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    shape: A `tf.TensorShape` or list of `ints`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

NcclBroadcast = ...
def nccl_broadcast_eager_fallback(input, shape, name, ctx):
  ...

def nccl_reduce(input, reduction, name=...):
  r"""Reduces `input` from `num_devices` using `reduction` to a single device.

  Reduces `input` from `num_devices` using `reduction` to a single device.

  The graph should be constructed so that all inputs have a valid device
  assignment, and the op itself is assigned one of these devices.

  input: The input to the reduction.
  data: the value of the reduction across all `num_devices` devices.
  reduction: the reduction operation to perform.

  Args:
    input: A list of at least 1 `Tensor` objects with the same type in: `half`, `float32`, `float64`, `int32`, `int64`.
    reduction: A `string` from: `"min", "max", "prod", "sum"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

NcclReduce = ...
def nccl_reduce_eager_fallback(input, reduction, name, ctx):
  ...
