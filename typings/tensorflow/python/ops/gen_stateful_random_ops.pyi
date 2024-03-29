"""
This type stub file was generated by pyright.
"""

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: stateful_random_ops.cc
"""
def non_deterministic_ints(shape, dtype=..., name=...):
  r"""Non-deterministically generates some integers.

  This op may use some OS-provided source of non-determinism (e.g. an RNG), so each execution will give different results.

  Args:
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.int64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

NonDeterministicInts = ...
def non_deterministic_ints_eager_fallback(shape, dtype, name, ctx):
  ...

def rng_read_and_skip(resource, alg, delta, name=...):
  r"""Advance the counter of a counter-based RNG.

  The state of the RNG after
  `rng_read_and_skip(n)` will be the same as that after `uniform([n])`
  (or any other distribution). The actual increment added to the
  counter is an unspecified implementation choice.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    alg: A `Tensor` of type `int32`. The RNG algorithm.
    delta: A `Tensor` of type `uint64`. The amount of advancement.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  ...

RngReadAndSkip = ...
def rng_read_and_skip_eager_fallback(resource, alg, delta, name, ctx):
  ...

def rng_skip(resource, algorithm, delta, name=...): # -> None:
  r"""Advance the counter of a counter-based RNG.

  The state of the RNG after
  `rng_skip(n)` will be the same as that after `stateful_uniform([n])`
  (or any other distribution). The actual increment added to the
  counter is an unspecified implementation detail.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    delta: A `Tensor` of type `int64`. The amount of advancement.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

RngSkip = ...
def rng_skip_eager_fallback(resource, algorithm, delta, name, ctx): # -> None:
  ...

def stateful_random_binomial(resource, algorithm, shape, counts, probs, dtype=..., name=...):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    algorithm: A `Tensor` of type `int64`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    counts: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    probs: A `Tensor`. Must have the same type as `counts`.
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulRandomBinomial = ...
def stateful_random_binomial_eager_fallback(resource, algorithm, shape, counts, probs, dtype, name, ctx):
  ...

def stateful_standard_normal(resource, shape, dtype=..., name=...):
  r"""Outputs random values from a normal distribution. This op is deprecated in favor of op 'StatefulStandardNormalV2'

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulStandardNormal = ...
def stateful_standard_normal_eager_fallback(resource, shape, dtype, name, ctx):
  ...

def stateful_standard_normal_v2(resource, algorithm, shape, dtype=..., name=...):
  r"""Outputs random values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulStandardNormalV2 = ...
def stateful_standard_normal_v2_eager_fallback(resource, algorithm, shape, dtype, name, ctx):
  ...

def stateful_truncated_normal(resource, algorithm, shape, dtype=..., name=...):
  r"""Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulTruncatedNormal = ...
def stateful_truncated_normal_eager_fallback(resource, algorithm, shape, dtype, name, ctx):
  ...

def stateful_uniform(resource, algorithm, shape, dtype=..., name=...):
  r"""Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulUniform = ...
def stateful_uniform_eager_fallback(resource, algorithm, shape, dtype, name, ctx):
  ...

def stateful_uniform_full_int(resource, algorithm, shape, dtype=..., name=...):
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers covering the whole range of `dtype`.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.uint64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulUniformFullInt = ...
def stateful_uniform_full_int_eager_fallback(resource, algorithm, shape, dtype, name, ctx):
  ...

def stateful_uniform_int(resource, algorithm, shape, minval, maxval, name=...):
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers in the range `[minval, maxval)`.
  The lower bound `minval` is included in the range, while the upper bound
  `maxval` is excluded.

  The random integers are slightly biased unless `maxval - minval` is an exact
  power of two.  The bias is small for values of `maxval - minval` significantly
  smaller than the range of the output (either `2^32` or `2^64`).

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    minval: A `Tensor`. Minimum value (inclusive, scalar).
    maxval: A `Tensor`. Must have the same type as `minval`.
      Maximum value (exclusive, scalar).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
  """
  ...

StatefulUniformInt = ...
def stateful_uniform_int_eager_fallback(resource, algorithm, shape, minval, maxval, name, ctx):
  ...

