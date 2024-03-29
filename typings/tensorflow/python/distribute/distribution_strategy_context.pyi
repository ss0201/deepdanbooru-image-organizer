"""
This type stub file was generated by pyright.
"""

import contextlib
from tensorflow.python.util.tf_export import tf_export

"""Utility to get tf.distribute.Strategy related contexts."""
distribute_lib = ...
class _ThreadMode:
  def __init__(self, dist, cross, replica) -> None:
    ...
  


class _CrossReplicaThreadMode(_ThreadMode):
  def __init__(self, strategy) -> None:
    ...
  


class _InReplicaThreadMode(_ThreadMode):
  def __init__(self, replica_ctx) -> None:
    ...
  


class _DefaultReplicaThreadMode(_ThreadMode):
  """Type of default value returned by `_get_per_thread_mode()`.

  Used when the thread-local stack is empty.
  """
  def __init__(self) -> None:
    ...
  


_variable_sync_on_read_context = ...
@tf_export("__internal__.distribute.variable_sync_on_read_context", v1=[])
@contextlib.contextmanager
def variable_sync_on_read_context(): # -> Generator[None, None, None]:
  """A context that forces SyncOnReadVariable to aggregate upon reading.

  This context is useful if one wants to read the aggregated value out of a
  SyncOnReadVariable in replica context. By default the aggregation is turned
  off per the definition of SyncOnReadVariable.

  When reading a SyncOnReadVariable in cross-replica context, aggregation is
  always turned on so there is no need for such context.

  By reading a SyncOnReadVariable, we mean:
    1. Convert the variable to a tensor using `convert_to_tensor`.
    2. Calling `variable.value()` or `variable.read_value()`.

  Example usage:

  ```
  strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
  with strategy.scope():
    v = tf.Variable(1.0, synchronization=tf.VariableSynchronization.ON_READ,
      aggregation=tf.VariableAggregation.SUM)

  def replica_fn():
    return v + 10.0

  non_aggregated = strategy.run(replica_fn)
  print(non_aggregated) # PerReplica: {0: 11.0, 1: 11.0}

  def replica_fn():
    with variable_sync_on_read_context():
      return v + 10.0

  aggregated = strategy.run(replica_fn)
  print(aggregated) # PerReplica: {0: 12.0, 1: 12.0}
  ```

  Yields:
    Context manager for aggregating SyncOnReadVariable upon reading.
  """
  ...

def in_variable_sync_on_read_context(): # -> Any | Literal[False]:
  ...

@tf_export("distribute.get_replica_context")
def get_replica_context(): # -> Any | Unknown:
  """Returns the current `tf.distribute.ReplicaContext` or `None`.

  Returns `None` if in a cross-replica context.

  Note that execution:

  1. starts in the default (single-replica) replica context (this function
     will return the default `ReplicaContext` object);
  2. switches to cross-replica context (in which case this will return
     `None`) when entering a `with tf.distribute.Strategy.scope():` block;
  3. switches to a (non-default) replica context inside `strategy.run(fn, ...)`;
  4. if `fn` calls `get_replica_context().merge_call(merge_fn, ...)`, then
     inside `merge_fn` you are back in the cross-replica context (and again
     this function will return `None`).

  Most `tf.distribute.Strategy` methods may only be executed in
  a cross-replica context, in a replica context you should use the
  API of the `tf.distribute.ReplicaContext` object returned by this
  method instead.

  ```
  assert tf.distribute.get_replica_context() is not None  # default
  with strategy.scope():
    assert tf.distribute.get_replica_context() is None

    def f():
      replica_context = tf.distribute.get_replica_context()  # for strategy
      assert replica_context is not None
      tf.print("Replica id: ", replica_context.replica_id_in_sync_group,
               " of ", replica_context.num_replicas_in_sync)

    strategy.run(f)
  ```

  Returns:
    The current `tf.distribute.ReplicaContext` object when in a replica context
    scope, else `None`.

    Within a particular block, exactly one of these two things will be true:

    * `get_replica_context()` returns non-`None`, or
    * `tf.distribute.is_cross_replica_context()` returns True.
  """
  ...

def get_cross_replica_context(): # -> Any | Unknown:
  """Returns the current tf.distribute.Strategy if in a cross-replica context.

  DEPRECATED: Please use `in_cross_replica_context()` and
  `get_strategy()` instead.

  Returns:
    Returns the current `tf.distribute.Strategy` object in a cross-replica
    context, or `None`.

    Exactly one of `get_replica_context()` and `get_cross_replica_context()`
    will return `None` in a particular block.
  """
  ...

@tf_export("distribute.in_cross_replica_context")
def in_cross_replica_context(): # -> bool:
  """Returns `True` if in a cross-replica context.

  See `tf.distribute.get_replica_context` for details.

  ```
  assert not tf.distribute.in_cross_replica_context()
  with strategy.scope():
    assert tf.distribute.in_cross_replica_context()

    def f():
      assert not tf.distribute.in_cross_replica_context()

    strategy.run(f)
  ```

  Returns:
    `True` if in a cross-replica context (`get_replica_context()` returns
    `None`), or `False` if in a replica context (`get_replica_context()` returns
    non-`None`).
  """
  ...

@tf_export("distribute.get_strategy")
def get_strategy(): # -> Any | Unknown:
  """Returns the current `tf.distribute.Strategy` object.

  Typically only used in a cross-replica context:

  ```
  if tf.distribute.in_cross_replica_context():
    strategy = tf.distribute.get_strategy()
    ...
  ```

  Returns:
    A `tf.distribute.Strategy` object. Inside a `with strategy.scope()` block,
    it returns `strategy`, otherwise it returns the default (single-replica)
    `tf.distribute.Strategy` object.
  """
  ...

@tf_export("distribute.has_strategy")
def has_strategy(): # -> bool:
  """Return if there is a current non-default `tf.distribute.Strategy`.

  ```
  assert not tf.distribute.has_strategy()
  with strategy.scope():
    assert tf.distribute.has_strategy()
  ```

  Returns:
    True if inside a `with strategy.scope():`.
  """
  ...

def get_strategy_and_replica_context(): # -> tuple[Unknown | Any, Unknown | Any]:
  ...

@tf_export("distribute.experimental_set_strategy")
def experimental_set_strategy(strategy): # -> None:
  """Set a `tf.distribute.Strategy` as current without `with strategy.scope()`.

  ```
  tf.distribute.experimental_set_strategy(strategy1)
  f()
  tf.distribute.experimental_set_strategy(strategy2)
  g()
  tf.distribute.experimental_set_strategy(None)
  h()
  ```

  is equivalent to:

  ```
  with strategy1.scope():
    f()
  with strategy2.scope():
    g()
  h()
  ```

  In general, you should use the `with strategy.scope():` API, but this
  alternative may be convenient in notebooks where you would have to put
  each cell in a `with strategy.scope():` block.

  Note: This should only be called outside of any TensorFlow scope to
  avoid improper nesting.

  Args:
    strategy: A `tf.distribute.Strategy` object or None.

  Raises:
    RuntimeError: If called inside a `with strategy.scope():`.
  """
  ...

@contextlib.contextmanager
def enter_or_assert_strategy(strategy): # -> Generator[None, None, None]:
  ...

_defaults = ...
_default_strategy_lock = ...
_default_replica_context_lock = ...
_default_replica_mode_lock = ...
get_distribution_strategy = ...
has_distribution_strategy = ...
