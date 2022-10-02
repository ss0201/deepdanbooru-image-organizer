"""
This type stub file was generated by pyright.
"""

import dataclasses

"""Test utilities."""
@dataclasses.dataclass
class TestClusterParams:
  cluster: dict
  max_num_worker: int
  max_num_ps: int
  ...


def get_cluster_def(cluster_params, num_workers, num_ps): # -> dict[str, Unknown]:
  ...

def gather(strategy, value): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
  """Gathers value from all workers.

  This is intended for tests before we implement an official all-gather API.

  Args:
    strategy: a `tf.distribute.Strategy`.
    value: a nested structure of n-dim `tf.distribute.DistributedValue` of
      `tf.Tensor`, or of a `tf.Tensor` if the strategy only has one replica.
      Cannot contain tf.sparse.SparseTensor.

  Returns:
    a (n+1)-dim `tf.Tensor`.
  """
  ...

def set_logical_devices_to_at_least(device, num): # -> None:
  """Create logical devices of at least a given number."""
  ...

def main(enable_v2_behavior=..., config_logical_devices=...): # -> None:
  """All-in-one main function for tf.distribute tests."""
  ...

def topological_sort_operations(operations): # -> dict[Unknown, Unknown]:
  """Topological sorts a list of operations.

  This does a topological sort of the operations in a graph. The edges include
  both data dependencies and control dependencies. Note that the edge goes from
  an operation to its dependencies.

  The sort is intentionally unstable, reversing orders of operations and
  dependencies on ties.

  Args:
    operations: a list of tf.Operation in the same graph.

  Returns:
    A map from a tf.Operation to its topological order.
  """
  ...

def assert_sequential_execution(order, operations): # -> None:
  """Asserts there's a deterministic execution order between the operations.

  Args:
    order: a map from a tf.Operation to its topological order.
    operations: a list of operations that should be executed sequentially. It
      can be given in any order.
  """
  ...

def get_running_threads(): # -> set[Unknown]:
  """Returns a set of all running thread names."""
  ...

def has_thread(prefix, running_threads): # -> bool:
  """Returns whether any 'running_threads' is prefixed with 'prefix'.

  Args:
    prefix: The prefix of the expected thread name.
    running_threads: A collection of the running thread names.
  """
  ...

def show_backref(target, max_depth=...): # -> str:
  """Returns a dot graph of all the objects that are referencing the target.

  A object referencing graph is useful to debug memory leak like circular
  reference. objgraph provides a good visualization of the memory graph than
  most python built-in utilities like gc.get_referrers(), which are not
  human-readable sometimes.

  The dot graph will be written to a string IO object, and can be rendered with
  graphviz in operating system.
  E.g. dot -Tpng {$dot_graph} -o output.png
  Args:
    target: The target object for the memory graph.
    max_depth: The maximum depth of the graph. By default 3 layers of references
      are used. Increases this a lot may result in the graph growing too big.

  Returns:
    A string that contains the object reference graph.
  Raises:
    NotImplementedError: if objgraph is not installed.
  """
  ...

def create_per_replica(strategy, value_list): # -> PerReplica:
  """Creates a PerReplica of Tensors from the value_list."""
  ...

def is_tpu_strategy(strategy): # -> bool:
  """Returns whether the strategy is a TPU strategy."""
  ...

