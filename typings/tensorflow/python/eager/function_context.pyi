"""
This type stub file was generated by pyright.
"""

from typing import Any, NamedTuple, Tuple
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_cache

"""Context information for a tf.function."""
class EagerContext(NamedTuple):
  parent_graph: Any
  device_functions: Any
  colocation_stack: Any
  in_cross_replica_context: Any
  variable_policy: Any
  xla_context_id: Any
  ...


def make_function_context() -> function_cache.FunctionContext:
  """Generates a FunctionContext based on current contextual info."""
  ...

def make_cache_key(args: Any, captures: Any = ...) -> Tuple[function_cache.FunctionCacheKey, trace_type.WeakrefDeletionObserver]:
  """Computes the cache key given the function arguments."""
  ...
