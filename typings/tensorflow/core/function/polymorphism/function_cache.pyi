"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, Hashable, NamedTuple, Optional, Sequence
from tensorflow.core.function import trace_type
from tensorflow.python.types import trace

"""Cache to manage concrete functions and their signatures."""
DELETE_WITH_WEAKREF = ...
class FunctionContext(NamedTuple):
  """Contains information regarding tf.function execution context."""
  context: Any
  ...


class CaptureSnapshot(trace.TraceType):
  """Store tf.function captures to accommodate its specific tracing logic.

  Captures are stored in mapping format, but its tracing logic is different from
  Python dict. When comparing types of two normal Python dicts in function
  argumenst, their keys are required to be the same. When comparing types for
  captures, keys can be different. This is because tf.function maintains a full
  list of captures and only a subset is active for each ConcreteFunction.
  But before dispatch, which captures are active is unknown, so all caputres are
  evaluated for comparison. Please also check `is_subtype_of` method.

  Attributes:
    mapping: A mapping from keys to corresponding TraceTypes of the dict values.
  """
  def __init__(self, mapping: Dict[Hashable, trace.TraceType]) -> None:
    ...
  
  def is_subtype_of(self, query: CaptureSnapshot) -> bool:
    """This method is used to check if `self` is a subtype of query.

    Typically, self represents an existing snapshot for a ConcreteFunction, and
    the query is a snapshot from all captures with runtime values. Keys in the
    query should be a superset of self.
    This method differs from default_types.Dict as this CaptureSnapshot doesn't
    require a full match of keys.

      For example:

      a = CaptureSnapshot({'x'=1, 'y'=2})
      b = CaptureSnapshot({'x'=1, 'y'=2, 'z'=3})
      assert not a.is_subtype_of(b)
      assert b.is_subtype_of(a)

    Args:
      query: A CaptureSnapshot instance that represents the current runtime
        values of all captures.

    Returns:
      A bool value represents the result.
    """
    ...
  
  def most_specific_common_supertype(self, types: Sequence[trace.TraceType]) -> Optional[CaptureSnapshot]:
    """See base class."""
    ...
  
  def __eq__(self, other: CaptureSnapshot) -> bool:
    ...
  
  def __hash__(self) -> int:
    ...
  


class FunctionCacheKey(trace.TraceType):
  """The unique key associated with a concrete function.

  Attributes:
    args_signature: A TraceType corresponding to the function arguments.
    captures_signature: A CaptureSnapshot corresponding to the function
      captures.
    call_context: The FunctionContext for when the args_signature was
      generated.
  """
  def __init__(self, args_signature: trace.TraceType, captures_signature: CaptureSnapshot, call_context: FunctionContext) -> None:
    ...
  
  def is_subtype_of(self, other: trace.TraceType) -> bool:
    ...
  
  def most_specific_common_supertype(self, others: Sequence[trace.TraceType]) -> Optional[FunctionCacheKey]:
    ...
  
  def __hash__(self) -> int:
    ...
  
  def __eq__(self, other) -> bool:
    ...
  
  def __repr__(self) -> str:
    ...
  


class FunctionCache:
  """A container for managing concrete functions."""
  __slots__ = ...
  def __init__(self) -> None:
    ...
  
  def lookup(self, key: FunctionCacheKey, use_function_subtyping: bool): # -> None:
    """Looks up a concrete function based on the key."""
    ...
  
  def delete(self, key: FunctionCacheKey): # -> bool:
    """Deletes a concrete function given the key it was added with."""
    ...
  
  def add(self, key: FunctionCacheKey, deletion_observer: trace_type.WeakrefDeletionObserver, concrete): # -> None:
    """Adds a new concrete function alongside its key.

    Args:
      key: A FunctionCacheKey object corresponding to the provided `concrete`.
      deletion_observer: A WeakrefDeletionObserver object for the `key`.
      concrete: The concrete function to be added to the cache.
    """
    ...
  
  def generalize(self, key: FunctionCacheKey) -> FunctionCacheKey:
    ...
  
  def clear(self): # -> None:
    """Removes all concrete functions from the cache."""
    ...
  
  def values(self): # -> list[Unknown]:
    """Returns a list of all `ConcreteFunction` instances held by this cache."""
    ...
  


