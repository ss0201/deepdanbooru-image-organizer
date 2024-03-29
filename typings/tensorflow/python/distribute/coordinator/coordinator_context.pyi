"""
This type stub file was generated by pyright.
"""

import contextlib

"""The execution context for ClusterCoordinator."""
cluster_coordinator = ...
_dispatch_context = ...
def get_current_dispatch_context(): # -> Any | None:
  ...

@contextlib.contextmanager
def with_dispatch_context(worker_obj): # -> Generator[None, None, None]:
  ...

class DispatchContext:
  """Context entered when executing a closure on a given worker."""
  def __init__(self, worker_obj) -> None:
    ...
  
  @property
  def worker(self): # -> Unknown:
    ...
  
  @property
  def worker_index(self):
    ...
  
  def maybe_rebuild_remote_values(self, remote_value): # -> None:
    ...
  
  def maybe_get_remote_value(self, ret): # -> Any:
    ...
  


