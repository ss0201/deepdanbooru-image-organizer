"""
This type stub file was generated by pyright.
"""

"""Contains utility functions used by summary ops in distribution strategy."""
def skip_summary(): # -> Tensor | Any | Literal[False] | None:
  """Determines if summary should be skipped.

  If using multiple replicas in distributed strategy, skip summaries on all
  replicas except the first one (replica_id=0).

  Returns:
    True if the summary is skipped; False otherwise.
  """
  ...
