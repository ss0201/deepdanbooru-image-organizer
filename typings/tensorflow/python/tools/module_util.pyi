"""
This type stub file was generated by pyright.
"""

"""Helper functions for modules."""
def get_parent_dir(module): # -> str:
  ...

def get_parent_dir_for_name(module_name): # -> LiteralString | None:
  """Get parent directory for module with the given name.

  Args:
    module_name: Module name for e.g.
      tensorflow_estimator.python.estimator.api._v1.estimator.

  Returns:
    Path to the parent directory if module is found and None otherwise.
    Given example above, it should return:
      /pathtoestimator/tensorflow_estimator/python/estimator/api/_v1.
  """
  ...

