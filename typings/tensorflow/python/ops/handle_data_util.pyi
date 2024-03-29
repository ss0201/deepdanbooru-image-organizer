"""
This type stub file was generated by pyright.
"""

"""Decorator to overrides the gradient for a function."""
get_resource_handle_data = ...
def copy_handle_data(source_t, target_t): # -> None:
  """Copies HandleData for variant and resource type tensors if available.

  The CppShapeInferenceResult::HandleData proto contains information about the
  shapes and types of the element tensors of resource/variant type tensors.
  We need to copy this across function boundaries, i.e., when capturing a
  placeholder or when returning a function tensor as output. If we don't do this
  the element tensors will have unknown shapes, e.g., if a TensorList variant
  tensor is captured as a placeholder, elements popped from that list would have
  unknown shape.

  Args:
    source_t: The tensor to copy HandleData from.
    target_t: The tensor to copy HandleData to.
  """
  ...

def set_handle_data(target_t, handle_data): # -> None:
  ...

