"""
This type stub file was generated by pyright.
"""

"""Utility for eagerly executing operations in parallel on multiple devices."""
_next_device_number = ...
_next_device_number_lock = ...
_all_parallel_devices = ...
def unpack(tensor):
  """Finds `tensor`'s parallel device and unpacks its components."""
  ...

class ParallelDevice:
  """A device which executes operations in parallel."""
  def __init__(self, components) -> None:
    """Creates a device which executes operations in parallel on `components`.

    Args:
      components: A list of device names. Each operation executed on the
        returned device executes on these component devices.

    Returns:
      A string with the name of the newly created device.
    """
    ...
  
  def pack(self, tensors): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
    """Create a tensor on the parallel device from a sequence of tensors.

    Args:
      tensors: A list of tensors, one per device in `self.components`. The list
        can contain composite tensors and nests (lists, dicts, etc. supported by
        `tf.nest`) with the same structure for each device, but every component
        of nests must already be a `tf.Tensor` or composite. Passing
        `tf.Variable` objects reads their value, it does not share a mutable
        reference between the packed and unpacked forms.

    Returns:
      A tensor placed on the ParallelDevice. For nested structures, returns a
      single structure containing tensors placed on the ParallelDevice (same
      structure as each component of `tensors`).

    Raises:
      ValueError: If the length of `tensors` does not match the number of
        component devices, or if there are non-tensor inputs.

    """
    ...
  
  def unpack(self, parallel_tensor): # -> list[Unknown | defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy]:
    """Unpack a parallel tensor into its components.

    Args:
      parallel_tensor: A tensor, composite tensor, or `tf.nest` of such placed
        on the ParallelDevice. Passing `tf.Variable` objects reads their value,
        it does not share a mutable reference between the packed and unpacked
        forms.

    Returns:
      A list with the same length as `self.components` each with the same
      structure as `parallel_tensor`, containing component tensors.

    """
    ...
  
  @property
  def device_ids(self): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
    """A parallel tensor with scalar integers numbering component devices.

    Each device ID is placed on its corresponding device, in the same order as
    the `components` constructor argument.

    Returns:
      A parallel tensor containing 0 on the first device, 1 on the second, etc.
    """
    ...
  
  def __enter__(self): # -> Self@ParallelDevice:
    """Runs ops in parallel, makes variables which save independent buffers."""
    ...
  
  def __exit__(self, typ, exc, tb): # -> None:
    ...
  


