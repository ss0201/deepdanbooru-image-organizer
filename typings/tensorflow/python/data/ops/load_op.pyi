"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""Implementation of LoadDataset in Python."""
nested_structure_coder = ...
def load(path, element_spec, compression, reader_func): # -> _LoadDataset:
  ...

class _LoadDataset(dataset_ops.DatasetSource):
  """A dataset that loads previously saved dataset."""
  def __init__(self, path, element_spec=..., compression=..., reader_func=...) -> None:
    ...
  
  @property
  def element_spec(self): # -> Any:
    ...
  

