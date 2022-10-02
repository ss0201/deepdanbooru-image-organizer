"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import tf_export

"""Resource management library."""
@tf_export(v1=['resource_loader.load_resource'])
def load_resource(path): # -> bytes:
  """Load the resource at given path, where path is relative to tensorflow/.

  Args:
    path: a string resource path relative to tensorflow/.

  Returns:
    The contents of that resource.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  ...

@tf_export(v1=['resource_loader.get_data_files_path'])
def get_data_files_path(): # -> str:
  """Get a direct path to the data files colocated with the script.

  Returns:
    The directory where files specified in data attribute of py_test
    and py_binary are stored.
  """
  ...

@tf_export(v1=['resource_loader.get_root_dir_with_all_resources'])
def get_root_dir_with_all_resources(): # -> str:
  """Get a root directory containing all the data attributes in the build rule.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary. Falls back to returning the same as get_data_files_path if it
    fails to detect a bazel runfiles directory.
  """
  ...

@tf_export(v1=['resource_loader.get_path_to_datafile'])
def get_path_to_datafile(path): # -> str:
  """Get the path to the specified file in the data dependencies.

  The path is relative to tensorflow/

  Args:
    path: a string resource path relative to tensorflow/

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  ...

@tf_export(v1=['resource_loader.readahead_file_path'])
def readahead_file_path(path, readahead=...):
  """Readahead files not implemented; simply returns given path."""
  ...
