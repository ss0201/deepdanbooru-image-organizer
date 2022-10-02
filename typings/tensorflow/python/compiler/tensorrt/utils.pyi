"""
This type stub file was generated by pyright.
"""

"""Exposes the Python wrapper conversion to trt_graph."""
def disable_non_trt_optimizers_in_rewriter_config(rewriter_config): # -> None:
  """Modifies rewriter_config to disable all non-TRT optimizations."""
  ...

def version_tuple_to_string(ver_tuple): # -> str:
  ...

def is_linked_tensorrt_version_greater_equal(major, minor=..., patch=...): # -> bool:
  ...

def is_loaded_tensorrt_version_greater_equal(major, minor=..., patch=...): # -> bool:
  ...

def is_experimental_feature_activated(feature_name): # -> bool:
  """Determines if a TF-TRT experimental feature is enabled.

  This helper function checks if an experimental feature was enabled using
  the environment variable `TF_TRT_EXPERIMENTAL_FEATURES=feature_1,feature_2`.

  Args:
    feature_name: Name of the feature being tested for activation.
  """
  ...

def get_node_compute_dtype(node): # -> str | list[str] | None:
  """Returns the compute DType of a GraphDef Node."""
  ...

def get_node_io_shapes(node, key): # -> list[Unknown]:
  """Returns the input/output shapes of a GraphDef Node."""
  ...

def get_trtengineop_io_dtypes(node, key): # -> str | list[str]:
  """Returns the input/output dtypes of a TRTEngineOp."""
  ...

def get_trtengineop_io_nodes_count(node, key): # -> int:
  """Returns the number of input/output nodes of a TRTEngineOp."""
  ...

def get_trtengineop_node_op_count(graphdef, node_name): # -> tuple[int | Unbound, defaultdict[Unknown, int]]:
  """Counts the number of nodes and OP types of a given TRTEngineOp."""
  ...

class DTypeIndex(dict):
  """Helper class to create an index of dtypes with incremental values."""
  def get_dtype_index(self, dtype):
    ...
  


def draw_graphdef_as_graphviz(graphdef, dot_output_filename): # -> None:
  """Exports a GraphDef to GraphViz format.

  - Step 1: Drawing Each Node of the compute GraphDef.
  - Step 2: Create nodes for each collected dtype in the graph.
  - Step 3: Creating invisible links to align properly the legend.

  Each node consequently mentions:
  - Op Type
  - Compute Dtype
  - Compute Device
  """
  ...

