"""
This type stub file was generated by pyright.
"""

"""Tensor Tracer report generation utilities."""
_TRACER_LOG_PREFIX = ...
_MARKER_SECTION_BEGIN = ...
_MARKER_SECTION_END = ...
_SECTION_NAME_CONFIG = ...
_SECTION_NAME_REASON = ...
_SECTION_NAME_OP_LIST = ...
_SECTION_NAME_TENSOR_LIST = ...
_SECTION_NAME_CACHE_INDEX_MAP = ...
_SECTION_NAME_GRAPH = ...
_SECTION_NAME_TENSOR_TRACER_CHECKPOINT = ...
_FIELD_NAME_VERSION = ...
_FIELD_NAME_DEVICE = ...
_FIELD_NAME_TRACE_MODE = ...
_FIELD_NAME_SUBMODE = ...
_FIELD_NAME_NUM_REPLICAS = ...
_FIELD_NAME_NUM_REPLICAS_PER_HOST = ...
_FIELD_NAME_NUM_HOSTS = ...
_FIELD_NAME_NUM_OPS = ...
_FIELD_NAME_NUM_TENSORS = ...
_FIELD_NAME_NUM_CACHE_INDICES = ...
_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED = ...
_CURRENT_VERSION = ...
_TT_REPORT_PROTO = ...
def topological_sort(g): # -> tuple[Literal[True], set[Unknown]] | tuple[Literal[False], list[Unknown]]:
  """Performs topological sort on the given graph.

  Args:
     g: the graph.

  Returns:
     A pair where the first element indicates if the topological
     sort succeeded (True if there is no cycle found; False if a
     cycle is found) and the second element is either the sorted
     list of nodes or the cycle of nodes found.
  """
  ...

class TensorTracerConfig:
  """Tensor Tracer config object."""
  def __init__(self) -> None:
    ...
  


class TensorTraceOrder:
  """Class that is responsible from storing the trace-id of the tensors."""
  def __init__(self, graph_order, traced_tensors) -> None:
    ...
  


def sort_tensors_and_ops(graph): # -> GraphWrapper:
  """Returns a wrapper that has consistent tensor and op orders."""
  ...

class OpenReportFile:
  """Context manager for writing report file."""
  def __init__(self, tt_parameters) -> None:
    ...
  
  def __enter__(self): # -> Open | None:
    ...
  
  def __exit__(self, unused_type, unused_value, unused_traceback): # -> None:
    ...
  


def proto_fingerprint(message_proto): # -> str:
  ...

class TTReportHandle:
  """Utility class responsible from creating a tensor tracer report."""
  def __init__(self) -> None:
    ...
  
  def instrument(self, name, explanation): # -> None:
    ...
  
  def instrument_op(self, op, explanation): # -> None:
    ...
  
  def instrument_tensor(self, tensor, explanation): # -> None:
    ...
  
  def create_report_proto(self, tt_config, tt_parameters, tensor_trace_order, tensor_trace_points, collected_signature_types): # -> TensorTracerReport:
    """Creates and returns a proto that stores tensor tracer configuration.

    Args:
      tt_config: TensorTracerConfig object holding information about the run
        environment (device, # cores, # hosts), and tensor tracer version
        information.
      tt_parameters: TTParameters objects storing the user provided parameters
        for tensor tracer.
      tensor_trace_order: TensorTraceOrder object storing a topological order of
        the graph.
      tensor_trace_points: Progromatically added trace_points/checkpoints.
      collected_signature_types: The signature types collected, e,g, norm,
        max, min, mean...
    Returns:
      TensorTracerReport proto.
    """
    ...
  
  def report_proto_path(self, trace_dir, summary_tag_name): # -> LiteralString:
    """Returns the path where report proto should be written.

    Args:
      trace_dir: String denoting the trace directory.
      summary_tag_name: Name of the unique tag that relates to
                        the report.
    Returns:
      A string denoting the path to the report proto.
    """
    ...
  
  def write_report_proto(self, report_path, report_proto, tt_parameters): # -> None:
    """Writes the given report proto under trace_dir."""
    ...
  
  def create_report(self, tt_config, tt_parameters, tensor_trace_order, tensor_trace_points): # -> None:
    """Creates a report file and writes the trace information."""
    ...
  

