"""
This type stub file was generated by pyright.
"""

from tensorflow.python.ops import op_selector
from tensorflow.python.util.tf_export import tf_export

"""Utility to lift subgraphs."""
UnliftableError = op_selector.UnliftableError
_InputMutation = ...
_ControlMutation = ...
@tf_export("__internal__.lift_to_graph", v1=[])
def lift_to_graph(tensors, graph, sources=..., disallowed_placeholders=..., add_sources=..., handle_captures=..., base_graph=..., op_map=...): # -> ObjectIdentityDictionary:
  """Copies the tensor and all its inputs recursively to the outer graph.

  Args:
    tensors: The Tensors to lift.
    graph: The graph to lift to.
    sources: Optional sequence of nodes to start from. If omitted the whole
      subgraph which feeds into `init_tensor` is lifted.
    disallowed_placeholders: An optional set of ops which may not appear in the
      lifted graph. Defaults to all placeholders.
    add_sources: A boolean indicating whether placeholders which are not in
      sources should be allowed.
    handle_captures: A boolean indicating whether to re-capture s in the new
      graph or simply create a vanilla placeholder.
    base_graph: The graph from which to lift ops. This will be inferred if not
      specified.
    op_map: A map contains all the existing nodes that have been lifted to the
      destination graph, so they won't be lifted and copied again.

  Returns:
    A mapping from ops in the current default graph to ops in `graph`.

  Raises:
    UnliftableError: If a placeholder blocks lifting.
  """
  ...

