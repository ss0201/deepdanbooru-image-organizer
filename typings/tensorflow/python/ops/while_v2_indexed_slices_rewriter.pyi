"""
This type stub file was generated by pyright.
"""

"""Methods for rewriting while_v2 grad functions with IndexedSlices output."""
def rewrite_grad_indexed_slices(grads, body_grad_graph, loop_vars, forward_inputs):
  """Handles special case of IndexedSlices returned from while gradient.

  Some gradient functions return IndexedSlices instead of a Tensor (e.g. the
  gradient of Gather ops). When this happens in the gradient of a while body,
  the resulting gradient body function will have mismatched inputs and outputs,
  since the input is a single Tensor, but the IndexedSlices gets unnested into
  three output Tensors.

  This function fixes this by rewriting the gradient body to have three inputs
  to match the three outputs, i.e., it effectively converts the input Tensor
  into an input IndexedSlices. It also returns new `loop_vars` to reflect the
  new inputs.

  Args:
    grads: the input gradient Tensors to the while gradient computation.
    body_grad_graph: _WhileBodyGradFuncGraph.
    loop_vars: list of Tensors. The inputs to body_grad_graph.
    forward_inputs: list of Tensors. The (flat) inputs to the forward-pass While
      op.

  Returns:
    The new loop_vars to pass to body_grad_graph.
  """
  ...

