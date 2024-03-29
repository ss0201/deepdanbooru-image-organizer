"""
This type stub file was generated by pyright.
"""

"""Utility functions for control flow.

This file is necessary to avoid cyclic dependencies between ops.py and
control_flow_ops.py.
"""
ENABLE_CONTROL_FLOW_V2 = ...
def enable_control_flow_v2(): # -> None:
  """Use control flow v2.

  Do not use this symbol. This will be removed.
  """
  ...

def EnableControlFlowV2(graph): # -> bool:
  """Returns whether control flow v2 should be used in `graph`."""
  ...

def IsInXLAContext(op): # -> bool:
  ...

def InXlaContext(graph): # -> bool:
  ...

def GraphOrParentsInXlaContext(graph): # -> bool:
  ...

def IsInWhileLoop(op): # -> bool:
  ...

def IsInCond(op): # -> bool:
  ...

def IsSwitch(op):
  """Return true if `op` is a Switch."""
  ...

def IsMerge(op):
  """Return true if `op` is a Merge."""
  ...

def IsLoopEnter(op):
  """Returns true if `op` is an Enter."""
  ...

def IsLoopExit(op):
  """Return true if `op` is an Exit."""
  ...

def IsCondSwitch(op): # -> bool:
  """Return true if `op` is the Switch for a conditional."""
  ...

def IsCondMerge(op): # -> bool:
  """Return true if `op` is the Merge for a conditional."""
  ...

def IsLoopSwitch(op): # -> bool:
  """Return true if `op` is the Switch for a while loop."""
  ...

def IsLoopMerge(op): # -> bool:
  """Return true if `op` is the Merge for a while loop."""
  ...

def IsLoopConstantEnter(op):
  """Return true iff op is a loop invariant."""
  ...

def GetLoopConstantEnter(value): # -> None:
  """Return the enter op if we can infer `value` to be a loop invariant."""
  ...

def GetOutputContext(op):
  """Return the control flow context for the output of an op."""
  ...

def GetContainingWhileContext(ctxt, stop_ctxt=...): # -> None:
  """Returns the first ancestor WhileContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a WhileContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext
    stop_ctxt: ControlFlowContext, optional. If provided, the search will end
      if it sees stop_ctxt.

  Returns:
    `ctxt` if `ctxt` is a WhileContext, the most nested WhileContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.  If `stop_ctxt` is not
    `None`, this returns `ctxt` if it matches `stop_ctxt` in its traversal.
  """
  ...

def GetContainingXLAContext(ctxt): # -> None:
  """Returns the first ancestor XLAContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a XLAContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext

  Returns:
    `ctxt` if `ctxt` is a XLAContext, the most nested XLAContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.
  """
  ...

def GetContainingCondContext(ctxt): # -> None:
  """Returns the first ancestor CondContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a CondContext, or None if `ctxt` is not in a cond.

  Args:
    ctxt: ControlFlowContext

  Returns:
    `ctxt` if `ctxt` is a CondContext, the most nested CondContext containing
    `ctxt`, or None if `ctxt` is not in a cond.
  """
  ...

def IsContainingContext(ctxt, maybe_containing_ctxt): # -> bool:
  """Returns true if `maybe_containing_ctxt` is or contains `ctxt`."""
  ...

def OpInContext(op, ctxt): # -> bool:
  ...

def TensorInContext(tensor, ctxt): # -> bool:
  ...

def CheckInputFromValidContext(op, input_op): # -> None:
  """Returns whether `input_op` can be used from `op`s context.

  Conceptually, only inputs from op's while context or any ancestor while
  context (including outside of any context) are valid. In practice, there are
  many other edge cases as well.

  Args:
    op: Operation
    input_op: Operation

  Raises:
    ValueError: if input_op is from an invalid context.
  """
  ...

def GetWhileContext(op):
  """Get the WhileContext to which this op belongs."""
  ...

