"""
This type stub file was generated by pyright.
"""

from tensorflow.python.autograph.pyct import cfg, transformer

"""Live variable analysis.

See https://en.wikipedia.org/wiki/Live_variable_analysis for a definition of
the following idioms: live variable, live in, live out, which are used
throughout this file.

This analysis attaches the following:
 * symbols that are live at the exit of control flow statements
 * symbols that are live at the entry of control flow statements

Requires activity analysis.
"""
class Analyzer(cfg.GraphVisitor):
  """CFG visitor that performs liveness analysis at statement level."""
  def __init__(self, graph, include_annotations) -> None:
    ...
  
  def init_state(self, _): # -> set[Unknown]:
    ...
  
  def visit_node(self, node):
    ...
  


class TreeAnnotator(transformer.Base):
  """Runs liveness analysis on each of the functions defined in the AST.

  If a function defined other local functions, those will have separate CFGs.
  However, dataflow analysis needs to tie up these CFGs to properly emulate the
  effect of closures. In the case of liveness, the parent function's live
  variables must account for the variables that are live at the entry of each
  subfunction. For example:

    def foo():
      # baz is live from here on
      def bar():
        print(baz)

  This analyzer runs liveness analysis on each individual function, accounting
  for the effect above.
  """
  def __init__(self, source_info, graphs, include_annotations) -> None:
    ...
  
  def visit(self, node): # -> stmt | Any | AST | list[Unknown] | tuple[Unknown, ...]:
    ...
  
  def visit_Lambda(self, node): # -> AST:
    ...
  
  def visit_FunctionDef(self, node): # -> AST:
    ...
  
  def visit_If(self, node): # -> AST:
    ...
  
  def visit_For(self, node): # -> AST:
    ...
  
  def visit_While(self, node): # -> AST:
    ...
  
  def visit_Try(self, node): # -> AST:
    ...
  
  def visit_ExceptHandler(self, node): # -> AST:
    ...
  
  def visit_With(self, node): # -> AST:
    ...
  
  def visit_Expr(self, node): # -> AST:
    ...
  


def resolve(node, source_info, graphs, include_annotations=...): # -> stmt | Any | AST | list[Unknown] | tuple[Unknown, ...]:
  """Resolves the live symbols at the exit of control flow statements.

  Args:
    node: ast.AST
    source_info: transformer.SourceInfo
    graphs: Dict[ast.FunctionDef, cfg.Graph]
    include_annotations: Bool, whether type annotations should be included in
      the analysis.
  Returns:
    ast.AST
  """
  ...
