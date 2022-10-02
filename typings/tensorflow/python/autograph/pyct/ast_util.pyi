"""
This type stub file was generated by pyright.
"""

import gast

"""AST manipulation utilities."""
class CleanCopier:
  """NodeTransformer-like visitor that copies an AST."""
  def __init__(self, preserve_annos) -> None:
    ...
  
  def copy(self, node): # -> list[Unknown] | tuple[Unknown, ...] | AST:
    """Returns a deep copy of node (excluding some fields, see copy_clean)."""
    ...
  


def copy_clean(node, preserve_annos=...): # -> list[Unknown] | tuple[Unknown, ...] | AST:
  """Creates a deep copy of an AST.

  The copy will not include fields that are prefixed by '__', with the
  exception of user-specified annotations.

  Args:
    node: ast.AST
    preserve_annos: Optional[Set[Hashable]], annotation keys to include in the
        copy
  Returns:
    ast.AST
  """
  ...

class SymbolRenamer(gast.NodeTransformer):
  """Transformer that can rename symbols to a simple names."""
  def __init__(self, name_map) -> None:
    ...
  
  def visit_Nonlocal(self, node): # -> Nonlocal:
    ...
  
  def visit_Global(self, node): # -> Global:
    ...
  
  def visit_Name(self, node): # -> AST:
    ...
  
  def visit_Attribute(self, node): # -> AST:
    ...
  
  def visit_FunctionDef(self, node): # -> AST:
    ...
  


def rename_symbols(node, name_map): # -> list[Any] | tuple[Any, ...] | Any:
  """Renames symbols in an AST. Requires qual_names annotations."""
  ...

def keywords_to_dict(keywords):
  """Converts a list of ast.keyword objects to a dict."""
  ...

class PatternMatcher(gast.NodeVisitor):
  """Matches a node against a pattern represented by a node."""
  def __init__(self, pattern) -> None:
    ...
  
  def compare_and_visit(self, node, pattern): # -> None:
    ...
  
  def no_match(self): # -> Literal[False]:
    ...
  
  def is_wildcard(self, p): # -> bool:
    ...
  
  def generic_visit(self, node): # -> Literal[False] | None:
    ...
  


def matches(node, pattern): # -> bool:
  """Basic pattern matcher for AST.

  The pattern may contain wildcards represented by the symbol '_'. A node
  matches a pattern if for every node in the tree, either there is a node of
  the same type in pattern, or a Name node with id='_'.

  Args:
    node: ast.AST
    pattern: ast.AST
  Returns:
    bool
  """
  ...

def apply_to_single_assignments(targets, values, apply_fn): # -> None:
  """Applies a function to each individual assignment.

  This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
  It tries to break down the unpacking if possible. In effect, it has the same
  effect as passing the assigned values in SSA form to apply_fn.

  Examples:

  The following will result in apply_fn(a, c), apply_fn(b, d):

      a, b = c, d

  The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):

      a, b = c

  The following will result in apply_fn(a, (b, c)):

      a = b, c

  It uses the visitor pattern to allow subclasses to process single
  assignments individually.

  Args:
    targets: Union[List[ast.AST, ...], Tuple[ast.AST, ...], ast.AST, should be
        used with the targets field of an ast.Assign node
    values: ast.AST
    apply_fn: Callable[[ast.AST, ast.AST], None], called with the
        respective nodes of each single assignment
  """
  ...

def parallel_walk(node, other): # -> Generator[tuple[AST | str | Unknown, AST | str | Unknown], None, None]:
  """Walks two ASTs in parallel.

  The two trees must have identical structure.

  Args:
    node: Union[ast.AST, Iterable[ast.AST]]
    other: Union[ast.AST, Iterable[ast.AST]]
  Yields:
    Tuple[ast.AST, ast.AST]
  Raises:
    ValueError: if the two trees don't have identical structure.
  """
  ...

