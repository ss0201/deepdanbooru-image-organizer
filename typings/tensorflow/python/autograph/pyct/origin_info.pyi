"""
This type stub file was generated by pyright.
"""

import collections
import gast

"""Container for origin source code information before AutoGraph compilation."""
class LineLocation(collections.namedtuple('LineLocation', ('filename', 'lineno'))):
  """Similar to Location, but without column information.

  Attributes:
    filename: Text
    lineno: int, 1-based
  """
  ...


class Location(collections.namedtuple('Location', ('filename', 'lineno', 'col_offset'))):
  """Encodes code location information.

  Attributes:
    filename: Text
    lineno: int, 1-based
    col_offset: int
    line_loc: LineLocation
  """
  @property
  def line_loc(self): # -> LineLocation:
    ...
  


class OriginInfo(collections.namedtuple('OriginInfo', ('loc', 'function_name', 'source_code_line', 'comment'))):
  """Container for information about the source code before conversion.

  Attributes:
    loc: Location
    function_name: Optional[Text]
    source_code_line: Text
    comment: Optional[Text]
  """
  def as_frame(self): # -> tuple[Unknown, Unknown, Unknown, Unknown]:
    """Returns a 4-tuple consistent with the return of traceback.extract_tb."""
    ...
  
  def __repr__(self): # -> LiteralString:
    ...
  


def create_source_map(nodes, code, filepath): # -> dict[Unknown, Unknown]:
  """Creates a source map between an annotated AST and the code it compiles to.

  Note: this function assumes nodes nodes, code and filepath correspond to the
  same code.

  Args:
    nodes: Iterable[ast.AST, ...], one or more AST modes.
    code: Text, the source code in which nodes are found.
    filepath: Text

  Returns:
    Dict[LineLocation, OriginInfo], mapping locations in code to locations
    indicated by origin annotations in node.
  """
  ...

class _Function:
  def __init__(self, name) -> None:
    ...
  


class OriginResolver(gast.NodeVisitor):
  """Annotates an AST with additional source information like file name."""
  def __init__(self, root_node, source_lines, comments_map, context_lineno, context_col_offset, filepath) -> None:
    ...
  
  def visit(self, node): # -> None:
    ...
  


def resolve(node, source, context_filepath, context_lineno, context_col_offset): # -> None:
  """Adds origin information to an AST, based on the source it was loaded from.

  This allows us to map the original source code line numbers to generated
  source code.

  Note: the AST may be a part of a larger context (e.g. a function is part of
  a module that may contain other things). However, this function does not
  assume the source argument contains the entire context, nor that it contains
  only code corresponding to node itself. However, it assumes that node was
  parsed from the given source code.
  For this reason, two extra arguments are required, and they indicate the
  location of the node in the original context.

  Args:
    node: gast.AST, the AST to annotate.
    source: Text, the source code representing node.
    context_filepath: Text
    context_lineno: int
    context_col_offset: int
  """
  ...

def resolve_entity(node, source, entity): # -> None:
  """Like resolve, but extracts the context information from an entity."""
  ...

def copy_origin(from_node, to_node): # -> None:
  """Copies the origin info from a node to another, recursively."""
  ...

