"""
This type stub file was generated by pyright.
"""

"""Converting AST to code and Python entities.

Adapted from Tangent.
"""
def load_source(source, delete_on_exit): # -> tuple[Unknown, str]:
  """Loads the given source code as a Python module."""
  ...

def load_ast(nodes, indentation=..., include_source_map=..., delete_on_exit=...): # -> tuple[Unknown, LiteralString, dict[Unknown, Unknown] | None]:
  """Loads the given AST as a Python module.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    nodes: Union[ast.AST, Iterable[ast.AST]], the code to compile, as an AST
      object.
    indentation: Text, the string to use for indentation.
    include_source_map: bool, whether return a source map.
    delete_on_exit: bool, whether to delete the temporary file used for
      compilation on exit.

  Returns:
    Tuple[module, Text, Dict[LineLocation, OriginInfo]], containing:
    the module containing the unparsed nodes, the source code corresponding to
    nodes, and the source map. Is include_source_map is False, the source map
    will be None.
  """
  ...

