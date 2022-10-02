"""
This type stub file was generated by pyright.
"""

from tensorflow.python.autograph.core import converter

"""Converter for list operations.

This includes converting Python lists to TensorArray/TensorList.
"""
class _Statement:
  def __init__(self) -> None:
    ...
  


class ListTransformer(converter.Base):
  """Converts lists and related operations to their TF counterpart."""
  def visit_List(self, node): # -> Any:
    ...
  
  def visit_Call(self, node): # -> list[Any] | Any | AST:
    ...
  
  def visit_FunctionDef(self, node): # -> FunctionDef:
    ...
  
  def visit_For(self, node): # -> For:
    ...
  
  def visit_While(self, node): # -> While:
    ...
  
  def visit_If(self, node): # -> If:
    ...
  
  def visit_With(self, node): # -> With:
    ...
  


def transform(node, ctx): # -> AST | list[Unknown] | tuple[Unknown, ...] | Any:
  ...
