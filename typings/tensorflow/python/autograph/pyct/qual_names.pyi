"""
This type stub file was generated by pyright.
"""

import collections
import gast

"""Utilities for manipulating qualified names.

A qualified name is a uniform way to refer to simple (e.g. 'foo') and composite
(e.g. 'foo.bar') syntactic symbols.

This is *not* related to the __qualname__ attribute used by inspect, which
refers to scopes.
"""
class CallerMustSetThis:
  ...


class Symbol(collections.namedtuple('Symbol', ['name'])):
  """Represents a Python symbol."""
  ...


class Literal(collections.namedtuple('Literal', ['value'])):
  """Represents a Python numeric literal."""
  def __str__(self) -> str:
    ...
  
  def __repr__(self): # -> str:
    ...
  


class QN:
  """Represents a qualified name."""
  def __init__(self, base, attr=..., subscript=...) -> None:
    ...
  
  def is_symbol(self): # -> bool:
    ...
  
  def is_simple(self): # -> bool:
    ...
  
  def is_composite(self): # -> bool:
    ...
  
  def has_subscript(self): # -> bool:
    ...
  
  def has_attr(self): # -> bool:
    ...
  
  @property
  def attr(self): # -> str | Literal:
    ...
  
  @property
  def parent(self): # -> QN:
    ...
  
  @property
  def owner_set(self): # -> set[Unknown]:
    """Returns all the symbols (simple or composite) that own this QN.

    In other words, if this symbol was modified, the symbols in the owner set
    may also be affected.

    Examples:
      'a.b[c.d]' has two owners, 'a' and 'a.b'
    """
    ...
  
  @property
  def support_set(self): # -> set[Unknown]:
    """Returns the set of simple symbols that this QN relies on.

    This would be the smallest set of symbols necessary for the QN to
    statically resolve (assuming properties and index ranges are verified
    at runtime).

    Examples:
      'a.b' has only one support symbol, 'a'
      'a[i]' has two support symbols, 'a' and 'i'
    """
    ...
  
  def __hash__(self) -> int:
    ...
  
  def __eq__(self, other) -> bool:
    ...
  
  def __lt__(self, other) -> bool:
    ...
  
  def __gt__(self, other) -> bool:
    ...
  
  def __str__(self) -> str:
    ...
  
  def __repr__(self): # -> str:
    ...
  
  def ssf(self):
    """Simple symbol form."""
    ...
  
  def ast(self):
    """AST representation."""
    ...
  


class QnResolver(gast.NodeTransformer):
  """Annotates nodes with QN information.

  Note: Not using NodeAnnos to avoid circular dependencies.
  """
  def visit_Name(self, node): # -> AST:
    ...
  
  def visit_Attribute(self, node): # -> AST:
    ...
  
  def visit_Subscript(self, node): # -> AST:
    ...
  


def resolve(node): # -> Any:
  ...

def from_str(qn_str): # -> Any | object:
  ...

