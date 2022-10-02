"""
This type stub file was generated by pyright.
"""

import abc

"""Add one or more `LinearOperators` efficiently."""
__all__ = []
def add_operators(operators, operator_name=..., addition_tiers=..., name=...): # -> list[Unknown]:
  """Efficiently add one or more linear operators.

  Given operators `[A1, A2,...]`, this `Op` returns a possibly shorter list of
  operators `[B1, B2,...]` such that

  ```sum_k Ak.matmul(x) = sum_k Bk.matmul(x).```

  The operators `Bk` result by adding some of the `Ak`, as allowed by
  `addition_tiers`.

  Example of efficient adding of diagonal operators.

  ```python
  A1 = LinearOperatorDiag(diag=[1., 1.], name="A1")
  A2 = LinearOperatorDiag(diag=[2., 2.], name="A2")

  # Use two tiers, the first contains an Adder that returns Diag.  Since both
  # A1 and A2 are Diag, they can use this Adder.  The second tier will not be
  # used.
  addition_tiers = [
      [_AddAndReturnDiag()],
      [_AddAndReturnMatrix()]]
  B_list = add_operators([A1, A2], addition_tiers=addition_tiers)

  len(B_list)
  ==> 1

  B_list[0].__class__.__name__
  ==> 'LinearOperatorDiag'

  B_list[0].to_dense()
  ==> [[3., 0.],
       [0., 3.]]

  B_list[0].name
  ==> 'Add/A1__A2/'
  ```

  Args:
    operators:  Iterable of `LinearOperator` objects with same `dtype`, domain
      and range dimensions, and broadcastable batch shapes.
    operator_name:  String name for returned `LinearOperator`.  Defaults to
      concatenation of "Add/A__B/" that indicates the order of addition steps.
    addition_tiers:  List tiers, like `[tier_0, tier_1, ...]`, where `tier_i`
      is a list of `Adder` objects.  This function attempts to do all additions
      in tier `i` before trying tier `i + 1`.
    name:  A name for this `Op`.  Defaults to `add_operators`.

  Returns:
    Subclass of `LinearOperator`.  Class and order of addition may change as new
      (and better) addition strategies emerge.

  Raises:
    ValueError:  If `operators` argument is empty.
    ValueError:  If shapes are incompatible.
  """
  ...

class _Hints:
  """Holds 'is_X' flags that every LinearOperator is initialized with."""
  def __init__(self, is_non_singular=..., is_positive_definite=..., is_self_adjoint=...) -> None:
    ...
  


class _Adder(metaclass=abc.ABCMeta):
  """Abstract base class to add two operators.

  Each `Adder` acts independently, adding everything it can, paying no attention
  as to whether another `Adder` could have done the addition more efficiently.
  """
  @property
  def name(self): # -> str:
    ...
  
  @abc.abstractmethod
  def can_add(self, op1, op2): # -> None:
    """Returns `True` if this `Adder` can add `op1` and `op2`.  Else `False`."""
    ...
  
  def add(self, op1, op2, operator_name, hints=...): # -> None:
    """Return new `LinearOperator` acting like `op1 + op2`.

    Args:
      op1:  `LinearOperator`
      op2:  `LinearOperator`, with `shape` and `dtype` such that adding to
        `op1` is allowed.
      operator_name:  `String` name to give to returned `LinearOperator`
      hints:  `_Hints` object.  Returned `LinearOperator` will be created with
        these hints.

    Returns:
      `LinearOperator`
    """
    ...
  


class _AddAndReturnScaledIdentity(_Adder):
  """Handles additions resulting in an Identity family member.

  The Identity (`LinearOperatorScaledIdentity`, `LinearOperatorIdentity`) family
  is closed under addition.  This `Adder` respects that, and returns an Identity
  """
  def can_add(self, op1, op2): # -> bool:
    ...
  


class _AddAndReturnDiag(_Adder):
  """Handles additions resulting in a Diag operator."""
  def can_add(self, op1, op2): # -> bool:
    ...
  


class _AddAndReturnTriL(_Adder):
  """Handles additions resulting in a TriL operator."""
  def can_add(self, op1, op2): # -> bool:
    ...
  


class _AddAndReturnMatrix(_Adder):
  """"Handles additions resulting in a `LinearOperatorFullMatrix`."""
  def can_add(self, op1, op2): # -> bool:
    ...
  


_IDENTITY = ...
_SCALED_IDENTITY = ...
_DIAG = ...
_TRIL = ...
_MATRIX = ...
_DIAG_LIKE = ...
_IDENTITY_FAMILY = ...
_EFFICIENT_ADD_TO_TENSOR = ...
SUPPORTED_OPERATORS = ...
_DEFAULT_ADDITION_TIERS = ...