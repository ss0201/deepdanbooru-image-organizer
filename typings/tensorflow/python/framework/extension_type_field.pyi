"""
This type stub file was generated by pyright.
"""

import collections
import enum

"""Meatadata about fields for user-defined ExtensionType classes."""
RESERVED_FIELD_NAMES = ...
class Sentinel:
  """Sentinel value that's not equal (w/ `is`) to any user value."""
  def __init__(self, name) -> None:
    ...
  
  def __repr__(self): # -> Unknown:
    ...
  


_NoneType = ...
class ExtensionTypeField(collections.namedtuple('ExtensionTypeField', ['name', 'value_type', 'default'])):
  """Metadata about a single field in a `tf.ExtensionType` object."""
  NO_DEFAULT = ...
  def __new__(cls, name, value_type, default=...): # -> Self@ExtensionTypeField:
    """Constructs a new ExtensionTypeField containing metadata for a single field.

    Args:
      name: The name of the new field (`str`).  May not be a reserved name.
      value_type: A python type expression constraining what values this field
        can take.
      default: The default value for the new field, or `NO_DEFAULT` if this
        field has no default value.

    Returns:
      A new `ExtensionTypeField`.

    Raises:
      TypeError: If the type described by `value_type` is not currently
          supported by `tf.ExtensionType`.
      TypeError: If `default` is specified and its type does not match
        `value_type`.
    """
    ...
  
  @staticmethod
  def is_reserved_name(name): # -> Literal[True]:
    """Returns true if `name` is a reserved name."""
    ...
  


def validate_field_value_type(value_type, in_mapping_key=..., allow_forward_references=...): # -> None:
  """Checks that `value_type` contains only supported type annotations.

  Args:
    value_type: The type annotation to check.
    in_mapping_key: True if `value_type` is nested in the key of a mapping.
    allow_forward_references: If false, then raise an exception if a
      `value_type` contains a forward reference (i.e., a string literal).

  Raises:
    TypeError: If `value_type` contains an unsupported type annotation.
  """
  ...

class _ConversionContext(enum.Enum):
  """Enum to indicate what kind of value is being converted.

  Used by `_convert_fields` and `_convert_value` and their helper methods.
  """
  VALUE = ...
  SPEC = ...
  DEFAULT = ...


def convert_fields(fields, field_values): # -> None:
  """Type-checks and converts each field in `field_values` (in place).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
  ...

def convert_fields_for_spec(fields, field_values): # -> None:
  """Type-checks and converts field values for a TypeSpec (in place).

  This is similar to `convert_fields`, except that we expect a `TypeSpec` for
  tensor-like types.  In particular, if the `value_type` of a field is
  `tf.Tensor` or a `CompositeTensor` subclass, then the corresponding value in
  `fields` is expected to contain a `TypeSpec` (rather than a value described by
  that `TypeSpec`).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
  ...

