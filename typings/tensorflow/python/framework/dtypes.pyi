"""
This type stub file was generated by pyright.
"""

import abc
from typing import Optional, Sequence, Type
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import _dtypes
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.core.function import trace_type

"""Library of dtypes (Tensor element types)."""
_np_bfloat16 = ...
class DTypeMeta(type(_dtypes.DType), abc.ABCMeta):
  ...


@tf_export("dtypes.DType", "DType")
class DType(_dtypes.DType, trace.TraceType, trace_type.Serializable, metaclass=DTypeMeta):
  """Represents the type of the elements in a `Tensor`.

  `DType`'s are used to specify the output data type for operations which
  require it, or to inspect the data type of existing `Tensor`'s.

  Examples:

  >>> tf.constant(1, dtype=tf.int64)
  <tf.Tensor: shape=(), dtype=int64, numpy=1>
  >>> tf.constant(1.0).dtype
  tf.float32

  See `tf.dtypes` for a complete list of `DType`'s defined.
  """
  __slots__ = ...
  @property
  def base_dtype(self): # -> DType | Self@DType:
    """Returns a non-reference `DType` based on this `DType`."""
    ...
  
  @property
  def real_dtype(self): # -> DType | Self@DType:
    """Returns the `DType` corresponding to this `DType`'s real part."""
    ...
  
  @property
  def as_numpy_dtype(self):
    """Returns a Python `type` object based on this `DType`."""
    ...
  
  @property
  def min(self): # -> Any:
    """Returns the minimum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    """
    ...
  
  @property
  def max(self): # -> Any:
    """Returns the maximum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    """
    ...
  
  @property
  def limits(self, clip_negative=...): # -> tuple[Unknown | Literal[0], Unknown]:
    """Return intensity limits, i.e.

    (min, max) tuple, of the dtype.
    Args:
      clip_negative : bool, optional If True, clip the negative range (i.e.
        return 0 for min intensity) even if the image dtype allows negative
        values. Returns
      min, max : tuple Lower and upper intensity limits.
    """
    ...
  
  def is_compatible_with(self, other): # -> bool:
    """Returns True if the `other` DType will be converted to this DType.

    The conversion rules are as follows:

    ```python
    DType(T)       .is_compatible_with(DType(T))        == True
    ```

    Args:
      other: A `DType` (or object that may be converted to a `DType`).

    Returns:
      True if a Tensor of the `other` `DType` will be implicitly converted to
      this `DType`.
    """
    ...
  
  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """See tf.types.experimental.TraceType base class."""
    ...
  
  def most_specific_common_supertype(self, types: Sequence[trace.TraceType]) -> Optional[DType]:
    """See tf.types.experimental.TraceType base class."""
    ...
  
  @classmethod
  def experimental_type_proto(cls) -> Type[types_pb2.SerializedDType]:
    """Returns the type of proto associated with DType serialization."""
    ...
  
  @classmethod
  def experimental_from_proto(cls, proto: types_pb2.SerializedDType) -> DType:
    """Returns a Dtype instance based on the serialized proto."""
    ...
  
  def experimental_as_proto(self) -> types_pb2.SerializedDType:
    """Returns a proto representation of the Dtype instance."""
    ...
  
  def __eq__(self, other) -> bool:
    """Returns True iff this DType refers to the same type as `other`."""
    ...
  
  def __ne__(self, other) -> bool:
    """Returns True iff self != other."""
    ...
  
  __hash__ = ...
  def __reduce__(self): # -> tuple[(type_value: Unknown) -> (DType | Unknown), tuple[Unknown]]:
    ...
  


dtype_range = ...
resource = ...
variant = ...
uint8 = ...
uint16 = ...
uint32 = ...
uint64 = ...
int8 = ...
int16 = ...
int32 = ...
int64 = ...
float16 = ...
half = ...
float32 = ...
float64 = ...
double = ...
complex64 = ...
complex128 = ...
string = ...
bool = ...
qint8 = ...
qint16 = ...
qint32 = ...
quint8 = ...
quint16 = ...
bfloat16 = ...
resource_ref = ...
variant_ref = ...
float16_ref = ...
half_ref = ...
float32_ref = ...
float64_ref = ...
double_ref = ...
int32_ref = ...
uint32_ref = ...
uint8_ref = ...
uint16_ref = ...
int16_ref = ...
int8_ref = ...
string_ref = ...
complex64_ref = ...
complex128_ref = ...
int64_ref = ...
uint64_ref = ...
bool_ref = ...
qint8_ref = ...
quint8_ref = ...
qint16_ref = ...
quint16_ref = ...
qint32_ref = ...
bfloat16_ref = ...
_INTERN_TABLE = ...
_TYPE_TO_STRING = ...
_STRING_TO_TF = ...
_np_qint8 = ...
_np_quint8 = ...
_np_qint16 = ...
_np_quint16 = ...
_np_qint32 = ...
np_resource = ...
_NP_TO_TF = ...
TF_VALUE_DTYPES = ...
_TF_TO_NP = ...
_QUANTIZED_DTYPES_NO_REF = ...
_QUANTIZED_DTYPES_REF = ...
QUANTIZED_DTYPES = ...
_PYTHON_TO_TF = ...
_ANY_TO_TF = ...
@tf_export("dtypes.as_dtype", "as_dtype")
def as_dtype(type_value): # -> DType:
  """Converts the given `type_value` to a `DType`.

  Note: `DType` values are interned. When passed a new `DType` object,
  `as_dtype` always returns the interned value.

  Args:
    type_value: A value that can be converted to a `tf.DType` object. This may
      currently be a `tf.DType` object, a [`DataType`
      enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
        a string type name, or a [`numpy.dtype`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html).

  Returns:
    A `DType` corresponding to `type_value`.

  Raises:
    TypeError: If `type_value` cannot be converted to a `DType`.
  """
  ...
