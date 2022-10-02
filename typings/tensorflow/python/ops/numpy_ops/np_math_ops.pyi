"""
This type stub file was generated by pyright.
"""

from tensorflow.python.ops.numpy_ops import np_utils

"""Mathematical operations."""
pi = ...
e = ...
inf = ...
@np_utils.np_doc_only('dot')
def dot(a, b): # -> Any | list[Unknown] | _basetuple | defaultdict[Unknown, Unknown] | ObjectProxy:
  ...

@np_utils.np_doc('add')
def add(x1, x2): # -> _dispatcher_for_logical_or | object:
  ...

@np_utils.np_doc('subtract')
def subtract(x1, x2):
  ...

@np_utils.np_doc('multiply')
def multiply(x1, x2): # -> _dispatcher_for_logical_and | object:
  ...

@np_utils.np_doc('true_divide')
def true_divide(x1, x2): # -> _dispatcher_for_real_div | object:
  ...

@np_utils.np_doc('divide')
def divide(x1, x2): # -> _dispatcher_for_real_div | object:
  ...

@np_utils.np_doc('floor_divide')
def floor_divide(x1, x2): # -> _dispatcher_for_floor_div | object:
  ...

@np_utils.np_doc('mod')
def mod(x1, x2): # -> _dispatcher_for_floor_mod | object:
  ...

@np_utils.np_doc('remainder')
def remainder(x1, x2): # -> _dispatcher_for_floor_mod | object:
  ...

@np_utils.np_doc('divmod')
def divmod(x1, x2): # -> tuple[Unknown | _dispatcher_for_floor_div | object, Unknown | _dispatcher_for_floor_mod | object]:
  ...

@np_utils.np_doc('maximum')
def maximum(x1, x2): # -> _dispatcher_for_relu | object:
  ...

@np_utils.np_doc('minimum')
def minimum(x1, x2): # -> _dispatcher_for_logical_and | object:
  ...

@np_utils.np_doc('clip')
def clip(a, a_min, a_max): # -> _dispatcher_for_logical_and | object | IndexedSlices:
  ...

@np_utils.np_doc('matmul')
def matmul(x1, x2): # -> Any | list[Unknown] | _basetuple | defaultdict[Unknown, Unknown] | ObjectProxy:
  ...

@np_utils.np_doc('tensordot')
def tensordot(a, b, axes=...):
  ...

@np_utils.np_doc_only('inner')
def inner(a, b): # -> Any | list[Unknown] | _basetuple | defaultdict[Unknown, Unknown] | ObjectProxy:
  ...

@np_utils.np_doc('cross')
def cross(a, b, axisa=..., axisb=..., axisc=..., axis=...): # -> Any | list[Unknown] | _basetuple | defaultdict[Unknown, Unknown] | ObjectProxy:
  ...

@np_utils.np_doc_only('vdot')
def vdot(a, b): # -> Any | list[Unknown] | _basetuple | defaultdict[Unknown, Unknown] | ObjectProxy:
  ...

@np_utils.np_doc('power')
def power(x1, x2):
  ...

@np_utils.np_doc('float_power')
def float_power(x1, x2):
  ...

@np_utils.np_doc('arctan2')
def arctan2(x1, x2): # -> _dispatcher_for_atan2 | object:
  ...

@np_utils.np_doc('nextafter')
def nextafter(x1, x2): # -> _dispatcher_for_next_after | object:
  ...

@np_utils.np_doc('heaviside')
def heaviside(x1, x2):
  ...

@np_utils.np_doc('hypot')
def hypot(x1, x2):
  ...

@np_utils.np_doc('kron')
def kron(a, b):
  ...

@np_utils.np_doc('outer')
def outer(a, b):
  ...

@np_utils.np_doc('logaddexp')
def logaddexp(x1, x2):
  ...

@np_utils.np_doc('logaddexp2')
def logaddexp2(x1, x2):
  ...

@np_utils.np_doc('polyval')
def polyval(p, x): # -> _dispatcher_for_broadcast_to | object | Tensor:
  ...

@np_utils.np_doc('isclose')
def isclose(a, b, rtol=..., atol=..., equal_nan=...): # -> bool | Any:
  ...

@np_utils.np_doc('allclose')
def allclose(a, b, rtol=..., atol=..., equal_nan=...):
  ...

@np_utils.np_doc('gcd')
def gcd(x1, x2): # -> Any:
  ...

@np_utils.np_doc('lcm')
def lcm(x1, x2):
  ...

@np_utils.np_doc('bitwise_and')
def bitwise_and(x1, x2): # -> SparseTensor | IndexedSlices | Tensor | Any:
  ...

@np_utils.np_doc('bitwise_or')
def bitwise_or(x1, x2): # -> SparseTensor | IndexedSlices | Tensor | Any:
  ...

@np_utils.np_doc('bitwise_xor')
def bitwise_xor(x1, x2): # -> SparseTensor | IndexedSlices | Tensor | Any:
  ...

@np_utils.np_doc('bitwise_not', link=np_utils.AliasOf('invert'))
def bitwise_not(x): # -> _dispatcher_for_logical_not | object:
  ...

@np_utils.np_doc('log')
def log(x): # -> _dispatcher_for_is_nan | object:
  ...

@np_utils.np_doc('exp')
def exp(x):
  ...

@np_utils.np_doc('sqrt')
def sqrt(x):
  ...

@np_utils.np_doc('abs', link=np_utils.AliasOf('absolute'))
def abs(x):
  ...

@np_utils.np_doc('absolute')
def absolute(x):
  ...

@np_utils.np_doc('fabs')
def fabs(x):
  ...

@np_utils.np_doc('ceil')
def ceil(x):
  ...

@np_utils.np_doc('floor')
def floor(x):
  ...

@np_utils.np_doc('conj')
def conj(x): # -> Tensor | Any:
  ...

@np_utils.np_doc('negative')
def negative(x): # -> _dispatcher_for_square | object:
  ...

@np_utils.np_doc('reciprocal')
def reciprocal(x): # -> _dispatcher_for_square | object:
  ...

@np_utils.np_doc('signbit')
def signbit(x):
  ...

@np_utils.np_doc('sin')
def sin(x): # -> _dispatcher_for_is_nan | object:
  ...

@np_utils.np_doc('cos')
def cos(x): # -> _dispatcher_for_is_nan | object:
  ...

@np_utils.np_doc('tan')
def tan(x): # -> _dispatcher_for_tan | object:
  ...

@np_utils.np_doc('sinh')
def sinh(x): # -> _dispatcher_for_sinh | object:
  ...

@np_utils.np_doc('cosh')
def cosh(x): # -> _dispatcher_for_cosh | object:
  ...

@np_utils.np_doc('tanh')
def tanh(x): # -> _dispatcher_for_is_nan | object:
  ...

@np_utils.np_doc('arcsin')
def arcsin(x): # -> _dispatcher_for_asin | object:
  ...

@np_utils.np_doc('arccos')
def arccos(x):
  ...

@np_utils.np_doc('arctan')
def arctan(x): # -> _dispatcher_for_atan | object:
  ...

@np_utils.np_doc('arcsinh')
def arcsinh(x): # -> _dispatcher_for_asinh | object:
  ...

@np_utils.np_doc('arccosh')
def arccosh(x): # -> _dispatcher_for_acosh | object:
  ...

@np_utils.np_doc('arctanh')
def arctanh(x): # -> _dispatcher_for_atanh | object:
  ...

@np_utils.np_doc('deg2rad')
def deg2rad(x):
  ...

@np_utils.np_doc('rad2deg')
def rad2deg(x):
  ...

_tf_float_types = ...
@np_utils.np_doc('angle')
def angle(z, deg=...):
  ...

@np_utils.np_doc('cbrt')
def cbrt(x):
  ...

@np_utils.np_doc('conjugate', link=np_utils.AliasOf('conj'))
def conjugate(x): # -> Tensor | Any:
  ...

@np_utils.np_doc('exp2')
def exp2(x):
  ...

@np_utils.np_doc('expm1')
def expm1(x): # -> _dispatcher_for_expm1 | object:
  ...

@np_utils.np_doc('fix')
def fix(x):
  ...

@np_utils.np_doc('iscomplex')
def iscomplex(x):
  ...

@np_utils.np_doc('isreal')
def isreal(x):
  ...

@np_utils.np_doc('iscomplexobj')
def iscomplexobj(x): # -> bool:
  ...

@np_utils.np_doc('isrealobj')
def isrealobj(x): # -> bool:
  ...

@np_utils.np_doc('isnan')
def isnan(x): # -> _dispatcher_for_is_nan | object:
  ...

nansum = ...
nanprod = ...
@np_utils.np_doc('nanmean')
def nanmean(a, axis=..., dtype=..., keepdims=...):
  ...

@np_utils.np_doc('isfinite')
def isfinite(x): # -> _dispatcher_for_acosh | object:
  ...

@np_utils.np_doc('isinf')
def isinf(x): # -> _dispatcher_for_is_inf | object:
  ...

@np_utils.np_doc('isneginf')
def isneginf(x):
  ...

@np_utils.np_doc('isposinf')
def isposinf(x):
  ...

@np_utils.np_doc('log2')
def log2(x): # -> Any:
  ...

@np_utils.np_doc('log10')
def log10(x): # -> Any:
  ...

@np_utils.np_doc('log1p')
def log1p(x): # -> _dispatcher_for_is_nan | object:
  ...

@np_utils.np_doc('positive')
def positive(x):
  ...

@np_utils.np_doc('sinc')
def sinc(x):
  ...

@np_utils.np_doc('square')
def square(x): # -> _dispatcher_for_square | object:
  ...

@np_utils.np_doc('diff')
def diff(a, n=..., axis=...): # -> Any | ndarray | defaultdict[Unknown, Unknown] | list[Unknown] | ObjectProxy | SparseTensor | IndexedSlices | _basetuple:
  ...

@np_utils.np_doc('equal')
def equal(x1, x2):
  ...

@np_utils.np_doc('not_equal')
def not_equal(x1, x2):
  ...

@np_utils.np_doc('greater')
def greater(x1, x2): # -> _dispatcher_for_greater | object:
  ...

@np_utils.np_doc('greater_equal')
def greater_equal(x1, x2): # -> _dispatcher_for_greater | object:
  ...

@np_utils.np_doc('less')
def less(x1, x2): # -> _dispatcher_for_greater | object:
  ...

@np_utils.np_doc('less_equal')
def less_equal(x1, x2): # -> _dispatcher_for_greater | object:
  ...

@np_utils.np_doc('array_equal')
def array_equal(a1, a2): # -> Any | list[Unknown] | _basetuple | defaultdict[Unknown, Unknown] | ObjectProxy:
  ...

@np_utils.np_doc('logical_and')
def logical_and(x1, x2): # -> _dispatcher_for_logical_and | object:
  ...

@np_utils.np_doc('logical_or')
def logical_or(x1, x2): # -> _dispatcher_for_logical_and | object:
  ...

@np_utils.np_doc('logical_xor')
def logical_xor(x1, x2): # -> _dispatcher_for_logical_and | object:
  ...

@np_utils.np_doc('logical_not')
def logical_not(x): # -> _dispatcher_for_logical_not | object:
  ...

@np_utils.np_doc('linspace')
def linspace(start, stop, num=..., endpoint=..., retstep=..., dtype=..., axis=...): # -> tuple[Unknown | SparseTensor | IndexedSlices | Tensor | Any, Unknown | Tensor | Any] | SparseTensor | IndexedSlices | Tensor | Any:
  ...

@np_utils.np_doc('logspace')
def logspace(start, stop, num=..., endpoint=..., base=..., dtype=..., axis=...): # -> SparseTensor | IndexedSlices | Tensor | Any:
  ...

@np_utils.np_doc('geomspace')
def geomspace(start, stop, num=..., endpoint=..., dtype=..., axis=...): # -> SparseTensor | IndexedSlices | Tensor | Any:
  ...

@np_utils.np_doc('ptp')
def ptp(a, axis=..., keepdims=...):
  ...

@np_utils.np_doc_only('concatenate')
def concatenate(arys, axis=...): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
  ...

@np_utils.np_doc_only('tile')
def tile(a, reps): # -> _dispatcher_for_tile | object:
  ...

@np_utils.np_doc('count_nonzero')
def count_nonzero(a, axis=...): # -> SparseTensor | IndexedSlices | Tensor | Any:
  ...

@np_utils.np_doc('argsort')
def argsort(a, axis=..., kind=..., order=...): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy | SparseTensor | IndexedSlices | Tensor | _basetuple:
  ...

@np_utils.np_doc('sort')
def sort(a, axis=..., kind=..., order=...):
  ...

@np_utils.np_doc('argmax')
def argmax(a, axis=...):
  ...

@np_utils.np_doc('argmin')
def argmin(a, axis=...):
  ...

@np_utils.np_doc('append')
def append(arr, values, axis=...): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
  ...

@np_utils.np_doc('average')
def average(a, axis=..., weights=..., returned=...): # -> tuple[Unknown | defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy | SparseTensor | IndexedSlices | Tensor | _basetuple, Unknown | _dispatcher_for_broadcast_to | object] | defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy | SparseTensor | IndexedSlices | Tensor | _basetuple:
  ...

@np_utils.np_doc('trace')
def trace(a, offset=..., axis1=..., axis2=..., dtype=...):
  ...

@np_utils.np_doc('meshgrid')
def meshgrid(*xi, **kwargs): # -> list[Unknown]:
  """This currently requires copy=True and sparse=False."""
  ...

@np_utils.np_doc_only('einsum')
def einsum(subscripts, *operands, **kwargs):
  ...

def enable_numpy_methods_on_tensor(): # -> None:
  """Adds additional NumPy methods on tf.Tensor class."""
  ...
