"""
This type stub file was generated by pyright.
"""

import distutils as _distutils
import inspect as _inspect
import logging as _logging
import os as _os
import site as _site
import sys as _sys
import typing as _typing
import tensorflow_io_gcs_filesystem as _tensorflow_io_gcs_filesystem
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
from tensorflow.python import tf2 as _tf2
from ._api.v2 import __internal__, __operators__, audio, autodiff, autograph, bitwise, compat, config, data, debugging, distribute, dtypes, errors, experimental, feature_column, graph_util, image, io, linalg, lite, lookup, math, mlir, nest, nn, profiler, quantization, queue, ragged, random, raw_ops, saved_model, sets, signal, sparse, strings, summary, sysconfig, test, tpu, train, types, version, xla
from tensorflow.python.data.ops.optional_ops import OptionalSpec
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.eager.def_function import function
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework.device_spec import DeviceSpecV2 as DeviceSpec
from tensorflow.python.framework.dtypes import DType, as_dtype, bfloat16, bool, complex128, complex64, double, float16, float32, float64, half, int16, int32, int64, int8, qint16, qint32, qint8, quint16, quint8, resource, string, uint16, uint32, uint64, uint8, variant
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.framework.indexed_slices import IndexedSlices, IndexedSlicesSpec
from tensorflow.python.framework.load_library import load_library, load_op_library
from tensorflow.python.framework.ops import Graph, Operation, RegisterGradient, Tensor, control_dependencies, convert_to_tensor_v2_with_dispatch as convert_to_tensor, device_v2 as device, get_current_name_scope, init_scope, inside_function, name_scope_v2 as name_scope, no_gradient
from tensorflow.python.framework.sparse_tensor import SparseTensor, SparseTensorSpec
from tensorflow.python.framework.tensor_conversion_registry import register_tensor_conversion_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray, constant_value as get_static_value, is_tf_type as is_tensor, make_tensor_proto
from tensorflow.python.framework.type_spec import TypeSpec, type_spec_from_value
from tensorflow.python.framework.versions import COMPILER_VERSION as __compiler_version__, CXX11_ABI_FLAG as __cxx11_abi_flag__, GIT_VERSION as __git_version__, MONOLITHIC_BUILD as __monolithic_build__, VERSION as __version__
from tensorflow.python.module.module import Module
from tensorflow.python.ops.array_ops import batch_to_space_v2 as batch_to_space, boolean_mask_v2 as boolean_mask, broadcast_dynamic_shape, broadcast_static_shape, concat, edit_distance, expand_dims_v2 as expand_dims, fill, fingerprint, gather_nd_v2 as gather_nd, gather_v2 as gather, guarantee_const, identity, meshgrid, newaxis, one_hot, ones, ones_like_v2 as ones_like, pad_v2 as pad, parallel_stack, rank, repeat, required_space_to_batch_paddings, reshape, reverse_sequence_v2 as reverse_sequence, searchsorted, sequence_mask, shape_n, shape_v2 as shape, size_v2 as size, slice, space_to_batch_v2 as space_to_batch, split, squeeze_v2 as squeeze, stack, stop_gradient, strided_slice, tensor_scatter_nd_update, transpose_v2 as transpose, unique, unique_with_counts, unstack, where_v2 as where, zeros, zeros_like_v2 as zeros_like
from tensorflow.python.ops.batch_ops import batch_function as nondifferentiable_batch_function
from tensorflow.python.ops.check_ops import assert_equal_v2 as assert_equal, assert_greater_v2 as assert_greater, assert_less_v2 as assert_less, assert_rank_v2 as assert_rank, ensure_shape
from tensorflow.python.ops.clip_ops import clip_by_global_norm, clip_by_norm, clip_by_value
from tensorflow.python.ops.control_flow_ops import Assert, case_v2 as case, cond_for_tf_v2 as cond, group, switch_case, tuple_v2 as tuple, while_loop_v2 as while_loop
from tensorflow.python.ops.critical_section_ops import CriticalSection
from tensorflow.python.ops.custom_gradient import custom_gradient, grad_pass_through, recompute_grad
from tensorflow.python.ops.functional_ops import foldl_v2 as foldl, foldr_v2 as foldr, scan_v2 as scan
from tensorflow.python.ops.gen_array_ops import bitcast, broadcast_to, extract_volume_patches, identity_n, reverse_v2 as reverse, scatter_nd, space_to_batch_nd, tensor_scatter_add as tensor_scatter_nd_add, tensor_scatter_max as tensor_scatter_nd_max, tensor_scatter_min as tensor_scatter_nd_min, tensor_scatter_sub as tensor_scatter_nd_sub, tile, unravel_index
from tensorflow.python.ops.gen_control_flow_ops import no_op
from tensorflow.python.ops.gen_data_flow_ops import dynamic_partition, dynamic_stitch
from tensorflow.python.ops.gen_linalg_ops import matrix_square_root
from tensorflow.python.ops.gen_logging_ops import timestamp
from tensorflow.python.ops.gen_math_ops import acosh, asin, asinh, atan, atan2, atanh, cos, cosh, greater, greater_equal, less, less_equal, logical_and, logical_not, logical_or, maximum, minimum, neg as negative, real_div as realdiv, sin, sinh, square, tan, tanh, truncate_div as truncatediv, truncate_mod as truncatemod
from tensorflow.python.ops.gen_nn_ops import approx_top_k
from tensorflow.python.ops.gen_random_index_shuffle_ops import random_index_shuffle
from tensorflow.python.ops.gen_string_ops import as_string
from tensorflow.python.ops.gradients_impl import HessiansV2 as hessians, gradients_v2 as gradients
from tensorflow.python.ops.gradients_util import AggregationMethod
from tensorflow.python.ops.histogram_ops import histogram_fixed_width, histogram_fixed_width_bins
from tensorflow.python.ops.init_ops_v2 import Constant as constant_initializer, Ones as ones_initializer, RandomNormal as random_normal_initializer, RandomUniform as random_uniform_initializer, Zeros as zeros_initializer
from tensorflow.python.ops.linalg_ops import eig, eigvals, eye, norm_v2 as norm
from tensorflow.python.ops.logging_ops import print_v2 as print
from tensorflow.python.ops.manip_ops import roll
from tensorflow.python.ops.map_fn import map_fn_v2 as map_fn
from tensorflow.python.ops.math_ops import abs, acos, add, add_n, argmax_v2 as argmax, argmin_v2 as argmin, cast, complex, cumsum, divide, equal, exp, floor, linspace_nd as linspace, matmul, multiply, not_equal, pow, range, reduce_all, reduce_any, reduce_logsumexp, reduce_max, reduce_mean, reduce_min, reduce_prod, reduce_sum, round, saturate_cast, scalar_mul_v2 as scalar_mul, sigmoid, sign, sqrt, subtract, tensordot, truediv
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor, RaggedTensorSpec
from tensorflow.python.ops.script_ops import eager_py_func as py_function, numpy_function
from tensorflow.python.ops.sort_ops import argsort, sort
from tensorflow.python.ops.special_math_ops import einsum
from tensorflow.python.ops.tensor_array_ops import TensorArray, TensorArraySpec
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.ops.variable_scope import variable_creator_scope
from tensorflow.python.ops.variables import Variable, VariableAggregationV2 as VariableAggregation, VariableSynchronization
from tensorflow.python.platform.tf_logging import get_logger
from tensorflow.python.compat import v2_compat as _compat
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi
from tensorflow_estimator.python.estimator.api._v2 import estimator as estimator
from keras.api._v2 import keras
from keras.api._v2.keras import initializers, losses, metrics, optimizers

"""
Top-level module of TensorFlow. By convention, we refer to this module as
`tf` instead of `tensorflow`, following the common practice of importing
TensorFlow via the command `import tensorflow as tf`.

The primary function of this module is to import all of the public TensorFlow
interfaces into a single place. The interfaces themselves are located in
sub-modules, as described below.

Note that the file `__init__.py` in the TensorFlow source code tree is actually
only a placeholder to enable test cases to run. The TensorFlow build replaces
this file with a file generated from [`api_template.__init__.py`](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/api_template.__init__.py)
"""
_API_MODULE = _sys.modules[__name__].bitwise
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
_current_module = ...
if nothasattr(_current_module, '__path__'):
  __path__ = ...
else:
  ...
if (_os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == 'true' or _os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == '1'):
  ...
_estimator_module = ...
estimator = ...
_module_dir = ...
if _module_dir:
  ...
_keras_module = ...
_keras = ...
_module_dir = ...
if _module_dir:
  ...
_major_api_version = ...
_site_packages_dirs = ...
if _site.ENABLE_USER_SITE and _site.USER_SITE is not None:
  ...
if 'getsitepackages' in dir(_site):
  ...
if 'sysconfig' in dir(_distutils):
  ...
_site_packages_dirs = ...
_current_file_location = ...
if _running_from_pip_package():
  _tf_dir = ...
  _kernel_dir = ...
if hasattr(_current_module, 'keras'):
  ...
if hasattr(_current_module, "keras"):
  ...
if _typing.TYPE_CHECKING:
  ...
_names_with_underscore = ...
__all__ = [_s for _s in dir() if not_s.startswith('_')]
