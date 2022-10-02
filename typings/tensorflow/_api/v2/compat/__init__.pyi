"""
This type stub file was generated by pyright.
"""

import sys as _sys
from . import v1, v2
from tensorflow.python.compat.compat import forward_compatibility_horizon, forward_compatible
from tensorflow.python.framework.tensor_shape import dimension_at_index, dimension_value
from tensorflow.python.util.compat import as_bytes, as_str, as_str_any, as_text, bytes_or_text_types, complex_types, integral_types, path_to_str, real_types

"""Compatibility functions.

The `tf.compat` module contains two sets of compatibility functions.

## Tensorflow 1.x and 2.x APIs

The `compat.v1` and `compat.v2` submodules provide a complete copy of both the
`v1` and `v2` APIs for backwards and forwards compatibility across TensorFlow
versions 1.x and 2.x. See the
[migration guide](https://www.tensorflow.org/guide/migrate) for details.

## Utilities for writing compatible code

Aside from the `compat.v1` and `compat.v2` submodules, `tf.compat` also contains
a set of helper functions for writing code that works in both:

* TensorFlow 1.x and 2.x
* Python 2 and 3


## Type collections

The compatibility module also provides the following aliases for common
sets of python types:

* `bytes_or_text_types`
* `complex_types`
* `integral_types`
* `real_types`

"""
