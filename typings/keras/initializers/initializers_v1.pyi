"""
This type stub file was generated by pyright.
"""

import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

"""Keras initializers for TF 1."""
_v1_zeros_initializer = ...
_v1_ones_initializer = ...
_v1_constant_initializer = ...
_v1_variance_scaling_initializer = ...
_v1_orthogonal_initializer = ...
_v1_identity = ...
_v1_glorot_uniform_initializer = ...
_v1_glorot_normal_initializer = ...
@keras_export(v1=["keras.initializers.RandomNormal", "keras.initializers.random_normal", "keras.initializers.normal"])
class RandomNormal(tf.compat.v1.random_normal_initializer):
    """Initializer that generates a normal distribution.

    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values to
        generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
      dtype: Default data type, used if no `dtype` argument is provided when
        calling the initializer. Only floating point types are supported.

    @compatibility(TF2)
    Although it is a legacy compat.v1 api,
    `tf.compat.v1.keras.initializers.RandomNormal` is compatible with eager
    execution and `tf.function`.

    To switch to native TF2, switch to using
    `tf.keras.initializers.RandomNormal` (not from `compat.v1`) and
    if you need to change the default dtype use
    `tf.keras.backend.set_floatx(float_dtype)`
    or pass the dtype when calling the initializer, rather than passing it
    when constructing the initializer.

    Random seed behavior:
    Also be aware that if you pass a seed to the TF2 initializer
    API it will reuse that same seed for every single initialization
    (unlike the TF1 initializer)

    #### Structural Mapping to Native TF2

    Before:

    ```python
    initializer = tf.compat.v1.keras.initializers.RandomNormal(
      mean=mean,
      stddev=stddev,
      seed=seed,
      dtype=dtype)

    weight_one = tf.Variable(initializer(shape_one))
    weight_two = tf.Variable(initializer(shape_two))
    ```

    After:

    ```python
    initializer = tf.keras.initializers.RandomNormal(
      mean=mean,
      # seed=seed,  # Setting a seed in the native TF2 API
                    # causes it to produce the same initializations
                    # across multiple calls of the same initializer.
      stddev=stddev)

    weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
    weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
    ```

    #### How to Map Arguments

    | TF1 Arg Name      | TF2 Arg Name    | Note                       |
    | :---------------- | :-------------- | :------------------------- |
    | `mean`            | `mean`          | No change to defaults |
    | `stddev`          | `stddev`        | No change to defaults |
    | `seed`            | `seed`          | Different random number generation |
    :                   :        : semantics (to change in a :
    :                   :        : future version). If set, the TF2 version :
    :                   :        : will use stateless random number :
    :                   :        : generation which will produce the exact :
    :                   :        : same initialization even across multiple :
    :                   :        : calls of the initializer instance. the :
    :                   :        : `compat.v1` version will generate new :
    :                   :        : initializations each time. Do not set :
    :                   :        : a seed if you need different          :
    :                   :        : initializations each time. Instead    :
    :                   :        : either set a global tf seed with      :
    :                   :        : `tf.random.set_seed` if you need      :
    :                   :        : determinism, or initialize each weight:
    :                   :        : with a separate initializer instance  :
    :                   :        : and a different seed.                 :
    | `dtype`           | `dtype`  | The TF2 native api only takes it    |
    :                   :      : as a `__call__` arg, not a constructor arg. :
    | `partition_info`  | -    |  (`__call__` arg in TF1) Not supported      |

    #### Example of fixed-seed behavior differences

    `compat.v1` Fixed seed behavior:

    >>> initializer = tf.compat.v1.keras.initializers.RandomNormal(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=False>

    After:

    >>> initializer = tf.keras.initializers.RandomNormal(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=True>

    @end_compatibility
    """
    def __init__(self, mean=..., stddev=..., seed=..., dtype=...) -> None:
        ...
    


@keras_export(v1=["keras.initializers.RandomUniform", "keras.initializers.random_uniform", "keras.initializers.uniform"])
class RandomUniform(tf.compat.v1.random_uniform_initializer):
    """Initializer that generates tensors with a uniform distribution.

    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range of
        random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range of
        random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
      dtype: Default data type, used if no `dtype` argument is provided when
        calling the initializer.

    @compatibility(TF2)
    Although it is a legacy `compat.v1` api,
    `tf.compat.v1.keras.initializers.RandomUniform` is compatible with eager
    execution and `tf.function`.

    To switch to native TF2, switch to using
    `tf.keras.initializers.RandomUniform` (not from `compat.v1`) and
    if you need to change the default dtype use
    `tf.keras.backend.set_floatx(float_dtype)`
    or pass the dtype when calling the initializer, rather than passing it
    when constructing the initializer.

    Random seed behavior:

    Also be aware that if you pass a seed to the TF2 initializer
    API it will reuse that same seed for every single initialization
    (unlike the TF1 initializer)

    #### Structural Mapping to Native TF2

    Before:

    ```python

    initializer = tf.compat.v1.keras.initializers.RandomUniform(
      minval=minval,
      maxval=maxval,
      seed=seed,
      dtype=dtype)

    weight_one = tf.Variable(initializer(shape_one))
    weight_two = tf.Variable(initializer(shape_two))
    ```

    After:

    ```python
    initializer = tf.keras.initializers.RandomUniform(
      minval=minval,
      maxval=maxval,
      # seed=seed,  # Setting a seed in the native TF2 API
                    # causes it to produce the same initializations
                    # across multiple calls of the same initializer.
      )

    weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
    weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
    ```

    #### How to Map Arguments

    | TF1 Arg Name      | TF2 Arg Name    | Note                       |
    | :---------------- | :-------------- | :------------------------- |
    | `minval`            | `minval`          | No change to defaults |
    | `maxval`          | `maxval`        | No change to defaults |
    | `seed`            | `seed`          | Different random number generation |
    :                    :        : semantics (to change in a :
    :                    :        : future version). If set, the TF2 version :
    :                    :        : will use stateless random number :
    :                    :        : generation which will produce the exact :
    :                    :        : same initialization even across multiple :
    :                    :        : calls of the initializer instance. the :
    :                    :        : `compat.v1` version will generate new :
    :                    :        : initializations each time. Do not set :
    :                    :        : a seed if you need different          :
    :                    :        : initializations each time. Instead    :
    :                    :        : either set a global tf seed with
    :                    :        : `tf.random.set_seed` if you need :
    :                    :        : determinism, or initialize each weight :
    :                    :        : with a separate initializer instance  :
    :                    :        : and a different seed.                 :
    | `dtype`           | `dtype`  | The TF2 native api only takes it  |
    :                   :      : as a `__call__` arg, not a constructor arg. :
    | `partition_info`  | -    |  (`__call__` arg in TF1) Not supported      |

    #### Example of fixed-seed behavior differences

    `compat.v1` Fixed seed behavior:

    >>> initializer = tf.compat.v1.keras.initializers.RandomUniform(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=False>

    After:

    >>> initializer = tf.keras.initializers.RandomUniform(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=True>

    @end_compatibility
    """
    def __init__(self, minval=..., maxval=..., seed=..., dtype=...) -> None:
        ...
    


@keras_export(v1=["keras.initializers.TruncatedNormal", "keras.initializers.truncated_normal"])
class TruncatedNormal(tf.compat.v1.truncated_normal_initializer):
    """Initializer that generates a truncated normal distribution.

    These values are similar to values from a `random_normal_initializer`
    except that values more than two standard deviations from the mean
    are discarded and re-drawn. This is the recommended initializer for
    neural network weights and filters.

    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values to
        generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
      dtype: Default data type, used if no `dtype` argument is provided when
        calling the initializer. Only floating point types are supported.

    @compatibility(TF2)
    Although it is a legacy compat.v1 api,
    `tf.compat.v1.keras.initializers.TruncatedNormal` is compatible with eager
    execution and `tf.function`.

    To switch to native TF2, switch to using
    `tf.keras.initializers.TruncatedNormal` (not from `compat.v1`) and
    if you need to change the default dtype use
    `tf.keras.backend.set_floatx(float_dtype)`
    or pass the dtype when calling the initializer, rather than passing it
    when constructing the initializer.

    Random seed behavior:
    Also be aware that if you pass a seed to the TF2 initializer
    API it will reuse that same seed for every single initialization
    (unlike the TF1 initializer)

    #### Structural Mapping to Native TF2

    Before:

    ```python
    initializer = tf.compat.v1.keras.initializers.TruncatedNormal(
      mean=mean,
      stddev=stddev,
      seed=seed,
      dtype=dtype)

    weight_one = tf.Variable(initializer(shape_one))
    weight_two = tf.Variable(initializer(shape_two))
    ```

    After:

    ```python
    initializer = tf.keras.initializers.TruncatedNormal(
      mean=mean,
      # seed=seed,  # Setting a seed in the native TF2 API
                    # causes it to produce the same initializations
                    # across multiple calls of the same initializer.
      stddev=stddev)

    weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
    weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
    ```

    #### How to Map Arguments

    | TF1 Arg Name      | TF2 Arg Name    | Note                       |
    | :---------------- | :-------------- | :------------------------- |
    | `mean`            | `mean`          | No change to defaults |
    | `stddev`          | `stddev`        | No change to defaults |
    | `seed`            | `seed`          | Different random number generation |
    :                    :        : semantics (to change in a :
    :                    :        : future version). If set, the TF2 version :
    :                    :        : will use stateless random number :
    :                    :        : generation which will produce the exact :
    :                    :        : same initialization even across multiple :
    :                    :        : calls of the initializer instance. the :
    :                    :        : `compat.v1` version will generate new :
    :                    :        : initializations each time. Do not set :
    :                    :        : a seed if you need different          :
    :                    :        : initializations each time. Instead    :
    :                    :        : either set a global tf seed with
    :                    :        : `tf.random.set_seed` if you need :
    :                    :        : determinism, or initialize each weight :
    :                    :        : with a separate initializer instance  :
    :                    :        : and a different seed.                 :
    | `dtype`           | `dtype`  | The TF2 native api only takes it  |
    :                   :      : as a `__call__` arg, not a constructor arg. :
    | `partition_info`  | -    |  (`__call__` arg in TF1) Not supported      |

    #### Example of fixed-seed behavior differences

    `compat.v1` Fixed seed behavior:

    >>> initializer = tf.compat.v1.keras.initializers.TruncatedNormal(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=False>

    After:

    >>> initializer = tf.keras.initializers.TruncatedNormal(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=True>

    @end_compatibility
    """
    def __init__(self, mean=..., stddev=..., seed=..., dtype=...) -> None:
        """Initializer that generates a truncated normal distribution.


        Args:
          mean: a python scalar or a scalar tensor. Mean of the random values to
            generate.
          stddev: a python scalar or a scalar tensor. Standard deviation of the
            random values to generate.
          seed: A Python integer. Used to create random seeds. See
            `tf.compat.v1.set_random_seed` for behavior.
          dtype: Default data type, used if no `dtype` argument is provided when
            calling the initializer. Only floating point types are supported.
        """
        ...
    


@keras_export(v1=["keras.initializers.lecun_normal"])
class LecunNormal(tf.compat.v1.variance_scaling_initializer):
    def __init__(self, seed=...) -> None:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


@keras_export(v1=["keras.initializers.lecun_uniform"])
class LecunUniform(tf.compat.v1.variance_scaling_initializer):
    def __init__(self, seed=...) -> None:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


@keras_export(v1=["keras.initializers.he_normal"])
class HeNormal(tf.compat.v1.variance_scaling_initializer):
    def __init__(self, seed=...) -> None:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


@keras_export(v1=["keras.initializers.he_uniform"])
class HeUniform(tf.compat.v1.variance_scaling_initializer):
    def __init__(self, seed=...) -> None:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


