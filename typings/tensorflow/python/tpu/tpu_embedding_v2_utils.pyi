"""
This type stub file was generated by pyright.
"""

import abc
import six
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, TypeVar, Union
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import init_ops_v2, variables as tf_variables
from tensorflow.python.util.tf_export import tf_export

"""Companion classes for mid level API for TPU Embeddings in TF2."""
TableVariable = TypeVar("TableVariable", sharded_variable.ShardedVariable, tf_variables.Variable)
SlotVarCreationFnType = Callable[[TableVariable, List[Text], List[init_ops_v2.Initializer]], Dict[Text, TableVariable]]
ClipValueType = Union[Tuple[float, float], float]
@six.add_metaclass(abc.ABCMeta)
class _Optimizer:
  """Base class for all optimizers, with common parameters."""
  def __init__(self, learning_rate: Union[float, Callable[[], float]], use_gradient_accumulation: bool, clip_weight_min: Optional[float], clip_weight_max: Optional[float], weight_decay_factor: Optional[float], multiply_weight_decay_factor_by_learning_rate: bool, clipvalue: Optional[ClipValueType] = ..., slot_variable_creation_fn: Optional[SlotVarCreationFnType] = ...) -> None:
    ...
  
  def __eq__(self, other: Any) -> Union[Any, bool]:
    ...
  
  def __hash__(self) -> int:
    ...
  


@tf_export("tpu.experimental.embedding.SGD")
class SGD(_Optimizer):
  """Optimization parameters for stochastic gradient descent for TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.SGD(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """
  def __init__(self, learning_rate: Union[float, Callable[[], float]] = ..., use_gradient_accumulation: bool = ..., clip_weight_min: Optional[float] = ..., clip_weight_max: Optional[float] = ..., weight_decay_factor: Optional[float] = ..., multiply_weight_decay_factor_by_learning_rate: bool = ..., clipvalue: Optional[ClipValueType] = ...) -> None:
    """Optimization parameters for stochastic gradient descent.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed. Weights are decayed by multiplying the weight
        by this factor each step.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tiple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction. Note if this is
        set, you may see a decrease in performance as  gradient accumulation
        will be enabled (it is normally off for SGD as it has no affect on
        accuracy). See
        'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for more
        information on gradient accumulation and its impact on tpu embeddings.
    """
    ...
  


@tf_export("tpu.experimental.embedding.Adagrad")
class Adagrad(_Optimizer):
  """Optimization parameters for Adagrad with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """
  def __init__(self, learning_rate: Union[float, Callable[[], float]] = ..., initial_accumulator_value: float = ..., use_gradient_accumulation: bool = ..., clip_weight_min: Optional[float] = ..., clip_weight_max: Optional[float] = ..., weight_decay_factor: Optional[float] = ..., multiply_weight_decay_factor_by_learning_rate: bool = ..., slot_variable_creation_fn: Optional[SlotVarCreationFnType] = ..., clipvalue: Optional[ClipValueType] = ...) -> None:
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      initial_accumulator_value: initial accumulator for Adagrad.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
          table variable, a list of slot names to create for it, and a list of
          initializers. This function should return a dict with the slot names
          as keys and the created variables as values with types matching the
          table variable. When set to None (the default), uses the built-in
          variable creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tuple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
    """
    ...
  


@tf_export("tpu.experimental.embedding.AdagradMomentum")
class AdagradMomentum(_Optimizer):
  """Optimization parameters for Adagrad + Momentum with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """
  def __init__(self, learning_rate: Union[float, Callable[[], float]] = ..., momentum: float = ..., use_nesterov: bool = ..., exponent: float = ..., beta2: float = ..., epsilon: float = ..., use_gradient_accumulation: bool = ..., clip_weight_min: Optional[float] = ..., clip_weight_max: Optional[float] = ..., weight_decay_factor: Optional[float] = ..., multiply_weight_decay_factor_by_learning_rate: bool = ..., slot_variable_creation_fn: Optional[SlotVarCreationFnType] = ..., clipvalue: Optional[ClipValueType] = ...) -> None:
    """Optimization parameters for Adagrad + Momentum.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      momentum: Moving average parameter for the momentum accumulator.
      use_nesterov: Whether to use the Nesterov variant of momentum. See
        Sutskever et al., 2013.
      exponent: Exponent for the Adagrad accumulator.
      beta2: Moving average parameter for the Adagrad accumulator.
      epsilon: initial accumulator for Adagrad accumulator.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
          table variable, a list of slot names to create for it, and a list of
          initializers. This function should return a dict with the slot names
          as keys and the created variables as values with types matching the
          table variable. When set to None (the default), uses the built-in
          variable creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tuple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
    """
    ...
  


@tf_export("tpu.experimental.embedding.FTRL")
class FTRL(_Optimizer):
  """Optimization parameters for FTRL with TPU embeddings.

  See Algorithm 1 of this
  [paper](https://research.google.com/pubs/archive/41159.pdf).

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.FTRL(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """
  def __init__(self, learning_rate: Union[float, Callable[[], float]] = ..., learning_rate_power: float = ..., l1_regularization_strength: float = ..., l2_regularization_strength: float = ..., beta: float = ..., initial_accumulator_value: float = ..., use_gradient_accumulation: bool = ..., clip_weight_min: Optional[float] = ..., clip_weight_max: Optional[float] = ..., weight_decay_factor: Optional[float] = ..., multiply_weight_decay_factor_by_learning_rate: bool = ..., slot_variable_creation_fn: Optional[SlotVarCreationFnType] = ..., clipvalue: Optional[ClipValueType] = ..., multiply_linear_by_learning_rate: bool = ..., allow_zero_accumulator: bool = ...) -> None:
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      learning_rate_power: A float value, must be less or equal to zero.
        Controls how the learning rate decreases during training. Use zero for a
        fixed learning rate.
      l1_regularization_strength: A float value, must be greater than or equal
        to zero.
      l2_regularization_strength: A float value, must be greater than or equal
        to zero.
      beta: A float value, representing the beta value from the paper.
      initial_accumulator_value: The starting value for accumulators. Only zero
        or positive values are allowed.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
          table variable, a list of slot names to create for it, and a list of
          initializers. This function should return a dict with the slot names
          as keys and the created variables as values with types matching the
          table variable. When set to None (the default), uses the built-in
          variable creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tuple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
      multiply_linear_by_learning_rate: If set to True, a modified formula is
        used for FTRL that treats the "linear" accumulator as being
        pre-multiplied by the learning rate (i.e., the accumulator named
        "linear" actually stores "linear * learning_rate"). Other than
        checkpoint compatibility, this is mathematically equivalent for a static
        learning rate; for a dynamic learning rate, it is nearly the same as
        long as the learning rate does not change quickly. The benefit of this
        is that the modified formula handles zero and near-zero learning rates
        without producing NaNs, improving flexibility for learning rate ramp-up.
      allow_zero_accumulator: If set to True, changes some internal formulas to
        allow zero and near-zero accumulator values at the cost of some
        performance; this only needs to be set if you are using an initial
        accumulator value of zero, which is uncommon.
    """
    ...
  


@tf_export("tpu.experimental.embedding.Adam")
class Adam(_Optimizer):
  """Optimization parameters for Adam with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  NOTE: By default this optimizer is lazy, i.e. it will not apply the gradient
  update of zero to rows that were not looked up. You can change this behavior
  by setting `lazy_adam` to `False`.

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adam(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """
  def __init__(self, learning_rate: Union[float, Callable[[], float]] = ..., beta_1: float = ..., beta_2: float = ..., epsilon: float = ..., lazy_adam: bool = ..., sum_inside_sqrt: bool = ..., use_gradient_accumulation: bool = ..., clip_weight_min: Optional[float] = ..., clip_weight_max: Optional[float] = ..., weight_decay_factor: Optional[float] = ..., multiply_weight_decay_factor_by_learning_rate: bool = ..., slot_variable_creation_fn: Optional[SlotVarCreationFnType] = ..., clipvalue: Optional[ClipValueType] = ...) -> None:
    """Optimization parameters for Adam.

    See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
    complete description of these parameters and their impacts on the optimizer
    algorithm.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      beta_1: A float value. The exponential decay rate for the 1st moment
        estimates.
      beta_2: A float value. The exponential decay rate for the 2nd moment
        estimates.
      epsilon: A small constant for numerical stability.
      lazy_adam: Use lazy Adam instead of Adam. Lazy Adam trains faster.
      sum_inside_sqrt: When this is true, the Adam update formula is changed
        from `m / (sqrt(v) + epsilon)` to `m / sqrt(v + epsilon**2)`. This
        option improves the performance of TPU training and is not expected to
        harm model quality.
      use_gradient_accumulation: Setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
          table variable, a list of slot names to create for it, and a list of
          initializers. This function should return a dict with the slot names
          as keys and the created variables as values with types matching the
          table variable. When set to None (the default), uses the built-in
          variable creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tiple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
    """
    ...
  


@tf_export("tpu.experimental.embedding.TableConfig")
class TableConfig:
  """Configuration data for one embedding table.

  This class holds the configuration data for a single embedding table. It is
  used as the `table` parameter of a
  `tf.tpu.experimental.embedding.FeatureConfig`. Multiple
  `tf.tpu.experimental.embedding.FeatureConfig` objects can use the same
  `tf.tpu.experimental.embedding.TableConfig` object. In this case a shared
  table will be created for those feature lookups.

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.

  """
  def __init__(self, vocabulary_size: int, dim: int, initializer: Optional[Callable[[Any], None]] = ..., optimizer: Optional[_Optimizer] = ..., combiner: Text = ..., name: Optional[Text] = ...) -> None:
    """Embedding table configuration.

    Args:
      vocabulary_size: Size of the table's vocabulary (number of rows).
      dim: The embedding dimension (width) of the table.
      initializer: A callable initializer taking one parameter, the shape of the
        variable that will be initialized. Will be called once per task, to
        initialize that task's shard of the embedding table. If not specified,
        defaults to `truncated_normal_initializer` with mean `0.0` and standard
        deviation `1/sqrt(dim)`.
      optimizer: An optional instance of an optimizer parameters class, instance
        of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`. It set will override the global
        optimizer passed to `tf.tpu.experimental.embedding.TPUEmbedding`.
      combiner: A string specifying how to reduce if there are multiple entries
        in a single row. Currently 'mean', 'sqrtn', 'sum' are supported, with
        'mean' the default. 'sqrtn' often achieves good accuracy, in particular
        with bag-of-words columns. For more information, see
        `tf.nn.embedding_lookup_sparse`.
      name: An optional string used to name the table. Useful for debugging.

    Returns:
      `TableConfig`.

    Raises:
      ValueError: if `vocabulary_size` is not a positive integer.
      ValueError: if `dim` is not a positive integer.
      ValueError: if `initializer` is specified and is not callable.
      ValueError: if `combiner` is not supported.
    """
    ...
  
  def __repr__(self): # -> str:
    ...
  


@tf_export("tpu.experimental.embedding.FeatureConfig")
class FeatureConfig:
  """Configuration data for one embedding feature.

  This class holds the configuration data for a single embedding feature. The
  main use is to assign features to `tf.tpu.experimental.embedding.TableConfig`s
  via the table parameter:

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.

  You can also specify the output shape for each feature. The output shape
  should be the expected activation shape excluding the table dimension. For
  dense and sparse tensor, the output shape should be the same as the input
  shape excluding the last dimension. For ragged tensor, the output shape can
  mismatch the input shape.

  NOTE: The `max_sequence_length` will be only used when the input tensor has
  rank 2 and the `output_shape` is not set in the feature config.

  When feeding features into `embedding.enqueue` they can be `tf.Tensor`s,
  `tf.SparseTensor`s or `tf.RaggedTensor`s. When the argument
  `max_sequence_length` is 0, the default, you should expect a output of
  `embedding.dequeue` for this feature of shape `(batch_size, dim)`. If
  `max_sequence_length` is greater than 0, the feature is embedded as a sequence
  and padded up to the given length. The shape of the output for this feature
  will be `(batch_size, max_sequence_length, dim)`.
  """
  def __init__(self, table: TableConfig, max_sequence_length: int = ..., validate_weights_and_indices: bool = ..., output_shape: Optional[Union[List[int], TensorShape]] = ..., name: Optional[Text] = ...) -> None:
    """Feature configuration.

    Args:
      table: An instance of `tf.tpu.experimental.embedding.TableConfig`,
        describing the table in which this feature should be looked up.
      max_sequence_length: If positive, the feature is a sequence feature with
        the corresponding maximum sequence length. If the sequence is longer
        than this, it will be truncated. If 0, the feature is not a sequence
        feature.
      validate_weights_and_indices: If true, uses safe_embedding_lookup during
        serving which ensures there are no empty rows and all weights and ids
        are positive at the expense of extra compute cost.
      output_shape: Optional argument to config the output shape of the feature
        activation. If provided, the feature feeding to the `embedding.enqueue`
        has to match the shape (for ragged tensor, the input shape and output
        shape can mismatch). If not provided, the shape can be either provided
        to the `embedding.build` or auto detected at the runtime.
      name: An optional name for the feature, useful for debugging.

    Returns:
      `FeatureConfig`.

    Raises:
      ValueError: if `table` is not an instance of
        `tf.tpu.experimental.embedding.TableConfig`.
      ValueError: if `max_sequence_length` not an integer or is negative.
    """
    ...
  
  def __repr__(self): # -> str:
    ...
  


def log_tpu_embedding_configuration(config: tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration) -> None:
  """Logs a TPUEmbeddingConfiguration proto across multiple statements.

  Args:
    config: TPUEmbeddingConfiguration proto to log.  Necessary because
      logging.info has a maximum length to each log statement, which
      particularly large configs can exceed.
  """
  ...
