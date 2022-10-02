"""
This type stub file was generated by pyright.
"""

import collections
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.util.tf_export import tf_export

"""This API defines FeatureColumn for sequential input.

NOTE: This API is a work in progress and will likely be changing frequently.
"""
def concatenate_context_input(context_input, sequence_input): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
  """Replicates `context_input` across all timesteps of `sequence_input`.

  Expands dimension 1 of `context_input` then tiles it `sequence_length` times.
  This value is appended to `sequence_input` on dimension 2 and the result is
  returned.

  Args:
    context_input: A `Tensor` of dtype `float32` and shape `[batch_size, d1]`.
    sequence_input: A `Tensor` of dtype `float32` and shape `[batch_size,
      padded_length, d0]`.

  Returns:
    A `Tensor` of dtype `float32` and shape `[batch_size, padded_length,
    d0 + d1]`.

  Raises:
    ValueError: If `sequence_input` does not have rank 3 or `context_input` does
      not have rank 2.
  """
  ...

@tf_export('feature_column.sequence_categorical_column_with_identity')
def sequence_categorical_column_with_identity(key, num_buckets, default_value=...): # -> SequenceCategoricalColumn:
  """Returns a feature column that represents sequences of integers.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  watches = sequence_categorical_column_with_identity(
      'watches', num_buckets=1000)
  watches_embedding = embedding_column(watches, dimension=10)
  columns = [watches_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    num_buckets: Range of inputs. Namely, inputs are expected to be in the
      range `[0, num_buckets)`.
    default_value: If `None`, this column's graph operations will fail for
      out-of-range inputs. Otherwise, this value must be in the range
      `[0, num_buckets)`, and will replace out-of-range inputs.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: if `num_buckets` is less than one.
    ValueError: if `default_value` is not in range `[0, num_buckets)`.
  """
  ...

@tf_export('feature_column.sequence_categorical_column_with_hash_bucket')
def sequence_categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=...): # -> SequenceCategoricalColumn:
  """A sequence of categorical terms where ids are set by hashing.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  tokens = sequence_categorical_column_with_hash_bucket(
      'tokens', hash_bucket_size=1000)
  tokens_embedding = embedding_column(tokens, dimension=10)
  columns = [tokens_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: `hash_bucket_size` is not greater than 1.
    ValueError: `dtype` is neither string nor integer.
  """
  ...

@tf_export('feature_column.sequence_categorical_column_with_vocabulary_file')
def sequence_categorical_column_with_vocabulary_file(key, vocabulary_file, vocabulary_size=..., num_oov_buckets=..., default_value=..., dtype=...): # -> SequenceCategoricalColumn:
  """A sequence of categorical terms where ids use a vocabulary file.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  states = sequence_categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
  states_embedding = embedding_column(states, dimension=10)
  columns = [states_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  """
  ...

@tf_export('feature_column.sequence_categorical_column_with_vocabulary_list')
def sequence_categorical_column_with_vocabulary_list(key, vocabulary_list, dtype=..., default_value=..., num_oov_buckets=...): # -> SequenceCategoricalColumn:
  """A sequence of categorical terms where ids use an in-memory list.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  colors = sequence_categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
      num_oov_buckets=2)
  colors_embedding = embedding_column(colors, dimension=3)
  columns = [colors_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    vocabulary_list: An ordered iterable defining the vocabulary. Each feature
      is mapped to the index of its value (if present) in `vocabulary_list`.
      Must be castable to `dtype`.
    dtype: The type of features. Only string and integer types are supported.
      If `None`, it will be inferred from `vocabulary_list`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a
      hash of the input value. A positive `num_oov_buckets` can not be specified
      with `default_value`.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: if `vocabulary_list` is empty, or contains duplicate keys.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: if `dtype` is not integer or string.
  """
  ...

@tf_export('feature_column.sequence_numeric_column')
def sequence_numeric_column(key, shape=..., default_value=..., dtype=..., normalizer_fn=...): # -> SequenceNumericColumn:
  """Returns a feature column that represents sequences of numeric data.

  Example:

  ```python
  temperature = sequence_numeric_column('temperature')
  columns = [temperature]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input features.
    shape: The shape of the input data per sequence id. E.g. if `shape=(2,)`,
      each example must contain `2 * sequence_length` values.
    default_value: A single value compatible with `dtype` that is used for
      padding the sparse data into a dense `Tensor`.
    dtype: The type of values.
    normalizer_fn: If not `None`, a function that can be used to normalize the
      value of the tensor after `default_value` is applied for parsing.
      Normalizer function takes the input `Tensor` as its argument, and returns
      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
      even though the most common use case of this function is normalization, it
      can be used for any kind of Tensorflow transformations.

  Returns:
    A `SequenceNumericColumn`.

  Raises:
    TypeError: if any dimension in shape is not an int.
    ValueError: if any dimension in shape is not a positive integer.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  ...

class SequenceNumericColumn(fc.SequenceDenseColumn, collections.namedtuple('SequenceNumericColumn', ('key', 'shape', 'default_value', 'dtype', 'normalizer_fn'))):
  """Represents sequences of numeric data."""
  @property
  def name(self):
    """See `FeatureColumn` base class."""
    ...
  
  @property
  def parse_example_spec(self): # -> dict[Unknown, VarLenFeature]:
    """See `FeatureColumn` base class."""
    ...
  
  def transform_feature(self, transformation_cache, state_manager):
    """See `FeatureColumn` base class.

    In this case, we apply the `normalizer_fn` to the input tensor.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Normalized input tensor.
    """
    ...
  
  @property
  def variable_shape(self): # -> TensorShape:
    """Returns a `TensorShape` representing the shape of sequence input."""
    ...
  
  def get_sequence_dense_tensor(self, transformation_cache, state_manager): # -> TensorSequenceLengthPair:
    """Returns a `TensorSequenceLengthPair`.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    ...
  
  @property
  def parents(self): # -> list[Unknown]:
    """See 'FeatureColumn` base class."""
    ...
  
  def get_config(self): # -> dict[str, Unknown]:
    """See 'FeatureColumn` base class."""
    ...
  
  @classmethod
  def from_config(cls, config, custom_objects=..., columns_by_name=...): # -> Self@SequenceNumericColumn:
    """See 'FeatureColumn` base class."""
    ...
  

