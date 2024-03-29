"""
This type stub file was generated by pyright.
"""

from keras.metrics.base_metric import Mean, MeanMetricWrapper, MeanTensor, Metric, Reduce, Sum, SumOverBatchSize, SumOverBatchSizeMetricWrapper, clone_metric, clone_metrics
from keras.metrics.metrics import AUC, Accuracy, BinaryAccuracy, BinaryCrossentropy, BinaryIoU, CategoricalAccuracy, CategoricalCrossentropy, CategoricalHinge, CosineSimilarity, FalseNegatives, FalsePositives, Hinge, IoU, KLDivergence, LogCoshError, MeanAbsoluteError, MeanAbsolutePercentageError, MeanIoU, MeanRelativeError, MeanSquaredError, MeanSquaredLogarithmicError, OneHotIoU, OneHotMeanIoU, Poisson, Precision, PrecisionAtRecall, Recall, RecallAtPrecision, RootMeanSquaredError, SensitivityAtSpecificity, SensitivitySpecificityBase, SparseCategoricalAccuracy, SparseCategoricalCrossentropy, SparseTopKCategoricalAccuracy, SpecificityAtSensitivity, SquaredHinge, TopKCategoricalAccuracy, TrueNegatives, TruePositives, _ConfusionMatrixConditionCount, _IoUBase, accuracy, binary_accuracy, binary_crossentropy, categorical_accuracy, categorical_crossentropy, categorical_hinge, cosine_similarity, hinge, kullback_leibler_divergence, logcosh, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_logarithmic_error, poisson, sparse_categorical_accuracy, sparse_categorical_crossentropy, sparse_top_k_categorical_accuracy, squared_hinge, top_k_categorical_accuracy
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object
from tensorflow.python.util.tf_export import keras_export

"""All Keras metrics."""
acc = ...
bce = ...
mse = ...
mae = ...
mape = ...
msle = ...
log_cosh = ...
cosine_proximity = ...
@keras_export("keras.metrics.serialize")
def serialize(metric): # -> Any | dict[str, Unknown] | None:
    """Serializes metric function or `Metric` instance.

    Args:
      metric: A Keras `Metric` instance or a metric function.

    Returns:
      Metric configuration dictionary.
    """
    ...

@keras_export("keras.metrics.deserialize")
def deserialize(config, custom_objects=...): # -> Any | None:
    """Deserializes a serialized metric class/function instance.

    Args:
      config: Metric configuration.
      custom_objects: Optional dictionary mapping names (strings) to custom
        objects (classes and functions) to be considered during deserialization.

    Returns:
        A Keras `Metric` instance or a metric function.
    """
    ...

@keras_export("keras.metrics.get")
def get(identifier): # -> Any | None:
    """Retrieves a Keras metric as a `function`/`Metric` class instance.

    The `identifier` may be the string name of a metric function or class.

    >>> metric = tf.keras.metrics.get("categorical_crossentropy")
    >>> type(metric)
    <class 'function'>
    >>> metric = tf.keras.metrics.get("CategoricalCrossentropy")
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    You can also specify `config` of the metric to this function by passing dict
    containing `class_name` and `config` as an identifier. Also note that the
    `class_name` must map to a `Metric` class

    >>> identifier = {"class_name": "CategoricalCrossentropy",
    ...               "config": {"from_logits": True}}
    >>> metric = tf.keras.metrics.get(identifier)
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    Args:
      identifier: A metric identifier. One of None or string name of a metric
        function/class or metric configuration dictionary or a metric function
        or a metric class instance

    Returns:
      A Keras metric as a `function`/ `Metric` class instance.

    Raises:
      ValueError: If `identifier` cannot be interpreted.
    """
    ...

