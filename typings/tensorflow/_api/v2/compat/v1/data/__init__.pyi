"""
This type stub file was generated by pyright.
"""

import sys as _sys
from . import experimental
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE, DatasetSpec, DatasetV1 as Dataset, INFINITE as INFINITE_CARDINALITY, UNKNOWN as UNKNOWN_CARDINALITY, get_legacy_output_classes as get_output_classes, get_legacy_output_shapes as get_output_shapes, get_legacy_output_types as get_output_types, make_initializable_iterator, make_one_shot_iterator
from tensorflow.python.data.ops.iterator_ops import Iterator
from tensorflow.python.data.ops.options import Options, ThreadingOptions
from tensorflow.python.data.ops.readers import FixedLengthRecordDatasetV1 as FixedLengthRecordDataset, TFRecordDatasetV1 as TFRecordDataset, TextLineDatasetV1 as TextLineDataset

"""`tf.data.Dataset` API for input pipelines.

See [Importing Data](https://tensorflow.org/guide/data) for an overview.

"""
