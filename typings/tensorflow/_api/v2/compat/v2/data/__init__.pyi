"""
This type stub file was generated by pyright.
"""

import sys as _sys
from . import experimental
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE, DatasetSpec, DatasetV2 as Dataset, INFINITE as INFINITE_CARDINALITY, UNKNOWN as UNKNOWN_CARDINALITY
from tensorflow.python.data.ops.iterator_ops import IteratorBase as Iterator, IteratorSpec
from tensorflow.python.data.ops.options import Options, ThreadingOptions
from tensorflow.python.data.ops.readers import FixedLengthRecordDatasetV2 as FixedLengthRecordDataset, TFRecordDatasetV2 as TFRecordDataset, TextLineDatasetV2 as TextLineDataset

"""`tf.data.Dataset` API for input pipelines.

See [Importing Data](https://tensorflow.org/guide/data) for an overview.

"""