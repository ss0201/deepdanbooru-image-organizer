"""
This type stub file was generated by pyright.
"""

import sys as _sys
from keras.preprocessing.sequence import TimeseriesGenerator, make_sampling_table, skipgrams
from keras.utils.data_utils import pad_sequences
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Utilities for preprocessing sequence data.

Deprecated: `tf.keras.preprocessing.sequence` APIs are not recommended for new
code. Prefer `tf.keras.utils.timeseries_dataset_from_array` and
the `tf.data` APIs which provide a much more flexible mechanisms for dealing
with sequences. See the [tf.data guide](https://www.tensorflow.org/guide/data)
for more details.

"""
if notisinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
