"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow.core.framework.summary_pb2 import Summary, SummaryDescription
from tensorflow.core.util.event_pb2 import Event, SessionLog, TaggedRunMetadata
from tensorflow.python.ops.summary_ops_v2 import all_v2_summary_ops, initialize
from tensorflow.python.summary.summary import audio, get_summary_description, histogram, image, merge, merge_all, scalar, tensor_summary, text
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.summary.writer.writer_cache import FileWriterCache

"""Operations for writing summary data, for use in analysis and visualization.

See the [Summaries and
TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) guide.

"""
