"""
This type stub file was generated by pyright.
"""

import sys as _sys
from . import embedding
from tensorflow.python.tpu.device_assignment import DeviceAssignment
from tensorflow.python.tpu.feature_column_v2 import embedding_column_v2 as embedding_column, shared_embedding_columns_v2 as shared_embedding_columns
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.tpu.tpu_embedding import AdagradParameters, AdamParameters, FtrlParameters, StochasticGradientDescentParameters
from tensorflow.python.tpu.tpu_hardware_feature import HardwareFeature
from tensorflow.python.tpu.tpu_strategy_util import initialize_tpu_system, shutdown_tpu_system
from tensorflow.python.tpu.tpu_system_metadata import TPUSystemMetadata

"""Public API for tf.tpu.experimental namespace.
"""
