"""
This type stub file was generated by pyright.
"""

import sys as _sys
from . import service
from tensorflow.python.data.experimental.ops.batching import dense_to_ragged_batch, dense_to_sparse_batch, map_and_batch, unbatch
from tensorflow.python.data.experimental.ops.cardinality import INFINITE as INFINITE_CARDINALITY, UNKNOWN as UNKNOWN_CARDINALITY, assert_cardinality, cardinality
from tensorflow.python.data.experimental.ops.counter import CounterV2 as Counter
from tensorflow.python.data.experimental.ops.distribute import SHARD_HINT
from tensorflow.python.data.experimental.ops.enumerate_ops import enumerate_dataset
from tensorflow.python.data.experimental.ops.error_ops import ignore_errors
from tensorflow.python.data.experimental.ops.from_list import from_list
from tensorflow.python.data.experimental.ops.get_single_element import get_single_element
from tensorflow.python.data.experimental.ops.grouping import Reducer, bucket_by_sequence_length, group_by_reducer, group_by_window
from tensorflow.python.data.experimental.ops.interleave_ops import choose_from_datasets_v2 as choose_from_datasets, parallel_interleave, sample_from_datasets_v2 as sample_from_datasets
from tensorflow.python.data.experimental.ops.io import load, save
from tensorflow.python.data.experimental.ops.iterator_ops import CheckpointInputPipelineHook, make_saveable_from_iterator
from tensorflow.python.data.experimental.ops.lookup_ops import DatasetInitializer, index_table_from_dataset, table_from_dataset
from tensorflow.python.data.experimental.ops.parsing_ops import parse_example_dataset
from tensorflow.python.data.experimental.ops.prefetching_ops import copy_to_device, prefetch_to_device
from tensorflow.python.data.experimental.ops.random_ops import RandomDatasetV2 as RandomDataset
from tensorflow.python.data.experimental.ops.readers import CsvDatasetV2 as CsvDataset, SqlDatasetV2 as SqlDataset, make_batched_features_dataset_v2 as make_batched_features_dataset, make_csv_dataset_v2 as make_csv_dataset
from tensorflow.python.data.experimental.ops.resampling import rejection_resample
from tensorflow.python.data.experimental.ops.scan_ops import scan
from tensorflow.python.data.experimental.ops.shuffle_ops import shuffle_and_repeat
from tensorflow.python.data.experimental.ops.snapshot import snapshot
from tensorflow.python.data.experimental.ops.take_while_ops import take_while
from tensorflow.python.data.experimental.ops.unique import unique
from tensorflow.python.data.experimental.ops.writers import TFRecordWriter
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE, enable_debug_mode, from_variant, get_structure, to_variant
from tensorflow.python.data.ops.iterator_ops import get_next_as_optional
from tensorflow.python.data.ops.optional_ops import Optional
from tensorflow.python.data.ops.options import AutoShardPolicy, AutotuneAlgorithm, AutotuneOptions, DistributeOptions, ExternalStatePolicy, OptimizationOptions, ThreadingOptions

"""Experimental API for building input pipelines.

This module contains experimental `Dataset` sources and transformations that can
be used in conjunction with the `tf.data.Dataset` API. Note that the
`tf.data.experimental` API is not subject to the same backwards compatibility
guarantees as `tf.data`, but we will provide deprecation advice in advance of
removing existing functionality.

See [Importing Data](https://tensorflow.org/guide/datasets) for an overview.

@@AutoShardPolicy
@@AutotuneAlgorithm
@@AutotuneOptions
@@CheckpointInputPipelineHook
@@Counter
@@CsvDataset
@@DatasetInitializer
@@DatasetStructure
@@DistributeOptions
@@ExternalStatePolicy
@@OptimizationOptions
@@Optional
@@OptionalStructure
@@RaggedTensorStructure
@@RandomDataset
@@Reducer
@@SparseTensorStructure
@@SqlDataset
@@Structure
@@TFRecordWriter
@@TensorArrayStructure
@@TensorStructure
@@ThreadingOptions

@@assert_cardinality
@@bucket_by_sequence_length
@@cardinality
@@choose_from_datasets
@@copy_to_device
@@dense_to_ragged_batch
@@dense_to_sparse_batch
@@distribute
@@enable_debug_mode
@@enumerate_dataset
@@from_list
@@from_variant
@@get_next_as_optional
@@get_single_element
@@get_structure
@@group_by_reducer
@@group_by_window
@@ignore_errors
@@index_table_from_dataset
@@load
@@make_batched_features_dataset
@@make_csv_dataset
@@make_saveable_from_iterator
@@map_and_batch
@@map_and_batch_with_legacy_function
@@parallel_interleave
@@parse_example_dataset
@@prefetch_to_device
@@rejection_resample
@@sample_from_datasets
@@save
@@scan
@@shuffle_and_repeat
@@snapshot
@@table_from_dataset
@@take_while
@@to_variant
@@unbatch
@@unique

@@AUTOTUNE
@@INFINITE_CARDINALITY
@@SHARD_HINT
@@UNKNOWN_CARDINALITY

"""
