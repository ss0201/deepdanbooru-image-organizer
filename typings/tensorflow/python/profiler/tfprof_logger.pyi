"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import tf_export

"""Logging tensorflow::tfprof::OpLogProto.

OpLogProto is used to add extra model information for offline analysis.
"""
TRAINABLE_VARIABLES = ...
REGISTERED_FLOP_STATS = ...
def merge_default_with_oplog(graph, op_log=..., run_meta=..., add_trace=..., add_trainable_var=...): # -> OpLogProto:
  """Merge the tfprof default extra info with caller's op_log.

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use
        default graph.
    op_log: OpLogProto proto.
    run_meta: RunMetadata proto used to complete shape information.
    add_trace: Whether to add op trace information.
    add_trainable_var: Whether to assign tf.compat.v1.trainable_variables() op
      type '_trainable_variables'.
  Returns:
    tmp_op_log: Merged OpLogProto proto.
  """
  ...

@tf_export(v1=['profiler.write_op_log'])
def write_op_log(graph, log_dir, op_log=..., run_meta=..., add_trace=...): # -> None:
  """Log provided 'op_log', and add additional model information below.

    The API also assigns ops in tf.compat.v1.trainable_variables() an op type
    called '_trainable_variables'.
    The API also logs 'flops' statistics for ops with op.RegisterStatistics()
    defined. flops calculation depends on Tensor shapes defined in 'graph',
    which might not be complete. 'run_meta', if provided, completes the shape
    information with best effort.

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use
        default graph.
    log_dir: directory to write the log file.
    op_log: (Optional) OpLogProto proto to be written. If not provided, an new
        one is created.
    run_meta: (Optional) RunMetadata proto that helps flops computation using
        run time shape information.
    add_trace: Whether to add python code trace information.
        Used to support "code" view.
  """
  ...

