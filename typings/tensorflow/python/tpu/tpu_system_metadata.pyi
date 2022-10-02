"""
This type stub file was generated by pyright.
"""

import collections
from tensorflow.python.util.tf_export import tf_export

"""TPU system metadata and associated tooling."""
_PINGING_MASTER_TIMEOUT_IN_MS = ...
_RETRY_TIMES = ...
_INITIAL_TPU_SYSTEM_TIMEOUT_IN_MS = ...
_DEFAULT_JOB_NAME = ...
_DEFAULT_COORDINATOR_JOB_NAME = ...
_LOCAL_MASTERS = ...
@tf_export('tpu.experimental.TPUSystemMetadata')
class TPUSystemMetadata(collections.namedtuple('TPUSystemMetadata', ['num_cores', 'num_hosts', 'num_of_cores_per_host', 'topology', 'devices'])):
  """Describes some metadata about the TPU system.

  Attributes:
    num_cores: interger. Total number of TPU cores in the TPU system.
    num_hosts: interger. Total number of hosts (TPU workers) in the TPU system.
    num_of_cores_per_host: interger. Number of TPU cores per host (TPU worker).
    topology: an instance of `tf.tpu.experimental.Topology`, which describes the
      physical topology of TPU system.
    devices: a tuple of strings, which describes all the TPU devices in the
      system.
  """
  def __new__(cls, num_cores, num_hosts, num_of_cores_per_host, topology, devices): # -> Self@TPUSystemMetadata:
    ...
  


def get_session_config_with_timeout(timeout_in_secs, cluster_def): # -> ConfigProto:
  """Returns a session given a timeout and a cluster configuration."""
  ...

def master_job(master, cluster_def): # -> Literal['tpu_worker'] | None:
  """Returns the canonical job name to use to place TPU computations on.

  Args:
    master: A `string` representing the TensorFlow master to use.
    cluster_def: A ClusterDef object describing the TPU cluster.

  Returns:
    A string containing the job name, or None if no job should be specified.

  Raises:
    ValueError: If the user needs to specify a tpu_job_name, because we are
      unable to infer the job name automatically, or if the user-specified job
      names are inappropriate.
  """
  ...
