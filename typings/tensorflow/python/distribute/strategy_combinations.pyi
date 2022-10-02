"""
This type stub file was generated by pyright.
"""

from tensorflow.python.distribute import collective_all_reduce_strategy

"""Strategy combinations for combinations.combine()."""
_TF_INTERNAL_API_PREFIX = ...
_did_connect_to_cluster = ...
_topology = ...
CollectiveAllReduceExtended = collective_all_reduce_strategy.CollectiveAllReduceExtended
MirroredStrategy = ...
CentralStorageStrategy = ...
OneDeviceStrategy = ...
CollectiveAllReduceStrategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy
DEFAULT_PARTITIONER = ...
_two_worker_pool = ...
_two_worker_pool_noshare = ...
_four_worker_pool = ...
default_strategy = ...
one_device_strategy = ...
one_device_strategy_gpu = ...
one_device_strategy_on_worker_1 = ...
one_device_strategy_gpu_on_worker_1 = ...
tpu_strategy = ...
tpu_strategy_packed_var = ...
tpu_strategy_spmd = ...
tpu_strategy_one_step = ...
tpu_strategy_one_core = ...
tpu_strategy_one_step_one_core = ...
cloud_tpu_strategy = ...
mirrored_strategy_with_one_cpu = ...
mirrored_strategy_with_one_gpu = ...
mirrored_strategy_with_gpu_and_cpu = ...
mirrored_strategy_with_two_cpus = ...
mirrored_strategy_with_two_gpus = ...
mirrored_strategy_with_two_gpus_no_merge_call = ...
mirrored_strategy_with_cpu_1_and_2 = ...
central_storage_strategy_with_two_gpus = ...
central_storage_strategy_with_gpu_and_cpu = ...
multi_worker_mirrored_2x1_cpu = ...
multi_worker_mirrored_2x1_gpu = ...
multi_worker_mirrored_2x1_gpu_noshare = ...
multi_worker_mirrored_2x2_gpu = ...
multi_worker_mirrored_2x2_gpu_no_merge_call = ...
multi_worker_mirrored_4x1_cpu = ...
def parameter_server_strategy_fn(name, num_workers, num_ps, required_gpus=..., variable_partitioner=...): # -> NamedDistribution:
  ...

parameter_server_strategy_3worker_2ps_cpu = ...
parameter_server_strategy_1worker_2ps_cpu = ...
parameter_server_strategy_3worker_2ps_1gpu = ...
parameter_server_strategy_1worker_2ps_1gpu = ...
graph_and_eager_modes = ...
def set_virtual_cpus_to_at_least(num_virtual_cpus): # -> None:
  ...

strategies_minus_tpu = ...
strategies_minus_default_and_tpu = ...
tpu_strategies = ...
all_strategies_minus_default = ...
all_strategies = ...
two_replica_strategies = ...
four_replica_strategies = ...
multidevice_strategies = ...
multiworker_strategies = ...
def strategy_minus_tpu_combinations(): # -> list[OrderedDict[Unknown, Unknown]]:
  ...

def tpu_strategy_combinations(): # -> list[OrderedDict[Unknown, Unknown]]:
  ...

def all_strategy_combinations(): # -> list[OrderedDict[Unknown, Unknown]]:
  ...

def all_strategy_minus_default_and_tpu_combinations(): # -> list[OrderedDict[Unknown, Unknown]]:
  ...

def all_strategy_combinations_minus_default(): # -> list[OrderedDict[Unknown, Unknown]]:
  ...

