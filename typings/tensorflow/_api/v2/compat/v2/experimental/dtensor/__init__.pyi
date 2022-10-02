"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow.dtensor.python.api import call_with_layout, check_layout, client_id, copy_to_mesh, device_name, fetch_layout, full_job_name, heartbeat_enabled, job_name, jobs, local_devices, num_clients, num_global_devices, num_local_devices, pack, relayout, run_on, unpack
from tensorflow.dtensor.python.d_checkpoint import DTensorCheckpoint
from tensorflow.dtensor.python.d_variable import DVariable
from tensorflow.dtensor.python.layout import Layout, MATCH, Mesh, UNSHARDED
from tensorflow.dtensor.python.mesh_util import barrier, create_distributed_mesh, create_mesh, dtensor_initialize_multi_client as initialize_multi_client
from tensorflow.dtensor.python.save_restore import enable_save_as_bf16, name_based_restore, name_based_save, sharded_save
from tensorflow.dtensor.python.tpu_util import dtensor_initialize_tpu_system as initialize_tpu_system, dtensor_shutdown_tpu_system as shutdown_tpu_system

"""Public API for tf.experimental.dtensor namespace.
"""