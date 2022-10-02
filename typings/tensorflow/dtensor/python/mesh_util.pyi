"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Tuple
from tensorflow.dtensor.python import layout
from tensorflow.python.util.tf_export import tf_export

"""Utilities to help with mesh creation."""
@tf_export('experimental.dtensor.create_mesh', v1=[])
def create_mesh(mesh_dims: Optional[List[Tuple[str, int]]] = ..., mesh_name: str = ..., devices: Optional[List[str]] = ..., device_type: Optional[str] = ...) -> layout.Mesh:
  """Creates a single-client mesh.

  If both `mesh_dims` and `devices` are specified, they must match each otehr.
  As a special case, when all arguments are missing, this creates a 1D CPU mesh
  with an empty name, assigning all available devices to that dimension.

  Args:
    mesh_dims: A list of (dim_name, dim_size) tuples. Defaults to a single
      batch-parallel dimension called 'x' using all devices. As a special case,
      a single-element mesh_dims whose dim_size is -1 also uses all devices.
    mesh_name: Name of the created mesh. Defaults to ''.
    devices: String representations of devices to use. This is the device part
      of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available logical devices.
    device_type: If `devices` is missing, the type of devices to use. Defaults
      to 'CPU'.

  Returns:
    A single-client mesh created from specified or default arguments.
  """
  ...

@tf_export('experimental.dtensor.create_distributed_mesh', v1=[])
def create_distributed_mesh(mesh_dims: List[Tuple[str, int]], mesh_name: str = ..., num_global_devices: Optional[int] = ..., num_clients: Optional[int] = ..., client_id: Optional[int] = ..., device_type: str = ...) -> layout.Mesh:
  """Creates a single- or multi-client mesh.

  For CPU and GPU meshes, users can choose to use fewer local devices than what
  is available. If any argument is missing, it will be extracted from
  environment variables. The default values for these environment variables
  create a mesh using all devices (common for unit tests).

  For TPU meshes, users should not specify any of the nullable arguments. The
  DTensor runtime will set these arguments automatically, using all TPU cores
  available in the entire cluster.

  Args:
    mesh_dims: A list of (dim_name, dim_size) tuples.
    mesh_name: Name of the created mesh. Defaults to ''.
    num_global_devices: Number of devices in the DTensor cluster. Defaults to
      the corresponding environment variable.
    num_clients: Number of clients in the DTensor cluster. Defaults to the
      corresponding environment variable, DTENSOR_NUM_CLIENTS.
    client_id: This client's ID. Defaults to the corresponding environment
      variable, DTENSOR_CLIENT_ID.
    device_type: Type of device to build the mesh for. Defaults to 'CPU'.

  Returns:
    A mesh created from specified or default arguments.
  """
  ...

@tf_export('experimental.dtensor.initialize_multi_client', v1=[])
def dtensor_initialize_multi_client(enable_coordination_service: Optional[bool] = ...) -> None:
  """Initializes Multi Client DTensor.

  The following environment variables controls the behavior of this function.
  If the variables are unset, DTensor will be configured to run in single-client
  mode.

  - DTENSOR_CLIENT_ID: integer, between 0 to num_clients - 1, to identify the
      client id of the current process.
  - DTENSOR_NUM_CLIENTS: integer, the number of clients.
  - DTENSOR_JOB_NAME: string, a hostname like string for the name of the dtensor
      job. The job name is used by TensorFlow in the job name section of
      the DeviceSpec.
  - DTENSOR_JOBS: string, a comma separated list. Each item in the list is
      of format `{hostname}:{port}` and the items must be sorted in alphabet
      order. The implication is the RPC port numbers of the clients from
      the same host must be ordered by the client ID.
      Examples of valid DTENSOR_JOBS values:
      - 4 clients on localhost:
        `localhost:10000,localhost:10001,localhost:10002,localhost:10003`
      - 2 clients on host1, 2 clients on host2
        `host1:10000,host1:10001,host2:10000,host2:10003`

  Args:
    enable_coordination_service: If true, enable distributed coordination
      service to make sure that workers know the devices on each other, a
      prerequisite for data transfer through cross-worker rendezvous.
  """
  ...

@tf_export('experimental.dtensor.barrier', v1=[])
def barrier(mesh: layout.Mesh, barrier_name: Optional[str] = ...): # -> None:
  """Runs a barrier on the mesh.

  Upon returning from the barrier, all operations run before the barrier
  would have completed across all clients. Currently we allocate a fully
  sharded tensor with mesh shape and run an all_reduce on it.

  Example:

  A barrier can be used before application exit to ensure completion of pending
  ops.

  ```python

  x = [1, 2, 3]
  x = dtensor.relayout(x, dtensor.Layout.batch_sharded(mesh, 'batch', 1))
  dtensor.barrier(mesh)

  # At this point all devices on all clients in the mesh have completed
  # operations before the barrier. Therefore it is OK to tear down the clients.
  sys.exit()
  ```

  Args:
    mesh: The mesh to run the barrier on.
    barrier_name: The name of the barrier. mainly used for logging purpose.
  """
  ...
