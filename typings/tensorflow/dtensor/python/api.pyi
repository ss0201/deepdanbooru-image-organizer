"""
This type stub file was generated by pyright.
"""

import contextlib
from typing import Any, Callable, List, Optional, Sequence
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.framework import device as tf_device, ops
from tensorflow.python.util.tf_export import tf_export

"""Core DTensor Python API."""
_DT_CLIENT_ID = ...
_DT_NUM_CLIENTS = ...
_DT_JOB_NAME = ...
_DT_JOBS = ...
_DT_HEARTBEAT_ENABLED = ...
_dtensor_singleton = ...
_dtensor_singleton_lock = ...
@tf_export("experimental.dtensor.call_with_layout", v1=[])
def call_with_layout(fn: Callable[..., Any], layout: Optional[layout_lib.Layout], *args, **kwargs) -> Any:
  """Calls a function in the DTensor device scope if `layout` is not None.

  If `layout` is not None, `fn` consumes DTensor(s) as input and produces a
  DTensor as output; a DTensor is a tf.Tensor with layout-related attributes.

  If `layout` is None, `fn` consumes and produces regular tf.Tensors.

  Args:
    fn: A supported TF API function such as tf.zeros.
    layout: Optional, the layout of the output DTensor.
    *args:  Arguments given to `fn`.
    **kwargs: Keyword arguments given to `fn`.

  Returns:
    The return value of `fn` transformed to a DTensor if requested.
  """
  ...

@tf_export("experimental.dtensor.run_on", v1=[])
@contextlib.contextmanager
def run_on(mesh: layout_lib.Mesh): # -> Generator[None, None, None]:
  """Runs enclosed functions in the DTensor device scope.

  This function returns a scope. All the ops and tf.functions in this scope will
  run on the DTensor device using the mesh provided.
  This is useful for wrapping any tf.function that doesn't take a DTensor as
  input but would like to produce DTensor as result. The scope will also make
  sure all small constants be replicated as DTensor.

  Args:
    mesh: A Mesh instance to extract a default mesh from.

  Yields:
    A context in which all ops and tf.functions will run on the DTensor device.
  """
  ...

@tf_export("experimental.dtensor.device_name", v1=[])
def device_name() -> str:
  """Returns the singleton DTensor device's name.

  This function can be used in the following way:

  ```python
  import tensorflow as tf

  with tf.device(dtensor.device_name()):
    # ...
  ```
  """
  ...

@tf_export("experimental.dtensor.copy_to_mesh", v1=[])
def copy_to_mesh(tensor: Any, layout: layout_lib.Layout, source_layout: Optional[layout_lib.Layout] = ...) -> ops.Tensor:
  """Copies a tf.Tensor onto the DTensor device with the given layout.

  Copies a regular tf.Tensor onto the DTensor device. Use the mesh attached to
  `layout` as target mesh. This method currently only supports replicated
  layouts. To get a DTensor with a sharded layout, use the `pack` method.

  Args:
    tensor: A regular tf.Tensor to be copied as a DTensor.
    layout: Target layout (and mesh) for the result DTensor.
    source_layout: Source layout of the tensor before copy, used for backward
      passes.

  Returns:
    A DTensor on the DTensor device with the given layout.
  """
  ...

@tf_export("experimental.dtensor.pack", v1=[])
def pack(tensors: Sequence[Any], layout: layout_lib.Layout) -> Any:
  """Packs `tf.Tensor` components into a DTensor.

  Packing and unpacking are inverse operations:

  ```
  * unpack(pack(tensors)) == tensors
  * pack(unpack(dtensor)) == dtensor
  ```

  1. For any DTensor on the mesh, `unpack` returns the raw components placed on
     each underlying device.
  2. Packing these raw components in the same order using `pack` returns a
     DTensor which should be identical to the original DTensor--both the content
     value and the layout.

  **Shape, Rank, and Scalars**: The rank of the DTensor is the same as the
  rank of its raw components, i.e., rank is preserved.  This leads to a
  consistent interpretation for packing scalar values into a DTensor. The only
  valid layout for a scalar value is fully replicated, and the individual
  components must be identical scalars.

  Each input `tensors[i]` will be copied to `layout.mesh.local_device[i]`
  if not already on the local device. Non-local components should not be passed
  to `pack`; use `copy_to_mesh` and `relayout` to place tensors on all global
  devices on a mesh.

  It is the caller's responsibility to ensure that the underlying values
  for `pack` adhere to the specified layout, and that only as many values are
  specified as there are local devices. Pack does not move data between clients.
  See examples below for more detail about layouts.

  For example, assume we have a mesh `[X(2), Y(3)]`, which has in total 6
  underlying devices. Futuremore, assume that the device location mapping is
  the following:

  ```
  device_ID  |  location X, Y
          0     0, 0
          1     0, 1
          2     0, 2
          3     1, 0
          4     1, 1
          5     1, 2
  ```

  1. For 1-D vector DTensor with shape `[128]` with layout `[mesh.X]` and value
     as `range(128)`, the raw components will have shape `[64]` each, and the
     raw components will be:

     ```
     device_ID  |  raw component
             0     range(0, 64)
             1     range(0, 64)
             2     range(0, 64)
             3     range(64, 128)
             4     range(64, 128)
             5     range(64, 128)
     ```

     This also means for a 1-D DTensor with shape `[2]` and layout `[mesh.X]`,
     the raw components have shape `[1]` rather than the shape for scalar values
     `[]`.

  2. For 2-D vector DTensor with shape `[2, 3]` with layout `[mesh.X, mesh.Y]`
     and value as `range(6)`, this is basically a fully-sharded DTensor.

     From global view, the content looks like
     ```
     [
       [0.0, 1.0, 2.0],
       [3.0, 4.0, 5.0],
     ]
     ```

     The raw components will have shape `[1, 1]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     [[0.0]]
             1     [[1.0]]
             2     [[2.0]]
             3     [[3.0]]
             4     [[4.0]]
             5     [[5.0]]
     ```

  3. For a scalar value `123.0` DTensor, it can only have one legitimate layout
     `[]` (no dimension, but fully replicated).

     The raw components will have shape `[]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     123.0
             1     123.0
             2     123.0
             3     123.0
             4     123.0
             5     123.0
     ```

     Again, caller of `pack` is expected to provide 6 identical value raw
     components with scalar shapes.

  4. For 3-D vector DTensor with shape `[2, 2, 3]` with layout
     `[X, unsharded, unsharded]` and value as `range(12)`,

     From global view, the content looks like:
     ```
     [
       [
         [0.0, 1.0, 2.0],
         [3.0, 4.0, 5.0],
       ],
       [
         [6.0, 7.0, 8.0],
         [9.0, 10., 11.],
       ],
     ]
     ```

     The raw components will have shape `[1, 2, 3]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     range(6).reshape([1, 2, 3])
             1     range(6).reshape([1, 2, 3])
             2     range(6).reshape([1, 2, 3])
             3     range(6, 12).reshape([1, 2, 3])
             4     range(6, 12).reshape([1, 2, 3])
             5     range(6, 12).reshape([1, 2, 3])
     ```

  Args:
    tensors: The list of local tensor components to pack into a DTensor.
    layout: The layout of the DTensor to be created.

  Returns:
    A DTensor created from the individual component tensors.

  Raises:
    RuntimeError: When `pack` is not called eagerly.
  """
  ...

@tf_export("experimental.dtensor.unpack", v1=[])
def unpack(tensor: Any) -> Sequence[Any]:
  """Unpacks a DTensor into `tf.Tensor` components.

  Packing and unpacking are inverse operations:

  ```
  * unpack(pack(tensors)) == tensors
  * pack(unpack(dtensor)) == dtensor
  ```

  1. For any DTensor on the mesh, `unpack` returns the raw components placed on
     each underlying device.
  2. Packing these raw components in the same order using `pack` returns a
     DTensor which should be identical to the original DTensor--both the content
     value and the layout.

  See the documentation for `pack` for more information about how packing and
  unpacking works.

  Args:
    tensor: The DTensor to unpack.

  Returns:
    The individual component tensors of the DTensor. This will include only the
    client-local components, i.e. the components placed on the local devices.

  Raises:
    RuntimeError: When `unpack` is not called eagerly.
  """
  ...

@tf_export("experimental.dtensor.fetch_layout", v1=[])
def fetch_layout(tensor: ops.Tensor) -> layout_lib.Layout:
  """Fetches the layout of a DTensor.

  Args:
    tensor: The DTensor whose layout is to be fetched.

  Returns:
    The `Layout` of this DTensor.

  Raises:
    RuntimeError: When not called eagerly.
  """
  ...

@tf_export("experimental.dtensor.check_layout", v1=[])
def check_layout(tensor: ops.Tensor, layout: layout_lib.Layout) -> None:
  """Asserts that the layout of the DTensor is `layout`.

  Args:
    tensor: A DTensor whose layout is to be checked.
    layout: The `Layout` to compare against.

  Raises:
    ValueError: If the layout of `tensor` does not match the supplied `layout`.
  """
  ...

@tf_export("experimental.dtensor.relayout", v1=[])
def relayout(tensor: ops.Tensor, layout: layout_lib.Layout) -> ops.Tensor:
  """Changes the layout of `tensor`.

  Changes the layout of `tensor` to `layout`. This is used to fine-tune the
  behavior of ops following/connected to `tensor`, such as choosing one SPMD
  expansion pattern over another. This works by forward propagating `layout`
  to connected TensorFlow computation graphs during layout propagation.

  Currently, only converting layouts from replicated to sharded or sharded to
  replicated per mesh dimension is supported. That is, "x, y" -> "unsharded, y"
  is supported, while "x, y" -> "z, y" is not supported.

  We also support a special "match" sharding spec, which instructs the relayout
  to act as an identity operation with respect to any sharding on these
  mesh dimensions.

  Relayout is internally lowered to a set of Split and/or AllToAll ops. When
  tensor layouts are converted from replicated to sharded, the cost is
  comparatively low because we only insert Split ops and no cross-device
  communication is needed. However, when tensor layouts are converted from
  sharded to replicated, cross-device communication may occur, causing potential
  performance impact.

  Args:
    tensor: A DTensor to specify a new layout for.
    layout: A Layout object specifying a new sharding spec.

  Returns:
    A DTensor output from the Relayout op.
  """
  ...

@tf_export("experimental.dtensor.client_id", v1=[])
def client_id() -> int:
  """Returns this client's ID."""
  ...

@tf_export("experimental.dtensor.num_clients", v1=[])
def num_clients() -> int:
  """Returns the number of clients in this DTensor cluster."""
  ...

@tf_export("experimental.dtensor.local_devices", v1=[])
def local_devices(device_type: str, for_client_id: Optional[int] = ...) -> List[tf_device.DeviceSpec]:
  """Returns a list of device specs of device_type attached to this client."""
  ...

@tf_export("experimental.dtensor.num_local_devices", v1=[])
def num_local_devices(device_type: str) -> int:
  """Returns the number of devices of device_type attached to this client."""
  ...

@tf_export("experimental.dtensor.num_global_devices", v1=[])
def num_global_devices(device_type: str) -> int:
  """Returns the number of devices of device_type in this DTensor cluster."""
  ...

@tf_export("experimental.dtensor.job_name", v1=[])
def job_name() -> str:
  """Returns the job name used by all clients in this DTensor cluster."""
  ...

@tf_export("experimental.dtensor.full_job_name", v1=[])
def full_job_name(task_id: Optional[int] = ...) -> str:
  """Returns the fully qualified TF job name for this or another task."""
  ...

@tf_export("experimental.dtensor.jobs", v1=[])
def jobs() -> List[str]:
  """Returns a list of job names of all clients in this DTensor cluster."""
  ...

@tf_export("experimental.dtensor.heartbeat_enabled", v1=[])
def heartbeat_enabled() -> bool:
  """Returns true if DTensor heartbeat service is enabled."""
  ...

