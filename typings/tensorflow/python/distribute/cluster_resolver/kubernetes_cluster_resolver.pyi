"""
This type stub file was generated by pyright.
"""

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.util.tf_export import tf_export

"""Implementation of Cluster Resolvers for Kubernetes."""
@tf_export('distribute.cluster_resolver.KubernetesClusterResolver')
class KubernetesClusterResolver(ClusterResolver):
  """ClusterResolver for Kubernetes.

  This is an implementation of cluster resolvers for Kubernetes. When given the
  the Kubernetes namespace and label selector for pods, we will retrieve the
  pod IP addresses of all running pods matching the selector, and return a
  ClusterSpec based on that information.

  Note: it cannot retrieve `task_type`, `task_id` or `rpc_layer`. To use it
  with some distribution strategies like
  `tf.distribute.experimental.MultiWorkerMirroredStrategy`, you will need to
  specify `task_type` and `task_id` by setting these attributes.

  Usage example with tf.distribute.Strategy:

    ```Python
    # On worker 0
    cluster_resolver = KubernetesClusterResolver(
        {"worker": ["job-name=worker-cluster-a", "job-name=worker-cluster-b"]})
    cluster_resolver.task_type = "worker"
    cluster_resolver.task_id = 0
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)

    # On worker 1
    cluster_resolver = KubernetesClusterResolver(
        {"worker": ["job-name=worker-cluster-a", "job-name=worker-cluster-b"]})
    cluster_resolver.task_type = "worker"
    cluster_resolver.task_id = 1
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
    ```
  """
  def __init__(self, job_to_label_mapping=..., tf_server_port=..., rpc_layer=..., override_client=...) -> None:
    """Initializes a new KubernetesClusterResolver.

    This initializes a new Kubernetes ClusterResolver. The ClusterResolver
    will attempt to talk to the Kubernetes master to retrieve all the instances
    of pods matching a label selector.

    Args:
      job_to_label_mapping: A mapping of TensorFlow jobs to label selectors.
        This allows users to specify many TensorFlow jobs in one Cluster
        Resolver, and each job can have pods belong with different label
        selectors. For example, a sample mapping might be
        ```
        {'worker': ['job-name=worker-cluster-a', 'job-name=worker-cluster-b'],
         'ps': ['job-name=ps-1', 'job-name=ps-2']}
        ```
      tf_server_port: The port the TensorFlow server is listening on.
      rpc_layer: (Optional) The RPC layer TensorFlow should use to communicate
        between tasks in Kubernetes. Defaults to 'grpc'.
      override_client: The Kubernetes client (usually automatically retrieved
        using `from kubernetes import client as k8sclient`). If you pass this
        in, you are responsible for setting Kubernetes credentials manually.

    Raises:
      ImportError: If the Kubernetes Python client is not installed and no
        `override_client` is passed in.
      RuntimeError: If autoresolve_task is not a boolean or a callable.
    """
    ...
  
  def master(self, task_type=..., task_id=..., rpc_layer=...): # -> str:
    """Returns the master address to use when creating a session.

    You must have set the task_type and task_id object properties before
    calling this function, or pass in the `task_type` and `task_id`
    parameters when using this function. If you do both, the function parameters
    will override the object properties.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.
    """
    ...
  
  def cluster_spec(self): # -> ClusterSpec:
    """Returns a ClusterSpec object based on the latest info from Kubernetes.

    We retrieve the information from the Kubernetes master every time this
    method is called.

    Returns:
      A ClusterSpec containing host information returned from Kubernetes.

    Raises:
      RuntimeError: If any of the pods returned by the master is not in the
        `Running` phase.
    """
    ...
  


