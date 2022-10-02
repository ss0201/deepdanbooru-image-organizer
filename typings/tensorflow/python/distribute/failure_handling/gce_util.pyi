"""
This type stub file was generated by pyright.
"""

import enum

"""Util of GCE specifics to ingegrate with WorkerPreemptionHandler."""
GCP_METADATA_HEADER = ...
_GCE_METADATA_URL_ENV_VARIABLE = ...
_RESTARTABLE_EXIT_CODE = ...
GRACE_PERIOD_GCE = ...
def gce_exit_fn():
  ...

def request_compute_metadata(path): # -> str:
  """Returns GCE VM compute metadata."""
  ...

def termination_watcher_function_gce(): # -> bool:
  ...

def on_gcp(): # -> bool:
  """Detect whether the current running environment is on GCP."""
  ...

@enum.unique
class PlatformDevice(enum.Enum):
  INTERNAL = ...
  GCE_GPU = ...
  GCE_TPU = ...
  GCE_CPU = ...
  UNSUPPORTED = ...


def detect_platform(): # -> Literal[PlatformDevice.GCE_GPU, PlatformDevice.GCE_TPU, PlatformDevice.GCE_CPU, PlatformDevice.INTERNAL]:
  """Returns the platform and device information."""
  ...

