"""
This type stub file was generated by pyright.
"""

from typing import Any, Iterable, Optional, Union
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.trackable import autotrackable

"""Base Class for TPU Embeddings Mid level APIs."""
class TPUEmbeddingBase(autotrackable.AutoTrackable):
  """The TPUEmbedding Base class.

  This class only contains the basic logic to check the feature config and table
  config for the tpu embedding mid level APIs.
  """
  def __init__(self, feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable], optimizer: Optional[tpu_embedding_v2_utils._Optimizer] = ...) -> None:
    """Creates the TPUEmbeddingBase object."""
    ...
  
  @property
  def embedding_tables(self):
    """Returns a dict of embedding tables, keyed by `TableConfig`."""
    ...
  
  def build(self): # -> None:
    """Create variables and slots variables for TPU embeddings."""
    ...
  
  def __call__(self, features: Any, weights: Optional[Any] = ...) -> Any:
    """Call the mid level api to do embedding lookup."""
    ...
  
  def embedding_lookup(self, features: Any, weights: Optional[Any] = ...) -> Any:
    """Lookup the embedding table using the input features."""
    ...
  

