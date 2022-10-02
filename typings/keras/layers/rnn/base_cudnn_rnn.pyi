"""
This type stub file was generated by pyright.
"""

from keras.layers.rnn.base_rnn import RNN

"""Base class for recurrent layers backed by cuDNN."""
class _CuDNNRNN(RNN):
    """Private base class for CuDNNGRU and CuDNNLSTM layers.

    Args:
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
          in addition to the output.
      go_backwards: Boolean (default False).
          If True, process the input sequence backwards and return the
          reversed sequence.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      time_major: Boolean (default False). If true, the inputs and outputs will
          be in shape `(timesteps, batch, ...)`, whereas in the False case, it
          will be `(batch, timesteps, ...)`.
    """
    def __init__(self, return_sequences=..., return_state=..., go_backwards=..., stateful=..., time_major=..., **kwargs) -> None:
        ...
    
    def call(self, inputs, mask=..., training=..., initial_state=...):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    
    @classmethod
    def from_config(cls, config): # -> Self@_CuDNNRNN:
        ...
    
    @property
    def trainable_weights(self): # -> list[Unknown]:
        ...
    
    @property
    def non_trainable_weights(self): # -> list[Unknown]:
        ...
    
    @property
    def losses(self): # -> list[Unknown]:
        ...
    
    def get_losses_for(self, inputs=...):
        ...
    


