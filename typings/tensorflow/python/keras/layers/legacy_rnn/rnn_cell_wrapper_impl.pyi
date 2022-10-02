"""
This type stub file was generated by pyright.
"""

"""Module contains the implementation of RNN cell wrappers."""
class DropoutWrapperBase:
  """Operator adding dropout to inputs and outputs of the given cell."""
  def __init__(self, cell, input_keep_prob=..., output_keep_prob=..., state_keep_prob=..., variational_recurrent=..., input_size=..., dtype=..., seed=..., dropout_state_filter_visitor=..., **kwargs) -> None:
    """Create a cell with added input, state, and/or output dropout.

    If `variational_recurrent` is set to `True` (**NOT** the default behavior),
    then the same dropout mask is applied at every step, as described in:
    [A Theoretically Grounded Application of Dropout in Recurrent
    Neural Networks. Y. Gal, Z. Ghahramani](https://arxiv.org/abs/1512.05287).

    Otherwise a different dropout mask is applied at every time step.

    Note, by default (unless a custom `dropout_state_filter` is provided),
    the memory state (`c` component of any `LSTMStateTuple`) passing through
    a `DropoutWrapper` is never modified.  This behavior is described in the
    above article.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is constant and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
      state_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
        State dropout is performed on the outgoing states of the cell. **Note**
        the state components to which dropout is applied when `state_keep_prob`
        is in `(0, 1)` are also determined by the argument
        `dropout_state_filter_visitor` (e.g. by default dropout is never applied
        to the `c` component of an `LSTMStateTuple`).
      variational_recurrent: Python bool.  If `True`, then the same dropout
        pattern is applied across all time steps per run call. If this parameter
        is set, `input_size` **must** be provided.
      input_size: (optional) (possibly nested tuple of) `TensorShape` objects
        containing the depth(s) of the input tensors expected to be passed in to
        the `DropoutWrapper`.  Required and used **iff** `variational_recurrent
        = True` and `input_keep_prob < 1`.
      dtype: (optional) The `dtype` of the input, state, and output tensors.
        Required and used **iff** `variational_recurrent = True`.
      seed: (optional) integer, the randomness seed.
      dropout_state_filter_visitor: (optional), default: (see below).  Function
        that takes any hierarchical level of the state and returns a scalar or
        depth=1 structure of Python booleans describing which terms in the state
        should be dropped out.  In addition, if the function returns `True`,
        dropout is applied across this sublevel.  If the function returns
        `False`, dropout is not applied across this entire sublevel.
        Default behavior: perform dropout on all terms except the memory (`c`)
          state of `LSTMCellState` objects, and don't try to apply dropout to
        `TensorArray` objects: ```
        def dropout_state_filter_visitor(s):
          if isinstance(s, LSTMCellState): # Never perform dropout on the c
            state. return LSTMCellState(c=False, h=True)
          elif isinstance(s, TensorArray): return False return True ```
      **kwargs: dict of keyword arguments for base layer.

    Raises:
      TypeError: if `cell` is not an `RNNCell`, or `keep_state_fn` is provided
        but not `callable`.
      ValueError: if any of the keep_probs are not between 0 and 1.
    """
    ...
  
  @property
  def wrapped_cell(self):
    ...
  
  @property
  def state_size(self):
    ...
  
  @property
  def output_size(self):
    ...
  
  def build(self, inputs_shape): # -> None:
    ...
  
  def zero_state(self, batch_size, dtype):
    ...
  
  def get_config(self): # -> dict[Unknown, Unknown]:
    """Returns the config of the dropout wrapper."""
    ...
  
  @classmethod
  def from_config(cls, config, custom_objects=...):
    ...
  


class ResidualWrapperBase:
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""
  def __init__(self, cell, residual_fn=..., **kwargs) -> None:
    """Constructs a `ResidualWrapper` for `cell`.

    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
          and outputs.
      **kwargs: dict of keyword arguments for base layer.
    """
    ...
  
  @property
  def state_size(self):
    ...
  
  @property
  def output_size(self):
    ...
  
  def zero_state(self, batch_size, dtype):
    ...
  
  def get_config(self): # -> dict[Unknown, Unknown]:
    """Returns the config of the residual wrapper."""
    ...
  
  @classmethod
  def from_config(cls, config, custom_objects=...):
    ...
  


class DeviceWrapperBase:
  """Operator that ensures an RNNCell runs on a particular device."""
  def __init__(self, cell, device, **kwargs) -> None:
    """Construct a `DeviceWrapper` for `cell` with device `device`.

    Ensures the wrapped `cell` is called with `tf.device(device)`.

    Args:
      cell: An instance of `RNNCell`.
      device: A device string or function, for passing to `tf.device`.
      **kwargs: dict of keyword arguments for base layer.
    """
    ...
  
  @property
  def state_size(self):
    ...
  
  @property
  def output_size(self):
    ...
  
  def zero_state(self, batch_size, dtype):
    ...
  
  def get_config(self): # -> dict[Unknown, Unknown]:
    ...
  


