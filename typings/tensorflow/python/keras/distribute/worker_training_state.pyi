"""
This type stub file was generated by pyright.
"""

"""Training state management."""
CKPT_SAVED_EPOCH = ...
CKPT_SAVED_EPOCH_UNUSED_VALUE = ...
class WorkerTrainingState:
  """Training state management class.

  This class provides apis for backing up and restoring the training state.
  This allows model and epoch information to be saved periodically and restore
  for fault-tolerance, also known as preemption-recovery purpose.
  """
  def __init__(self, model, checkpoint_dir) -> None:
    ...
  
  def back_up(self, epoch): # -> None:
    """Back up the current state of training into a checkpoint file.

    Args:
      epoch: The current epoch information to be saved.
    """
    ...
  
  def restore(self): # -> None:
    """Restore the training state from the backed up checkpoint file.

    Returns:
      True if the training state is successfully restored. False if the training
      state doesn't need to be restored, or error occurred so it can't.
    """
    ...
  
  def delete_backup(self): # -> None:
    """Delete the backup directories.

    Delete the backup directories which should not exist after `fit()`
    successfully finishes.
    """
    ...
  
  def maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
    """Maybe load initial epoch from ckpt considering possible worker recovery.

    When `_ckpt_saved_epoch` attribute exists and is not
    `CKPT_SAVED_EPOCH_UNUSED_VALUE`, this is under multi-worker training setting
    and indicates the worker is recovering from previous failure. In this case,
    infer `initial_epoch` from `self._ckpt_saved_epoch` to continue previous
    unfinished training from certain epoch.

    Args:
      initial_epoch: The original initial_epoch user passes in in `fit()`.
      mode: The mode for running `model.fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """
    ...
  


