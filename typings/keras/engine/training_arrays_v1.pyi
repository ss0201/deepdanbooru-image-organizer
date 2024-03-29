"""
This type stub file was generated by pyright.
"""

from keras.engine import training_utils_v1

"""Part of the Keras training engine related to plain array data."""
def model_iteration(model, inputs, targets=..., sample_weights=..., batch_size=..., epochs=..., verbose=..., callbacks=..., val_inputs=..., val_targets=..., val_sample_weights=..., shuffle=..., initial_epoch=..., steps_per_epoch=..., validation_steps=..., validation_freq=..., mode=..., validation_in_fit=..., prepared_feed_values_from_dataset=..., steps_name=..., **kwargs):
    """Loop function for arrays of data with modes TRAIN/TEST/PREDICT.

    Args:
        model: Keras Model instance.
        inputs: Either a list or dictionary of arrays, or a dataset instance.
        targets: List/dictionary of input arrays.
        sample_weights: Optional list of sample weight arrays.
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: 0, 1, or 2. Verbosity mode.
          0 = silent, 1 = progress bar, 2 = one line per epoch.
          Note that the progress bar is not particularly useful when
          logged to a file, so verbose=2 is recommended when not running
          interactively (eg, in a production environment).
        callbacks: List of callbacks to be called during training
        val_inputs: Either a list or dictionary of arrays, or a dataset
          instance.
        val_targets: List/dictionary of target arrays.
        val_sample_weights: Optional list of sample weight arrays.
        shuffle: Whether to shuffle the data at the beginning of each epoch
          concatenation of list the display names of the outputs of `f` and the
          list of display names of the outputs of `f_val`.
        initial_epoch: Epoch at which to start training (useful for resuming a
          previous training run)
        steps_per_epoch: Total number of steps (batches of samples) before
          declaring one epoch finished and starting the next epoch. Ignored with
          the default value of `None`.
        validation_steps: Number of steps to run validation for (only if doing
          validation from data tensors). Ignored with the default value of
          `None`.
        validation_freq: Only relevant if validation data is provided. Integer
          or `collections.abc.Container` instance (e.g. list, tuple, etc.). If
          an integer, specifies how many training epochs to run before a new
          validation run is performed, e.g. `validation_freq=2` runs validation
          every 2 epochs. If a Container, specifies the epochs on which to run
          validation, e.g. `validation_freq=[1, 2, 10]` runs validation at the
          end of the 1st, 2nd, and 10th epochs.
        mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.
        validation_in_fit: if true, then this method is invoked from within
          training iteration (for validation). In the case where `val_inputs` is
          a dataset, this flag indicates that its iterator and feed values are
          already created so should properly reuse resources.
        prepared_feed_values_from_dataset: if True, `inputs` is a list of feed
          tensors returned from `_prepare_feed_values` call on the validation
          dataset, so do not call it again on `inputs`. Should only be used for
          inline validation (i.e., only if `validation_in_fit` is also True).
        steps_name: The string name of the steps argument, either `steps`,
          `validation_steps`, or `steps_per_epoch`. Only used for error message
          formatting.
        **kwargs: Additional arguments for backwards compatibility.

    Returns:
        - In TRAIN mode: `History` object.
        - In TEST mode: Evaluation metrics.
        - In PREDICT mode: Outputs of the Model called on inputs.

    Raises:
        ValueError: in case of invalid arguments.
    """
    ...

fit_loop = ...
test_loop = ...
predict_loop = ...
class ArrayLikeTrainingLoop(training_utils_v1.TrainingLoop):
    """TrainingLoop that handle inputs like array.

    This is the default handler for most of the input data types, includes
    symbolic tensors or Numpy array-like, Datasets and iterators in graph mode
    (since they generate symbolic tensors). This Function is used to handle
    model with `run_eagerly` = False.
    """
    def fit(self, model, x=..., y=..., batch_size=..., epochs=..., verbose=..., callbacks=..., validation_split=..., validation_data=..., shuffle=..., class_weight=..., sample_weight=..., initial_epoch=..., steps_per_epoch=..., validation_steps=..., validation_freq=..., **kwargs):
        ...
    
    def evaluate(self, model, x=..., y=..., batch_size=..., verbose=..., sample_weight=..., steps=..., callbacks=..., **kwargs):
        ...
    
    def predict(self, model, x, batch_size=..., verbose=..., steps=..., callbacks=..., **kwargs):
        ...
    


