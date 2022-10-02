"""
This type stub file was generated by pyright.
"""

from keras.engine import training_utils_v1

"""Part of the Keras training engine related to distributed training."""
def experimental_tpu_fit_loop(model, dataset, epochs=..., verbose=..., callbacks=..., initial_epoch=..., steps_per_epoch=..., val_dataset=..., validation_steps=..., validation_freq=...):
    """Fit loop for training with TPU tf.distribute.Strategy.

    Args:
        model: Keras Model instance.
        dataset: Dataset that returns inputs and targets
        epochs: Number of times to iterate over the data
        verbose: Integer, Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        val_dataset: Dataset for validation data.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.abc.Container` instance (e.g. list, tuple, etc.). If
            an integer, specifies how many training epochs to run before a new
            validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.

    Returns:
        Returns `None`.

    Raises:
        ValueError: in case of invalid arguments.
    """
    ...

def experimental_tpu_test_loop(model, dataset, verbose=..., steps=..., callbacks=...): # -> float | list[float]:
    """Test loop for evaluating with TPU tf.distribute.Strategy.

    Args:
        model: Keras Model instance.
        dataset: Dataset for input data.
        verbose: Integer, Verbosity mode 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring predictions finished.
            Ignored with the default value of `None`.
        callbacks: List of callbacks to be called during training

    Returns:
        Scalar loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the outputs.
    """
    ...

def experimental_tpu_predict_loop(model, dataset, verbose=..., steps=..., callbacks=...): # -> Any | list[Unknown]:
    """Predict loop for predicting with TPU tf.distribute.Strategy.

    Args:
        model: Keras Model instance.
        dataset: Dataset for input data.
        verbose: Integer, Verbosity mode 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring `_predict_loop` finished.
            Ignored with the default value of `None`.
        callbacks: List of callbacks to be called during training

    Returns:
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    """
    ...

class DistributionSingleWorkerTrainingLoop(training_utils_v1.TrainingLoop):
    """Training loop for distribution strategy with single worker."""
    def fit(self, model, x=..., y=..., batch_size=..., epochs=..., verbose=..., callbacks=..., validation_split=..., validation_data=..., shuffle=..., class_weight=..., sample_weight=..., initial_epoch=..., steps_per_epoch=..., validation_steps=..., validation_freq=..., **kwargs):
        """Fit loop for Distribution Strategies."""
        ...
    
    def evaluate(self, model, x=..., y=..., batch_size=..., verbose=..., sample_weight=..., steps=..., callbacks=..., **kwargs): # -> float | list[float]:
        """Evaluate loop for Distribution Strategies."""
        ...
    
    def predict(self, model, x, batch_size=..., verbose=..., steps=..., callbacks=..., **kwargs): # -> Any | list[Unknown]:
        """Predict loop for Distribution Strategies."""
        ...
    


class DistributionMultiWorkerTrainingLoop(training_utils_v1.TrainingLoop):
    """Training loop for distribution strategy with multiple worker."""
    def __init__(self, single_worker_loop) -> None:
        ...
    
    def fit(self, *args, **kwargs): # -> None:
        ...
    
    def evaluate(self, *args, **kwargs): # -> None:
        ...
    
    def predict(self, *args, **kwargs):
        ...
    


