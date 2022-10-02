"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import keras_export

"""Boston housing price regression dataset."""
@keras_export("keras.datasets.boston_housing.load_data")
def load_data(path=..., test_split=..., seed=...): # -> tuple[tuple[ndarray[Unknown, Unknown], ndarray[Unknown, Unknown]], tuple[ndarray[Unknown, Unknown], ndarray[Unknown, Unknown]]]:
    """Loads the Boston Housing dataset.

    This is a dataset taken from the StatLib library which is maintained at
    Carnegie Mellon University.

    **WARNING:** This dataset has an ethical problem: the authors of this
    dataset included a variable, "B", that may appear to assume that racial
    self-segregation influences house prices. As such, we strongly discourage
    the use of this dataset, unless in the context of illustrating ethical
    issues in data science and machine learning.

    Samples contain 13 attributes of houses at different locations around the
    Boston suburbs in the late 1970s. Targets are the median values of
    the houses at a location (in k$).

    The attributes themselves are defined in the
    [StatLib website](http://lib.stat.cmu.edu/datasets/boston).

    Args:
      path: path where to cache the dataset locally
          (relative to `~/.keras/datasets`).
      test_split: fraction of the data to reserve as test set.
      seed: Random seed for shuffling the data
          before computing the test split.

    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train, x_test**: numpy arrays with shape `(num_samples, 13)`
      containing either the training samples (for x_train),
      or test samples (for y_train).

    **y_train, y_test**: numpy arrays of shape `(num_samples,)` containing the
      target scalars. The targets are float scalars typically between 10 and
      50 that represent the home prices in k$.
    """
    ...
