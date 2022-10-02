"""
This type stub file was generated by pyright.
"""

from ..base import BaseEstimator
from ._base import SelectorMixin

class VarianceThreshold(SelectorMixin, BaseEstimator):
    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <variance_threshold>`.

    Parameters
    ----------
    threshold : float, default=0
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SelectFromModel: Meta-transformer for selecting features based on
        importance weights.
    SelectPercentile : Select features according to a percentile of the highest
        scores.
    SequentialFeatureSelector : Transformer that performs Sequential Feature
        Selection.

    Notes
    -----
    Allows NaN in the input.
    Raises ValueError if no feature in X meets the variance threshold.

    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::

        >>> from sklearn.feature_selection import VarianceThreshold
        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = VarianceThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """
    def __init__(self, threshold=...) -> None:
        ...
    
    def fit(self, X, y=...): # -> Self@VarianceThreshold:
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number of features.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        ...
    


