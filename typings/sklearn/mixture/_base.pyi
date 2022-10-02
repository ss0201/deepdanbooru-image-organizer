"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta
from ..base import BaseEstimator, DensityMixin

"""Base class for mixture models."""
class BaseMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """
    def __init__(self, n_components, tol, reg_covar, max_iter, n_init, init_params, random_state, warm_start, verbose, verbose_interval) -> None:
        ...
    
    def fit(self, X, y=...): # -> Self@BaseMixture:
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            The fitted mixture.
        """
        ...
    
    def fit_predict(self, X, y=...):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        ...
    
    def score_samples(self, X): # -> tuple[Any, Any | Unbound] | Any:
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        ...
    
    def score(self, X, y=...): # -> Any:
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        ...
    
    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        ...
    
    def predict_proba(self, X): # -> Any:
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        ...
    
    def sample(self, n_samples=...): # -> tuple[ndarray[Unknown, Unknown], Unknown]:
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """
        ...
    

