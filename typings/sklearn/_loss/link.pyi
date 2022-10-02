"""
This type stub file was generated by pyright.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

"""
Module contains classes for invertible (and differentiable) link functions.
"""
@dataclass
class Interval:
    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool
    def __post_init__(self): # -> None:
        """Check that low <= high"""
        ...
    
    def includes(self, x): # -> bool:
        """Test whether all values of x are in interval range.

        Parameters
        ----------
        x : ndarray
            Array whose elements are tested to be in interval range.

        Returns
        -------
        result : bool
        """
        ...
    


class BaseLink(ABC):
    """Abstract base class for differentiable, invertible link functions.

    Convention:
        - link function g: raw_prediction = g(y_pred)
        - inverse link h: y_pred = h(raw_prediction)

    For (generalized) linear models, `raw_prediction = X @ coef` is the so
    called linear predictor, and `y_pred = h(raw_prediction)` is the predicted
    conditional (on X) expected value of the target `y_true`.

    The methods are not implemented as staticmethods in case a link function needs
    parameters.
    """
    is_multiclass = ...
    interval_y_pred = ...
    @abstractmethod
    def link(self, y_pred, out=...): # -> None:
        """Compute the link function g(y_pred).

        The link function maps (predicted) target values to raw predictions,
        i.e. `g(y_pred) = raw_prediction`.

        Parameters
        ----------
        y_pred : array
            Predicted target values.
        out : array
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        Returns
        -------
        out : array
            Output array, element-wise link function.
        """
        ...
    
    @abstractmethod
    def inverse(self, raw_prediction, out=...): # -> None:
        """Compute the inverse link function h(raw_prediction).

        The inverse link function maps raw predictions to predicted target
        values, i.e. `h(raw_prediction) = y_pred`.

        Parameters
        ----------
        raw_prediction : array
            Raw prediction values (in link space).
        out : array
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        Returns
        -------
        out : array
            Output array, element-wise inverse link function.
        """
        ...
    


class IdentityLink(BaseLink):
    """The identity link function g(x)=x."""
    def link(self, y_pred, out=...):
        ...
    
    inverse = ...


class LogLink(BaseLink):
    """The log link function g(x)=log(x)."""
    interval_y_pred = ...
    def link(self, y_pred, out=...): # -> Any:
        ...
    
    def inverse(self, raw_prediction, out=...): # -> Any:
        ...
    


class LogitLink(BaseLink):
    """The logit link function g(x)=logit(x)."""
    interval_y_pred = ...
    def link(self, y_pred, out=...): # -> Any:
        ...
    
    def inverse(self, raw_prediction, out=...): # -> Any:
        ...
    


class MultinomialLogit(BaseLink):
    """The symmetric multinomial logit function.

    Convention:
        - y_pred.shape = raw_prediction.shape = (n_samples, n_classes)

    Notes:
        - The inverse link h is the softmax function.
        - The sum is over the second axis, i.e. axis=1 (n_classes).

    We have to choose additional constraints in order to make

        y_pred[k] = exp(raw_pred[k]) / sum(exp(raw_pred[k]), k=0..n_classes-1)

    for n_classes classes identifiable and invertible.
    We choose the symmetric side constraint where the geometric mean response
    is set as reference category, see [2]:

    The symmetric multinomial logit link function for a single data point is
    then defined as

        raw_prediction[k] = g(y_pred[k]) = log(y_pred[k]/gmean(y_pred))
        = log(y_pred[k]) - mean(log(y_pred)).

    Note that this is equivalent to the definition in [1] and implies mean
    centered raw predictions:

        sum(raw_prediction[k], k=0..n_classes-1) = 0.

    For linear models with raw_prediction = X @ coef, this corresponds to
    sum(coef[k], k=0..n_classes-1) = 0, i.e. the sum over classes for every
    feature is zero.

    Reference
    ---------
    .. [1] Friedman, Jerome; Hastie, Trevor; Tibshirani, Robert. "Additive
        logistic regression: a statistical view of boosting" Ann. Statist.
        28 (2000), no. 2, 337--407. doi:10.1214/aos/1016218223.
        https://projecteuclid.org/euclid.aos/1016218223

    .. [2] Zahid, Faisal Maqbool and Gerhard Tutz. "Ridge estimation for
        multinomial logit models with symmetric side constraints."
        Computational Statistics 28 (2013): 1017-1034.
        http://epub.ub.uni-muenchen.de/11001/1/tr067.pdf
    """
    is_multiclass = ...
    interval_y_pred = ...
    def symmetrize_raw_prediction(self, raw_prediction):
        ...
    
    def link(self, y_pred, out=...): # -> Any:
        ...
    
    def inverse(self, raw_prediction, out=...):
        ...
    


_LINKS = ...
