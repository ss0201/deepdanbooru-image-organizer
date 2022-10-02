"""
This type stub file was generated by pyright.
"""

from tensorflow.python.ops.distributions import distribution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

"""The Laplace distribution class."""
__all__ = ["Laplace", "LaplaceWithSoftplusScale"]
@tf_export(v1=["distributions.Laplace"])
class Laplace(distribution.Distribution):
  """The Laplace distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; mu, sigma) = exp(-|x - mu| / sigma) / Z
  Z = 2 sigma
  ```

  where `loc = mu`, `scale = sigma`, and `Z` is the normalization constant.

  Note that the Laplace distribution can be thought of two exponential
  distributions spliced together "back-to-back."

  The Lpalce distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Laplace(loc=0, scale=1)
  Y = loc + scale * X
  ```

  """
  @deprecation.deprecated("2019-01-01", "The TensorFlow Distributions library has moved to " "TensorFlow Probability " "(https://github.com/tensorflow/probability). You " "should update all references to use `tfp.distributions` " "instead of `tf.distributions`.", warn_once=True)
  def __init__(self, loc, scale, validate_args=..., allow_nan_stats=..., name=...) -> None:
    """Construct Laplace distribution with parameters `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g., `loc / scale` is a valid operation).

    Args:
      loc: Floating point tensor which characterizes the location (center)
        of the distribution.
      scale: Positive floating point tensor which characterizes the spread of
        the distribution.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc` and `scale` are of different dtype.
    """
    ...
  
  @property
  def loc(self): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
    """Distribution parameter for the location."""
    ...
  
  @property
  def scale(self): # -> defaultdict[Unknown, Unknown] | Any | list[Unknown] | ObjectProxy:
    """Distribution parameter for scale."""
    ...
  


class LaplaceWithSoftplusScale(Laplace):
  """Laplace with softplus applied to `scale`."""
  @deprecation.deprecated("2019-01-01", "Use `tfd.Laplace(loc, tf.nn.softplus(scale)) " "instead.", warn_once=True)
  def __init__(self, loc, scale, validate_args=..., allow_nan_stats=..., name=...) -> None:
    ...
  


