"""
This type stub file was generated by pyright.
"""

"""
Our own implementation of the Newton algorithm

Unlike the scipy.optimize version, this version of the Newton conjugate
gradient solver uses only one function call to retrieve the
func value, the gradient value and a callable for the Hessian matvec
product. If the function call is very expensive (e.g. for logistic
regression with large design matrix), this approach gives very
significant speedups.
"""
class _LineSearchError(RuntimeError):
    ...

