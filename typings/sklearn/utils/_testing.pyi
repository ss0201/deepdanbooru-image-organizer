"""
This type stub file was generated by pyright.
"""

import contextlib
import unittest
from sklearn.utils import deprecated

"""Testing utilities."""
__all__ = ["assert_raises", "assert_raises_regexp", "assert_array_equal", "assert_almost_equal", "assert_array_almost_equal", "assert_array_less", "assert_approx_equal", "assert_allclose", "assert_run_python_script", "SkipTest"]
_dummy = ...
assert_raises = ...
SkipTest = unittest.case.SkipTest
assert_dict_equal = ...
assert_raises_regex = ...
assert_raises_regexp = ...
@deprecated("`assert_warns` is deprecated in 1.0 and will be removed in 1.2." "Use `pytest.warns` instead.")
def assert_warns(warning_class, func, *args, **kw):
    """Test that a certain warning occurs.

    .. deprecated:: 1.0
        `assert_warns` is deprecated in 1.0 and will be removed in 1.2.
        Use `pytest.warns` instead.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`

    Returns
    -------
    result : the return value of `func`

    """
    ...

@deprecated("`assert_warns_message` is deprecated in 1.0 and will be removed in 1.2." "Use `pytest.warns` instead.")
def assert_warns_message(warning_class, message, func, *args, **kw):
    """Test that a certain warning occurs and with a certain message.

    .. deprecated:: 1.0
        `assert_warns_message` is deprecated in 1.0 and will be removed in 1.2.
        Use `pytest.warns` instead.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    message : str or callable
        The message or a substring of the message to test for. If callable,
        it takes a string as the argument and will trigger an AssertionError
        if the callable returns `False`.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`.

    Returns
    -------
    result : the return value of `func`

    """
    ...

def assert_no_warnings(func, *args, **kw):
    """
    Parameters
    ----------
    func
    *args
    **kw
    """
    ...

def ignore_warnings(obj=..., category=...): # -> ((*args: Unknown, **kwargs: Unknown) -> Unknown) | _IgnoreWarnings:
    """Context manager and decorator to ignore warnings.

    Note: Using this (in both variants) will clear all warnings
    from all python modules loaded. In case you need to test
    cross-module-warning-logging, this is not your tool of choice.

    Parameters
    ----------
    obj : callable, default=None
        callable where you want to ignore the warnings.
    category : warning class, default=Warning
        The category to filter. If Warning, all categories will be muted.

    Examples
    --------
    >>> import warnings
    >>> from sklearn.utils._testing import ignore_warnings
    >>> with ignore_warnings():
    ...     warnings.warn('buhuhuhu')

    >>> def nasty_warn():
    ...     warnings.warn('buhuhuhu')
    ...     print(42)

    >>> ignore_warnings(nasty_warn)()
    42
    """
    ...

class _IgnoreWarnings:
    """Improved and simplified Python warnings context manager and decorator.

    This class allows the user to ignore the warnings raised by a function.
    Copied from Python 2.7.5 and modified as required.

    Parameters
    ----------
    category : tuple of warning class, default=Warning
        The category to filter. By default, all the categories will be muted.

    """
    def __init__(self, category) -> None:
        ...
    
    def __call__(self, fn): # -> (*args: Unknown, **kwargs: Unknown) -> Unknown:
        """Decorator to catch and hide warnings without visual nesting."""
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def __enter__(self): # -> None:
        ...
    
    def __exit__(self, *exc_info): # -> None:
        ...
    


def assert_raise_message(exceptions, message, function, *args, **kwargs): # -> None:
    """Helper function to test the message raised in an exception.

    Given an exception, a callable to raise the exception, and
    a message string, tests that the correct exception is raised and
    that the message is a substring of the error thrown. Used to test
    that the specific message thrown during an exception is correct.

    Parameters
    ----------
    exceptions : exception or tuple of exception
        An Exception object.

    message : str
        The error message or a substring of the error message.

    function : callable
        Callable object to raise error.

    *args : the positional arguments to `function`.

    **kwargs : the keyword arguments to `function`.
    """
    ...

def assert_allclose(actual, desired, rtol=..., atol=..., equal_nan=..., err_msg=..., verbose=...): # -> None:
    """dtype-aware variant of numpy.testing.assert_allclose

    This variant introspects the least precise floating point dtype
    in the input argument and automatically sets the relative tolerance
    parameter to 1e-4 float32 and use 1e-7 otherwise (typically float64
    in scikit-learn).

    `atol` is always left to 0. by default. It should be adjusted manually
    to an assertion-specific value in case there are null values expected
    in `desired`.

    The aggregate tolerance is `atol + rtol * abs(desired)`.

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional, default=None
        Relative tolerance.
        If None, it is set based on the provided arrays' dtypes.
    atol : float, optional, default=0.
        Absolute tolerance.
    equal_nan : bool, optional, default=True
        If True, NaNs will compare equal.
    err_msg : str, optional, default=''
        The error message to be printed in case of failure.
    verbose : bool, optional, default=True
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    numpy.testing.assert_allclose

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils._testing import assert_allclose
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> assert_allclose(x, y, rtol=1e-5, atol=0)
    >>> a = np.full(shape=10, fill_value=1e-5, dtype=np.float32)
    >>> assert_allclose(a, 1e-5)
    """
    ...

def assert_allclose_dense_sparse(x, y, rtol=..., atol=..., err_msg=...): # -> None:
    """Assert allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : {array-like, sparse matrix}
        First array to compare.

    y : {array-like, sparse matrix}
        Second array to compare.

    rtol : float, default=1e-07
        relative tolerance; see numpy.allclose.

    atol : float, default=1e-9
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.

    err_msg : str, default=''
        Error message to raise.
    """
    ...

def set_random_state(estimator, random_state=...): # -> None:
    """Set random state of an estimator if it has the `random_state` param.

    Parameters
    ----------
    estimator : object
        The estimator.
    random_state : int, RandomState instance or None, default=0
        Pseudo random number generator state.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    ...

def check_skip_network(): # -> None:
    ...

class TempMemmap:
    """
    Parameters
    ----------
    data
    mmap_mode : str, default='r'
    """
    def __init__(self, data, mmap_mode=...) -> None:
        ...
    
    def __enter__(self): # -> Any | list[Any]:
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb): # -> None:
        ...
    


def create_memmap_backed_data(data, mmap_mode=..., return_folder=..., aligned=...): # -> Any | list[Any] | tuple[Any | list[Any], str]:
    """
    Parameters
    ----------
    data
    mmap_mode : str, default='r'
    return_folder :  bool, default=False
    aligned : bool, default=False
        If True, if input is a single numpy array and if the input array is aligned,
        the memory mapped array will also be aligned. This is a workaround for
        https://github.com/joblib/joblib/issues/563.
    """
    ...

def check_docstring_parameters(func, doc=..., ignore=...):
    """Helper to check docstring.

    Parameters
    ----------
    func : callable
        The function object to test.
    doc : str, default=None
        Docstring if it is passed manually to the test.
    ignore : list, default=None
        Parameters to ignore.

    Returns
    -------
    incorrect : list
        A list of string describing the incorrect results.
    """
    ...

def assert_run_python_script(source_code, timeout=...): # -> None:
    """Utility to check assertions in an independent Python subprocess.

    The script provided in the source code should return 0 and not print
    anything on stderr or stdout.

    This is a port from cloudpickle https://github.com/cloudpipe/cloudpickle

    Parameters
    ----------
    source_code : str
        The Python source code to execute.
    timeout : int, default=60
        Time in seconds before timeout.
    """
    ...

def raises(expected_exc_type, match=..., may_pass=..., err_msg=...): # -> _Raises:
    """Context manager to ensure exceptions are raised within a code block.

    This is similar to and inspired from pytest.raises, but supports a few
    other cases.

    This is only intended to be used in estimator_checks.py where we don't
    want to use pytest. In the rest of the code base, just use pytest.raises
    instead.

    Parameters
    ----------
    excepted_exc_type : Exception or list of Exception
        The exception that should be raised by the block. If a list, the block
        should raise one of the exceptions.
    match : str or list of str, default=None
        A regex that the exception message should match. If a list, one of
        the entries must match. If None, match isn't enforced.
    may_pass : bool, default=False
        If True, the block is allowed to not raise an exception. Useful in
        cases where some estimators may support a feature but others must
        fail with an appropriate error message. By default, the context
        manager will raise an exception if the block does not raise an
        exception.
    err_msg : str, default=None
        If the context manager fails (e.g. the block fails to raise the
        proper exception, or fails to match), then an AssertionError is
        raised with this message. By default, an AssertionError is raised
        with a default error message (depends on the kind of failure). Use
        this to indicate how users should fix their estimators to pass the
        checks.

    Attributes
    ----------
    raised_and_matched : bool
        True if an exception was raised and a match was found, False otherwise.
    """
    ...

class _Raises(contextlib.AbstractContextManager):
    def __init__(self, expected_exc_type, match, may_pass, err_msg) -> None:
        ...
    
    def __exit__(self, exc_type, exc_value, _): # -> bool:
        ...
    


class MinimalClassifier:
    """Minimal classifier implementation with inheriting from BaseEstimator.

    This estimator should be tested with:

    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """
    _estimator_type = ...
    def __init__(self, param=...) -> None:
        ...
    
    def get_params(self, deep=...): # -> dict[str, Unknown | None]:
        ...
    
    def set_params(self, **params): # -> Self@MinimalClassifier:
        ...
    
    def fit(self, X, y): # -> Self@MinimalClassifier:
        ...
    
    def predict_proba(self, X): # -> ndarray[Unknown, Unknown]:
        ...
    
    def predict(self, X):
        ...
    
    def score(self, X, y): # -> Any:
        ...
    


class MinimalRegressor:
    """Minimal regressor implementation with inheriting from BaseEstimator.

    This estimator should be tested with:

    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """
    _estimator_type = ...
    def __init__(self, param=...) -> None:
        ...
    
    def get_params(self, deep=...): # -> dict[str, Unknown | None]:
        ...
    
    def set_params(self, **params): # -> Self@MinimalRegressor:
        ...
    
    def fit(self, X, y): # -> Self@MinimalRegressor:
        ...
    
    def predict(self, X): # -> Any:
        ...
    
    def score(self, X, y): # -> float | ndarray[Unknown, Unknown]:
        ...
    


class MinimalTransformer:
    """Minimal transformer implementation with inheriting from
    BaseEstimator.

    This estimator should be tested with:

    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """
    def __init__(self, param=...) -> None:
        ...
    
    def get_params(self, deep=...): # -> dict[str, Unknown | None]:
        ...
    
    def set_params(self, **params): # -> Self@MinimalTransformer:
        ...
    
    def fit(self, X, y=...): # -> Self@MinimalTransformer:
        ...
    
    def transform(self, X, y=...): # -> Any:
        ...
    
    def fit_transform(self, X, y=...): # -> Any:
        ...
    


