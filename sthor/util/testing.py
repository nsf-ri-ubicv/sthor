import numpy as np
from numpy.testing.utils import assert_allclose

# modified from:
# https://github.com/numpy/numpy/blob/master/numpy/testing/utils.py
RTOL = 1e-3
ATOL = 1e-6

def assert_allclose_round(actual, desired, rtol=RTOL, atol=ATOL,
                          err_msg='', verbose=True):
    """Round arrays before passing them to `assert_allclose`
    See `assert_allclose` for more details.
    """
    decimal = int(-np.log10(atol))
    _actual = np.round(actual, decimal)
    _desired = np.round(desired, decimal)
    assert np.isfinite(_actual).all()
    assert np.isfinite(_desired).all()
    assert_allclose(_actual, _desired, rtol=rtol, atol=atol, err_msg=err_msg,
                    verbose=verbose)
