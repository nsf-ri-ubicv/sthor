"""Filterbank Correlation Operation."""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilver <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['fbcorr']


import numpy as np
from scipy.signal import correlate

DEFAULT_STRIDE = 1


def fbcorr(arr_in, arr_fb,
           stride=DEFAULT_STRIDE,
           arr_out=None):
    """XXX: docstring"""

    # -- Temporary constraints
    # XXX: make fbcorr n-dimensional
    assert arr_in.ndim == 3
    assert arr_fb.ndim == 4

    # -- check arguments
    assert arr_in.dtype == arr_fb.dtype

    inh, inw, ind = arr_in.shape
    fbh, fbw, fbd, fbn = arr_fb.shape

    assert fbd == ind

    out_shape = (inh - fbh + 1), (inw - fbw + 1), fbn

    # -- Create output array if necessary
    if arr_out is None:
        arr_out = np.empty(out_shape, dtype=arr_in.dtype)

    assert arr_out.dtype == arr_in.dtype
    assert arr_out.shape == out_shape

    # -- Correlate !
    for di in xrange(fbn):
        filt = arr_fb[..., di]
        out = correlate(arr_in, filt, mode='valid')
        arr_out[..., di] = out[..., 0]

    return arr_out
