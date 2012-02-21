"""Filterbank Correlation Operation.

XXX: docstring
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD 3-clause


import numpy as np
from scipy.signal import correlate


def fbcorr(arr_in, arr_fb, arr_out=None, stride=1, mode='valid'):
    """XXX: docstring"""

    # -- check arguments
    assert arr_in.dtype == arr_fb.dtype
    assert mode in ('valid', 'same')

    inh, inw, ind = arr_in.shape
    fbn, fbh, fbw, fbd = arr_fb.shape

    if mode == 'valid':
        out_shape = (inh - fbh + 1), (inw - fbw + 1), fbn
    elif mode == 'same':
        out_shape = inh, inw, fbn

    # -- Create output array if necessary
    if arr_out is None:
        arr_out = np.empty(out_shape, dtype=arr_in.dtype)

    assert arr_out.dtype == arr_in.dtype
    assert arr_out.shape == out_shape

    # -- Correlate !
    for di in xrange(fbn):
        arr_out[:, :, di] = correlate(arr_in, arr_fb[di], mode=mode)

    return arr_out
