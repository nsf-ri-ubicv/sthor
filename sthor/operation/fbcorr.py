"""5D Filterbank Correlation Operation.

XXX: docstring
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD 3-clause


import numpy as np
from scipy.signal import correlate

def fbcorr(arr_in, arr_fb, arr_out=None, stride=(1, 1, 1, 1)):
    """XXX: docstring
    XXX: doctest
    """

    # -- parse and check arguments
    assert arr_in.ndim == 5
    assert arr_fb.ndim == 5
    assert arr_fb.dtype == arr_in.dtype
    assert len(stride) == 4

    in_s, in_t, in_h, in_w, in_d = arr_in.shape
    fb_n, fb_t, fb_h, fb_w, fb_d = arr_fb.shape
    assert fb_t <= in_t
    assert fb_h <= in_h
    assert fb_w <= in_w
    assert fb_d == in_d

    stride = np.array(stride, dtype=int)
    st_t, st_h, st_w, st_d = stride
    assert (stride > 0).all()

    out_s = in_s
    out_t = (in_t - fb_t) / st_t
    out_h = (in_h - fb_h) / st_h
    out_w = (in_w - fb_w) / st_w
    out_d = (fb_n - 1) / st_d
    out_shape = out_s, out_t, out_h, out_w, out_d

    # -- Create arr_out if necessary
    if arr_out is None:
        arr_out = np.empty(out_shape, dtype=arr_in.dtype)

    assert arr_out.shape == out_shape
    assert arr_out.dtype == arr_in.dtype

    # -- Correlate !
    for si in xrange(out_s):
        for ni in xrange(fb_n):
            response = correlate(arr_in[si], arr_fb[ni], mode='valid')
            response = response[::st_t, ::st_h, ::st_w]
            arr_out[si, :, :, :, ni] = response

    return arr_out
