"""Filterbank Correlation Operation"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['fbcorr3']

import numpy as np
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1


def fbcorr3(arr_in, arr_fb, stride=DEFAULT_STRIDE, arr_out=None):
    """3D Filterbank Correlation
    XXX: docstring
    """

    assert arr_in.ndim == 3
    assert arr_fb.ndim == 4
    assert arr_fb.dtype == arr_in.dtype

    inh, inw, ind = arr_in.shape
    fbh, fbw, fbd, fbn = arr_fb.shape

    assert fbn > 1
    assert fbh <= inh
    assert fbw <= inw
    assert fbd == ind

    if arr_out is not None:
        assert arr_out.dtype == arr_in.dtype
        assert arr_out.shape == (1 + (inh - fbh) / stride,
                                 1 + (inw - fbw) / stride,
                                 fbn)

    # -- reshape arr_in
    arr_inr = view_as_windows(arr_in, (fbh, fbw, fbd))[::stride, ::stride]
    outh, outw = arr_inr.shape[:2]
    arr_inrm = arr_inr.reshape(outh * outw, -1)

    # -- reshape arr_fb
    arr_fbm = arr_fb.reshape((fbh * fbw * fbd, fbn))

    # -- correlate !
    arr_out = np.dot(arr_inrm, arr_fbm)
    arr_out = arr_out.reshape(outh, outw, -1)

    assert arr_out.dtype == arr_in.dtype  # XXX: should go away

    return arr_out


try:
    fbcorr3 = profile(fbcorr3)
except NameError:
    pass
