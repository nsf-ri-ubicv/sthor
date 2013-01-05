"""Filterbank Correlation Operation"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['fbcorr5_incpt']

import numpy as np
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1


def fbcorr5_incpt(arr_in, arr_fb, incpts, stride=DEFAULT_STRIDE, arr_out=None):

    assert arr_in.ndim == 5
    assert arr_fb.ndim == 4
    assert arr_fb.dtype == arr_in.dtype

    in_imgs, in_tiles, inh, inw, ind = arr_in.shape
    fbh, fbw, fbd, fbn = arr_fb.shape

    f_size = fbh * fbw * fbd

    assert fbn > 1
    assert fbh <= inh
    assert fbw <= inw
    assert fbd == ind

    assert incpts.shape[0] == fbn

    if arr_out is not None:
        assert arr_out.dtype == arr_in.dtype
        assert arr_out.shape == (in_imgs,
                                 in_tiles,
                                 1 + (inh - fbh) / stride,
                                 1 + (inw - fbw) / stride,
                                 fbn)

    # -- reshape arr_in
    arr_inr = view_as_windows(arr_in, (1, 1, fbh, fbw, fbd))[::stride, ::stride]

    n_imgs, n_tiles, outh, outw = arr_inr.shape[:4]

    assert n_imgs == in_imgs
    assert n_tiles == in_tiles

    arr_inrm = arr_inr.reshape(n_imgs * n_tiles * outh * outw, f_size)

    # -- reshape arr_fb
    arr_fbm = arr_fb.reshape((f_size, fbn))

    # -- correlate !
    arr_out = np.dot(arr_inrm, arr_fbm)
    # -- intercept
    arr_out += incpts

    arr_out += 3.6

    #print 'mean filter response',  arr_out.mean()
    #import pdb; pdb.set_trace()

    arr_out = arr_out.reshape(n_imgs, n_tiles, outh, outw, -1)

    return arr_out