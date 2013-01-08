"""Filterbank Correlation Operation"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['fbcorr5']

import numpy as np
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1


def fbcorr5(arr_in, arr_fb, stride=DEFAULT_STRIDE, 
            arr_out=None, f_mean=None, f_std=None, f_proj=None,
            f_intercept=None, f_offset=None):

    """5D Filterbank Correlation
    XXX: docstring
    """

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

    if f_mean is not None:
        arr_inrm = arr_inrm - f_mean

    if f_std is not None:
        arr_inrm = arr_inrm * (1. / f_std)

    if f_proj is not None:

        # Contrast normalization
        p_mean = arr_inrm.mean(axis=1)
        p_std = arr_inrm.std(axis=1)
        p_std[p_std == 0.] = 1.

        arr_inrm = (arr_inrm.T - p_mean)
        arr_inrm = (arr_inrm / p_std).T

        # Projection
        arr_inrm = np.dot(arr_inrm, f_proj)

    # -- reshape arr_fb
    arr_fbm = arr_fb.reshape((f_size, fbn))

    # -- correlate !
    arr_out = np.dot(arr_inrm, arr_fbm)

    if f_intercept is not None:
        arr_out += f_intercept

    if f_offset is not None:
        arr_out += f_offset

    arr_out = arr_out.reshape(n_imgs, n_tiles, outh, outw, -1)

    assert arr_out.dtype == arr_in.dtype  # XXX: should go away

    return arr_out

#try:
#    fbcorr5 = profile(fbcorr5)
#except NameError:
#    pass