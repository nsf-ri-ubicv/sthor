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
            arr_out=None, f_mean=None, f_std=None):

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

    if arr_inrm.shape[1] == 3201:
        print 'ok'
        # -- kmeans and slm_zca workaround
        #assert f_mean is None

        # Contrast normalization
        p_mean = arr_inrm.mean(axis=1)
        p_std = arr_inrm.std(axis=1)
        p_std[p_std == 0.] = 1.

        arr_inrm_new = (arr_inrm.T - p_mean)
        arr_inrm_new = (arr_inrm_new / p_std).T
        arr_inrm = arr_inrm_new

    # -- subtract mean and divide by std. dev. feature-wise
    if f_mean is not None:

        f_mean = f_mean.reshape(f_size)
        arr_inr_new = arr_inrm - f_mean

        # -- slm_zca workaround
        #arr_inr_new /= f_std

        arr_inr_new = np.dot(arr_inr_new, f_std)

        arr_inrm = arr_inr_new

    # -- reshape arr_fb
    arr_fbm = arr_fb.reshape((f_size, fbn))

    # -- correlate !
    arr_out = np.dot(arr_inrm, arr_fbm)
    arr_out = arr_out.reshape(n_imgs, n_tiles, outh, outw, -1)

    assert arr_out.dtype == arr_in.dtype  # XXX: should go away

    return arr_out


#try:
#    fbcorr5 = profile(fbcorr5)
#except NameError:
#    pass