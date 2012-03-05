"""Filterbank Normalized Cross-Correlation Operation"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['fbncc3']

import numpy as np
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1
DEFAULT_THRESHOLD = 1.0
DEFAULT_NORMALIZE_FILTERS = True

EPSILON = 1e-6


import numexpr as ne
assert ne.use_vml

print ne.set_num_threads(8)

def fbncc3(arr_in, arr_fb,
           normalize_filters=DEFAULT_NORMALIZE_FILTERS,
           threshold=DEFAULT_THRESHOLD,
           stride=DEFAULT_STRIDE, arr_out=None):
    """3D Filterbank Normalized Cross-Correlation

    XXX: docstring
    """

    # -- Basic checks
    assert arr_in.ndim == 3
    assert arr_fb.ndim == 4

    inh, inw, ind = arr_in.shape
    fbh, fbw, fbd, fbn = arr_fb.shape

    assert fbh <= inh
    assert fbw <= inw
    assert fbd == ind

    nb_size = np.prod(arr_fb.shape[:-1])

    # -- Prepare arr_out
    if arr_out is not None:
        assert arr_out.dtype == arr_in.dtype
        assert arr_out.shape == (inh - fbh + 1, inw - fbw + 1, ind)

    ys = fbh / 2
    xs = fbw / 2
    _arr_out = arr_in[ys:-ys, xs:-xs]

    # -- Normalize input array
    arr_inr = view_as_windows(arr_in, arr_fb.shape[:-1])[::stride, ::stride]
    inr_sum = np.apply_over_axes(np.sum, arr_inr, (3, 4, 5))
    inr_ssq = np.apply_over_axes(np.sum, ne.evaluate('arr_inr ** 2'), (3, 4, 5))

    #inr_num = arr_inr - inr_sum / nb_size
    inr_num = ne.evaluate('arr_inr - inr_sum / nb_size')
    in_div = np.sqrt((inr_ssq - (inr_sum ** 2.0) / nb_size).clip(0, np.inf))
    np.putmask(in_div, in_div < threshold, 1.0)
    inr = ne.evaluate("inr_num / in_div")

    inrm = inr.reshape(-1, nb_size)

    # -- Normalize filters
    arr_fbr = arr_fb.reshape(inrm.shape[-1], -1)

    if normalize_filters:

        fb_sum = arr_fbr.sum(0)
        fb_ssq = (arr_fbr ** 2.).sum(0)

        fb_num = arr_fbr - fb_sum / nb_size
        fb_div = np.sqrt((fb_ssq - (fb_sum ** 2.0) / nb_size).clip(0, np.inf))
        np.putmask(fb_div, fb_div < EPSILON, EPSILON)  # avoid zero division

        fbrm = fb_num / fb_div

    else:
        fbrm = arr_fbr

    # -- Correlate !
    _arr_out = np.dot(inrm, fbrm)

    # -- Reshape back to 3D
    _arr_out = _arr_out.reshape(arr_inr.shape[:2] + (fbn,))

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    assert arr_out.dtype == arr_in.dtype

    return arr_out


try:
    fbncc3 = profile(fbncc3)
except NameError:
    pass


def main():
    arr = np.random.randn(32, 32, 3).astype('f')

    fbl = [
        # -- layer 1
        np.random.randn(5, 5, 3, 64).astype('f'),
        np.random.randn(5, 5, 64, 64).astype('f'),
        # -- layer 2
        np.random.randn(5, 5, 64, 128).astype('f'),
        np.random.randn(5, 5, 128, 128).astype('f'),
        # -- layer 3
        np.random.randn(5, 5, 128, 256).astype('f'), 
        np.random.randn(5, 5, 256, 256).astype('f'), 
    ]

    N = 10
    import time
    start = time.time()
    for i in xrange(N):
        tmp = arr.copy()
        for fi, fb in enumerate(fbl):
            print tmp.shape, fb.shape
            tmp = fbncc3(tmp, fb)
            #if (fi + 1) % 2 == 0:
                #tmp = fbncc3(tmp, fb, stride=2)
            #else:
                #tmp = fbncc3(tmp, fb)
            print tmp.shape
    end = time.time()
    print N / (end - start)


if __name__ == '__main__':
    main()
