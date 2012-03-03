"""Local Normalization Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['lcdnorm3']

import numpy as np

from skimage.util.shape import view_as_windows
import numexpr as ne

EPSILON = 1e-6
DEFAULT_STRIDE = 1
DEFAULT_THRESHOLD = 1.0


def lcdnorm3(arr_in, neighborhood, threshold=DEFAULT_THRESHOLD,
            stride=DEFAULT_STRIDE, arr_out=None):
    """3D Local Contrast Divisive Normalization

    XXX: docstring
    """

    assert arr_in.ndim == 3
    assert len(neighborhood) == 2

    inh, inw, ind = arr_in.shape

    nbh, nbw = neighborhood
    nb_size = nbh * nbw * ind

    if arr_out is not None:
        assert arr_out.shape == (inh - nbh + 1, inw - nbw + 1, ind)

    # -- prepare arr_out
    ys = nbh / 2
    xs = nbw / 2
    _arr_out = arr_in[ys:-ys, xs:-xs]

    inrv = view_as_windows(arr_in, neighborhood + (ind,))
    inrv = inrv.reshape(inrv.shape[:2] + (1, -1,))

    # -- local sums
    arr_sum = inrv.sum(-1)

    # -- local sums of squares
    arr_ssq = ne.evaluate('inrv ** 2.0')
    arr_ssq = arr_ssq.sum(-1)

    # -- remove the mean
    _arr_out = ne.evaluate('_arr_out - arr_sum / nb_size')

    # -- divide by the euclidean norm
    l2norms = (arr_ssq - (arr_sum ** 2.) / nb_size).clip(0, np.inf)
    l2norms = np.sqrt(l2norms) + EPSILON
    # XXX: use numpy-1.7.0 copyto()
    np.putmask(l2norms, l2norms < threshold, 1)
    _arr_out = ne.evaluate('_arr_out / l2norms')

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    return arr_out


def ldnorm3(arr_in, neighborhood, threshold=DEFAULT_THRESHOLD,
            stride=DEFAULT_STRIDE, arr_out=None):
    """3D Local Divisive Normalization

    XXX: docstring
    """

    assert arr_in.ndim == 3
    assert len(neighborhood) == 2

    inh, inw, ind = arr_in.shape

    nbh, nbw = neighborhood

    if arr_out is not None:
        assert arr_out.shape == (inh - nbh + 1, inw - nbw + 1, ind)

    # -- prepare arr_out
    ys = nbh / 2
    xs = nbw / 2
    _arr_out = arr_in[ys:-ys, xs:-xs]

    inrv = view_as_windows(arr_in, neighborhood + (ind,))
    inrv = inrv.reshape(inrv.shape[:2] + (1, -1,))

    # -- local sums of squares
    arr_ssq = ne.evaluate('inrv ** 2.0')
    arr_ssq = arr_ssq.sum(-1)

    # -- divide by the euclidean norm
    l2norms = arr_ssq
    l2norms = np.sqrt(l2norms) + EPSILON
    # XXX: use numpy-1.7.0 copyto()
    np.putmask(l2norms, l2norms < threshold, 1)
    _arr_out = ne.evaluate('_arr_out / l2norms')

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    return arr_out
