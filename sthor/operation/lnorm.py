"""Local Normalization Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['ldnorm3', 'lcdnorm3']

import numpy as np

from skimage.util.shape import view_as_windows

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

    # -- local sums
    arr_sum = arr_in.sum(-1)
    arr_sum = view_as_windows(arr_sum, (1, nbw)).sum(-1)[:, ::stride, 0]
    arr_sum = view_as_windows(arr_sum, (nbh, 1)).sum(-2)[::stride, :]

    # -- local sums of squares
    arr_ssq = (arr_in ** 2.0).sum(-1)
    arr_ssq = view_as_windows(arr_ssq, (1, nbw)).sum(-1)[:, ::stride, 0]
    arr_ssq = view_as_windows(arr_ssq, (nbh, 1)).sum(-2)[::stride, :]

    # -- remove the mean
    _arr_out = _arr_out - arr_sum / nb_size

    # -- divide by the euclidean norm
    l2norms = (arr_ssq - (arr_sum ** 2.) / nb_size).clip(0, np.inf)
    l2norms = np.sqrt(l2norms) + EPSILON
    # XXX: use numpy-1.7.0 copyto()
    np.putmask(l2norms, l2norms < threshold, 1)
    _arr_out = _arr_out / l2norms

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    return arr_out

try:
    lcdnorm3 = profile(lcdnorm3)
except NameError:
    pass


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

    # -- divide by the euclidean norm
    arr_ssq = (arr_in ** 2.0).sum(-1)
    arr_ssq = view_as_windows(arr_ssq, (1, nbw)).sum(-1)[:, ::stride, 0]
    arr_ssq = view_as_windows(arr_ssq, (nbh, 1)).sum(-2)[::stride, :]
    l2norms = np.sqrt(arr_ssq) + EPSILON
    # XXX: use numpy-1.7.0 copyto()
    np.putmask(l2norms, l2norms < threshold, 1)
    _arr_out = _arr_out / l2norms

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    return arr_out
