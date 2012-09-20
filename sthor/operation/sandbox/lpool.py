"""Local Pooling Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['lpool5']

import numpy as np
from skimage.util.shape import view_as_windows

import numexpr as ne
#ne.use_vml=True
if not ne.use_vml:
    import warnings
    warnings.warn("numexpr is NOT using Intel VML!")

# --
DEFAULT_STRIDE = 1
DEFAULT_ORDER = 1.0


def lpool5(arr_in, neighborhood,
           order=DEFAULT_ORDER,
           stride=DEFAULT_STRIDE, arr_out=None):
    """5D Local Pooling Operation

    XXX: docstring
    """
    assert arr_in.ndim == 5
    assert len(neighborhood) == 2

    order = np.array([order], dtype=arr_in.dtype)
    stride = np.int(stride)

    in_imgs, in_tiles, inh, inw, ind = arr_in.shape
    nbh, nbw = neighborhood
    assert nbh <= inh
    assert nbw <= inw

    if arr_out is not None:
        assert arr_out.dtype == arr_in.dtype
        assert arr_out.shape == (in_imgs,
                                 in_tiles,
                                 1 + (inh - nbh) / stride,
                                 1 + (inw - nbw) / stride,
                                 ind)

    _arr_out = ne.evaluate('arr_in ** order')
    _arr_out = view_as_windows(_arr_out, (1, 1, 1, nbw, 1))
    _arr_out = ne.evaluate('sum(_arr_out, 8)')[:, :, :, ::stride, :, 0, 0, 0, 0]
    _arr_out = view_as_windows(_arr_out, (1, 1, nbh, 1, 1))
    _arr_out = ne.evaluate('sum(_arr_out, 7)')[:, :, ::stride, :, :, 0, 0, 0, 0]
    # Note that you need to use '1' and not '1.0' so that the dtype of
    # the exponent does not change (i.e. get promoted)
    _arr_out = ne.evaluate('_arr_out ** (1 / order)')

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    assert arr_out.shape[0] == in_imgs
    assert arr_out.shape[1] == in_tiles

    assert arr_out.dtype == arr_in.dtype

    return arr_out

try:
    lpool5 = profile(lpool5)
except NameError:
    pass

