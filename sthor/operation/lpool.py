"""Local Pooling Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['lpool3']

import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1
DEFAULT_ORDER = 1.0

assert ne.use_vml
ne.set_num_threads(8)


def lpool3(arr_in, neighborhood,
           order=DEFAULT_ORDER,
           stride=DEFAULT_STRIDE, arr_out=None):
    """3D Local Pooling Operation

    XXX: docstring
    """

    assert arr_in.ndim == 3
    assert len(neighborhood) == 2

    order = np.float32(order)
    stride = np.int(stride)

    inh, inw, ind = arr_in.shape
    nbh, nbw = neighborhood
    assert nbh <= inh
    assert nbw <= inw

    _arr_out = ne.evaluate('arr_in ** order')
    _arr_out = np.squeeze(view_as_windows(_arr_out, (1, nbw, 1)).sum(-2))[:, ::stride]
    _arr_out = np.squeeze(view_as_windows(_arr_out, (nbh, 1, 1)).sum(-3))[::stride, :]
    _arr_out = ne.evaluate('_arr_out ** (1.0 / order)')

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    return arr_out

try:
    lpool3 = profile(lpool3)
except NameError:
    pass

