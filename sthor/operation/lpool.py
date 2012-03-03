"""Local Pooling Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['lpool3']

import numpy as np
#import numexpr as ne
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1
DEFAULT_ORDER = 1.0


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

    xp = arr_in ** order
    win_shape = neighborhood + (1,)
    xpr = view_as_windows(xp, win_shape)[::stride, ::stride]
    xprm = xpr.reshape(xpr.shape[:3] + (-1,))
    xprms = xprm.sum(-1)
    arr_out = xprms ** (1.0 / order)

    return arr_out
