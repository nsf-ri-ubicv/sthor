"""Padding functions"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np


def filter_pad2d(arr_in, filter_shape2d, constant=0,
                 reverse_padding=False):
    """Returns a padded array with constant values for the padding. The
    first two dimensions of the input array are padded, not the third
    one.

    Parameters
    ----------

    `arr_in`: array-like, shape = [height, width, depth]
        input 3D array to pad along the first two dimensions

    `filter_shape2d`: 2-tuple
        size of the "filter" from which we decide how many units on both
        sides of the array needs to be added

    `constant`: float
        value for the padding

    `reverse_padding`: bool
        in this case the array is not padded but rather the aprons are
        removed directly from the array

    Returns
    -------
    `arr_out`: array-like
        padded input array
    """

    assert arr_in.ndim == 3
    assert len(filter_shape2d) == 2

    fh, fw = np.array(filter_shape2d, dtype=int)
    inh, inw, ind = arr_in.shape

    if fh % 2 == 0:
        h_left, h_right = fh / 2, fh / 2 - 1
    else:
        h_left, h_right = fh / 2, fh / 2
    if fw % 2 == 0:
        w_left, w_right = fw / 2, fw / 2 - 1
    else:
        w_left, w_right = fw / 2, fw / 2

    # -- dimensions of the output array
    if reverse_padding:
        h_new = inh - h_left - h_right
        w_new = inw - w_left - w_right
    else:
        h_new = h_left + inh + h_right
        w_new = w_left + inw + w_right

    assert h_new >= 1
    assert w_new >= 1

    # -- makes sure the padding value is of the same type
    #    as the input array elements
    arr_out = np.empty((h_new, w_new, ind), dtype=arr_in.dtype)
    arr_out[:] = constant

    if reverse_padding:
        arr_out[:] = arr_in[h_left:inh - h_right,
                            w_left:inw - w_right]

    else:
        arr_out[h_left:h_new - h_right,
                w_left:w_new - w_right, :] = arr_in

    return arr_out
