"""Padding of a 3D array along the first two dimensions"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np


def pad(arr_in, neighborhood, pad_val=0.):
    """Returns a padded array with constant values for the
    padding. The first two dimensions of the input array
    are padded, not the third one.

    Parameters
    ----------

    `arr_in`: array-like
        input 3D array to pad along the first two dimensions

    `neighborhood`: 2-tuple
        size of the "filter" from which we decide how many
        units on both sides of the array needs to be added

    `pad_val`: float
        value for the padding

    Returns
    -------

    `arr_out`: array-like
        padded input array
    """

    fh, fw = int(neighborhood[0]), int(neighborhood[1])

    if fh % 2 == 0:
        h_left, h_right = fh / 2, fh / 2 - 1
    else:
        h_left, h_right = fh / 2, fh / 2
    if fw % 2 == 0:
        w_left, w_right = fw / 2, fw / 2 - 1
    else:
        w_left, w_right = fw / 2, fw / 2

    # -- dimensions of the output array
    h_new = h_left + arr_in.shape[0] + h_right
    w_new = w_left + arr_in.shape[1] + w_right

    # -- makes sure the padding value is of the same type
    #    as the input array elements
    pad_val = np.array([pad_val], dtype=arr_in.dtype)[0]

    arr_out = pad_val * np.ones((h_new, w_new, arr_in.shape[2]),
                                dtype=arr_in.dtype)

    arr_out[h_left:h_new - h_right,
            w_left:w_new - w_right, :] = arr_in

    return arr_out
