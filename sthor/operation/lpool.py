"""
Local Pooling Operation.

Easy-to-understand (but slow) naive numpy implementation.
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#
# License: Proprietary

__all__ = ['LPoolScipyNaive']

# -- Imports
import numpy as np
from pythor3.operation.common.lsum import lsum
from pythor3.operation.lpool_.plugins import LPoolPlugin

# -- Default parameters
from pythor3.operation.lpool_ import DEFAULT_KER_SHAPE

# -- Contracts
from pythor3.operation.lpool_ import (
    assert_preconditions_on_data,
    assert_postconditions_on_properties,
    assert_postconditions_on_data)


class LPoolScipyNaive(LPoolPlugin):

    def _get_tmp_shape(self, arr_in,
                       ker_shape=DEFAULT_KER_SHAPE,
                      ):
        """XXX: docstring"""
        in_h, in_w = arr_in.shape[:2]
        tmp_h = in_h - ker_shape[0] + 1
        tmp_w = in_w - ker_shape[1] + 1
        if arr_in.ndim == 3:
            in_d = arr_in.shape[2]
            tmp_d = in_d
            tmp_shape = tmp_h, tmp_w, tmp_d
        else:
            tmp_shape = tmp_h, tmp_w
        return tmp_shape

    def run(self):
        """XXX: doctring"""

        tmp_shape = self._get_tmp_shape(self.arr_in, ker_shape=self.ker_shape)
        self.arr_tmp = np.atleast_3d(np.empty(tmp_shape, self.arr_in.dtype))

        # input array
        arr_in = self.arr_in
        if arr_in.ndim == 2:
            _arr_in = arr_in[:, :, None]
        else:
            _arr_in = arr_in[:]
        in_h, in_w, in_d = _arr_in.shape
        assert_preconditions_on_data(arr_in)

        dtype = self.arr_in.dtype

        # output array
        arr_out = self.arr_out
        if arr_in.ndim == 2:
            _arr_out = arr_out[:, :, None]
        else:
            _arr_out = arr_out[:]

        # temporary array
        arr_tmp = self.arr_tmp

        # operation parameters
        ker_shape = self.ker_shape
        order = self.order
        stride = self.stride

        # -- get input data
        src = _arr_in[:]

        # -- power
        if order != 1:
            src = src.astype(np.float64) ** order
            src = src.astype(np.float32)

        # -- local sum
        for di in xrange(in_d):
            slice2d = lsum(src[:, :, di], ker_shape, mode='valid')
            arr_tmp[:, :, di] = slice2d.astype(dtype)

        # -- root
        if order != 1:
            arr_tmp[arr_tmp < 0] = 0
            arr_tmp = arr_tmp ** (1. / order)

        # -- output
        _arr_out = arr_tmp[::stride, ::stride]
        if arr_in.ndim == 2:
            _arr_out = _arr_out[:, :, 0]

        # -- Contracts: postconditions
        assert_postconditions_on_properties(arr_in, _arr_out, ker_shape, stride)
        assert_postconditions_on_data(_arr_out)

        arr_out[:] = _arr_out

        return arr_out
