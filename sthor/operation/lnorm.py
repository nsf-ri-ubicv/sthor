"""
Local Normalization Operation.

Easy-to-understand (but slow) naive numpy implementation.
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#
# License: Proprietary

__all__ = ['LNormScipyNaive']

# -- Imports
import numpy as np
from pythor3.operation.common.lsum import lsum
from pythor3.operation.lnorm_.plugins import LNormPlugin

# -- Small epsilon
from pythor3.operation.lnorm_ import EPSILON

# -- Contracts
from pythor3.operation.lnorm_ import (
    assert_postconditions_on_properties,
    assert_postconditions_on_data,
)


class LNormScipyNaive(LNormPlugin):

    def run(self):
        """XXX: docstring"""

        arr_in = self.arr_in
        #assert_preconditions_on_data(arr_in)
        if arr_in.ndim == 2:
            _arr_in = arr_in[:, :, None]
        else:
            _arr_in = arr_in

        arr_out = self.arr_out
        if arr_out.ndim == 2:
            _arr_out = arr_out[:, :, None]
        else:
            _arr_out = arr_out

        ker_h, ker_w = inker_shape = self.inker_shape
        outker_shape = self.outker_shape
        dtype = self.arr_in.dtype
        out_shape = _arr_out.shape

        remove_mean = self.remove_mean
        div_method = self.div_method
        threshold = self.threshold
        stretch = self.stretch

        # input (min/max)
        arr_src = _arr_in[:].copy()

        # ---------------------------------------------------------------------
        # compute corresponding numerator (arr_num) and divisor (arr_div)
        # ---------------------------------------------------------------------
        # -- handle outker_shape=inker_shape (full)
        if outker_shape == inker_shape:
            # -- sum kernel
            in_d = _arr_in.shape[-1]
            kshape = list(inker_shape) + [in_d]
            ker = np.ones(kshape, dtype=dtype)
            size = float(ker.size)

            # -- compute sum-of-square
            arr_sq = arr_src ** 2.
            assert np.isfinite(arr_sq).all()

            arr_ssq = lsum(arr_sq, kshape, mode='valid').astype(dtype)
            assert np.isfinite(arr_ssq).all()

            # -- compute arr_num and arr_div
            # preparation
            ys = inker_shape[0] / 2
            xs = inker_shape[1] / 2
            arr_out_h, arr_out_w, arr_out_d = out_shape[-3:]
            hs = arr_out_h
            ws = arr_out_w

            # compute 'euclidean' (magnitude) divisor (norm = 1)
            if div_method == 'euclidean':
                # with mean substraction
                if remove_mean:
                    arr_sum = lsum(arr_src, kshape, mode='valid').astype(dtype)
                    arr_num = arr_src[ys:ys + hs, xs:xs + ws] \
                            - (arr_sum / size)
                    val = (arr_ssq - (arr_sum ** 2.) / size)
                    # to avoid sqrt of negative numbers
                    np.putmask(val, val < 0, 0)
                    arr_div = np.sqrt(val) + EPSILON

                # without mean substraction
                else:
                    arr_num = arr_src[ys:ys + hs, xs:xs + ws]
                    # arr_ssq should not have any value < 0
                    # however, it can happen (e.g. with fftconvolve)
                    # so we ensure to set these values to 0
                    np.putmask(arr_ssq, arr_ssq < 0., 0.)
                    arr_div = np.sqrt(arr_ssq) + EPSILON

            # or compute 'std' (standard deviation) divisor (var = 1)
            elif div_method == 'std':
                arr_sum = lsum(arr_src, kshape, mode='valid').astype(dtype)
                # with mean substraction
                if remove_mean:
                    arr_num = arr_src[ys:ys + hs, xs:xs + ws] \
                            - (arr_sum / size)

                # without mean substraction
                else:
                    arr_num = arr_src[ys:ys + hs, xs:xs + ws]

                val = (arr_ssq / size - (arr_sum / size) ** 2.)
                # to avoid sqrt of a negative number
                np.putmask(val, val < 0., 0.)
                arr_div = np.sqrt(val) + EPSILON

            else:
                raise ValueError("div_method='%s' not understood" % div_method)

        # ---------------------------------------------------------------------
        # -- handle outker_shape=(0,0) (per depth dim) *NOT TESTED*
        elif outker_shape == (0, 0):
            # -- output shape
            in_h, in_w, in_d = _arr_in.shape[-3:]
            kin_h, kin_w = inker_shape
            arr_out_h = (in_h - kin_h + 1)
            arr_out_w = (in_w - kin_w + 1)
            arr_out_d = in_d
            arr_out_shape = arr_out_h, arr_out_w, arr_out_d

            # -- sum kernel
            ker = np.ones(inker_shape, dtype=dtype)
            size = float(ker.size)

            # -- compute sum-of-square
            arr_sq = arr_src ** 2.
            arr_ssq = lsum(arr_sq, inker_shape + (1,), mode='valid')
            arr_ssq = arr_ssq.astype(dtype)

            # -- compute arr_num and arr_div
            # preparation
            ys = inker_shape[0] / 2
            xs = inker_shape[1] / 2
            arr_out_h, arr_out_w, arr_out_d = out_shape[-3:]
            hs = arr_out_h
            ws = arr_out_w

            def get_arr_sum():
                arr_sum = np.empty(arr_out_shape, dtype=dtype)
                for d in xrange(in_d):
                    slice2d = lsum(arr_src[:, :, d], inker_shape, mode='valid')
                    slice2d = slice2d.astype(dtype)
                    arr_sum[:, :, d] = slice2d
                return arr_sum

            # compute 'euclidean' (magnitude) divisor (norm = 1)
            if div_method == 'euclidean':
                # with mean substraction
                if remove_mean:
                    arr_sum = get_arr_sum()
                    arr_num = arr_src[ys:ys + hs, xs:xs + ws] \
                            - (arr_sum / size)
                    val = (arr_ssq - (arr_sum ** 2.) / size)
                    # to avoid sqrt of a negative number
                    np.putmask(val, val < 0., 0.)
                    arr_div = np.sqrt(val) + EPSILON

                # without mean substraction
                else:
                    arr_num = arr_src[ys:ys + hs, xs:xs + ws]
                    arr_div = np.sqrt(arr_ssq) + EPSILON

            # or compute 'std' (standard deviation) divisor (var = 1)
            elif div_method == 'std':
                arr_sum = get_arr_sum()
                # with mean substraction
                if remove_mean:
                    arr_num = arr_src[ys:ys + hs, xs:xs + ws] \
                            - (arr_sum / size)

                # without mean substraction
                else:
                    arr_num = arr_src[ys:ys + hs, xs:xs + ws]

                val = (arr_ssq / size - (arr_sum / size) ** 2.)
                # to avoid sqrt of a negative number
                np.putmask(val, val < 0., 0.)
                arr_div = np.sqrt(val) + EPSILON
            else:
                raise ValueError("div_method '%s' not understood" % div_method)
        else:
            raise ValueError(
                'inker_shape=%s and outker_shape=%s not understood'
                % (inker_shape, outker_shape)
            )

        # ---------------------------------------------------------------------
        # apply normalization
        # ---------------------------------------------------------------------
        if stretch != 1:
            arr_num *= stretch
            arr_div *= stretch

        # volume threshold
        assert np.isfinite(arr_div).all()
        np.putmask(arr_div, arr_div < (threshold + EPSILON), 1.)

        # output (min/max)
        assert np.isfinite(arr_num).all()
        assert np.isfinite(arr_div).all()
        _arr_out[:] = (arr_num / arr_div)

        if arr_in.ndim == 2:
            _arr_out.shape = _arr_out.shape[:2]

        # -- Contracts: postconditions
        assert_postconditions_on_properties(_arr_in, _arr_out, inker_shape)
        assert_postconditions_on_data(_arr_out)

        return _arr_out
