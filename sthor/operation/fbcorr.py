"""
Filterbank Correlation Operation.

Easy-to-understand (but slow) naive numpy implementation.
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#
# License: Proprietary


# XXX: FBCorrScipyNaive should avoid duplicated code in its methods

# -- Imports
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import correlate as ndicorr
from pythor3.operation.fbcorr_.plugins import FBCorrPlugin

# -- Default parameters
from pythor3.operation.fbcorr_ import (
    DEFAULT_STRIDE,
    DEFAULT_MIN_OUT,
    DEFAULT_MAX_OUT,
    DEFAULT_MODE)


class FBCorrScipyNaive(FBCorrPlugin):

    def __call__(self, arr_in, arr_fb, arr_out,
                 min_out=DEFAULT_MIN_OUT, max_out=DEFAULT_MAX_OUT,
                 stride=DEFAULT_STRIDE, mode=DEFAULT_MODE):
        """XXX: doctring"""

        _arr_in = arr_in
        _arr_fb = arr_fb
        _arr_out = arr_out

        dtype = arr_in.dtype

        in_h, in_w, in_d = _arr_in.shape
        fb_n, fb_h, fb_w, fb_d = _arr_fb.shape

        if fb_h == 1:
            slice_h = slice(None, None, None)
        else:
            h_fb_h = int(fb_h / 2)
            slice_h = slice(h_fb_h, -h_fb_h, None)

        if fb_w == 1:
            slice_w = slice(None, None, None)
        else:
            h_fb_w = int(fb_w / 2)
            slice_w = slice(h_fb_w, -h_fb_w, None)

        # -- handle one dot product
        if _arr_in.shape == _arr_fb.shape[1:]:
            # for each filter
            for i in xrange(fb_n):
                filt = _arr_fb[i]
                _arr_out[:, :, i] = np.dot(_arr_in[:].ravel(), filt.ravel())

        # -- handle multiple dot products
        else:
            # hardcoded heuristic for performance
            # use fftconvolve when the filters are > 13
            if max(fb_h, fb_w) > 13:
                for odi in xrange(fb_n):
                    # reshape filter
                    filt = _arr_fb[odi].ravel()
                    filt = np.flipud(filt)
                    filt = np.reshape(filt, (fb_h, fb_w, fb_d))
                    # process
                    result = fftconvolve(_arr_in[:], filt, mode=mode).real
                    result = result.astype(dtype)
                    out = np.reshape(result, result.shape[0:2])
                    # store
                    _arr_out[:, :, odi] = out

            # otherwise use ndimage.correlate
            else:
                _arr_out[:] = 0
                for odi in xrange(fb_n):
                    filt = _arr_fb[odi]
                    # process
                    for fdi in xrange(filt.shape[2]):
                        ai = _arr_in[:, :, fdi]
                        af = filt[:, :, fdi]
                        partial = ndicorr(ai, af, mode='constant')
                        if mode is 'valid':
                            _arr_out[:, :, odi] += partial[slice_h, slice_w]
                        elif mode is 'same':
                            _arr_out[:, :, odi] += partial
                        else:
                            raise NotImplementedError("Unsupported fbcorr mode '%s'" % mode)
                        #_arr_out[:, :, odi] += correlate2d(ai, af, mode=mode)

        # -- output (clip)
        if min_out is not None:
            np.putmask(_arr_out, _arr_out < min_out, min_out)
        if max_out is not None:
            np.putmask(_arr_out, _arr_out > max_out, max_out)

        return _arr_out
