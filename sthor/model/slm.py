"""Sequential Layered Model"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows

from sthor.operation import lcdnorm3
from sthor.operation import fbcorr3
from sthor.operation import lpool3

from sthor.util.pad import filter_pad2d

from pprint import pprint

DTYPE = np.float32

# ----------------
# Helper functions
# ----------------

def _get_ops_nbh_nbw_stride(description):

    to_return = []

    for layer in description:
        for operation in layer:

            if operation[0] == 'lnorm':
                nbh, nbw = operation[1]['kwargs']['inker_shape']
                to_return += [('lnorm', nbh, nbw, 1)]

            elif operation[0] == 'fbcorr':
                nbh, nbw = operation[1]['initialize']['filter_shape']
                to_return += [('fbcorr', nbh, nbw, 1)]

            else:
                nbh, nbw = operation[1]['kwargs']['ker_shape']
                striding = operation[1]['kwargs']['stride']
                to_return += [('lpool', nbh, nbw, striding)]

    return to_return


def _get_receptive_field_shape(description):

    ops_param = _get_ops_nbh_nbw_stride(description)

    # -- in this case the SLM does nothing so
    #    it is the identity
    if len(ops_param) == 0:
        return (1, 1)

    # -- otherwise we processed the image with some
    #    operations and we go backwards to estimate
    #    the global receptive field of the SLM
    else:
        # -- we start from a single pixel-wide shape
        out_h, out_w = 1, 1

        # -- then we compute what was the shape before
        #    we applied the preceding operation
        for _, nbh, nbw, s in reversed(ops_param):
            in_h = (out_h - 1) * s + nbh
            in_w = (out_w - 1) * s + nbw
            out_h, out_w = in_h, in_w
        return (out_h, out_w)

# --------------
# SLM base class
# --------------

class SequentialLayeredModel(object):

    def __init__(self, in_shape, description):
        """XXX: docstring for __init__"""

        pprint(description)
        self.description = description
        self.n_layers = len(description)

        self.in_shape = in_shape

        self.filterbanks = {}

        self.ops_nbh_nbw_stride = _get_ops_nbh_nbw_stride(description)
        self.receptive_field_shape = _get_receptive_field_shape(description)

        try:
            self.process = profile(self.process)
        except NameError:
            pass


    def process(self, arr_in,
                pad_apron=False,
                interleave_stride=False):
        """XXX: docstring for process"""

        rcpt_field = self.receptive_field_shape
        description = self.description
        input_shape = arr_in.shape

        assert input_shape[:2] == self.in_shape
        assert len(input_shape) == 2 or len(input_shape) == 3

        if len(input_shape) == 3:
            tmp_out = arr_in
        elif len(input_shape) == 2:
            tmp_out = arr_in[..., np.newaxis]
        else:
            raise ValueError("The input array should be 2D or 3D")

        # -- first we initialize some variables to be used in
        #    the processing of the feature maps
        h, w = self.in_shape
        Y, X = np.mgrid[:h, :w]

        tmp_out_l = [(tmp_out, X, Y)]

        nbh_nbw_stride_l = self.ops_nbh_nbw_stride

        # -- loop over all the SLM operations in order
        op_counter = 0

        for layer_idx, layer_desc in enumerate(description):
            for op_idx, (op_name, op_params) in enumerate(layer_desc):

                kwargs = op_params['kwargs']

                tmp_l = []

                _, nbh, nbw, stride = nbh_nbw_stride_l[op_counter]

                for arr, X, Y in tmp_out_l:
                    tmp_l += self._process_one_op(arr, X, Y,
                                       layer_idx, op_idx, kwargs,
                                       op_params,
                                       op_name, nbh, nbw, stride,
                                       pad_apron=pad_apron,
                                       interleave_stride=interleave_stride)

                tmp_out_l = tmp_l
                op_counter += 1

        # -- now we need to possibly interleave the arrays
        #    in ``tmp_out_l``
        if interleave_stride:

            out_shape = (h, w, tmp_out_l[0][0].shape[-1])
            arr_out = np.empty(out_shape, dtype=arr_in.dtype)
            Y_ref, X_ref = np.mgrid[:h, :w]
            Y_int, X_int = np.zeros((h, w), dtype=np.int), \
                           np.zeros((h, w), dtype=np.int)

            if pad_apron:

                for arr, Xc, Yc in tmp_out_l:

                    anchor_h, anchor_w = Yc[0, 0], Xc[0, 0]
                    stride_h = Yc[1, 0] - Yc[0, 0]
                    stride_w = Xc[0, 1] - Xc[0, 0]

                    arr_out[anchor_h::stride_h, anchor_w::stride_w, ...] = arr
                    X_int[anchor_h::stride_h, anchor_w::stride_w] = Xc
                    Y_int[anchor_h::stride_h, anchor_w::stride_w] = Yc

                assert (X_int == X_ref).all()
                assert (Y_int == Y_ref).all()

                return arr_out

            else:

                X_int = filter_pad2d(X_int[..., np.newaxis], rcpt_field,
                                     reverse_padding=True).squeeze()
                Y_int = filter_pad2d(Y_int[..., np.newaxis], rcpt_field,
                                     reverse_padding=True).squeeze()
                X_ref = filter_pad2d(X_ref[..., np.newaxis], rcpt_field,
                                     reverse_padding=True).squeeze()
                Y_ref = filter_pad2d(Y_ref[..., np.newaxis], rcpt_field,
                                     reverse_padding=True).squeeze()
                arr_out = filter_pad2d(arr_out, rcpt_field,
                                       reverse_padding=True)

                offset_Y = Y_ref.min()
                offset_X = X_ref.min()

                for arr, Xc, Yc in tmp_out_l:

                    anchor_h, anchor_w = Yc[0, 0] - offset_Y, \
                                         Xc[0, 0] - offset_X
                    stride_h = Yc[1, 0] - Yc[0, 0]
                    stride_w = Xc[0, 1] - Xc[0, 0]

                    arr_out[anchor_h::stride_h, anchor_w::stride_w, ...] = arr
                    X_int[anchor_h::stride_h, anchor_w::stride_w] = Xc
                    Y_int[anchor_h::stride_h, anchor_w::stride_w] = Yc

                assert (X_int == X_ref).all()
                assert (Y_int == Y_ref).all()

                return arr_out

        else:

            assert len(tmp_out_l) == 1
            arr_out, _, _ = tmp_out_l[0]

            return arr_out

    def _process_one_op(self, arr, X, Y,
                        layer_idx, op_idx, kwargs,
                        op_params,
                        op_name, nbh, nbw, stride,
                        pad_apron=False,
                        interleave_stride=False):

        out_l = []

        # -- here we compute the pixel coordinates of
        #    the central pixel in a patch
        hc, wc = nbh / 2, nbw / 2

        if pad_apron:

            arr = filter_pad2d(arr, (nbh, nbw))
            X = np.squeeze(filter_pad2d(X[..., np.newaxis], (nbh, nbw),
                                        constant=-1))
            Y = np.squeeze(filter_pad2d(Y[..., np.newaxis], (nbh, nbw),
                                        constant=-1))

        if interleave_stride:

            for i in xrange(stride):
                for j in xrange(stride):

                    arr_out_ij = self._get_feature_map(arr[i::, j::, ...],
                                                  layer_idx, op_idx, kwargs,
                                                  op_params, op_name)
                    X_out_ij = view_as_windows(X[i::, j::],
                                               (nbh, nbw))[::stride, ::stride,
                                                           hc, wc]
                    Y_out_ij = view_as_windows(Y[i::, j::],
                                               (nbh, nbw))[::stride, ::stride,
                                                           hc, wc]
                    out_l += [(arr_out_ij, X_out_ij, Y_out_ij)]
        else:

            arr_out = self._get_feature_map(arr, layer_idx, op_idx, kwargs,
                                            op_params, op_name)
            X_out = view_as_windows(X, (nbh, nbw))[::stride, ::stride, hc, wc]
            Y_out = view_as_windows(Y, (nbh, nbw))[::stride, ::stride, hc, wc]
            out_l += [(arr_out, X_out, Y_out)]

        return out_l


    def _get_feature_map(self, tmp_in,
                         layer_idx, op_idx, kwargs,
                         op_params, op_name):

        if op_name == 'lnorm':

            inker_shape = kwargs['inker_shape']
            outker_shape = kwargs['outker_shape']
            remove_mean = kwargs['remove_mean']
            stretch = kwargs['stretch']
            threshold = kwargs['threshold']

            # SLM PLoS09 / FG11 constraints:
            assert inker_shape == outker_shape

            tmp_out = lcdnorm3(tmp_in, inker_shape,
                               contrast=remove_mean,
                               stretch=stretch,
                               threshold=threshold)

        elif op_name == 'fbcorr':

            max_out = kwargs['max_out']
            min_out = kwargs['min_out']

            fbkey = layer_idx, op_idx
            if fbkey not in self.filterbanks:
                initialize = op_params['initialize']
                if isinstance(initialize, np.ndarray):
                    fb = initialize
                    if len(fb.shape) == 3:
                        fb = fb[..., np.newaxis]
                else:
                    filter_shape = list(initialize['filter_shape'])
                    generate = initialize['generate']
                    n_filters = initialize['n_filters']

                    fb_shape = [n_filters] + filter_shape + [tmp_in.shape[-1]]

                    # generate filterbank data
                    method_name, method_kwargs = generate
                    assert method_name == 'random:uniform'

                    rseed = method_kwargs.get('rseed', None)
                    rng = np.random.RandomState(rseed)

                    fb = rng.uniform(size=fb_shape)

                    for fidx in xrange(n_filters):
                        filt = fb[fidx]
                        # zero-mean, unit-l2norm
                        filt -= filt.mean()
                        filt_norm = np.linalg.norm(filt)
                        assert filt_norm != 0
                        filt /= filt_norm
                        fb[fidx] = filt

                fb = np.ascontiguousarray(np.rollaxis(fb, 0, 4)).astype(DTYPE)
                self.filterbanks[fbkey] = fb
                print fb.shape

            fb = self.filterbanks[fbkey]

            # -- filter
            assert tmp_in.dtype == np.float32
            tmp_out = fbcorr3(tmp_in, fb)

            # -- activation
            min_out = -np.inf if min_out is None else min_out
            max_out = +np.inf if max_out is None else max_out
            # insure that the type is right before calling numexpr
            min_out = np.array([min_out], dtype=tmp_in.dtype)
            max_out = np.array([max_out], dtype=tmp_in.dtype)
            # call numexpr
            tmp_out = ne.evaluate('where(tmp_out < min_out, min_out, tmp_out)')
            tmp_out = ne.evaluate('where(tmp_out > max_out, max_out, tmp_out)')
            assert tmp_out.dtype == tmp_in.dtype


        elif op_name == 'lpool':

            ker_shape = kwargs['ker_shape']
            order = kwargs['order']
            stride = kwargs['stride']

            tmp_out = lpool3(tmp_in, ker_shape, order=order, stride=stride)

        else:
            raise ValueError("operation '%s' not understood" % op_name)

        assert tmp_out.dtype == tmp_in.dtype
        assert tmp_out.dtype == np.float32

        return tmp_out
