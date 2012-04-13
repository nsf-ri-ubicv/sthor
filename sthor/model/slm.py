"""Sequential Layered Model"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np
import numexpr as ne

from sthor.operation import lcdnorm3
from sthor.operation import fbcorr3
from sthor.operation import lpool3

from pprint import pprint

DTYPE = np.float32


class SequentialLayeredModel(object):

    def __init__(self, in_shape, description):
        """XXX: docstring for __init__"""

        pprint(description)
        self.description = description

        self.in_shape = in_shape

        self.filterbanks = {}

        try:
            self.process = profile(self.process)
        except NameError:
            pass

    def get_n_layers(self):

        if not hasattr(self, '_n_layers'):

            nlayers = len(self.description)
            self._n_layers = nlayers
            return self._n_layers

        else:

            return self._n_layers

    def get_ops_nbh_nbw_stride(self):

        if not hasattr(self, '_ops_nbh_nbw_stride'):

            to_return = []

            for layer in self.description:
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

            self._ops_nbh_nbw_stride = to_return
            return self._ops_nbh_nbw_stride

        else:

            return self._ops_nbh_nbw_stride

    def get_receptive_field_shape(self):

        if not hasattr(self, '_receptive_field_shape'):

            ops_param = self.get_ops_nbh_nbw_stride()

            # -- in this case the SLM does nothing so
            #    it is the identity
            if len(ops_param) == 0:
                self._receptive_field_shape = (1, 1)
                return self._receptive_field_shape

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
                self._receptive_field_shape = (out_h, out_w)
                return self._receptive_field_shape

        else:

            return self._receptive_field_shape

    def process(self, arr_in, with_aprons=True,
                process_whole_image=False):
        """XXX: docstring for process"""

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

        # -- first we initialize a list of 3-tuple
        h, w = tmp_out.shape[:2]
        Y, X = np.mgrid[:h, :w]
        maps = [(tmp_out, X, Y)]

        nbh_nbw_stride_l = self.get_ops_nbh_nbw_stride()

        for layer_idx, layer_desc in enumerate(description):

            for op_idx, (op_name, op_params) in enumerate(layer_desc):

                tmp_in = tmp_out

                kwargs = op_params['kwargs']

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
