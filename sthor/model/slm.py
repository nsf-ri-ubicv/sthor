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

        self.n_layers = len(self.description) - 1

        self.in_shape = in_shape

        self.filterbanks = {}

        try:
            self.process = profile(self.process)
        except NameError:
            pass

    def _extract_nb_and_stride(self):
        """this tiny helper function extracts the neighborhood
        parameters (i.e. `inker_shape` for `lnorm`, `filter_shape`
        for `fbcorr` and `ker_shape` for `lpool`) along with the
        striding parameters (here only for `lpool`)"""

        final_list = []
        for layer in self.description:
            for operation in layer:
                if operation[0] == 'lnorm':
                    nbh, nbw = operation[1]['kwargs']['inker_shape']
                    final_list += [(nbh, nbw, 1)]
                elif operation[0] == 'fbcorr':
                    nbh, nbw = operation[1]['initialize']['filter_shape']
                    final_list += [(nbh, nbw, 1)]
                else:
                    nbh, nbw = operation[1]['kwargs']['ker_shape']
                    striding = operation[1]['kwargs']['stride']
                    final_list += [(nbh, nbw, striding)]

        return final_list

    def _layer_output_2Dshape(self,
                              nlayers):
        """this helper function computes the expected shape
        of a feature array after having been processed by
        a predefined number of layers
        """

        # -- simple check on the number of layers
        assert 0 <= nlayers <= self.n_layers

        layer_param = self._extract_nb_and_stride()
        nmax = 1 + 3 * nlayers
        h, w = self.in_shape[:2]

        for parameters in layer_param[:nmax]:
            nbh, nbw, s = parameters
            h = 1 + (h - nbh) / s
            w = 1 + (w - nbw) / s

        return (h, w)

    def rcp_field_central_px_coords(self,
                                    nlayers,
                                    x_coords=-1,
                                    y_coords=-1):
        """Given nlayers (which gives the level at which we look
        at the output of the SLM), and given the coordinates of a
        certain amount of chosen pixel coordinates on the output
        of that layer, this routine will compute the central point
        coordinates of each receptive fields.
        """

        h_max, w_max = self._layer_output_2Dshape(nlayers)

        # -- checks on number of layers
        assert 0 <= nlayers <= self.n_layers

        # -- by default we want to consider all the pixel values
        #    at the layer requested
        if x_coords == -1 and y_coords == -1:
            x_coords, y_coords = np.mgrid[0:h_max:1, 0:w_max:1]
            x_coords = x_coords.ravel()
            y_coords = y_coords.ravel()

        # -- checks on input coordinate arrays
        x_coords = np.array(x_coords).astype(int)
        y_coords = np.array(y_coords).astype(int)
        assert x_coords.ndim == 1
        assert y_coords.ndim == 1
        assert x_coords.size == y_coords.size

        # -- checks on the coordinate ranges
        assert 0 <= x_coords.min()
        assert x_coords.max() < h_max
        assert 0 <= y_coords.min()
        assert y_coords.max() < w_max

        layer_param = self._extract_nb_and_stride()
        nmax = 1 + 3 * nlayers

        for parameters in reversed(layer_param[:nmax]):
            nbh, nbw, s = parameters
            x_coords = nbh / 2 + x_coords * s
            y_coords = nbw / 2 + y_coords * s

        return (x_coords, y_coords)

    def process(self, arr_in):
        """XXX: docstring for process"""

        description = self.description

        assert arr_in.shape == self.in_shape
        input_shape = arr_in.shape
        assert len(input_shape) == 2 or len(input_shape) == 3

        if len(input_shape) == 3:
            tmp_out = arr_in
        elif len(input_shape) == 2:
            tmp_out = arr_in[..., np.newaxis]
        else:
            raise ValueError("The input array should be 2D or 3D")

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
