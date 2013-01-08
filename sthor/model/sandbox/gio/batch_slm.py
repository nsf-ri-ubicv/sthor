"""Hierarchical Generative-Discriminative Model"""

# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import time
import numpy as np
import numexpr as ne

from pprint import pprint

from sthor.operation.sandbox import lcdnorm5
from sthor.operation.sandbox import fbcorr5
from sthor.operation.sandbox import lpool5

DTYPE = np.float32
MAX_MEM_GB = 20.
PARTITION_SIZE = 35

# ----------------
# Helper functions
# ----------------


def _get_rf_shape_stride_in_interval(slm_description,
                                     ly_ini, op_ini,
                                     ly_end, op_end,
                                     depth_ini):

    # -- P.S.: the inverval determined by ly_ini, op_ini, 
    #          ly_end, and op_end is inclusive.

    # -- asserts there is at least one operation in the interval
    assert len(slm_description) >= ly_end >= ly_ini >= 0
    assert op_ini >= 0
    if ly_end == ly_ini:
        assert len(slm_description[ly_end]) >= op_end >= op_ini 

    # -- uppermost unit shape and stride size
    out_h, out_w, out_d, out_s = 1, 1, depth_ini, 1

    for ly_idx in reversed(xrange(ly_ini, ly_end + 1)):
        if ly_idx == ly_ini:
            it_op_ini = op_ini
        else:
            it_op_ini = 0
        if ly_idx == ly_end:
            it_op_end = op_end + 1
        else:
            it_op_end = len(slm_description[ly_idx])


        for op_idx in reversed(xrange(it_op_ini, it_op_end)):

            operation = slm_description[ly_idx][op_idx]

            if operation[0] == 'lnorm':
                nbh, nbw = operation[1]['kwargs']['inker_shape']
                s = 1
            elif operation[0] == 'fbcorr':
                nbh, nbw = operation[1]['initialize']['filter_shape']
                s = 1
                out_d = operation[1]['initialize']['n_filters']
            elif operation[0] == 'lpool':
                nbh, nbw = operation[1]['kwargs']['ker_shape']
                s = operation[1]['kwargs']['stride']

            # -- compute what was the shape before applied the 
            #    preceding operation
            in_h = (out_h - 1) * s + nbh
            in_w = (out_w - 1) * s + nbw
            out_h, out_w, out_s = in_h, in_w, out_s * s

    return (out_h, out_w, out_d, out_s)

def _get_shape_stride_by_layer(slm_description, in_shape):

    to_return = []
    tmp_shape = in_shape

    for layer_idx, layer_desc in enumerate(slm_description):

        # -- get receptive field layer-wise
        rfh, rfw, rfd, rfs = _get_rf_shape_stride_in_interval(
                             slm_description,
                             layer_idx, 0,
                             layer_idx, len(layer_desc)-1,
                             tmp_shape[-1])

        tmp_shape = ((tmp_shape[0] - rfh) / rfs + 1,
                     (tmp_shape[1] - rfw) / rfs + 1,
                     rfd)

        to_return += [(tmp_shape[0], tmp_shape[1], rfh, rfw, rfd, rfs)]

    return to_return


# --------------
# SLM base class
# --------------

class BatchSequentialLayeredModel(object):

    def __init__(self, in_shape, description, filterbanks=None,
                 fb_mean=None, fb_std=None, fb_proj=None,
                 fb_intercept=None, fb_offset=None):

        pprint(description)

        self.description = description

        self.in_shape = in_shape

        self.n_layers = len(description)

        self.shape_stride_by_layer = _get_shape_stride_by_layer(description,
                                                                in_shape)

        if filterbanks is not None:
            self.filterbanks = filterbanks

            self.fb_mean = fb_mean
            self.fb_std = fb_std
            self.fb_proj = fb_proj
            self.fb_intercept = fb_intercept
            self.fb_offset = fb_offset
        else:
            if self.n_layers in (3, 4):
                self.filterbanks = [[]] # layer 0 has no filter bank
            else:
                self.filterbanks = []

            self.fb_mean = None
            self.fb_std = None
            self.fb_proj = None
            self.fb_intercept = None
            self.fb_offset = None

            # -- set filters
            self._fit()

        # -- this is the working array, that will be used throughout object
        #    methods. its purpose is to avoid a large memory footprint due
        #    to modularization.
        self.arr_w = None

    def _fit(self):

        desc = self.description

        # -- filter bank must be empty
        assert len(self.filterbanks) in (0,1)

        for layer_idx in xrange(len(self.filterbanks), len(desc)):

            if layer_idx == 0:
                n_f_in = self.in_shape[-1]
            else:
                n_f_in = self.shape_stride_by_layer[layer_idx-1][4]

            l_desc = desc[layer_idx]
            op_name, f_desc = l_desc[0] # fg11-type slm

            if op_name != 'fbcorr':
                self.filterbanks += []
                continue

            f_init = f_desc['initialize']
            f_shape = f_init['filter_shape'] + (n_f_in,)
            n_filters = f_init['n_filters']

            generate = f_init['generate']
            method_name, method_kwargs = generate
            assert method_name == 'random:uniform'

            rseed = method_kwargs.get('rseed', None)
            self.rng_f = np.random.RandomState(rseed)

            fb_shape = (n_filters,) + f_shape

            fb = self.rng_f.uniform(low=-1.0, high=1.0, size=fb_shape)

            # -- zero-mean, unit-l2norm
            for f_idx in xrange(n_filters):
                filt = fb[f_idx]
                filt -= filt.mean()
                filt_norm = np.linalg.norm(filt)
                assert filt_norm != 0
                filt /= filt_norm
                fb[f_idx] = filt

            fb = np.ascontiguousarray(np.rollaxis(fb, 0, 4)).astype(DTYPE)
            self.filterbanks += [fb.copy()]

        assert len(self.filterbanks) == len(desc)

        return


    def transform(self, arr_in):

        input_shape = arr_in.shape
        n_imgs = input_shape[0]
        assert input_shape[1:] == self.in_shape
        assert len(input_shape) == 3 or len(input_shape) == 4

        if len(input_shape) == 3:
            arr_in = arr_in[..., np.newaxis]

        desc = self.description

        # assert that the model is ready to be used
        assert len(self.filterbanks) == self.n_layers

        # -- amount of memory that the output array is going to demand
        mem_arr_out = self._mem_layer(self.n_layers-1, n_imgs)

        if mem_arr_out > MAX_MEM_GB:
            raise ValueError("size of output array is greater than %s; " +
                             "the maximum allowed" % MAX_MEM_GB)

        # -- determine if and how arr_in is partitioned
        n_partitions, partitions = self._get_partitions(0, len(desc)-1, n_imgs)

        for part_idx, (part_init, part_end) in enumerate(partitions):

            t1 = time.time()

            if n_partitions == 1:
                self.arr_w = arr_in
                self.arr_w.shape = (n_imgs, 1) + self.arr_w.shape[1:]
            else:
                if part_idx == 0:
                    # -- initialize arr_out
                    [l_h, l_w, _, _, l_d, _] = \
                                    self.shape_stride_by_layer[self.n_layers-1]

                    arr_out = np.empty((n_imgs, 1, l_h, l_w, l_d), dtype=DTYPE)

                self.arr_w = arr_in[part_init:part_end]
                self.arr_w.shape = (part_end - part_init, 1) + \
                                    self.arr_w.shape[1:]

            for layer_idx, l_desc in enumerate(desc):
                self._transform_layer(layer_idx, l_desc)

            if n_partitions == 1:
                arr_out = self.arr_w
            else:
                arr_out[part_init:part_end] = self.arr_w

            t_elapsed = time.time() - t1
            print 'Partiton %d out of %d processed in %g seconds...' % (
                  part_idx + 1, n_partitions, t_elapsed)

        self.arr_w = None
        return arr_out

    def _get_fb_data(self, layer_idx, fb_data):

        if fb_data is not None:
            return fb_data[layer_idx]
        else:
            return None

    # -- transform self.arr_w according to slm layer description.
    def _transform_layer(self, layer_idx, l_desc):

        if layer_idx == 0:
            l_h, l_w, l_d = self.in_shape
        else:
            [l_h, l_w, _, _, l_d, _] = self.shape_stride_by_layer[layer_idx-1]

        # -- assert layer input shape
        assert self.arr_w.shape[2:] == (l_h, l_w, l_d)

        for op_idx, (op_name, op_params) in enumerate(l_desc):
            kwargs = op_params['kwargs']

            if op_name == 'fbcorr':
                fb = self.filterbanks[layer_idx]

                f_mean = self._get_fb_data(layer_idx, self.fb_mean)
                f_std = self._get_fb_data(layer_idx, self.fb_std)
                f_proj = self._get_fb_data(layer_idx, self.fb_proj)
                f_intercept = self._get_fb_data(layer_idx, self.fb_intercept)
                f_offset = self._get_fb_data(layer_idx, self.fb_offset)

            else:
                fb = None

                f_mean = None
                f_std = None
                f_proj = None
                f_intercept = None
                f_offset = None

            self.arr_w = self._process_one_op(op_name, kwargs, self.arr_w, fb,
                                              f_mean, f_std, f_proj,
                                              f_intercept, f_offset)

        return


    def _process_one_op(self, op_name, kwargs, arr_in, fb=None,
                        f_mean=None, f_std=None, f_proj=None,
                        f_intercept=None, f_offset=None):

        if op_name == 'lnorm':

            inker_shape = kwargs['inker_shape']
            outker_shape = kwargs['outker_shape']
            remove_mean = kwargs['remove_mean']
            stretch = kwargs['stretch']
            threshold = kwargs['threshold']

            # SLM PLoS09 / FG11 constraints:
            assert inker_shape == outker_shape

            tmp_out = lcdnorm5(arr_in, inker_shape,
                               contrast=remove_mean,
                               stretch=stretch,
                               threshold=threshold)

        elif op_name == 'fbcorr':

            assert fb is not None

            max_out = kwargs['max_out']
            min_out = kwargs['min_out']

            # -- filter
            assert arr_in.dtype == np.float32

            tmp_out = fbcorr5(arr_in, fb,
                              f_mean=f_mean, f_std=f_std, f_proj=f_proj,
                              f_intercept=f_intercept, f_offset=f_offset)

            # -- activation
            min_out = -np.inf if min_out is None else min_out
            max_out = +np.inf if max_out is None else max_out
            # insure that the type is right before calling numexpr
            min_out = np.array([min_out], dtype=arr_in.dtype)
            #min_out2 = np.array([0.0], dtype=arr_in.dtype)
            max_out = np.array([max_out], dtype=arr_in.dtype)
            # call numexpr
            #tmp_out = ne.evaluate('where(tmp_out < min_out, min_out2, tmp_out)')
            tmp_out = ne.evaluate('where(tmp_out < min_out, min_out, tmp_out)')
            tmp_out = ne.evaluate('where(tmp_out > max_out, max_out, tmp_out)')
            assert tmp_out.dtype == arr_in.dtype

        elif op_name == 'lpool':

            ker_shape = kwargs['ker_shape']
            order = kwargs['order']
            stride = kwargs['stride']

            tmp_out = lpool5(arr_in, ker_shape, order=order, stride=stride)

        else:
            raise ValueError("operation '%s' not understood" % op_name)

        assert tmp_out.dtype == arr_in.dtype
        assert tmp_out.dtype == np.float32

        return tmp_out


    def _mem_layer(self, layer_idx, n_imgs):

        [l_h, l_w, _, _, l_d, _] = self.shape_stride_by_layer[layer_idx]
        l_shape = (n_imgs, l_h, l_w, l_d)

        return self._mem_shape(l_shape)


    def _mem_shape(self, shape):

        mem = np.ndarray((1), dtype=DTYPE).nbytes
        for d in shape:
            mem *= d
        # -- transform max_mem to gigabytes
        return float(mem) / 1024**3


    # -- determine if and how a hypothetical input array is going to be
    #    partitioned because its transformaiton exceeds memory limit
    def _get_partitions(self, layer_init, layer_end, n_imgs):

        part_size = PARTITION_SIZE
        n_partitions = int(n_imgs / float(part_size) + 1)

        partitions = []
        part_init = 0

        # -- compute partition indices
        for part_i in xrange(n_partitions):
            partitions += [[part_init, part_init +  part_size]]
            part_init += part_size

        assert n_partitions == len(partitions)

        if n_partitions > 0:
            partitions[n_partitions-1][1] = n_imgs

        return n_partitions, partitions