"""Hierarchical Generative-Discriminative Model"""

# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os
import time
import numpy as np
import numexpr as ne

from skimage.util.shape import view_as_windows

from sthor.operation.sandbox import lcdnorm5
from sthor.operation.sandbox import fbcorr5
from sthor.operation.sandbox import lpool5

from pprint import pprint

DTYPE = np.float32
MAX_MEM_GB = 2.

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
    tmp_shape = in_shape + (1,)

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

class HierarchicalGenDiscModel(object):

    """Hierarchical Generative and/or Discriminative Model (HGDM - just a name)

    TODO: docstring.

    Parameters
    ----------
    in_shape : shape of arrays to be considered as input for the model.

    slm_description : slm-type model description.

    hgdm_description : additional description related to HGDM. For each layer,
        it specifies if the filters are tied (like SLM) or tiled and what  
        technique is going to be used to learn the filters (e.g., k-means, 
        PCA or PLS)

    model_path : path where fitlers and neural images are going to be stored.

    model_prefix : model identifer to be considered as prefix in the file
        names.
    """

    def __init__(self, in_shape, slm_description, hgdm_description, 
                 model_path, model_prefix):

        pprint(slm_description)
        pprint(hgdm_description)

        self.slm_description = slm_description
        self.hgdm_description = hgdm_description

        self.in_shape = in_shape
        self.model_path = model_path
        self.model_prefix = model_prefix

        self.n_layers = len(slm_description)
        assert len(slm_description) == len(hgdm_description)

        self.filterbanks = [[]] # layer 0 has no filter bank
        self.shape_stride_by_layer = _get_shape_stride_by_layer(slm_description,
                                                                in_shape)

        self._load_filters()

        # -- this is the working array, that will be used throughout object
        #    methods. its purpose is to avoid a large memory footprint due
        #    to modularization.
        self.arr_w = None

    # -- sequentially learn filters of each layer whenever necessary (i.e.,
    #    if they were not already learned.
    def fit(self, arr_in, y=None):

        input_shape = arr_in.shape
        n_imgs = input_shape[0]
        assert input_shape[1:3] == self.in_shape
        assert len(input_shape) == 3 or len(input_shape) == 4

        slm_desc = self.slm_description
        hgdm_desc = self.hgdm_description

        # -- layer from which the learning process is going to start
        layers_fb = len(self.filterbanks)
        assert layers_fb > 0 # -- there must be at least a null one

        if layers_fb in (1,2,3):

            if layers_fb == 1:
                if len(input_shape) == 3:
                    self.arr_w = arr_in[..., np.newaxis]
                else:
                    self.arr_w = arr_in

                # -- adds one dimension to the input array to cope with the
                # -- 5d operation functions
                self.arr_w.shape = (n_imgs, 1) + self.arr_w.shape[1:]
            else:
                self.arr_w = None

            for layer_idx in xrange(layers_fb, len(slm_desc)):

                fb = self._learn_filters(layer_idx, y)
                self.filterbanks += [fb.copy()]

        assert len(self.filterbanks) == len(slm_desc) == len(hgdm_desc)

        return


    def transform(self, arr_in):

        input_shape = arr_in.shape
        n_imgs = input_shape[0]
        assert input_shape[1:3] == self.in_shape
        assert len(input_shape) == 3 or len(input_shape) == 4

        if len(input_shape) == 3:
            arr_in = arr_in[..., np.newaxis]

        slm_desc = self.slm_description
        hgdm_desc = self.hgdm_description

        # assert that the model is ready to be used
        assert len(self.filterbanks) == self.n_layers

        # -- amount of memory that the output array is going to demand
        mem_arr_out = self._mem_layer(self.n_layers-1, n_imgs)

        if mem_arr_out > MAX_MEM_GB:
            raise ValueError("size of output array is greater than '%s'; " +
                             "the maximum allowed" % MAX_MEM_GB)

        # -- determine if and how arr_in is partitioned
        n_partitions, partitions = self._get_partitions(0, len(slm_desc)-1,
                                                        n_imgs)

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

            for layer_idx, (slm_l_desc, hgdm_l_desc) in \
                                        enumerate(zip(slm_desc, hgdm_desc)):

                if layer_idx > 0:
                    hgdm_l_arch = hgdm_l_desc['architecture']
                else:
                    hgdm_l_arch = None

                self._transform_layer(layer_idx, slm_l_desc, hgdm_l_arch)

            if n_partitions == 1:
                arr_out = self.arr_w
            else:
                arr_out[part_init:part_end] = self.arr_w

            t_elapsed = time.time() - t1
            print 'Partiton %d out of %d processed in %g seconds...' % (
                  part_idx + 1, n_partitions, t_elapsed)

        self.arr_w = None
        return arr_out

    # -- learn filters and apply model to the input images
    def fit_transform(self, arr_in, y=None):

        n_imgs = arr_in.shape[0]
        self.fit(arr_in, y)

        # -- amount of memory that the output array is going to demand
        mem_arr_out = self._mem_layer(self.n_layers-1, n_imgs)

        if mem_arr_out > MAX_MEM_GB:

            pprint("size of output array is greater than '%s'; " +
                   "the maximum allowed. filters were learned, but" +
                   "input array was not transformed" % MAX_MEM_GB)
        else:
            # -- if we have the penultimate layer output in disk, we use this
            #    data and only process last layer

            layer_idx = self.n_layers - 1
            slm_l_desc = self.slm_description[layer_idx]
            hgdm_l_desc = self.hgdm_description[layer_idx]
            hgdm_l_arch = hgdm_l_desc['architecture']

            input_fnames, partitions = \
                         self._get_neural_imgs_fnames(layer_idx, 'read', n_imgs)
            n_partitions = len(input_fnames)

            if n_partitions == 0:
                arr_out = self.transform(arr_in)
            elif n_partitions == 1:
                self.arr_w = np.load(input_fnames[0])
                self._transform_layer(layer_idx, slm_l_desc, hgdm_l_arch)
                arr_out = self.arr_w
            else:
                # -- initialize arr_out
                [l_h, l_w, _, _, l_d, _] = \
                                self.shape_stride_by_layer[layer_idx]
                arr_out = np.empty((n_imgs, 1, l_h, l_w, l_d), dtype=DTYPE)

                for fname, (part_init, part_end) in zip(input_fnames, 
                                                        partitions):

                    self.arr_w = np.load(fname)

                    n_imgs_part = part_end - part_init
                    assert n_imgs_part == self.arr_w.shape[0]

                    self._transform_layer(layer_idx, slm_l_desc, hgdm_l_arch)

                    arr_out[part_init:part_end] = self.arr_w

        self.arr_w = None
        return arr_out

    # -- load existing filters according to model_prefix and hgdm_description
    def _load_filters(self):

        slm_desc = self.slm_description
        hgdm_desc = self.hgdm_description

        for layer_idx, (slm_l_desc, hgdm_l_desc) in enumerate(
                                                    zip(slm_desc, hgdm_desc)):
            if layer_idx == 0:
                continue

            filter_fname = self._get_filter_fname(layer_idx)
            
            if os.path.exists(filter_fname):
                fb = np.load(filter_fname)
            else:
                break

            f_init = slm_l_desc[0][1]['initialize']
            f_shape = f_init['filter_shape']
            hgdm_l_arch = hgdm_l_desc['architecture']

            n_f_in = self.shape_stride_by_layer[layer_idx-1][4]
            n_f_out = self.shape_stride_by_layer[layer_idx][4]
            assert n_f_out == f_init['n_filters']        


            if hgdm_l_arch == 'tiled':
                n_f_h, n_f_w = self.shape_stride_by_layer[layer_idx][:2]
            else:
                n_f_h, n_f_w = (1,1)

            fb_shape = (n_f_h, n_f_w, n_f_out) + f_shape + (n_f_in,)
            assert fb.shape == fb_shape

            self.filterbanks += [fb.copy()]


    def _learn_filters(self, layer_idx, y):

        slm_l_desc = self.slm_description[layer_idx]
        hgdm_l_desc = self.hgdm_description[layer_idx]
        learn_algo = hgdm_l_desc['f_learn_algo']
        hgdm_l_arch = hgdm_l_desc['architecture']

        f_desc = slm_l_desc[0][1] # fg11-type slm
        f_init = f_desc['initialize']
        f_shape = f_init['filter_shape']

        n_f_in = self.shape_stride_by_layer[layer_idx-1][4]
        n_f_out = self.shape_stride_by_layer[layer_idx][4]
        assert n_f_out == f_init['n_filters']        

        if hgdm_l_arch == 'tiled':
            n_f_h, n_f_w = self.shape_stride_by_layer[layer_idx][:2]
        else:
            n_f_h, n_f_w = (1,1)

        fb_shape = (n_f_h, n_f_w, n_f_out) + f_shape + (n_f_in,)
        fb = np.empty(fb_shape, dtype=DTYPE)

        # -- if learn_algo is slm, than self.arr_w has no effect; we just work 
        #    with the associated shapes
        if learn_algo == 'slm':

            generate = f_init['generate']
            method_name, method_kwargs = generate
            assert method_name == 'random:uniform'

            rseed = method_kwargs.get('rseed', None)
            rng = np.random.RandomState(rseed)

            fb = rng.uniform(size=fb_shape)

            for t_y in xrange(n_f_h):
                for t_x in xrange(n_f_w):
                    for f_idx in xrange(n_f_out):

                        filt = fb[t_y, t_x, f_idx]
                        # zero-mean, unit-l2norm
                        filt -= filt.mean()
                        filt_norm = np.linalg.norm(filt)
                        assert filt_norm != 0
                        filt /= filt_norm
                        fb[t_y, t_x, f_idx] = filt

            fb = np.ascontiguousarray(np.rollaxis(fb, 2, 6)).astype(DTYPE)

        else:
            print 'Not implemented!'
            # -- a) get file names and partitions associated with previous
            #       layer (input to this layer)
            # -- b) for each file, sample patches to train filters
            #       layer_idx
            # -- c) learn filters
            # -- d) get input once more as in (a), get file names and partitions
            #       associated with the output of this layers. process input
            #       and save to outpu accordingly.


        return fb

    # -- transform self.arr_w according to slm and hgdm layer descriptions.
    def _transform_layer(self, layer_idx, slm_l_desc, hgdm_l_arch=None):

        n_imgs = self.arr_w.shape[0]

        if layer_idx == 0:
            layer_in_shape = self.in_shape + (1,)
        else:
            [tiles_h_in, tiles_w_in, _, _, tiles_d_in, _] = \
                                        self.shape_stride_by_layer[layer_idx-1]
            layer_in_shape = (tiles_h_in, tiles_w_in, tiles_d_in)

        # -- assert layer input shape
        assert self.arr_w.shape[2:] == layer_in_shape

        if hgdm_l_arch == 'tiled':

            [tiles_h_out, tiles_w_out, rfh, rfw, tiles_d_out, rfs] = \
                                        self.shape_stride_by_layer[layer_idx]

            arr_in_tiled = view_as_windows(self.arr_w, (1, 1, rfh, rfw, 
                                           layer_in_shape[2]))
            arr_in_tiled = arr_in_tiled[:, :, ::rfs, ::rfs]

            assert arr_in_tiled.shape[:4] == (n_imgs, 1, 
                                              tiles_h_out, tiles_w_out)

            arr_out = np.empty((n_imgs, 1, tiles_h_out, tiles_w_out, 
                                tiles_d_out), dtype=DTYPE)

            for t_y in xrange(tiles_h_out):
                for t_x in xrange(tiles_w_out):

                    self.arr_w = arr_in_tiled[:, :, t_y, t_x, 0, 0, 0] 
                    fb = self.filterbanks[layer_idx][t_y, t_x]

                    self._transf_layer_helper(slm_l_desc, fb)
                    arr_out[:, :, t_y, t_x] = self.arr_w[:, :, 0, 0]

            self.arr_w = arr_out

        else:
            if layer_idx > 0:
                fb = self.filterbanks[layer_idx][0, 0]
            else:
                fb = None

            self._transf_layer_helper(slm_l_desc, fb)

        return


    def _transf_layer_helper(self, slm_l_desc, fb):

        for op_idx, (op_name, op_params) in enumerate(slm_l_desc):
            kwargs = op_params['kwargs']

            if op_name == 'fbcorr':
                fb_par = fb
            else:
                fb_par = None

            self.arr_w = self._process_one_op(op_name, kwargs, 
                                              self.arr_w, fb_par)

        return


    def _process_one_op(self, op_name, kwargs, arr_in, fb=None):

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
            tmp_out = fbcorr5(arr_in, fb)

            # -- activation
            min_out = -np.inf if min_out is None else min_out
            max_out = +np.inf if max_out is None else max_out
            # insure that the type is right before calling numexpr
            min_out = np.array([min_out], dtype=arr_in.dtype)
            max_out = np.array([max_out], dtype=arr_in.dtype)
            # call numexpr
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

    def _get_neural_imgs_fnames(self, layer_idx, mode, n_imgs=0):

        assert mode in ('read', 'write')

        path = self.model_path
        model_prefix = self.model_prefix
        hgdm_desc = self.hgdm_description
        filter_key = ''

        for l_idx in xrange(layer_idx):
            if l_idx == 0:
                continue
            filter_key += hgdm_desc[l_idx]['f_key'] + '.'

        f_basename  = model_prefix + '.neural_imgs.' + filter_key
        n_partitions, partitions = self._get_partitions(layer_idx, layer_idx, 
                                                        n_imgs)
        fnames = []

        for part_idx in xrange(n_partitions):
            fname = f_basename + 'part.' + str(part_idx) + '.npy'
            fname = os.path.join(path, fname)

            if mode == 'read':
                if not os.path.exists(fname):
                    fnames = []
                    partitions = []
                    break

            fnames += [fname]

        return fnames, partitions


    def _get_filter_fname(self, layer_idx):

        path = self.model_path
        model_prefix = self.model_prefix
        hgdm_desc = self.hgdm_description
        filter_key = ''

        for l_idx in xrange(layer_idx):
            if l_idx == 0:
                continue
            filter_key += hgdm_desc[l_idx]['f_key'] + '.'

        fname  = model_prefix + '.filters.' + filter_key + 'npy'
        fname  = os.path.join(path, fname)

        return fname


    def _max_mem_transform(self, layer_init, layer_end, n_imgs):

        assert layer_init in (0,1,2,3)
        assert layer_end in (0,1,2,3)

        slm_desc = self.slm_description
        hgdm_desc = self.hgdm_description

        if layer_init == 0:
            l_h, l_w, l_d = self.in_shape + (1,)
        else:
            [l_h, l_w, _, _, l_d, _] = \
                                        self.shape_stride_by_layer[layer_init-1]

        op_d_ant = 1
        max_mem = self._mem_shape((n_imgs, l_h, l_w, l_d))

        for l_idx, (slm_l_desc, hgdm_l_desc) in enumerate(zip(
                                            slm_desc[layer_init:layer_end+1],
                                            hgdm_desc[layer_init:layer_end+1])):

            [_, _, rf_h, rf_w, _, _] = self.shape_stride_by_layer[l_idx]

            for operation in slm_l_desc:

                # -- compute shape by operation
                if operation[0] == 'fbcorr':
                    op_h, op_w = operation[1]['initialize']['filter_shape']
                    op_d = operation[1]['initialize']['n_filters']
                    op_s = 1

                    op_elems = op_h * op_w * op_d_ant
                    op_d_ant = op_d
                elif operation[0] == 'lpool':
                    op_h, op_w = operation[1]['kwargs']['ker_shape']
                    op_d = l_d
                    op_s = operation[1]['kwargs']['stride']

                    op_elems = max(op_h, op_w)
                elif operation[0] == 'lnorm':
                    op_h, op_w = operation[1]['kwargs']['inker_shape']
                    op_d = l_d
                    op_s = 1

                    op_elems = max(op_h, op_w)

                l_h = (l_h - op_h) / op_s + 1
                l_w = (l_w - op_w) / op_s + 1
                l_d = op_d

                if l_idx > 0 and hgdm_l_desc['architecture'] == 'tiled':
                    rf_h = (rf_h - op_h) / op_s + 1
                    rf_w = (rf_w - op_w) / op_s + 1
                    op_shape = (n_imgs, rf_h, rf_w, max(l_d, op_elems))
                else:
                    op_shape = (n_imgs, l_h, l_w, max(l_d, op_elems))

                cur_mem = self._mem_shape(op_shape)
                if cur_mem > max_mem:
                    max_mem = cur_mem

            cur_mem = self._mem_shape((n_imgs, l_h, l_w, l_d))
            if cur_mem > max_mem:
                max_mem = cur_mem

        return max_mem

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

        # -- amount of memory needed to transform n_imgs
        mem_transform = self._max_mem_transform(layer_init, layer_end, n_imgs)

        n_partitions = int(mem_transform / MAX_MEM_GB + 1.)
        part_size = n_imgs / n_partitions

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