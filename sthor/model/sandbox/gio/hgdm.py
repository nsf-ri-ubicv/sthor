"""Hierarchical Generative-Discriminative Model"""

# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os
import time
import numpy as np
import numexpr as ne

from pprint import pprint

from skimage.util.shape import view_as_windows

from sthor.operation.sandbox import lcdnorm5
from sthor.operation.sandbox import fbcorr5
from sthor.operation.sandbox import lpool5

from sklearn.decomposition import PCA
from pls import pls


DTYPE = np.float32
MAX_MEM_GB = 2.
PARTITION_SIZE = 100
N_PATCHES_P_IMG = 3
PATCH_SAMPLE_SEED = 21

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
        self.filterbanks_mean = [[]]
        self.filterbanks_std = [[]]

        self.shape_stride_by_layer = _get_shape_stride_by_layer(slm_description,
                                                                in_shape)

        self._rng_p = np.random.RandomState(PATCH_SAMPLE_SEED)

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

                # -- reshape array to cope with the 5d operation functions
                self.arr_w.shape = (n_imgs, 1) + self.arr_w.shape[1:]

            for layer_idx in xrange(layers_fb, len(slm_desc)):

                # -- transform previous layer
                self._transf_layer_by_parts(layer_idx-1, n_imgs, True)

                # -- learn filters for the current layer
                fb, fb_mean, fb_std = \
                                self._sample_learn_filters(layer_idx, y, n_imgs)
                self.filterbanks += [fb.copy()]

                if fb_mean is not None:
                    self.filterbanks_mean += [fb_mean.copy()]
                    self.filterbanks_std += [fb_std.copy()]
                else:
                    self.filterbanks_mean += [None]
                    self.filterbanks_std += [None]

                # -- save filters in disk
                fname, fname_mean, fname_std = self._get_filter_fname(layer_idx)

                if os.path.exists(fname):
                    raise ValueError('filter learned already exist in disk')
                else:
                    np.save(fname, fb)

                    if fb_mean is not None:
                        np.save(fname_mean, fb_mean)
                        np.save(fname_std, fb_std)

        assert len(self.filterbanks) == len(slm_desc) == len(hgdm_desc)

        self.arr_w = None
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

        layer_idx = self.n_layers - 1

        # -- amount of memory that the output array is going to demand
        mem_arr_out = self._mem_layer(layer_idx, n_imgs)

        if mem_arr_out > MAX_MEM_GB:

            pprint("size of output array is greater than '%s'; " +
                   "the maximum allowed. filters were learned, but" +
                   "input array was not transformed" % MAX_MEM_GB)
        else:

            # -- process last layer - the only one not processed while fitting
            #    the model
            self._transf_layer_by_parts(layer_idx, n_imgs, False)

            arr_out = self.arr_w

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

            learn_algo = hgdm_l_desc['f_learn_algo']

            fname, fname_mean, fname_std = self._get_filter_fname(layer_idx)
            
            if os.path.exists(fname):
                if learn_algo != 'slm':
                    if os.path.exists(fname_mean):
                        fb_mean = np.load(fname_mean)
                        fb_std = np.load(fname_std)
                    else:
                        break
                else:
                    fb_mean = None
                    fb_std = None

                fb = np.load(fname)
            else:
                break

            f_init = slm_l_desc[0][1]['initialize']
            hgdm_l_arch = hgdm_l_desc['architecture']

            n_f_in = self.shape_stride_by_layer[layer_idx-1][4]
            n_f_out = self.shape_stride_by_layer[layer_idx][4]
            assert n_f_out == f_init['n_filters']

            f_shape = f_init['filter_shape'] + (n_f_in,)

            if hgdm_l_arch == 'tiled':
                n_f_h, n_f_w = self.shape_stride_by_layer[layer_idx][:2]
            else:
                n_f_h, n_f_w = (1,1)

            fb_shape = (n_f_h, n_f_w) + f_shape + (n_f_out,)
            assert fb.shape == fb_shape

            if learn_algo != 'slm':
                assert fb_mean.shape == (n_f_h, n_f_w) + f_shape
                assert fb_std.shape == (n_f_h, n_f_w) + f_shape
                self.filterbanks_mean += [fb_mean.copy()]
                self.filterbanks_std += [fb_std.copy()]
            else:
                self.filterbanks_mean += [None]
                self.filterbanks_std += [None]

            self.filterbanks += [fb.copy()]


    def _sample_learn_filters(self, layer_idx, y, n_imgs):

        assert layer_idx in (1,2,3)

        [n_f_h_in, n_f_w_in, _, _, n_f_in, _] = \
                                         self.shape_stride_by_layer[layer_idx-1]

        slm_l_desc = self.slm_description[layer_idx]
        hgdm_l_desc = self.hgdm_description[layer_idx]
        learn_algo = hgdm_l_desc['f_learn_algo']
        hgdm_l_arch = hgdm_l_desc['architecture']

        f_desc = slm_l_desc[0][1] # fg11-type slm
        f_init = f_desc['initialize']
        f_shape = f_init['filter_shape'] + (n_f_in,)
        n_filters = f_init['n_filters']

        if hgdm_l_arch == 'tied':
            n_f_h_out, n_f_w_out = (1,1)
            rf_h, rf_w, rf_s = (n_f_h_in, n_f_w_in, 1)
            n_f_out = self.shape_stride_by_layer[layer_idx][4]
        elif hgdm_l_arch == 'tiled':
            [n_f_h_out, n_f_w_out, rf_h, rf_w, n_f_out, rf_s] = \
                                           self.shape_stride_by_layer[layer_idx]

        assert n_f_out == n_filters

        fb_shape = (n_f_h_out, n_f_w_out, n_filters) + f_shape

        if learn_algo == 'slm':
            fb_mean, fb_std = None, None 
        else:
            arr_learn = np.empty((n_imgs * N_PATCHES_P_IMG,) + f_shape, 
                                 dtype=DTYPE)
            y_learn = np.empty((n_imgs * N_PATCHES_P_IMG,), dtype=y.dtype)

            fb = np.empty(fb_shape, dtype=DTYPE)
            fb_mean = np.empty((n_f_h_out, n_f_w_out) + f_shape, dtype=DTYPE)
            fb_std = np.empty((n_f_h_out, n_f_w_out) + f_shape, dtype=DTYPE)

        for t_y in xrange(n_f_h_out):
            for t_x in xrange(n_f_w_out):

                t1 = time.time()

                if learn_algo == 'slm':

                    generate = f_init['generate']
                    method_name, method_kwargs = generate
                    assert method_name == 'random:uniform'

                    rseed = method_kwargs.get('rseed', None)
                    rng_f = np.random.RandomState(rseed)

                    fb = rng_f.uniform(low=-1.0, high=1.0, size=fb_shape)

                    # -- remove filter mean
                    for f_idx in xrange(n_filters):
                        filt = fb[t_y, t_x, f_idx]
                        filt -= filt.mean()
                        fb[t_y, t_x, f_idx] = filt

                else:

                    input_fnames, partitions = \
                         self._get_neural_imgs_fnames(layer_idx-1, 
                                                      'read', n_imgs)

                    assert len(partitions) >= 1

                    p_idx = 0
                    p_end_idx = 0

                    for fname, (part_init, part_end) in zip(input_fnames, 
                                                            partitions):
                        n_imgs_part = part_end - part_init
                        arr_p = np.load(fname)
                        assert n_imgs_part == arr_p.shape[0]

                        arr_rf = view_as_windows(arr_p, (1, 1, 
                                                         rf_h, rf_w, n_f_in))
                        arr_rf = arr_rf[:, :, ::rf_s, ::rf_s]

                        assert arr_rf.shape[:4] == (n_imgs_part, 1, 
                                                    n_f_h_out, n_f_w_out)

                        # -- select only the receptive field where filters are 
                        #    going to be learned
                        arr_rf = arr_rf[:, :, t_y, t_x]
                        arr_rf.shape = n_imgs_part, rf_h, rf_w, n_f_in

                        p_init_idx = p_idx * N_PATCHES_P_IMG
                        p_end_idx = (p_idx + n_imgs_part) * N_PATCHES_P_IMG

                        arr_learn[p_init_idx:p_end_idx], \
                        y_learn[p_init_idx:p_end_idx] = \
                             self._sample_rf(arr_rf, 
                                             y[p_idx:p_idx+n_imgs_part],
                                             f_shape)

                        p_idx += n_imgs_part

                    assert p_end_idx == n_imgs * N_PATCHES_P_IMG
                    print arr_learn.shape

                    # -- learn filters for the sampled receptive field
                    fb[t_y, t_x], fb_mean[t_y, t_x], fb_std[t_y, t_x] = \
                                    self._learn_filters(arr_learn, y_learn, 
                                                        learn_algo, n_filters)
                    print arr_learn.shape

                # -- normalize filter to unit-l2norm
                for f_idx in xrange(n_filters):

                    filt = fb[t_y, t_x, f_idx]
                    # zero-mean, unit-l2norm
                    #filt -= filt.mean()
                    filt_norm = np.linalg.norm(filt)
                    assert filt_norm != 0
                    filt /= filt_norm
                    fb[t_y, t_x, f_idx] = filt

                t_elapsed = time.time() - t1
                print 'Filters from tile %d out of %d learned in %g seconds...'\
                       % (t_y * n_f_h_out + t_x + 1,
                          n_f_h_out * n_f_w_out, t_elapsed)

        fb = np.ascontiguousarray(np.rollaxis(fb, 2, 6)).astype(DTYPE)

        return fb, fb_mean, fb_std


    def _sample_rf(self, arr_rf, y, f_shape):

        n_imgs, rf_h, rf_w, rf_d = arr_rf.shape
        f_h, f_w, f_d = f_shape

        assert f_h <= rf_h and f_w <= rf_w and f_d == rf_d

        arr_out = np.empty((n_imgs * N_PATCHES_P_IMG,) + f_shape, dtype=DTYPE)
        y_out = np.empty((n_imgs * N_PATCHES_P_IMG,), dtype=y.dtype)

        rnd_patches = True

        for i, rf_img in enumerate(arr_rf):
            for j in xrange(N_PATCHES_P_IMG):

                if rnd_patches:
                    p_y = self._rng_p.random_integers(low=0, high=rf_h-f_h)
                    p_x = self._rng_p.random_integers(low=0, high=rf_w-f_w)
                else:
                    # -- get patch in the central part of the receptive field
                    assert N_PATCHES_P_IMG == 1
                    p_y = (rf_h - f_h) / 2
                    p_x = (rf_w - f_w) / 2

                i_out = i * N_PATCHES_P_IMG + j

                arr_out[i_out] = rf_img[p_y:p_y+f_h,p_x:p_x+f_w].copy()
                y_out[i_out] = y[i]

        return arr_out, y_out


    def _learn_filters(self, X, y, learn_algo, n_filters):

        n_train, f_h, f_w, f_d = X.shape

        X = X.copy()
        X.shape = n_train, -1

        n_proj_vectors = (n_filters + 1) / 2

        f_mean, f_std = _get_norm_info(X)
        X = _preprocess_features(X, f_mean, f_std)
        assert(not np.isnan(np.ravel(X)).any())
        assert(not np.isinf(np.ravel(X)).any())

        if learn_algo == 'pls':

            proj_vectors, _, _ = pls(X, y, n_proj_vectors, class_specific=False)
            proj_vectors = proj_vectors.T

        elif learn_algo == 'pca':

            pca = PCA(n_components=n_proj_vectors)
            pca.fit(X=X)
            proj_vectors = pca.components_

        filters = np.empty((n_filters, f_h * f_w * f_d), dtype=DTYPE)

        for i_f in xrange(n_proj_vectors):
            filt = proj_vectors[i_f].copy()
            for i_neg in xrange(2):
                i_f_neg = i_f * 2 + i_neg
                if i_f_neg < n_filters:
                    filters[i_f_neg]= filt
                filt = -filt

        filters.shape = n_filters, f_h, f_w, f_d
        f_mean.shape = f_h, f_w, f_d
        f_std.shape = f_h, f_w, f_d

        return filters, f_mean, f_std

    # -- transform train neural images from layer x to layer x+1. both the 
    #    input (except for layer 0) and the output are stored in disk
    #    (possibly in partitions) for two reasons: 
    #    (1) they may be used later to train models that are similiar up 
    #        to layer x
    #    (2) they do not fit in memory
    def _transf_layer_by_parts(self, layer_idx, n_imgs, write_output=True):

        slm_l_desc = self.slm_description[layer_idx]
        hgdm_l_desc = self.hgdm_description[layer_idx]

        if layer_idx > 0:
            hgdm_l_arch = hgdm_l_desc['architecture']
        else:
            hgdm_l_arch = None

        # -- height, width, and depth of this layer's output
        [l_h, l_w, _, _, l_d, _] = self.shape_stride_by_layer[layer_idx]

        if layer_idx == 0:
            input_fnames = [['']]
            parts_in = [[0, n_imgs]]
        else:
            input_fnames, parts_in = \
                       self._get_neural_imgs_fnames(layer_idx-1, 'read', n_imgs)

        if write_output:
            output_fnames, parts_out = \
                        self._get_neural_imgs_fnames(layer_idx, 'write', n_imgs)
        else:
            output_fnames = [['']]
            parts_out = [[0, n_imgs]]


        assert len(parts_in) >= 1
        #assert len(parts_out) >= 1
        #assert write_output or len(parts_out) == 1

        i_idx = 0
        n_imgs_transf = 0

        if layer_idx == 0:
            arr_p_i = self.arr_w
        else:
            i_fname = input_fnames[i_idx]
            arr_p_i = np.load(i_fname)

        p_i_size = parts_in[i_idx][1] - parts_in[i_idx][0]
        p_i_idx = 0

        for o_fname, (p_o_init, p_o_end) in zip(output_fnames, parts_out):

            p_o_size = p_o_end - p_o_init
            p_o_idx = 0

            arr_p_o = np.empty((p_o_size, 1, l_h, l_w, l_d), dtype=DTYPE)

            while p_o_idx < p_o_size:

                i_pend = p_i_size - p_i_idx
                o_pend = p_o_size - p_o_idx
                n_imgs_to_transf = min(i_pend, o_pend)

                self.arr_w = arr_p_i[p_i_idx:p_i_idx+n_imgs_to_transf]
                self._transform_layer(layer_idx, slm_l_desc, hgdm_l_arch)

                arr_p_o[p_o_idx:p_o_idx+n_imgs_to_transf] = self.arr_w

                p_i_idx += n_imgs_to_transf
                p_o_idx += n_imgs_to_transf

                n_imgs_transf += n_imgs_to_transf

                if p_i_idx == p_i_size:

                    i_idx +=1
                    if i_idx < len(parts_in):
                        i_fname = input_fnames[i_idx]
                        arr_p_i = np.load(i_fname)
                        p_i_size = parts_in[i_idx][1] - parts_in[i_idx][0]
                        p_i_idx = 0

                elif p_i_idx > p_i_size:
                    raise ValueError('Index error while processing layer ' + 
                                     'by parts')

            assert p_o_idx ==  p_o_size, ('Index error while processing ' + 
                                          'layer by parts')

            if write_output:
                np.save(o_fname, arr_p_o)
            else:
                self.arr_w = arr_p_o

        if n_imgs_transf > 0:
            assert n_imgs == n_imgs_transf

        return

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

                    if self.filterbanks_mean[layer_idx] is not None:
                        fb_mean = self.filterbanks_mean[layer_idx][t_y, t_x]
                        fb_std = self.filterbanks_std[layer_idx][t_y, t_x]

                    self._transf_layer_helper(slm_l_desc, fb, fb_mean, fb_std)
                    arr_out[:, :, t_y, t_x] = self.arr_w[:, :, 0, 0]

            self.arr_w = arr_out

        else:
            if layer_idx > 0:
                fb = self.filterbanks[layer_idx][0, 0]

                if self.filterbanks_mean[layer_idx] is not None:

                    fb_mean = self.filterbanks_mean[layer_idx][0, 0]
                    fb_std = self.filterbanks_std[layer_idx][0, 0]
                else:
                    fb_mean, fb_std = None, None
            else:
                fb, fb_mean, fb_std = None, None, None

            self._transf_layer_helper(slm_l_desc, fb, fb_mean, fb_std)

        return


    def _transf_layer_helper(self, slm_l_desc, fb, fb_mean, fb_std):

        for op_idx, (op_name, op_params) in enumerate(slm_l_desc):
            kwargs = op_params['kwargs']

            if op_name == 'fbcorr':
                fb_par = fb
                fb_mean_par = fb_mean
                fb_std_par = fb_std
            else:
                fb_par, fb_mean_par, fb_std_par = None, None, None

            self.arr_w = self._process_one_op(op_name, kwargs, self.arr_w,
                                              fb_par, fb_mean_par, fb_std_par)

        return


    def _process_one_op(self, op_name, kwargs, arr_in,
                        fb=None, f_mean=None, f_std=None):

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

            tmp_out = fbcorr5(arr_in, fb, f_mean=f_mean, f_std=f_std)

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

    def _get_neural_imgs_fnames(self, layer_idx, mode, n_imgs):

        assert mode in ('read', 'write')

        path = self.model_path
        model_prefix = self.model_prefix
        hgdm_desc = self.hgdm_description
        filter_key = ''

        for l_idx in xrange(layer_idx+1):
            if l_idx == 0:
                continue
            filter_key += hgdm_desc[l_idx]['f_key'] + '.'

        f_basename  = model_prefix + '.neural_imgs.' + filter_key
        n_partitions, partitions = self._get_partitions(layer_idx, layer_idx, 
                                                        n_imgs)
        fnames = []

        files_exist = True

        for part_idx in xrange(n_partitions):
            fname = f_basename + 'part.' + '%03d' % part_idx + '.npy'
            fname = os.path.join(path, fname)

            if not os.path.exists(fname):
                files_exist = False

            fnames += [fname]

        if ((mode == 'read' and not files_exist) or 
            (mode == 'write' and files_exist)):

            fnames = []
            partitions = []

        return fnames, partitions


    def _get_filter_fname(self, layer_idx):

        path = self.model_path
        model_prefix = self.model_prefix
        hgdm_desc = self.hgdm_description
        filter_key = ''

        for l_idx in xrange(layer_idx+1):
            if l_idx == 0:
                continue
            filter_key += hgdm_desc[l_idx]['f_key'] + '.'

        fname  = model_prefix + '.filters.' + filter_key + 'npy'
        fname_mean  = model_prefix + '.filters_mean.' + filter_key + 'npy'
        fname_std  = model_prefix + '.filters_std.' + filter_key + 'npy'
        
        fname  = os.path.join(path, fname)
        fname_mean  = os.path.join(path, fname_mean)
        fname_std  = os.path.join(path, fname_std)

        return fname, fname_mean, fname_std


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


def _preprocess_features(features, fmean, fstd):

    assert features[0].shape == fmean.shape
    assert features[0].shape == fstd.shape

    features_norm = features - fmean
    features_norm /= fstd

    return features_norm

MEAN_MAX_NPOINTS = 2000
STD_MAX_NPOINTS = 2000

def _get_norm_info(train_features):
    npoints = train_features.shape[0]

    print "Preprocessing scaling parameters..."
    if npoints < MEAN_MAX_NPOINTS:
        fmean = train_features.mean(0)
    else:
        # - try to optimize memory usage...
        sel = train_features[:MEAN_MAX_NPOINTS]
        fmean = np.empty_like(sel[0,:])

        np.add.reduce(sel, axis=0, dtype="float32", out=fmean)

        curr = np.empty_like(fmean)
        npoints_done = MEAN_MAX_NPOINTS
        while npoints_done < npoints:

            sel = train_features[npoints_done:npoints_done + MEAN_MAX_NPOINTS]
            np.add.reduce(sel, axis=0, dtype="float32", out=curr)
            np.add(fmean, curr, fmean)
            npoints_done += MEAN_MAX_NPOINTS

        fmean /= npoints

    if npoints < STD_MAX_NPOINTS:
        fstd = train_features.std(0)
    else:
        # - try to optimize memory usage...
        sel = train_features[:MEAN_MAX_NPOINTS]

        mem = np.empty_like(sel)
        curr = np.empty_like(mem[0,:])

        seln = sel.shape[0]
        np.subtract(sel, fmean, mem[:seln])
        np.multiply(mem[:seln], mem[:seln], mem[:seln])
        fstd = np.add.reduce(mem[:seln], axis=0, dtype="float32")

        npoints_done = MEAN_MAX_NPOINTS
        while npoints_done < npoints:

            sel = train_features[npoints_done:npoints_done + MEAN_MAX_NPOINTS]
            seln = sel.shape[0]
            np.subtract(sel, fmean, mem[:seln])
            np.multiply(mem[:seln], mem[:seln], mem[:seln])
            np.add.reduce(mem[:seln], axis=0, dtype="float32", out=curr)
            np.add(fstd, curr, fstd)

            npoints_done += MEAN_MAX_NPOINTS

        fstd = np.sqrt(fstd / npoints)

    fstd[fstd == 0] = 1
    return (fmean, fstd)
