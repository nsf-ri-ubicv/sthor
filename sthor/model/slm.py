"""Sequential Layered Model"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np
import numexpr as ne

from sthor.operation import lcdnorm3
from sthor.operation import fbcorr
from sthor.operation import lpool3

from pprint import pprint


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

    def process(self, arr_in):
        """XXX: docstring for process"""

        description = self.description
        filterbanks = self.filterbanks

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
                    if fbkey not in filterbanks:
                        initialize = op_params['initialize']
                        filter_shape = initialize['filter_shape']
                        generate = initialize['generate']
                        n_filters = initialize['n_filters']

                        fb_shape = (n_filters,) + filter_shape + (tmp_in.shape[-1],)

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

                        fb = np.ascontiguousarray(np.rollaxis(fb, 0, 4)).astype('f')

                        filterbanks[fbkey] = fb
                        print fb.shape

                    fb = filterbanks[fbkey]

                    # -- filter
                    assert tmp_in.dtype == np.float32
                    tmp_out = fbcorr(tmp_in, fb)

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


def main():

    import genson
    from os import path
    from numpy.testing import assert_allclose
    from pythor3.utils.testing import assert_allclose_round
    from thoreano.slm import TheanoSLM

    mypath = path.dirname(__file__)

    RTOL = 1e-2
    ATOL = 1e-4

    # -- Raise exceptions on floating-point errors
    np.seterr(all='raise')

    gt_time = 0
    gv_time = 0

    iter = 0
    while True:
        with open(path.join(mypath, 'plos09.gson')) as fin:
            gen = genson.loads(fin.read())

        desc = gen.next()
        genson.default_random_seed = 1

        ## -- L3 1st
        #desc = [
            #[('lnorm',
              #{'kwargs': {'inker_shape': [9, 9],
                          #'outker_shape': [9, 9],
                          #'remove_mean': False,
                          #'stretch': 10,
                          #'threshold': 1}})],
            #[('fbcorr',
              ##{'initialize': ['426b269c1bfeec366992218fb6e0cb5252cd7f69',
                              ##(64, 3, 3)],
              #{'initialize': {'filter_shape': (3, 3),
                              #'generate': ('random:uniform', {'rseed': 42}),
                              #'n_filters': 64},
               #'kwargs': {'max_out': None, 'min_out': 0}}),
             #('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 1, 'stride': 2}}),
             #('lnorm',
              #{'kwargs': {'inker_shape': [5, 5],
                          #'outker_shape': [5, 5],
                          #'remove_mean': False,
                          #'stretch': 0.10000000000000001,
                          #'threshold': 1}})],
            #[('fbcorr',
              ##{'initialize': ['9f1a2ad385682d076a7feacd923e50c330df4e29',
                              ##(128, 5, 5, 64)],
              #{'initialize': {'filter_shape': (5, 5),
                              #'generate': ('random:uniform', {'rseed': 42}),
                              #'n_filters': 128},
               #'kwargs': {'max_out': None, 'min_out': 0}}),
             #('lpool', {'kwargs': {'ker_shape': [5, 5], 'order': 1, 'stride': 2}}),
             #('lnorm',
              #{'kwargs': {'inker_shape': [7, 7],
                          #'outker_shape': [7, 7],
                          #'remove_mean': False,
                          #'stretch': 1,
                          #'threshold': 1}})],
            #[('fbcorr',
              ##{'initialize': ['d79b5af0732b177b2a9170288ba8f73727b56354',
                              ##(256, 5, 5, 128)],
              #{'initialize': {'filter_shape': (5, 5),
                              #'generate': ('random:uniform', {'rseed': 42}),
                              #'n_filters': 256},
               #'kwargs': {'max_out': None, 'min_out': 0}}),
             #('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 10, 'stride': 2}}),
             #('lnorm',
              #{'kwargs': {'inker_shape': [3, 3],
                          #'outker_shape': [3, 3],
                          #'remove_mean': False,
                          #'stretch': 10,
                          #'threshold': 1}})]
        #]

        in_shape = 200, 200, 1

        from pythor3 import model
        print 'create gt slm'
        try:
            slm_gt = model.slm.SequentialLayeredModel(
                in_shape, desc,
                plugin='passthrough',
                plugin_kwargs={
                    'plugin_mapping':{
                        'all':{
                            #'plugin':'scipy_naive',
                            #'plugin_kwargs': {},
                            'plugin':'cthor',
                            #'plugin_kwargs': {'variant': 'icc:sse:tbb'},
                            'plugin_kwargs': {'variant': 'sse:tbb'},
                        }
                    }
                })
        except ValueError, err:
            print err
            continue

        print 'create gv slm'
        slm_gv = SequentialLayeredModel(in_shape, desc)
        #slm_gt = TheanoSLM((200, 200, 1), desc)
        #slm_gv = TheanoSLM((200, 200, 1), desc)

        #from scipy import misc
        #a = misc.lena()
        #a = misc.imresize(a, (200, 200)) / 1.0
        #a.shape = a.shape[:2] + (1,)
        #a -= a.min()
        #a /= a.max()
        #a = a.astype('f')

        #print a.dtype
        #raise
        a = np.random.randn(200, 200, 1).astype('f')
        a -= a.min()
        a /= a.max()
        a = a.astype('f')

        import time
        W = 2
        for i in xrange(W):
            slm_gt.process(a)
            slm_gv.process(a)

        N = 10
        for i in xrange(N):
            iter += 1
            print 'iter', iter
            np.random.seed(iter)
            #np.random.seed(20)
            a = np.random.randn(200, 200, 1).astype('f')
            a -= a.min()
            a /= a.max()
            a = a.astype('f')

            #print 'a.mean()', a.mean()
            #print 'np.linalg.norm(a)', np.linalg.norm(a)

            x = a.copy()
            start = time.time()
            gt = slm_gt.process(x)
            gt_time += time.time() - start

            x = a.copy()
            start = time.time()
            gv = slm_gv.process(x)
            gv_time += time.time() - start

            print 'norm', np.linalg.norm(gv - gt)
            print 'abs max', np.absolute(gv - gt).max()
            tmp = np.absolute(gv - gt)
            print tmp.ravel().argmax()
            print gv[tmp == tmp.max()]
            print gt[tmp == tmp.max()]
            #assert np.absolute(gv - gt).max() < 1e-3
            #assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)
            #assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        #print 'gv fps', N / gv_time
        #print 'gt fps', N / gt_time
        print 'speedup', gt_time / gv_time

if __name__ == '__main__':
    main()
