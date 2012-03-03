"""Sequential Layered Model"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np

from sthor.operation import ldnorm3, lcdnorm3
from sthor.operation import fbcorr
from sthor.operation import lpool3

from pythor3.operation import fbcorr as pt3fbcorr
from pythor3.operation import lpool as pt3lpool
from pythor3.operation import lnorm as pt3lnorm

nfs = [64, 128, 256]
nb = (9, 9)


from pprint import pprint
from pythor3 import plugin_library


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

                #print op_name
                #print op_params

                if op_name == 'lnorm':

                    kwargs = op_params['kwargs']

                    inker_shape = kwargs['inker_shape']
                    outker_shape = kwargs['outker_shape']
                    remove_mean = kwargs['remove_mean']
                    stretch = kwargs['stretch']
                    threshold = kwargs['threshold']

                    # SLM PLoS09 / FG11 constraints:
                    assert inker_shape == outker_shape

                    tmp_in = tmp_out * stretch

                    if remove_mean:
                        tmp_out = lcdnorm3(tmp_in, inker_shape, threshold=threshold)
                    else:
                        tmp_out = ldnorm3(tmp_in, inker_shape, threshold=threshold)

                elif op_name == 'fbcorr':

                    tmp_in = tmp_out

                    kwargs = op_params['kwargs']
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

                        fb = np.ascontiguousarray(np.rollaxis(fb, 0, 4))

                        filterbanks[fbkey] = fb
                        print fb.shape

                    fb = filterbanks[fbkey]

                    # -- filter
                    tmp_out = fbcorr(tmp_in, fb)

                    # -- activation
                    min_out = -np.inf if min_out is None else min_out
                    max_out = +np.inf if max_out is None else max_out
                    tmp_out = tmp_out.clip(min_out, max_out)

                elif op_name == 'lpool':

                    tmp_in = tmp_out

                    kwargs = op_params['kwargs']
                    ker_shape = kwargs['ker_shape']
                    order = kwargs['order']
                    stride = kwargs['stride']

                    tmp_out = lpool3(tmp_in, ker_shape, order=order, stride=stride)

                else:
                    raise ValueError("operation '%s' not understood" % op_name)

                #print tmp_out.shape

        return tmp_out


def main():

    import genson
    from os import path
    from numpy.testing import assert_allclose

    mypath = path.dirname(__file__)

    RTOL = 1e-2
    ATOL = 1e-4

    while True:
        with open(path.join(mypath, 'plos09.gson')) as fin:
            gen = genson.loads(fin.read())

        desc = gen.next()

    #desc = \
    #[[('lnorm',
       #{'kwargs': {'inker_shape': (5, 5),
                   #'outker_shape': (5, 5),
                   #'remove_mean': True,
                   #'stretch': 1,
                   #'threshold': 0.1}})],
     #[('fbcorr',
       #{'initialize': {'filter_shape': (7, 7),
                       #'generate': ('random:uniform', {'rseed': 42}),
                       #'n_filters': 64},
        #'kwargs': {'max_out': None, 'min_out': None}}),
      #('lpool', {'kwargs': {'ker_shape': (5, 5), 'order': 1, 'stride': 2}}),
      #('lnorm',
       #{'kwargs': {'inker_shape': (5, 5),
                   #'outker_shape': (5, 5),
                   #'remove_mean': False,
                   #'stretch': 0.1,
                   #'threshold': 0.1}})
     #],
     #[('fbcorr',
       #{'initialize': {'filter_shape': (5, 5),
                       #'generate': ('random:uniform', {'rseed': 42}),
                       #'n_filters': 64},
        #'kwargs': {'max_out': None, 'min_out': None}}),
      #('lpool', {'kwargs': {'ker_shape': (5, 5), 'order': 2, 'stride': 2}}),
      #('lnorm',
       #{'kwargs': {'inker_shape': (3, 3),
                   #'outker_shape': (3, 3),
                   #'remove_mean': True,
                   #'stretch': 10,
                   #'threshold': 0.1}})
     #],
     #[('fbcorr',
       #{'initialize': {'filter_shape': (9, 9),
                       #'generate': ('random:uniform', {'rseed': 42}),
                       #'n_filters': 16},
        #'kwargs': {'max_out': 1, 'min_out': None}}),
      #('lpool', {'kwargs': {'ker_shape': (5, 5), 'order': 2, 'stride': 2}}),
      #('lnorm',
       #{'kwargs': {'inker_shape': (3, 3),
                   #'outker_shape': (3, 3),
                   #'remove_mean': True,
                   #'stretch': 1,
                   #'threshold': 1}})]
    #]

        in_shape = 200, 200, 1

        from pythor3 import model
        slm_gt = model.slm.SequentialLayeredModel(
            in_shape, desc,
            plugin='passthrough',
            plugin_kwargs={
                'plugin_mapping':{
                    'all':{
                        'plugin':'cthor',
                        #'plugin_kwargs': {'variant': 'icc:sse:tbb'},
                        'plugin_kwargs': {'variant': 'sse:tbb'},
                    }
                }
            })

        slm_gv = SequentialLayeredModel(in_shape, desc)


        import time
        N = 10
        gt_time = 0
        gv_time = 0
        for i in xrange(N):
            a = np.random.randn(200, 200, 1).astype('f')
            a -= a.min()
            a /= a.max()

            start = time.time()
            try:
                gt = slm_gt.process(a)
            except ValueError:
                continue
            gt_time += time.time() - start

            start = time.time()
            gv = slm_gv.process(a)
            gv_time += time.time() - start

            print 'abs max', np.absolute(gv - gt).max()
            #assert np.absolute(gv - gt).max() < 1e-3
            assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        print 'gv fps', N / gv_time
        print 'gt fps', N / gt_time
        print 'speedup', gt_time / gv_time

if __name__ == '__main__':
    main()
