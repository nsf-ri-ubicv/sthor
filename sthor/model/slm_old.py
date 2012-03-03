"""Sequential Layered Model"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np

from sthor.operation import lcdnorm3
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

    def __init__(self, input_shape, description):
        """XXX: docstring for __init__"""

        pprint(description)
        print input_shape
        assert len(input_shape) == 2 or len(input_shape) == 3

        self.description = description
        self.filterbanks = []

    def process(self, arr_in):
        """XXX: docstring for process"""

        description = self.description

        for layer_idx, layer_desc in enumerate(description):

            #operation_l = []

            for op_name, op_params in layer_desc:

                print op_name
                print op_params

                #curr_arr_in = last_arr_out
                #op_d = {}
                #assert op_name in ALLOWED_OPERATIONS

                #pclass = plugin_library['operation.' + op_name][plugin_mapping[op_name]['plugin']]
                #args = [curr_arr_in]

                #if op_name == 'fbcorr':
                    #curr_arr_fb = self._get_filterbank(curr_arr_in,
                                                       #op_params['initialize'])
                    #args += [curr_arr_fb]
                    #op_d['arr_fb'] = curr_arr_fb

                #kwargs = op_params['kwargs']

                #log.info(pclass)
                ##print op_params, kwargs
                #pobj = pclass(**plugin_mapping[op_name]['plugin_kwargs'])
                #curr_arr_out = pobj.prepare(*args, **kwargs)
                #pobj.specialize(*args, arr_out=curr_arr_out, **kwargs)

                #op_d['name'] = op_name
                #op_d['arr_in'] = curr_arr_in
                #op_d['arr_out'] = curr_arr_out
                #op_d['plugin_instance'] = pobj

                #operation_l += [op_d]

                #last_arr_out = curr_arr_out

            #layers += [operation_l]

            #return arr_out

def pt3slm(arr_in):

    assert arr_in.ndim == 3
    inh, inw, ind = arr_in.shape


    for nf in nfs:
        fbshape = (nf,) + nb + (arr_in.shape[-1],)
        fb1 = np.random.randn(*fbshape).astype('f')
        n1 = pt3lnorm(arr_in, inker_shape=nb, outker_shape=nb, threshold=1.0,
                      plugin='cthor', plugin_kwargs=dict(variant='sse:tbb'))
        f1 = pt3fbcorr(n1, fb1,
                       plugin='cthor', plugin_kwargs=dict(variant='sse:tbb'))
        p1 = pt3lpool(f1, ker_shape=nb, order=2, stride=2,
                      plugin='cthor', plugin_kwargs=dict(variant='sse:tbb'))
        arr_in = p1

    return p1

def slm(arr_in):

    assert arr_in.ndim == 3
    inh, inw, ind = arr_in.shape

    fbs = []
    fbd = arr_in.shape[-1]

    for nf in nfs:
        fbshape = nb + (fbd, nf)
        print 'generating', fbshape, 'filterbank'
        fb = np.random.randn(*fbshape).astype('f')
        fbs += [fb]
        fbd = nf
 
    #for nf in nfs:
    for fb in fbs:
        n1 = lcdnorm3(arr_in, nb, threshold=1.0)
        f1 = fbcorr(n1, fb)
        p1 = lpool3(f1, nb, order=2, stride=2)
        arr_in = p1

    return p1

def main():


    desc = \
    [[('lnorm',
       {'kwargs': {'inker_shape': (9, 9),
                   'outker_shape': (9, 9),
                   'remove_mean': False,
                   'stretch': 1,
                   'threshold': 0.10000000000000001}})],
     [('fbcorr',
       {'initialize': {'filter_shape': (9, 9),
                       'generate': ('random:uniform', {'rseed': 42}),
                       'n_filters': 64},
        'kwargs': {'max_out': None, 'min_out': None}}),
      ('lpool', {'kwargs': {'ker_shape': (5, 5), 'order': 1, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': (5, 5),
                   'outker_shape': (5, 5),
                   'remove_mean': False,
                   'stretch': 0.10000000000000001,
                   'threshold': 0.10000000000000001}})],
     [('fbcorr',
       {'initialize': {'filter_shape': (9, 9),
                       'generate': ('random:uniform', {'rseed': 42}),
                       'n_filters': 64},
        'kwargs': {'max_out': None, 'min_out': None}}),
      ('lpool', {'kwargs': {'ker_shape': (9, 9), 'order': 10, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': (3, 3),
                   'outker_shape': (3, 3),
                   'remove_mean': True,
                   'stretch': 10,
                   'threshold': 0.10000000000000001}})],
     [('fbcorr',
       {'initialize': {'filter_shape': (9, 9),
                       'generate': ('random:uniform', {'rseed': 42}),
                       'n_filters': 16},
        'kwargs': {'max_out': 1, 'min_out': None}}),
      ('lpool', {'kwargs': {'ker_shape': (5, 5), 'order': 2, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': (3, 3),
                   'outker_shape': (3, 3),
                   'remove_mean': False,
                   'stretch': 1,
                   'threshold': 1}})],
     [('fbcorr',
       {'initialize': {'filter_shape': (9, 9),
                       'generate': ('random:uniform', {'rseed': 42}),
                       'n_filters': 64},
        'kwargs': {'max_out': 1, 'min_out': None}}),
      ('lpool', {'kwargs': {'ker_shape': (9, 9), 'order': 10, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': (5, 5),
                   'outker_shape': (5, 5),
                   'remove_mean': True,
                   'stretch': 0.10000000000000001,
                   'threshold': 10}})]]

    in_shape = 200, 200
    slm = SequentialLayeredModel(in_shape, desc)
    raise

    a = np.random.randn(200, 200, 1).astype('f')

    import time
    N = 10
    start = time.time()
    for i in xrange(N):
        out = slm(a)
        #out = pt3slm(a)
        print out.shape
    end = time.time()
    print N / (end - start)

if __name__ == '__main__':
    main()
