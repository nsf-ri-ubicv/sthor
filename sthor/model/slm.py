import numpy as np

from sthor.operation import lcdnorm3
from sthor.operation import fbcorr
from sthor.operation import lpool3

from pythor3.operation import fbcorr as pt3fbcorr
from pythor3.operation import lpool as pt3lpool
from pythor3.operation import lnorm as pt3lnorm


def pt3slm(arr_in):

    assert arr_in.ndim == 3
    inh, inw, ind = arr_in.shape

    #nfs = [32, 64, 128]
    nfs = [64, 128, 256]
    nb = (9, 9)

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

    #nfs = [32, 64, 128]
    nfs = [64, 128, 256]
    #nb = (9, 9)
    nb = (5, 5)
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
