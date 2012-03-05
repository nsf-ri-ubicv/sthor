import numpy as np
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1


def fbcorr(arr_in, arr_fb, stride=DEFAULT_STRIDE, arr_out=None):
    """3D Filterbank Correlation
    XXX: docstring
    """
    assert arr_in.dtype == np.float32

    assert arr_in.ndim == 3
    assert arr_fb.ndim == 4

    inh, inw, ind = arr_in.shape
    fbh, fbw, fbd, fbn = arr_fb.shape

    assert fbn > 1
    assert fbh <= inh
    assert fbw <= inw
    assert fbd == ind

    # -- reshape arr_in
    arr_inr = view_as_windows(arr_in, (fbh, fbw, fbd))
    outh, outw = arr_inr.shape[:2]
    arr_inrm = arr_inr.reshape(outh * outw, -1)

    # -- reshape arr_fb
    arr_fbm = arr_fb.reshape((fbh * fbw * fbd, fbn))

    # -- correlate !
    arr_out = np.dot(arr_inrm, arr_fbm)
    arr_out = arr_out.reshape(outh, outw, -1)

    assert arr_out.dtype == np.float32
    return arr_out


try:
    fbcorr = profile(fbcorr)
except NameError:
    pass


#def main():
    ##arr_in = np.random.randn(200, 200, 64).astype('f')
    #arr_in = np.random.randn(20, 20, 64).astype('f')
    #fb = np.random.randn(9, 9, 64, 128).astype('f')

    #fb2 = np.ascontiguousarray(fb.transpose(3, 0, 1, 2).copy())
    #print fb2.shape
    #print fb.shape
 
    #from pythor3.operation import fbcorr as fbcorr_pt3

    #import time
    #N = 10
    #start = time.time()
    #for i in xrange(N):
        #print i
        ##gv = fbcorr(arr_in, fb)
        #gt = fbcorr_pt3(arr_in, fb2, plugin='cthor', plugin_kwargs=dict(variant='sse:tbb'))[:]
        ##print np.linalg.norm(gv - gt)
    #end = time.time()
    #fps = N / (end - start)
    #print fps
    #tim = 1. / fps
    #print tim

    #flops = np.cumprod(arr_in.shape[:2] + fb.shape)[-1] * 2
    #gflops = (flops / 1e9)
    #print 'gflops / sec', 1. * gflops / tim

#if __name__ == '__main__':
    #main()
