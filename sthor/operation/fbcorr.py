import numpy as np
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1


def fbcorr(arr_in, arr_fb, stride=DEFAULT_STRIDE, arr_out=None):
    """3D Filterbank Correlation
    XXX: docstring
    """

    assert arr_in.ndim == 3
    assert arr_fb.ndim == 4

    inh, inw, ind = arr_in.shape
    fbn, fbh, fbw, fbd = arr_fb.shape

    assert fbn > 1
    assert fbh <= inh
    assert fbw <= inw
    assert fbd == ind

    # -- reshape arr_in
    arr_inr = view_as_windows(arr_in, (fbh, fbw, 1))
    outh, outw = arr_inr.shape[:2]
    arr_inrm = arr_inr.reshape(outh * outw, -1)

    # -- reshape arr_fb
    arr_fb = arr_fb.transpose((0, 3, 1, 2))
    arr_fbm = arr_fb.reshape(fbn, -1)

    # -- correlate !
    #print 'shape', arr_inrm.shape, arr_fbm.T.shape
    arr_out = np.dot(arr_inrm, arr_fbm.T)
    arr_out = arr_out.reshape(outh, outw, -1)

    return arr_out


try:
    fbcorr = profile(fbcorr)
except NameError:
    pass


def main():
    arr_in = np.random.randn(200, 200, 64).astype('f')
    fb = np.random.randn(128, 8, 8, 64).astype('f')

    #from pythor3.operation import fbcorr as fbcorr_pt3

    import time
    N = 10
    start = time.time()
    for i in xrange(N):
        print i
        print fbcorr(arr_in, fb).shape
        #pt3_fbcorr(a, fb, plugin='cthor', plugin_kwargs=dict(variant='sse:tbb'))
    end = time.time()
    fps = N / (end - start)
    print fps
    tim = 1. / fps
    print tim

    flops = np.cumprod(arr_in.shape[:2] + fb.shape)[-1] * 2
    gflops = (flops / 1e9)
    print 'gflops / sec', 1. * gflops / tim

if __name__ == '__main__':
    main()
