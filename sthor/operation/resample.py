import scipy as sp
import numpy as np
import pylab as pl
from scipy import ndimage as ndi
from skimage.util.shape import view_as_blocks


def upsample(arr_in, arr_out):

    assert arr_in.ndim == 3
    assert arr_out.ndim == 3

    inh, inw, ind = arr_in.shape
    outh, outw, outd = arr_out.shape

    for oy in range(outh):
        iy = (inh - 1.) * (oy / (outh - 1.))
        for ox in range(outw):
            ix = (inw - 1.) * (ox / (outw - 1.))
            for oz in range(outd):
                iz = (ind - 1.) * (oz / (outd - 1.))
                arr_out[oy, ox, oz] = arr_in[iy, ix, iz]

    return arr_out


def demo_upsample_nearest():
    a = sp.misc.lena() / 1.
    a = sp.misc.lena() / 1.
    a.shape = a.shape[:2] + (1,)
    print a.shape
    #print np.tile(a, 2).shape
    #a = np.dstack((a, -a))
    N = 96
    a = np.tile(a, N)
    a[:,:,95] = -a[:,:,95]

    #r = np.tile(a, (2, 2, 1))
    #np.kron(a, np.ones((2,2,1))).shape

    # -- loop
    #a = a[:, :, 0].reshape(256, 256, 1)
    #r = np.empty((1024, 1024, 1))
    #r[0::2, 0::2] = a
    #r[0::2, 1::2] = a
    #r[1::2, 0::2] = a
    #r[1::2, 1::2] = a

    # -- block view
    r = np.empty((1024, 1024, N))
    b = view_as_blocks(r, (2, 2, 1))
    print b.shape

    a2 = a.reshape(a.shape + (1, 1, 1))
    #a[:, :, :, np.newaxis, np.newaxis, np.newaxis]
    b[:] = a2
    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')


    #b2 = b.swapaxes(1, 3).reshape(r.shape)
    b2 = b.transpose((0, 3, 1, 4, 2, 5)).reshape(r.shape)

    pl.matshow(b2[:,:,0])
    pl.matshow(b2[:,:,1])
    pl.matshow(b2[:,:,95])
    pl.show()
