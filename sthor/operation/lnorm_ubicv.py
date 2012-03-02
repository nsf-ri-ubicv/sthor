import numpy as np
from skimage.util.shape import view_as_windows

import numexpr as ne

n_cores = ne.detect_number_of_cores()
ne.set_num_threads(n_cores)


EPSILON = 1e-4


def lnorm(arr_in, inker_shape=(3, 3), threshold=1, mode='valid'):

    assert arr_in.ndim == 3
    assert mode == 'valid'
    assert len(inker_shape) == 2
    assert threshold > 0

    inh, inw, ind = arr_in.shape

    ikh, ikw = inker_shape
    nb_size = ikh * ikw * ind

    kshape = (ikh, ikw, ind)
    inrv = view_as_windows(arr_in, kshape)
    inrv = inrv.reshape(inrv.shape[:2] + (1, -1,))

    # -- local sums
    arr_sum = inrv.sum(-1)

    # -- local sums of squares
    arr_ssq = ne.evaluate('inrv ** 2.0')
    arr_ssq = arr_ssq.sum(-1)

    # -- remove the mean
    ys = inker_shape[0] / 2
    xs = inker_shape[1] / 2
    arr_slice = arr_in[ys:-ys, xs:-xs]
    arr_out = arr_slice - arr_sum / nb_size
    arr_out = ne.evaluate('arr_slice - arr_sum / nb_size')

    # -- divide by the euclidean norm
    l2norms = (arr_ssq - (arr_sum ** 2.) / nb_size).clip(0, np.inf)
    l2norms = np.sqrt(l2norms)# + EPSILON
    # XXX: use numpy-1.7.0 copyto()
    np.putmask(l2norms, l2norms < threshold, 1)
    arr_out = ne.evaluate('arr_out / l2norms')

    return arr_out

try:
    lnorm = profile(lnorm)
except NameError:
    pass

def main():
    a = np.random.randn(100, 100, 32).astype('f')

    N = 10
    import time
    start = time.time()
    for i in xrange(N):
        print i
        out = lnorm(a, (5, 5))
        #print out.shape
    end = time.time()
    print N / (end - start)

if __name__ == '__main__':
    main()
#
