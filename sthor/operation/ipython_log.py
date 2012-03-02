import numpy as np
from numpy.testing import assert_allclose
from pythor3.operation import lnorm

from lnorm_ubicv import lnorm as lnorm_ubicv

rtol = 1e-3
atol = 1e-6

dl = [1, 4, 8, 16, 3, 5, 2]
sl = [1e-1, 1, 1e1]
#tl = [0]#, 0.1, 0.2, 0.5, 1.0, 10.]
#tl = [0.1, 0.2, 0.5, 1.0, 10.]
tl = [1e-1, 1, 1e1]
#tl = [0.8, 1.0, 10.0, 1e2, 1e3, 1e6]
#tl = [0, 0.1, 0.5, 0.8, 1.0, 10.0, 1e2, 1e3, 1e6]

for _ in xrange(10):

    for d in dl:

        for s in sl:

            for t in tl:

                print d, s, t

                arr = np.random.randn(3, 3, d).astype('float32')

                #gv = lnorm(s * arr, threshold=t)[:]
                gv = lnorm_ubicv(s * arr, threshold=t)[:]
                gt = lnorm(arr, stretch=s, threshold=t, remove_mean=True)[:]

                assert_allclose(gv, gt, atol=atol, rtol=rtol)
