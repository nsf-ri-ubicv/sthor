import time
import numpy as np
from skimage.util.shape import view_as_windows

np.random.seed(42)
a = np.random.randn(512, 512).astype('f')
fb = np.random.randn(512, 32, 32).astype('f')

start = time.time()
out = np.tensordot(view_as_windows(a, fb.shape[1:]), fb, axes=[(2, 3), (1, 2)])
end = time.time()
print out

duration = end - start
gflops = (np.prod(out.shape) * np.prod(fb.shape[1:]) * 2) / (1000**3.)
print gflops
gflops_per_sec = gflops / duration
print 'gflops/sec', gflops_per_sec
