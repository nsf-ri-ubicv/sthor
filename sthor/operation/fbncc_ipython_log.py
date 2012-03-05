import numpy as np
from skimage.util.shape import view_as_windows

a = np.random.randn(100, 100, 4).astype('f')
fb = np.random.randn(5, 5, 4, 32).astype('f')

nb_size = np.prod(fb.shape[:-1])

# -- normalize input array
rv = view_as_windows(a, fb.shape[:-1])
arr_sum = np.apply_over_axes(np.sum, rv, (3, 4, 5))
arr_ssq = np.apply_over_axes(np.sum, rv ** 2.0, (3, 4, 5))

num = rv - arr_sum / nb_size
div = np.sqrt((arr_ssq - (arr_sum ** 2.0) / nb_size).clip(0, np.inf))

inr = num / div
inrm = inr.reshape(inr.shape[0] * inr.shape[1], -1)

print inrm.mean(1).mean()
print np.linalg.norm(inrm[1])
print np.linalg.norm(inrm[2])

# -- normalize filters
fbm = fb.reshape(inrm.shape[-1], -1)

fb_sum = fbm.sum(0)
fb_ssq = (fbm ** 2.).sum(0)

fb_num = fbm - fb_sum / nb_size

fb_div = np.sqrt((fb_ssq - (fb_sum ** 2.0) / nb_size).clip(0, np.inf))
fbm = fb_num / fb_div

print fbm.mean(1).mean()
print np.linalg.norm(fbm[:, 0])

out = np.dot(inrm, fbm)
