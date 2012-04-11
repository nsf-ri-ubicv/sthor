import numpy as np
from numpy.lib.stride_tricks import as_strided

for dtype in ('float32', 'float16'):
    print 'trying with', dtype

    arr_in = np.random.randn(10, 10, 4).astype(dtype)

    window_shape = (5, 5, 1)
    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple(arr_shape - window_shape + 1) + tuple(window_shape)
    new_strides = arr_in.strides + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)
