from os import path
import time
from glob import glob
import numpy as np
from scipy import misc

from sthor.model import slm
import parameters

from skimage.util.shape import view_as_blocks

# -- find images
pattern = '/home/npinto/datasets/hongmajaj/images/*/*.png'
fnames = glob(pattern)
print len(fnames), 'images found!'


# -- process
desc_ext_l = [
    (
        parameters.fg11_ht_l3_1_description,
        '.fg11_ht_l3_1.npy', False, False, None, None,
    ),
    (
        parameters.zstone12_ht_l3_prime_description,
        '.zstone12_ht_l3_prime.npy', False, False, None, None,
    ),
    (
        parameters.zstone12_ht_l3_prime_description,
        '.zstone12_ht_l3_prime'
        '_64X'
        '_pad_apron=True'
        '_interleave_stride=True'
        '_block=8x8'
        '_reduce=mean.npy',
        True, True, (8, 8), 'mean'
    ),
]
in_shape = (200, 200)

for desc, ext, pad_apron, interleave_stride, block, reduce in desc_ext_l:
    print ext, pad_apron, interleave_stride, block, reduce

    model = slm.SequentialLayeredModel(
        in_shape=in_shape, description=desc)

    start = time.time()
    cumtime = 0
    for i, fname in enumerate(fnames):
        if 'zz_blank' in fname:
            continue
        out_fname = fname + ext
        if path.exists(out_fname):
            continue
        arr = misc.imread(fname, flatten=True)
        arr = misc.imresize(arr, in_shape).astype('float32')
        arr -= arr.min()
        arr /= arr.max()
        farr = model.transform(
            arr,
            pad_apron=pad_apron, interleave_stride=interleave_stride
            )
        if reduce is not None:
            assert block is not None
            farr_b = view_as_blocks(farr, block + (1,))
            farr_br = farr_b.reshape(farr_b.shape[:3] + (-1,))
            if reduce == 'mean':
                farr_brm = farr_br.mean(-1)
            elif reduce == 'min':
                farr_brm = farr_br.min(-1)
            elif reduce == 'max':
                farr_brm = farr_br.max(-1)
            elif reduce == 'median':
                farr_brm = np.median(farr_br, axis=-1)
            else:
                raise ValueError("'%s' reduce not understood" % reduce)
            farr = farr_brm
        print farr.shape, out_fname
        np.save(out_fname, farr)
        end = time.time()
        fps = (i + 1.) / (end - start)
        eta = (len(fnames) - i) / fps
        print '%d/%d: fps=%.2f, eta=%ds' % ((i+1), len(fnames), fps, eta)
