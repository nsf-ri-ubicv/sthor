import numpy as np
from sthor.model.slm_new import SequentialLayeredModel
import genson

from IPython import embed

# -- loading PLOS09-type SLM parameter ranges
with open('plos09.gson') as fin:
    gen = genson.loads(fin.read())

# -- extract SLM parameter range description
desc = gen.next()
genson.default_random_seed = 7
in_shape = 512, 512, 1
h, w, d = in_shape

# -- create random SLM model
slm = SequentialLayeredModel(in_shape, desc)

# -- fake (i.e. 'random') image normalized between 0 and 1
arr_in = np.random.randn(h, w, d).astype('f')
arr_in -= arr_in.min()
arr_in /= arr_in.max()

# -- compute feature array
feature_array = slm.process(arr_in)
print feature_array.shape
embed()
