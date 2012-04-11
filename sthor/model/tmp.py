#!/usr/bin/env python

import numpy as np
import scipy as sp
import time
from pprint import pprint

#from dldata import hongmajaj  # Hong & Majaj Neuronal Datasets module
from pythor3.model import SequentialLayeredModel
import genson

with open('plos09-l4-400x400.gson') as fin:
    model_desc = genson.loads(fin.read()).next()

pprint(model_desc)
raise

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RSEED = 42
IMAGE_LIMIT = 100#None#300

# -- random number generator
rng = np.random.RandomState(RSEED)

# -- dataset object
ds = hongmajaj.V4ChaboTitoVar03wh(verify_sha1=False)
categories = hongmajaj.CATEGORIES
time_window = (70, 170)

# -- imgdl
# don't use zz_blank images
imgdl = np.array([
    imgd
    for imgd in ds.meta
    if 'blank' not in imgd['annotations']['labels']
])


# -- shuffle data
ridx = rng.permutation(len(imgdl))
print ridx
imgdl = imgdl[ridx]
imgdl = imgdl[:IMAGE_LIMIT]
n_samples = len(imgdl)

model = None

def _imgd_to_img_arr(imgd):
    fname = imgd['filename']
    arr = sp.misc.imread(fname).mean(2)
    arr = sp.misc.imresize(arr, (200, 200))
    arr -= arr.mean()
    arr /= arr.std()
    return arr

features_cache = {}
neurons_cache = {}

def _imgdl_to_features(model_desc, imgdl):
    global model

    log.info("Processing features: %d images to process..." % len(imgdl))

    if model is None:
        arr_in = _imgd_to_img_arr(imgdl[0])
        model = SequentialLayeredModel(
            arr_in.shape, model_desc,
            plugin='passthrough',
            plugin_kwargs={
                'plugin_mapping':{
                    'all':{
                        'plugin':'cthor',
                        #'plugin_kwargs': {'variant': 'sse:tbb'},
                        'plugin_kwargs': {'variant': 'sse'},
                    }
                }
            })

    features = []
    n_samples = len(imgdl)
    start = time.time()
    for img_i, imgd in enumerate(imgdl):
        key = imgd['filename']
        if key not in features_cache:
            arr_in = _imgd_to_img_arr(imgd)
            arr_out = model.process(arr_in)
            features_cache[key] = arr_out
        else:
            arr_out = features_cache[key]
        features.append(arr_out)
        n_done = img_i + 1
        elapsed = time.time() - start
        if n_done % 10 == 0:
        #if elapsed % 2 == 0:
            status = ("Progress: %d/%d images [%.1f%% @ %.1f fps]"
                      % (n_done, n_samples, 100. * n_done / n_samples, n_done / elapsed))
            #status += chr(8) * (len(status) + 1)
            log.info(status)

    features = np.array(features)
    return features


def _imgdl_to_neurons(ds, imgdl, time_window=(70, 170)):
    """Returns neurons' activity"""
    log.info("Processing neurons: %d images to process..." % len(imgdl))
    xl = [ds.get_neurons(imgd, time_window) for imgd in imgdl]
    neurons = np.array(xl).mean(1)  # average all reps
    return neurons

# -- Estimators
X = _imgdl_to_features(model_desc, imgdl)
log.info(X.shape)
X.shape = X.shape[0], -1
log.info(X.shape)

Y = _imgdl_to_neurons(ds, imgdl, time_window=time_window)
log.info(Y.shape)
Y.shape = Y.shape[0], -1
log.info(Y.shape)

from idc_ps import idc_ps
ps = idc_ps(X, Y)
