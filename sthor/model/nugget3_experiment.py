#!/usr/bin/env python

import numpy as np
import scipy as sp
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import pylab as pl
import time
from pprint import pprint

import hongmajaj  # Hong & Majaj Neuronal Datasets module
from pythor3.model import SequentialLayeredModel
#from params_tmp import model_desc
import genson

with open('plos09-l4-400x400.gson') as fin:
    model_desc = genson.loads(fin.read()).next()

pprint(model_desc)

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#from joblib import Memory
#mem = Memory(cachedir='_cache')
#import cPickle as pkl

RSEED = 42
#N_FOLDS = 2
IMAGE_LIMIT = 100#None#300
#NEURON_LIMIT = None

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

# -- random number generator
rng = np.random.RandomState(RSEED)

# -- dataset object
#ds = hongmajaj.V4ChaboTitoVar03wh(verify_sha1=False)
#ds = hongmajaj.V4ChaboTitoVar03wh(verify_sha1=False)
ds = hongmajaj.V4ChaboTitoVar03wh(verify_sha1=False)
categories = hongmajaj.CATEGORIES
time_window = (70, 170)

# -- imgdl
# don't use zz_blank images
#from make_hashable import make_hashable
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
    #arr = sp.misc.imread(fname).mean(2)#[::8, ::8].mean(2)
    arr = sp.misc.imread(fname).mean(2)##[::8, ::8].mean(2)
    arr = sp.misc.imresize(arr, (400, 400))
    arr -= arr.mean()
    arr /= arr.std()
    return arr

#model_desc = None
#imgdl = None

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
#clf = linear_model.RidgeCV()
#clf = linear_model.LassoCV()
#scaler = preprocessing.Scaler()

X = _imgdl_to_features(model_desc, imgdl)
log.info(X.shape)
X.shape = X.shape[0], -1
log.info(X.shape)

Y = _imgdl_to_neurons(ds, imgdl, time_window=time_window)
log.info(Y.shape)
Y.shape = Y.shape[0], -1
log.info(Y.shape)

#nridx = rng.permutation(Y.shape[1])
#Y = Y[:, nridx]
#X = Y[:, ::2]
#Y = Y[:, 1::2]

#print X
#scaler.fit(X)

#print scaler.mean_
#import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
#X = preprocessing.Scaler().fit_transform(X)
EPSILON = 1e-3

#import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')


#log.info(">>> Z-scoring features...")
#X -= X.mean(0)
#X_std = X.std(0)
#X_std[X_std == 0] = 1
#X_std[np.abs(X_std) < EPSILON] = 1
#X /= X_std

#log.info(">>> Z-scoring neurons...")
#Y -= Y.mean(0)
#Y_std = Y.std(0)
#Y_std[Y_std == 0] = 1
#Y_std[np.abs(Y_std) < EPSILON] = 1
#Y /= Y_std

#Y = preprocessing.Scaler().fit_transform(Y)
#import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

from idc_ps import idc_ps
ps = idc_ps(X, Y)

## -- Cross Validation
#cvkf = cross_validation.KFold(n_samples, N_FOLDS)

#y_gv_l = []
#y_gt_l = []
#for trn, tst in cvkf:
    ## -- imgdl splits
    #imgdl_trn = imgdl[trn]
    #imgdl_tst = imgdl[tst]

    ## -- training
    #log.info(">>> Training...")
    #X_trn = _imgdl_to_features(model_desc, imgdl_trn)
    #y_trn = _imgdl_to_neurons(ds, imgdl_trn, time_window=time_window)
    #from idc_ps import idc_ps
    #print idc_ps(X_trn, y_trn)
    #print idc_ps(y_trn, y_trn)
    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
    ##X_trn = y_trn

    #X_trn = scaler.fit_transform(X_trn)
    #clf.fit(X_trn, y_trn)

    ## -- testing
    #log.info(">>> Testing...")
    #X_tst = _imgdl_to_features(model_desc, imgdl_tst)
    #y_gt = _imgdl_to_neurons(ds, imgdl_tst, time_window=time_window)
    ##X_tst = y_gt

    #X_tst = scaler.transform(X_tst)
    #y_gv = clf.predict(X_tst)

    #y_gv_l.append(y_gv)
    #y_gt_l.append(y_gt)

#y_gv_a = np.array(y_gv_l)
#y_gt_a = np.array(y_gt_l)

#import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')


#rmse = np.sqrt(((y_gv_a - y_gt_a) ** 2.).mean())
#log.info("RMSE = %.3f" % rmse)

##cc = np.corrcoef(y_gv_a, y_gt_a)[0, 1]
##log.info("CC = %.3f" % cc)

#pl.scatter(y_gv_a.ravel(), y_gt_a.ravel())
#pl.show()
