"""
Unit Test Suite for Filterbank Correlation Operation.

This suite tests basic functionalities of the `fbcorr` operation.
It should be used by all plugins/implementations of the operation.
"""

# TODO: complete the test suite.
# TODO: make nosetests verbose more explicit to help debug when tests fail

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#
# License: Proprietary


# -- Imports
from nose.tools import assert_raises
from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest
from pythor3.utils.testing import assert_allclose
import numpy as np

try:
    from scipy import lena
except:
    from scipy.misc import lena

import types

from pythor3.operation import fbcorr
from pythor3.array import Array
from pythor3.utils.hashing import get_pkl_sha1

# make fbcorr non-lazy for the purposes of these tests
import functools
fbcorr = functools.partial(fbcorr, lazy=False)

from pythor3.operation.lnorm_.tests.test_unit import info

# -- Raise exceptions on floating-point errors
np.seterr(all='raise')

# -- Global Variables
DTYPE = 'float32'
RTOL = 1e-3
ATOL = 1e-6


def get_suite(plugin, plugin_kwargs=None, tags=None):

    if plugin_kwargs is None:
        plugin_kwargs = {}

    if tags is None:
        tags = []

    print ">>> testing plugin '%s' with kwargs '%s'" % \
            (plugin, plugin_kwargs)

    def test_input_d_1():

        arr_in = np.zeros((2, 3, 1), dtype=DTYPE)
        arr_fb = np.zeros((4, 1, 1, 1), dtype=DTYPE)
        arr_out = np.zeros((2, 3, 4), dtype=DTYPE)

        arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        gt = np.array([[[  0.,   0.,   0.,   0.],
                        [  0.,   1.,   2.,   3.],
                        [  0.,   2.,   4.,   6.]],

                       [[  0.,   3.,   6.,   9.],
                        [  0.,   4.,   8.,  12.],
                        [  0.,   5.,  10.,  15.]]], dtype=DTYPE)

        fbcorr(arr_in, arr_fb, arr_out=arr_out,
               plugin=plugin, plugin_kwargs=plugin_kwargs)
        assert_allclose(arr_out[:], gt, rtol=RTOL, atol=ATOL)

        arr_out = fbcorr(arr_in, arr_fb,
                         plugin=plugin, plugin_kwargs=plugin_kwargs)
        assert_allclose(arr_out[:], gt, rtol=RTOL, atol=ATOL)


    def test_one_dot():

        arr_in = np.zeros((3, 3, 4), dtype=DTYPE)
        arr_fb = np.zeros((8, 3, 3, 4), dtype=DTYPE)
        arr_out = np.zeros((1, 1, 8), dtype=DTYPE)

        arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        gt = np.array([[[  14910.,   37590.,   60270.,   82950.,
                           105630.,  128310., 150990.,  173670.]]],
                      dtype=DTYPE)

        fbcorr(arr_in, arr_fb, arr_out=arr_out,
               plugin=plugin, plugin_kwargs=plugin_kwargs)
        assert_allclose(arr_out[:], gt, rtol=RTOL, atol=ATOL)

        arr_out = fbcorr(arr_in, arr_fb,
                         plugin=plugin, plugin_kwargs=plugin_kwargs)
        assert_allclose(arr_out[:], gt, rtol=RTOL, atol=ATOL)


    def test_no_clip_2d():

        arr_in = np.zeros((20, 30), dtype=DTYPE)
        arr_fb = np.zeros((8, 3, 3), dtype=DTYPE)
        arr_out = np.zeros((18, 28, 8), dtype=DTYPE)

        min_out = None
        max_out = None

        arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        gt = np.array([[  4182.,  12363.,  20544.,  28725.,
                         36906.,  45087.,  53268.,  61449.]
                       ,
                       [  5334.,  16107.,  26880.,  37653.,
                        48426.,  59199.,  69972.,   80745.]],
                      dtype=DTYPE)
        idx = [2, 3], [10, 12]

        fbcorr(arr_in, arr_fb, arr_out=arr_out,
               min_out=min_out, max_out=max_out,
               plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = fbcorr(arr_in, arr_fb,
                         min_out=min_out, max_out=max_out,
                         plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


    def test_no_clip_3d():

        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        arr_fb = np.zeros((8, 1, 1, 4), dtype=DTYPE)
        arr_out = np.zeros((20, 30, 8), dtype=DTYPE)

        min_out = None
        max_out = None

        arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        gt = np.array([[  1694.,   6198.,  10702.,  15206.,
                          19710.,  24214., 28718.,  33222.],
                       [  2462.,   9014.,  15566.,  22118.,
                          28670.,  35222., 41774.,  48326.]],
                      dtype=DTYPE)
        idx = [2, 3], [10, 12]

        fbcorr(arr_in, arr_fb, arr_out=arr_out,
               min_out=min_out, max_out=max_out,
               plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]

        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)
        arr_out = fbcorr(arr_in, arr_fb,
                         min_out=min_out, max_out=max_out,
                         plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


    def test_min_out():

        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        arr_fb = np.zeros((8, 1, 1, 4), dtype=DTYPE)
        arr_out = np.zeros((20, 30, 8), dtype=DTYPE)

        min_out = 24214
        max_out = None

        arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        gt = np.array([[ 24214.,  24214.,  24214.,  30326.,
                         39310.,  48294.,  57278.,  66262.],
                       [ 24214.,  24214.,  24214.,  24214.,
                         28670.,  35222.,  41774., 48326.]],
                      dtype=DTYPE)
        idx = [4, 3], [20, 12]

        fbcorr(arr_in, arr_fb, arr_out=arr_out,
               min_out=min_out, max_out=max_out,
               plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = fbcorr(arr_in, arr_fb,
                         min_out=min_out, max_out=max_out,
                         plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


    def test_max_out():

        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        arr_fb = np.zeros((8, 1, 1, 4), dtype=DTYPE)
        arr_out = np.zeros((20, 30, 8), dtype=DTYPE)

        arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        min_out = None
        max_out = 24214

        gt = np.array([[  1694.,   6198.,  10702.,  15206.,
                         19710.,  24214.,  24214.,  24214.],
                       [  6206.,  22742.,  24214.,  24214.,
                         24214.,  24214.,  24214.,  24214.]],
                      dtype=DTYPE)
        idx = [2, 8], [10, 18]

        fbcorr(arr_in, arr_fb, arr_out=arr_out,
               min_out=min_out, max_out=max_out,
               plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = fbcorr(arr_in, arr_fb,
                         min_out=min_out, max_out=max_out,
                         plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


    def test_lena_npy_array():

        arr_in = lena()[::32, ::32].astype(DTYPE)
        arr_fb = np.empty((4, 3, 3), dtype=DTYPE)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        idx = [[4, 2], [4, 2]]

        gt = np.array(
            [[  5138.,  16667.,  28196.,  39725.],
             [  4232.,  13160.,  22088.,  31016.]],
            dtype=DTYPE)

        arr_out = fbcorr(arr_in, arr_fb,
                         plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


    def test_lena_npy_array_float64():

        arr_in = lena()[::32, ::32].astype(np.float64)
        arr_fb = np.empty((4, 3, 3), dtype=np.float64)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        idx = [[4, 2], [4, 2]]

        gt = np.array(
            [[  5138.,  16667.,  28196.,  39725.],
             [  4232.,  13160.,  22088.,  31016.]],
            dtype=np.float64)

        try:
            arr_out = fbcorr(arr_in, arr_fb,
                             plugin=plugin, plugin_kwargs=plugin_kwargs)
            gv = arr_out[idx]
            assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)
        except NotImplementedError, err:
            print "Got: NotImplementedError '%s', skipping..." % err
            raise SkipTest


    def test_lena_npy_array_non_C_contiguous():

        arr_in = lena()[::32, ::32].astype(DTYPE)
        arr_in = np.asfortranarray(arr_in)
        arr_fb = np.empty((4, 3, 3), dtype=DTYPE)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)
        arr_fb = np.asfortranarray(arr_fb)

        idx = [[4, 2], [4, 2]]

        gt = np.array(
            [[  5138.,  16667.,  28196.,  39725.],
             [  4232.,  13160.,  22088.,  31016.]],
            dtype=DTYPE)

        try:
            arr_out = fbcorr(arr_in, arr_fb,
                             plugin=plugin, plugin_kwargs=plugin_kwargs)
            gv = arr_out[idx]
            assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)
        except NotImplementedError, err:
            print "Got: NotImplementedError '%s', skipping..." % err
            raise SkipTest

    def test_lena_pt3_array():

        lena32 = lena()[::32, ::32].astype(DTYPE)/255.
        arr_in = Array(lena32.shape, dtype=DTYPE)
        arr_in[:] = lena32

        arr_fb = Array((4, 3, 3), dtype=DTYPE)
        arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

        idx = [[4, 2], [4, 2]]

        gt = np.array(
            [[  20.14902115,   65.36077881,  110.57255554,  155.78433228],
             [  16.59607887,   51.60784531,   86.61960602,  121.63137817]],
            dtype=DTYPE)

        arr_out = fbcorr(arr_in, arr_fb,
                         plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    suite = {}
    if plugin_kwargs:
        plugin_kwarg_hash = '__kwargs_hash_' + get_pkl_sha1(plugin_kwargs)
    else:
        plugin_kwarg_hash = ""

    for key, value in locals().iteritems():
        if isinstance(value, types.FunctionType) and key.startswith('test_'):
            func = value
            func.__name__ += '__plugin_%s' % plugin +  plugin_kwarg_hash
            suite[func.__name__] = attr(*tags)(info(plugin, plugin_kwargs, func))

    return suite


globals().update(get_suite('default'))


def test_error_plugin():
    for name, test in get_suite('error').iteritems():
        yield assert_raises, KeyError, test
