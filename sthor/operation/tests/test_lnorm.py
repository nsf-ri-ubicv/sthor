"""
Unit Test Suite for Local Normalization Operation.

This suite tests basic functionalities of the `lnorm` operation.
It should be used by all plugins/implementations of the operation.
"""

# TODO: complete the test suite.
# TODO: make nosetests verbose more explicit to help debug when tests fail

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#
# License: Proprietary


# -- Imports
from nose.tools import assert_raises
from nose.plugins.skip import SkipTest
from nose.plugins.attrib import attr
from pythor3.utils.testing import assert_allclose
import numpy as np

try:
    from scipy.misc import lena
except:
    from scipy import lena

import types

from pythor3.operation import lnorm
from pythor3.array import Array
from pythor3.utils.hashing import get_pkl_sha1

# -- Raise exceptions on floating-point errors
np.seterr(all='raise')

# -- Global Variables
DTYPE = 'float32'
RTOL = 1e-3
ATOL = 1e-6

# make lnorm non-lazy for the purposes of these tests
import functools
lnorm = functools.partial(lnorm, lazy=False)

# TODO: move this to operation/common, rename and refactor as a real decorator
def info(plugin, plugin_kwargs, func):

    def wrapper():
        # TODO: more information
        # TODO: use logging !
        print
        print ">>> '%s'" % func.__name__
        print "> testing plugin '%s' with plugin_kwargs '%s'" % \
                (plugin, plugin_kwargs)
        func()

    wrapper.__name__ = func.__name__
    return wrapper


def get_suite(plugin, plugin_kwargs=None, tags=None):

    if plugin_kwargs is None:
        plugin_kwargs = {}

    if tags is None:
        tags = []

    def test_input_d_1_default_remove_mean_threshold_stretch():

        arr_in = np.zeros((20, 30, 1), dtype=DTYPE)
        inker_shape = 5, 5
        arr_out = np.zeros((16, 26, 1), dtype=DTYPE)
        np.random.seed(42)
        data = np.random.randn(np.prod(arr_in.shape))
        arr_in[:] = data.reshape(arr_in.shape)

        idx = [[4, 3], [20, 12]]
        gt = np.array([[0.20177312], [0.21249016]], dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=inker_shape,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=inker_shape,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    def test_input_d_4_div_euclidean_remove_mean_false_default_rest():

        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 3], [20, 12]]
        gt = np.array([[ 0.13273999, -0.09456467,
                        -0.01975331, -0.04648187],
                       [ 0.00148955, -0.00257985,
                         0.02118244, -0.01543736]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=inker_shape,
              div_method='euclidean', remove_mean=False,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=inker_shape,
                        div_method='euclidean', remove_mean=False,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    def test_div_euclidean_remove_mean_false_threshold_1_stretch_1e_2():
        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 3], [20, 12]]
        gt = np.array([[ 0.01255756, -0.00894607,
                        -0.00186872, -0.00439731],
                       [ 0.00013929, -0.00024125,
                         0.00198085, -0.0014436 ]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=inker_shape,
              div_method='euclidean', remove_mean=False, threshold=1, stretch=1e-2,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=inker_shape,
                        div_method='euclidean', remove_mean=False, threshold=1, stretch=1e-2,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    def test_div_euclidean_remove_mean_true():

        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 3], [20, 12]]

        gt = np.array([[  1.27813682e-01, -9.97862518e-02,
                         -2.48777084e-02, -5.16409911e-02],
                       [ -2.00690944e-02, -2.42322776e-02,
                          7.76741435e-05, -3.73861268e-02]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=inker_shape,
              div_method='euclidean', remove_mean=True,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=inker_shape,
                        div_method='euclidean', remove_mean=True,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    def test_div_std_remove_mean_false():

        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 8], [20, 12]]

        gt = np.array([[  1.32899761, -0.94678491,
                         -0.19777086, -0.46537822],
                       [  1.67757177,  0.42027149,
                         -0.70711917, -0.05593578]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=inker_shape,
              div_method='std', remove_mean=False,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=inker_shape,
                        div_method='std', remove_mean=False,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    def test_div_std_remove_mean_true():

        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 8], [20, 12]]
        gt = np.array([[  1.27801514, -0.99776751,
                         -0.2487534 , -0.51636076],
                       [  1.42037416,  0.16307378,
                         -0.9643169 , -0.31313351]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=inker_shape,
              div_method='std', remove_mean=True,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=inker_shape,
                        div_method='std', remove_mean=True,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    def test_outker_shape_0_div_mag_remove_mean_false():
        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        outker_shape = 0, 0
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 3], [20, 12]]

        gt = np.array([[ 0.24052431, -0.18180957,
                        -0.04978044, -0.0898783 ],
                       [ 0.00301287, -0.00500357,
                         0.04109935, -0.03260877]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=outker_shape,
              div_method='euclidean', remove_mean=False,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=outker_shape,
                        div_method='euclidean', remove_mean=False,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    def test_outker_shape_0_div_mag_remove_mean_true():
        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        outker_shape = 0, 0
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 3], [20, 12]]

        gt = np.array([[ 0.18866782, -0.17986178,
                        -0.05663793, -0.06177634],
                       [-0.00420652, -0.03951693,
                        -0.0673274 , -0.05859426]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=outker_shape,
              div_method='euclidean', remove_mean=True,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=outker_shape,
                        div_method='euclidean', remove_mean=True,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    def test_outker_shape_0_div_std_remove_mean_false():
        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        outker_shape = 0, 0
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 8], [20, 12]]

        gt = np.array([[ 1.26222396, -0.90901738,
                        -0.24902068, -0.45406818],
                       [ 1.54160333,  0.49371463,
                         -0.80440265, -0.05310058]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=outker_shape,
              div_method='std', remove_mean=False,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=outker_shape,
                        div_method='std', remove_mean=False,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


    def test_outker_shape_0_div_std_remove_mean_true():
        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        inker_shape = 5, 5
        outker_shape = 0, 0
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 8], [20, 12]]
        gt = np.array([[ 0.94326323, -0.89923584,
                        -0.28315943, -0.30885619],
                       [ 1.27807069,  0.63492846,
                         -1.23798132, -0.50979644]],
                      dtype=DTYPE)

        lnorm(arr_in, arr_out=arr_out,
              inker_shape=inker_shape, outker_shape=outker_shape,
              div_method='std', remove_mean=True,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lnorm(arr_in,
                        inker_shape=inker_shape, outker_shape=outker_shape,
                        div_method='std', remove_mean=True,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


    def test_lena_npy_array():

        arr_in = lena()[::32, ::32].astype(DTYPE)

        idx = [[4, 2], [4, 2]]

        gt = np.array([0.2178068, 0.30647671],
                      dtype=DTYPE)

        arr_out = lnorm(arr_in,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


    def test_lena_npy_array_non_C_contiguous():

        arr_in = lena()[::32, ::32].astype(DTYPE)
        arr_in = np.asfortranarray(arr_in)

        idx = [[4, 2], [4, 2]]

        gt = np.array([0.2178068, 0.30647671],
                      dtype=DTYPE)


        try:
            arr_out = lnorm(arr_in,
                            plugin=plugin, plugin_kwargs=plugin_kwargs)
            gv = arr_out[idx]
            assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)
        except NotImplementedError:
            raise SkipTest

    def test_lena_pt3_array():

        lena32 = lena()[::32, ::32].astype(DTYPE)/255.
        arr_in = Array(lena32.shape, dtype=DTYPE)
        arr_in[:] = lena32

        idx = [[4, 2], [4, 2]]

        gt = np.array([0.21779411,  0.30645376],
                      dtype=DTYPE)

        arr_out = lnorm(arr_in,
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
            func.__name__ += '__plugin_%s' % plugin + plugin_kwarg_hash
            func = attr(*tags)(info(plugin, plugin_kwargs, func))
            suite[func.__name__] = func

    return suite


globals().update(get_suite('default'))


def test_error_plugin():
    for name, test in get_suite('error').iteritems():
        yield assert_raises, KeyError, test
