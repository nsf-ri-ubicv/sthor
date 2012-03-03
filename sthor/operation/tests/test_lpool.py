"""Test Suite for Local Pooling Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD


# -- Imports
from nose.tools import assert_raises, assert_equals
from nose.plugins.skip import SkipTest
from nose.plugins.attrib import attr
from numpy.testing import assert_array_equal
from pythor3.utils.testing import assert_allclose_round
import numpy as np

try:
    from scipy.misc import lena
except:
    from scipy import lena

import types

from pythor3.operation import lpool
from pythor3.array import Array
from pythor3.utils.hashing import get_pkl_sha1

# -- Global Variables
DTYPE = 'float32'
RTOL = 1e-3
ATOL = 1e-6

# make lpool non-lazy for the purposes of these tests
import functools
lpool = functools.partial(lpool, lazy=False)

# XXX: 'info' needs to be factored out to a common module
from pythor3.operation.lnorm_.tests.test_unit import info


def get_suite(plugin, plugin_kwargs=None, tags=None):

    if plugin_kwargs is None:
        plugin_kwargs = {}

    if tags is None:
        tags = []

    def test_default_lena():

        arr_in = lena().astype(np.float32) / 1.
        idx = [[4, 2], [4, 2]]

        gt = np.array([1442.,  1455.],
                      dtype=DTYPE)

        arr_out = lpool(arr_in,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        assert_equals(arr_out.ndim, arr_in.ndim)
        gv = arr_out[idx]
        print gv.shape
        print gv

        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


    def test_default_lena_non_C_contiguous():

        arr_in = lena() / 1.
        arr_in = np.asfortranarray(arr_in)
        idx = [[4, 2], [4, 2]]

        gt = np.array([1442.,  1455.],
                      dtype=DTYPE)

        try:
            arr_out = lpool(arr_in,
                            plugin=plugin, plugin_kwargs=plugin_kwargs)
            assert_equals(arr_out.ndim, arr_in.ndim)
            gv = arr_out[idx]
            print gv.shape
            print gv

            assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)
        except NotImplementedError:
            raise SkipTest

    def test_default_input_d_1():

        arr_in = np.zeros((20, 30, 1), dtype=DTYPE)
        ker_shape = 5, 5
        arr_out = np.zeros((16, 26, 1), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 8], [20, 12]]

        gt = np.array([[-3.59586287],
                       [ 9.16217232]],
                      dtype=DTYPE)

        lpool(arr_in, arr_out=arr_out,
              ker_shape=ker_shape,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lpool(arr_in,
                        ker_shape=ker_shape,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


    def test_ker_shape_5():
        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        ker_shape = 5, 5
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

        idx = [[4, 8], [20, 12]]

        gt = np.array([[  7.93315649,  -0.24066222,
                          0.64046526,  -3.51567388],
                       [  7.14803362,  -2.99622989,
                          9.5001564 ,  11.99116325]],
                      dtype=DTYPE)

        lpool(arr_in, arr_out=arr_out,
              ker_shape=ker_shape,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lpool(arr_in,
                        ker_shape=ker_shape,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


    def test_ker_shape_5_order_2():
        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        ker_shape = 5, 5
        arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)
        order = 2

        idx = [[4, 8], [20, 12]]

        gt = np.array([[ 5.22081137,  4.9204731 ,
                         3.75381684,  4.892416  ],
                       [ 5.60951042,  4.28514147,
                         4.77592659,  5.77252817]],
                      dtype=DTYPE)

        lpool(arr_in, arr_out=arr_out,
              ker_shape=ker_shape, order=order,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lpool(arr_in,
                        ker_shape=ker_shape, order=order,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


    def test_ker_shape_5_order_2_stride_2():

        arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
        ker_shape = 5, 5
        arr_out = np.zeros((8, 13, 4), dtype=DTYPE)
        np.random.seed(42)
        arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)
        order = 2
        stride = 2

        idx = [[4, 3], [10, 6]]

        gt = np.array([[ 4.81159449,  5.68032312,
                         5.07941389,  6.04614496],
                       [ 4.87372255,  4.47074938,
                         4.07878876,  4.43441534]],
                      dtype=DTYPE)

        lpool(arr_in, arr_out=arr_out,
              ker_shape=ker_shape, order=order, stride=stride,
              plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)

        arr_out = lpool(arr_in,
                        ker_shape=ker_shape, order=order, stride=stride,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


    def test_strides_self():

        arr_in = lena() / 1.
        arr_in = arr_in[:12, :12]

        for stride in xrange(1, 12):

            try:
                arr_out = lpool(arr_in,
                                plugin=plugin, plugin_kwargs=plugin_kwargs)
                gt = arr_out[::stride, ::stride]
                gv = lpool(arr_in, stride=stride,
                           plugin=plugin, plugin_kwargs=plugin_kwargs)
                assert_array_equal(gv, gt)
            except NotImplementedError:
                raise SkipTest

    def test_strides_scipy_naive():

        arr_in = lena() / 1.
        arr_in = arr_in[:12, :12]

        for stride in xrange(1, 12):
            try:
                gt = lpool(arr_in, stride=stride,
                           plugin='scipy_naive')
                gv = lpool(arr_in, stride=stride,
                           plugin=plugin, plugin_kwargs=plugin_kwargs)
                assert_array_equal(gv, gt)
            except NotImplementedError:
                raise SkipTest

    def test_test_lena_npy_array():

        arr_in = lena()[::32, ::32].astype(DTYPE)

        idx = [[4, 2], [4, 2]]

        gt = np.array([1280.99987793, 992.],
                      dtype=DTYPE)

        arr_out = lpool(arr_in,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


    def test_test_lena_pt3_array():

        lena32 = lena()[::32, ::32].astype(DTYPE)/255.
        arr_in = Array(lena32.shape, dtype=DTYPE)
        arr_in[:] = lena32

        idx = [[4, 2], [4, 2]]

        gt = np.array([5.02353001,  3.89019608],
                      dtype=DTYPE)

        arr_out = lpool(arr_in,
                        plugin=plugin, plugin_kwargs=plugin_kwargs)
        gv = arr_out[idx]
        assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


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
