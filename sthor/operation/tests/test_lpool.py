"""Test Suite for Local Pooling Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD


from nose.tools import assert_equals
from numpy.testing import assert_array_equal
from sthor.util.testing import assert_allclose_round
import numpy as np

from scipy.misc import lena

from sthor.operation import lpool3

# -- Global Variables
DTYPE = 'float32'
RTOL = 1e-3
ATOL = 1e-6


def test_default_lena():

    arr_in = lena().astype(np.float32) / 1.
    arr_in.shape = arr_in.shape[:2] + (1,)
    idx = [[4, 2], [4, 2]]
    neighborhood = 3, 3

    gt = np.array([[1442.], [1455.]],
                  dtype=DTYPE)

    arr_out = lpool3(arr_in, neighborhood)
    assert_equals(arr_out.ndim, arr_in.ndim)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


def test_default_lena_non_C_contiguous():

    arr_in = lena() / 1.
    arr_in.shape = arr_in.shape[:2] + (1,)
    neighborhood = 3, 3
    arr_in = np.asfortranarray(arr_in)

    idx = [[4, 2], [4, 2]]

    gt = np.array([[1442.], [1455.]],
                  dtype=DTYPE)

    arr_out = lpool3(arr_in, neighborhood)
    assert_equals(arr_out.ndim, arr_in.ndim)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


def test_default_input_d_1():

    arr_in = np.zeros((20, 30, 1), dtype=DTYPE)
    neighborhood = 5, 5
    arr_out = np.zeros((16, 26, 1), dtype=DTYPE)
    np.random.seed(42)
    arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

    idx = [[4, 8], [20, 12]]

    gt = np.array([[-3.59586287],
                   [9.16217232]],
                  dtype=DTYPE)

    lpool3(arr_in, neighborhood, arr_out=arr_out)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)

    arr_out = lpool3(arr_in, neighborhood)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


def test_neighborhood_5():

    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    neighborhood = 5, 5
    arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
    np.random.seed(42)
    arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

    idx = [[4, 8], [20, 12]]

    gt = np.array([[7.93315649, -0.24066222,
                      0.64046526, -3.51567388],
                   [7.14803362, -2.99622989,
                      9.5001564,  11.99116325]],
                  dtype=DTYPE)

    lpool3(arr_in, neighborhood, arr_out=arr_out)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)

    arr_out = lpool3(arr_in, neighborhood)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


def test_neighborhood_5_order_2():

    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    neighborhood = 5, 5
    arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
    np.random.seed(42)
    arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)
    order = 2

    idx = [[4, 8], [20, 12]]

    gt = np.array([[5.22081137, 4.9204731,
                    3.75381684, 4.892416],
                   [5.60951042, 4.28514147,
                    4.77592659, 5.77252817]],
                  dtype=DTYPE)

    lpool3(arr_in, neighborhood, order=order, arr_out=arr_out)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)

    arr_out = lpool3(arr_in, neighborhood, order=order)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


def test_neighborhood_order_2_stride_2():

    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    neighborhood = 5, 5
    arr_out = np.zeros((8, 13, 4), dtype=DTYPE)
    np.random.seed(42)
    arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)
    order = 2
    stride = 2

    idx = [[4, 3], [10, 6]]

    gt = np.array([[4.81159449,  5.68032312,
                     5.07941389,  6.04614496],
                   [4.87372255,  4.47074938,
                     4.07878876,  4.43441534]],
                  dtype=DTYPE)

    lpool3(arr_in, neighborhood, order=order, stride=stride, arr_out=arr_out)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)

    arr_out = lpool3(arr_in, neighborhood, order=order, stride=stride)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)


def test_strides_self():

    arr_in = lena() / 1.
    arr_in = arr_in[:12, :12]
    arr_in.shape = arr_in.shape[:2] + (1,)
    neighborhood = 3, 3

    for stride in xrange(1, 12):

        arr_out = lpool3(arr_in, neighborhood)
        gt = arr_out[::stride, ::stride]
        gv = lpool3(arr_in, neighborhood, stride=stride)
        assert_array_equal(gv, gt)


def test_test_lena_npy_array():

    arr_in = lena()[::32, ::32].astype(DTYPE)
    arr_in.shape = arr_in.shape[:2] + (1,)
    neighborhood = 3, 3

    idx = [[4, 2], [4, 2]]

    gt = np.array([[1280.99987793], [992.]],
                  dtype=DTYPE)

    arr_out = lpool3(arr_in, neighborhood)
    gv = arr_out[idx]
    assert_allclose_round(gv, gt, rtol=RTOL, atol=ATOL)
