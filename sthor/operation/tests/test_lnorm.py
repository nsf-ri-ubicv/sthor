"""Unit Test Suite for Local Normalization Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD


from numpy.testing import assert_allclose
import numpy as np
from scipy.misc import lena

from sthor.operation import lcdnorm3

# -- Raise exceptions on floating-point errors
np.seterr(all='raise')

# -- Global Variables
DTYPE = 'float32'
RTOL = 1e-3
ATOL = 1e-6


def test_input_d_1_default():

    arr_in = np.zeros((20, 30, 1), dtype=DTYPE)
    neighborhood = 5, 5
    arr_out = np.zeros((16, 26, 1), dtype=DTYPE)
    np.random.seed(42)
    arr_in[:] = np.random.randn(*arr_in.shape)

    idx = [[4, 3], [20, 12]]

    gt = np.array([[0.23064923],
                   [0.20238316]], dtype=DTYPE)

    lcdnorm3(arr_in, neighborhood, arr_out=arr_out)
    gv = arr_out[idx]
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    arr_out = lcdnorm3(arr_in, neighborhood)
    gv = arr_out[idx]
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


def test_input_d_4_default():

    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    neighborhood = 5, 5
    arr_out = np.zeros((16, 26, 4), dtype=DTYPE)
    np.random.seed(42)
    arr_in[:] = np.random.randn(np.prod(arr_in.shape)).reshape(arr_in.shape)

    idx = [[4, 3], [20, 12]]

    gt = np.array([[+1.27813682e-01, -9.97862518e-02,
                    -2.48777084e-02, -5.16409911e-02],
                   [-2.00690944e-02, -2.42322776e-02,
                     7.76741435e-05, -3.73861268e-02]],
                  dtype=DTYPE)

    lcdnorm3(arr_in, neighborhood, arr_out=arr_out)
    gv = arr_out[idx]
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

    arr_out = lcdnorm3(arr_in, neighborhood)
    gv = arr_out[idx]
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


def test_lena_npy_array():

    arr_in = lena()[::32, ::32].astype(DTYPE)
    arr_in.shape = arr_in.shape[:2] + (1,)
    neighborhood = 3, 3

    idx = [[4, 2], [4, 2]]

    gt = np.array([[-0.53213698], [-0.0816204]],
                  dtype=DTYPE)

    arr_out = lcdnorm3(arr_in, neighborhood)
    gv = arr_out[idx]
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


def test_lena_npy_array_non_C_contiguous():

    arr_in = lena()[::32, ::32].astype(DTYPE)
    arr_in = np.asfortranarray(arr_in)
    arr_in.shape = arr_in.shape[:2] + (1,)
    neighborhood = 3, 3

    idx = [[4, 2], [4, 2]]

    gt = np.array([[-0.53213698], [-0.0816204]],
                  dtype=DTYPE)

    arr_out = lcdnorm3(arr_in, neighborhood)
    gv = arr_out[idx]
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)
