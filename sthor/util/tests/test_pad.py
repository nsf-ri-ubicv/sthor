"""Test suite for ``pad.py``
"""

import numpy as np
from numpy.testing import assert_allclose
from sthor.util.pad import filter_pad2d
from pytest import raises

DTYPE = np.float32
RTOL = 1e-6
ATOL = 1e-6

def test_wrong_input_array_dimension():

    arr = np.random.randn(10, 5, 5, 2).astype(DTYPE)
    filter_shape = (5, 5)

    raises(AssertionError, filter_pad2d, arr, filter_shape)


def test_wrong_filter_shape():

    arr = np.random.randn(10, 5, 5).astype(DTYPE)
    filter_shape = (5, 2, 2)

    raises(AssertionError, filter_pad2d, arr, filter_shape)


def test_arr_3_3_1_filter_2_2():

    pad_val = 1.23

    arr = np.arange(3*3*1).reshape((3, 3, 1)).astype(DTYPE)
    filter_shape = (2, 2)

    ref = np.array([[[pad_val], [pad_val], [pad_val], [pad_val]],
                    [[pad_val], [0.], [1.], [2.]],
                    [[pad_val], [3.], [4.], [5.]],
                    [[pad_val], [6.], [7.], [8.]]])

    res = filter_pad2d(arr, filter_shape, constant=pad_val)
    assert_allclose(res, ref, rtol=RTOL, atol=ATOL)


def test_arr_4_4_2_filter_3_3():

    pad_val = 0.98

    arr = np.arange(2*2*2).reshape((2, 2, 2)).astype(DTYPE)
    filter_shape = (3, 3)

    ref = np.array([[[pad_val, pad_val],
                     [pad_val, pad_val],
                     [pad_val, pad_val],
                     [pad_val, pad_val]],
                    [[pad_val, pad_val],
                     [0., 1.],
                     [2., 3.],
                     [pad_val, pad_val]],
                    [[pad_val, pad_val],
                     [4., 5.],
                     [6., 7.],
                     [pad_val, pad_val]],
                    [[pad_val, pad_val],
                     [pad_val, pad_val],
                     [pad_val, pad_val],
                     [pad_val, pad_val]]]).astype(DTYPE)

    res = filter_pad2d(arr, filter_shape, constant=pad_val)
    assert_allclose(res, ref, rtol=RTOL, atol=ATOL)
