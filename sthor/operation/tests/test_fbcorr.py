"""Test Suite for Filterbank Correlation Operation."""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

from nose.tools import raises
from numpy.testing import assert_allclose

import numpy as np

from sthor.operation import fbcorr3

# -- Raise exceptions on floating-point errors
np.seterr(all='raise')

# -- Global Variables
DTYPE = np.float32
RTOL = 1e-3
ATOL = 1e-6

# -- general formula to compute an element from
#    the tensor fbcorr(arr_in, arr_fb)

def get_fbcorr_element(arr_in, arr_fb, i, j, k):

    inh, inw, ind = arr_in.shape
    fh, fw = arr_fb.shape[:2]

    item = 0.
    for p in xrange(i, i + fh):
        for q in xrange(j, j + fw):
            for l in xrange(0, ind):
                item += arr_in[p, q, l] * arr_fb[p - i,
                                                 q - j,
                                                 l, k]
    return item

# -----------------------------------------------------------------------------
# -- Unit Tests
# -----------------------------------------------------------------------------

def test_constant_arr_in_and_fb():

    a = 1.234
    b = 2.345

    inh, inw, ind = 4, 5, 6
    fbh, fbw, fbd, fn = 2, 2, 6, 4

    arr_in = a * np.ones(inh*inw*ind).reshape(
                        (inh, inw, ind)).astype(DTYPE)
    arr_fb = b * np.ones(fbh*fbw*fbd*fn).reshape(
                        (fbh, fbw, fbd, fn)).astype(DTYPE)

    outh, outw, outd = (inh - fbh + 1), (inw - fbw + 1), fn

    constant = a * b * fbh * fbw * fbd

    ref = constant * np.ones(outh*outw*outd).reshape(
                            (outh, outw, outd)).astype(DTYPE)
    res = fbcorr3(arr_in, arr_fb)
    assert_allclose(res, ref, rtol=RTOL, atol=ATOL)


def test_constant_arr_in_arange_fb():

    a = 0.987

    inh, inw, ind = 6, 7, 4
    fbh, fbw, fbd, fn = 2, 3, 4, 4

    arr_in = a * np.ones(inh*inw*ind).reshape(
                        (inh, inw, ind)).astype(DTYPE)
    arr_fb = np.arange(fbh*fbw*fbd*fn).reshape(
                      (fbh, fbw, fbd, fn)).astype(DTYPE)

    outh, outw, outd = (inh - fbh + 1), (inw - fbw + 1), fn

    alpha = a * fbh * fbw * ind
    beta = 0.5 * (fn * (ind - 1)
                + fn * ind * (fbw - 1)
                + fn * ind * fbw * (fbh - 1))

    ref = np.zeros((outh, outw, outd), dtype=DTYPE)
    for k in xrange(outd):
        ref[:, :, k] = alpha * (k + beta)

    res = fbcorr3(arr_in, arr_fb)
    assert_allclose(res, ref, rtol=RTOL, atol=ATOL)


def test_arange_arr_in_constant_fb():

    b = 2.345

    inh, inw, ind = 8, 7, 3
    fbh, fbw, fbd, fn = 1, 5, 3, 8

    arr_in = np.arange(inh*inw*ind).reshape(
                      (inh, inw, ind)).astype(DTYPE)
    arr_fb = b * np.ones(fbh*fbw*fbd*fn).reshape(
                        (fbh, fbw, fbd, fn)).astype(DTYPE)

    outh, outw, outd = (inh - fbh + 1), (inw - fbw + 1), fn

    alpha = 0.5 * b * fbh * fbw * ind
    ref = np.zeros((outh, outw, outd), dtype=DTYPE)
    for i in xrange(outh):
        for j in xrange(outw):
            ref[i, j, :] = alpha * (ind - 1
                                  + ind * (2 * j + fbw - 1)
                                  + ind * inw * (2 * i + fbh - 1))

    res = fbcorr3(arr_in, arr_fb)
    assert_allclose(res, ref, rtol=RTOL, atol=ATOL)


def test_arange_arr_in_and_fb():

    inh, inw, ind = 8, 7, 3
    fbh, fbw, fbd, fn = 1, 5, 3, 8

    arr_in = np.arange(inh*inw*ind).reshape(
                      (inh, inw, ind)).astype(DTYPE)
    arr_fb = np.arange(fbh*fbw*fbd*fn).reshape(
                      (fbh, fbw, fbd, fn)).astype(DTYPE)

    outh, outw, outd = (inh - fbh + 1), (inw - fbw + 1), fn

    alpha = 0.5 * fbh * fbw * ind
    ref = np.zeros((outh, outw, outd), dtype=DTYPE)
    for i in xrange(outh):
        for j in xrange(outw):
            for k in xrange(outd):
                ref[i, j, k] = alpha * (
                1. / 3. * (3 * i + 2 * fbh - 1) *
                          (inw * ind ** 2 * fn * fbw * (fbh - 1)) \
              + ind * inw * (2 * i + fbh - 1) *
                          (0.5 * ind * fn * (fbw - 1) + \
                           0.5 * fn * (ind - 1) + k) \
              + 1. / 3. * (3 * j + 2 * fbw - 1) *
                          (ind ** 2 * fn * (fbw - 1)) \
              + ind * (2 * j + fbw - 1) *
                          (0.5 * ind * fn * fbw * (fbh - 1) + 0.5 * fn *
                              (ind - 1) + k) \
              + fn * (ind - 1) *
                          (0.5 * fbw * (fbh - 1) + 0.5 * ind * (fbw - 1) \
                        +  1. / 3. * (2 * ind - 1)) + (ind - 1) * k)

    res = fbcorr3(arr_in, arr_fb)
    assert_allclose(res, ref, rtol=RTOL, atol=ATOL)


def test_constant_arr_in_random_arr_fb():

    a = 4.827

    inh, inw, ind = 8, 7, 3
    fbh, fbw, fbd, fn = 3, 5, 3, 8

    arr_in = a * np.ones(inh*inw*ind).reshape(
                        (inh, inw, ind)).astype(DTYPE)
    arr_fb = np.random.randn(fbh*fbw*fbd*fn).reshape(
                            (fbh, fbw, fbd, fn)).astype(DTYPE)

    outh, outw, outd = (inh - fbh + 1), (inw - fbw + 1), fn

    ref = np.zeros((outh, outw, outd), dtype=DTYPE)
    for k in xrange(arr_fb.shape[-1]):
        constant = a * arr_fb[..., k].sum()
        ref[..., k] = constant

    res = fbcorr3(arr_in, arr_fb)
    assert_allclose(res, ref, rtol=RTOL, atol=ATOL)


def test_random_arr_in_and_arr_fb():

    inh, inw, ind = 8, 7, 3
    fbh, fbw, fbd, fn = 3, 5, 3, 8

    arr_in = np.random.randn(inh*inw*ind).reshape(
                            (inh, inw, ind)).astype(DTYPE)
    arr_fb = np.random.randn(fbh*fbw*fbd*fn).reshape(
                            (fbh, fbw, fbd, fn)).astype(DTYPE)

    outh, outw, outd = (inh - fbh + 1), (inw - fbw + 1), fn

    ref = np.zeros((outh, outw, outd), dtype=DTYPE)
    for i in xrange(outh):
        for j in xrange(outw):
            for k in xrange(outd):
                ref[i, j, k] = get_fbcorr_element(arr_in, arr_fb,
                                                  i, j, k)

    res = fbcorr3(arr_in, arr_fb)
    assert_allclose(res, ref, rtol=RTOL, atol=ATOL)


def test_input_d_1():

    arr_in = np.zeros((2, 3, 1), dtype=DTYPE)
    arr_fb = np.zeros((1, 1, 1, 4), dtype=DTYPE)

    arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
    arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

    gt = np.array([[[  0.,   0.,   0.,   0.],
                    [  0.,   1.,   2.,   3.],
                    [  0.,   2.,   4.,   6.]],

                   [[  0.,   3.,   6.,   9.],
                    [  0.,   4.,   8.,  12.],
                    [  0.,   5.,  10.,  15.]]], dtype=DTYPE)

    arr_out = fbcorr3(arr_in, arr_fb)
    assert_allclose(arr_out, gt, rtol=RTOL, atol=ATOL)

    arr_out = fbcorr3(arr_in, arr_fb)
    assert_allclose(arr_out, gt, rtol=RTOL, atol=ATOL)


def test_one_dot():

    arr_in = np.zeros((3, 3, 4), dtype=DTYPE)
    arr_fb = np.zeros((3, 3, 4, 8), dtype=DTYPE)

    arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
    arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

    gt = np.array([[[  119280., 119910., 120540., 121170.,
                       121800., 122430., 123060., 123690.]]],
                  dtype=DTYPE)

    arr_out = fbcorr3(arr_in, arr_fb)
    assert_allclose(arr_out, gt, rtol=RTOL, atol=ATOL)

    arr_out = fbcorr3(arr_in, arr_fb)
    assert_allclose(arr_out, gt, rtol=RTOL, atol=ATOL)


def test_arr_out_2d():

    arr_in = np.zeros((20, 30, 1), dtype=DTYPE)
    arr_fb = np.zeros((3, 3, 1, 8), dtype=DTYPE)
    arr_out = np.zeros((18, 28, 8), dtype=DTYPE)

    arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
    arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

    idx = [2, 3], [10, 12]

    gt = np.zeros((len(idx), arr_out.shape[-1]), dtype=DTYPE)
    for i in xrange(gt.shape[0]):
        for k in xrange(gt.shape[1]):
            gt[i, k] = get_fbcorr_element(arr_in, arr_fb,
                                          idx[0][i], idx[1][i], k)

    arr_out = fbcorr3(arr_in, arr_fb)
    gv = arr_out[idx]
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


def test_arr_out_3d():

    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    arr_fb = np.zeros((1, 1, 4, 8), dtype=DTYPE)
    arr_out = np.zeros((20, 30, 8), dtype=DTYPE)

    arr_in[:] = np.arange(np.prod(arr_in.shape)).reshape(arr_in.shape)
    arr_fb[:] = np.arange(np.prod(arr_fb.shape)).reshape(arr_fb.shape)

    idx = [2, 3], [10, 12]

    gt = np.zeros((len(idx), arr_out.shape[-1]), dtype=DTYPE)
    for i in xrange(gt.shape[0]):
        for k in xrange(gt.shape[1]):
            gt[i, k] = get_fbcorr_element(arr_in, arr_fb,
                                          idx[0][i], idx[1][i], k)

    arr_out = fbcorr3(arr_in, arr_fb)
    gv = arr_out[idx]
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)

# -----------------------------------------------------------------------------
# -- Test Errors
# -----------------------------------------------------------------------------

@raises(AssertionError)
def test_error_arr_in_ndim():
    arr_in = np.zeros((1, 4, 4, 4), dtype=DTYPE)
    arr_fb = np.zeros((1, 1, 4, 8), dtype=DTYPE)
    fbcorr3(arr_in, arr_fb)

@raises(AssertionError)
def test_error_arr_fb_ndim():
    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    arr_fb = np.zeros((3, 3, 4), dtype=DTYPE)
    fbcorr3(arr_in, arr_fb)

@raises(AssertionError)
def test_error_too_few_filters():
    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    arr_fb = np.zeros((3, 3, 4, 1), dtype=DTYPE)
    fbcorr3(arr_in, arr_fb)

@raises(AssertionError)
def test_error_arr_fb_h_too_big():
    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    arr_fb = np.zeros((21, 1, 4, 8), dtype=DTYPE)
    fbcorr3(arr_in, arr_fb)

@raises(AssertionError)
def test_error_arr_fb_w_too_big():
    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    arr_fb = np.zeros((1, 31, 4, 8), dtype=DTYPE)
    fbcorr3(arr_in, arr_fb)

@raises(AssertionError)
def test_error_arr_fb_d_too_small():
    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    arr_fb = np.zeros((1, 1, 1, 8), dtype=DTYPE)
    fbcorr3(arr_in, arr_fb)

@raises(AssertionError)
def test_error_arr_fb_d_too_large():
    arr_in = np.zeros((20, 30, 4), dtype=DTYPE)
    arr_fb = np.zeros((1, 1, 5, 8), dtype=DTYPE)
    fbcorr3(arr_in, arr_fb)
