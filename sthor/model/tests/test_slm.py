#!/usr/bin/env python

"""
Test suite for ```slm_dev```
"""

import numpy as np
from numpy.testing import assert_allclose
from sthor.model.slm import SequentialLayeredModel
RTOL = 1e-5
ATOL = 1e-6

L3_first_desc = [
    [('lnorm',
      {'kwargs': {'inker_shape': [9, 9],
                  'outker_shape': [9, 9],
                  'remove_mean': False,
                  'stretch': 10,
                  'threshold': 1}})],
    [('fbcorr',
      {'initialize': {'filter_shape': (3, 3),
                      'generate': ('random:uniform', {'rseed': 42}),
                      'n_filters': 64},
       'kwargs': {'max_out': None, 'min_out': 0}}),
     ('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 1, 'stride': 2}}),
     ('lnorm',
      {'kwargs': {'inker_shape': [5, 5],
                  'outker_shape': [5, 5],
                  'remove_mean': False,
                  'stretch': 0.1,
                  'threshold': 1}})],
    [('fbcorr',
      {'initialize': {'filter_shape': (5, 5),
                      'generate': ('random:uniform', {'rseed': 42}),
                      'n_filters': 128},
       'kwargs': {'max_out': None, 'min_out': 0}}),
     ('lpool', {'kwargs': {'ker_shape': [5, 5], 'order': 1, 'stride': 2}}),
     ('lnorm',
      {'kwargs': {'inker_shape': [7, 7],
                  'outker_shape': [7, 7],
                  'remove_mean': False,
                  'stretch': 1,
                  'threshold': 1}})],
    [('fbcorr',
      {'initialize': {'filter_shape': (5, 5),
                      'generate': ('random:uniform', {'rseed': 42}),
                      'n_filters': 256},
       'kwargs': {'max_out': None, 'min_out': 0}}),
     ('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 10, 'stride': 2}}),
     ('lnorm',
      {'kwargs': {'inker_shape': [3, 3],
                  'outker_shape': [3, 3],
                  'remove_mean': False,
                  'stretch': 10,
                  'threshold': 1}})]
]

def test_no_description():

    in_shape = 256, 256
    desc = []
    slm = SequentialLayeredModel(in_shape, desc)

    assert slm.n_layers == 0
    assert slm.ops_nbh_nbw_stride == []
    assert slm.receptive_field_shape == (1, 1)


def test_L3_first_desc():

    in_shape = 256, 256
    desc = L3_first_desc
    slm = SequentialLayeredModel(in_shape, desc)

    assert slm.n_layers == 4
    assert slm.ops_nbh_nbw_stride == [('lnorm', 9, 9, 1),
                                      ('fbcorr', 3, 3, 1),
                                      ('lpool', 7, 7, 2),
                                      ('lnorm', 5, 5, 1),
                                      ('fbcorr', 5, 5, 1),
                                      ('lpool', 5, 5, 2),
                                      ('lnorm', 7, 7, 1),
                                      ('fbcorr', 5, 5, 1),
                                      ('lpool', 7, 7, 2),
                                      ('lnorm', 3, 3, 1)]
    assert slm.receptive_field_shape == (121, 121)


def test_null_image_same_size_as_receptive_field():

    in_shape = 121, 121
    desc = L3_first_desc
    slm = SequentialLayeredModel(in_shape, desc)

    img = np.zeros(in_shape).astype('f')

    features = slm.process(img)

    assert features.shape == (1, 1, 256)
    assert features.sum() == 0.


def test_zero_input_image_no_pad_no_interleave():

    in_shape = 200, 200
    desc = L3_first_desc
    slm = SequentialLayeredModel(in_shape, desc)

    img = np.zeros(in_shape).astype('f')

    features = slm.process(img, pad_apron=False, interleave_stride=False)

    assert features.shape == (10, 10, 256)
    assert features.sum() == 0.


def test_zero_input_image_with_pad_no_interleave():

    in_shape = 200, 200
    desc = L3_first_desc
    slm = SequentialLayeredModel(in_shape, desc)

    img = np.zeros(in_shape).astype('f')

    features = slm.process(img, pad_apron=True, interleave_stride=False)

    assert features.shape == (25, 25, 256)
    assert features.sum() == 0.


def test_zero_input_image_no_pad_with_interleave():

    in_shape = 200, 200
    desc = L3_first_desc
    slm = SequentialLayeredModel(in_shape, desc)

    img = np.zeros(in_shape).astype('f')

    features = slm.process(img, pad_apron=False, interleave_stride=True)

    assert features.shape == (80, 80, 256)
    assert features.sum() == 0.


def test_zero_input_image_with_pad_with_interleave():

    in_shape = 200, 200
    desc = L3_first_desc
    slm = SequentialLayeredModel(in_shape, desc)

    img = np.zeros(in_shape).astype('f')

    features = slm.process(img, pad_apron=True, interleave_stride=True)

    assert features.shape == (200, 200, 256)
    assert features.sum() == 0.


def test_outout_with_interleave_and_stride_and_no_interleave():

    in_shape = 200, 200
    desc = L3_first_desc
    slm = SequentialLayeredModel(in_shape, desc)

    img = np.random.randn(200, 200).astype('f')

    full_features = slm.process(img, pad_apron=True, interleave_stride=True)
    features = slm.process(img, pad_apron=True, interleave_stride=False)

    assert_allclose(features, full_features[::8, ::8], rtol=RTOL, atol=ATOL)
