from copy import deepcopy

import pyll
choice = pyll.scope.choice

# -- lnorm
_lnorm_shape = choice([
    (3, 3),
    (5, 5),
    (7, 7),
    (9, 9),
])
_lnorm = {
    'kwargs': {
        'inker_shape': _lnorm_shape,
        'outker_shape' : _lnorm_shape,
        'remove_mean' : choice([False, True]),
        'stretch' : choice([.1, 1., 10.]),
        'threshold' : choice([.1, 1., 10.])
    }
}


# -- fbcorr
_fbcorr_shape = choice([
    (3, 3),
    (5, 5),
    (7, 7),
    (9, 9),
])
_fbcorr = {
    'initialize': {
        'filter_shape': _fbcorr_shape,
        'n_filters': choice([16, 32, 64]),
        'generate': ('random:uniform', {'rseed': 42}),
    },
    'kwargs': {
        'min_out': choice([None, 0]),
        'max_out': choice([1, None]),
    }
}
_fbcorr1 = deepcopy(_fbcorr)
_fbcorr1['initialize']['n_filters'] = choice([16, 32, 64])
_fbcorr2 = deepcopy(_fbcorr)
_fbcorr2['initialize']['n_filters'] = choice([16, 32, 64, 128])
_fbcorr3 = deepcopy(_fbcorr)
_fbcorr3['initialize']['n_filters'] = choice([16, 32, 64, 128, 256])


# -- lpool
_lpool_shape = choice([
    (3, 3),
    (5, 5),
    (7, 7),
    (9, 9),
])
_lpool = {
    'kwargs': {
        'ker_shape': _lpool_shape,
        'order': choice([1., 2., 10.]),
        'stride': 2,
    }
}


# --
plos09_l3_description_pyll = [
    [
        ('lnorm', deepcopy(_lnorm)),
    ],
    [
        ('fbcorr', deepcopy(_fbcorr1)),
        ('lpool', deepcopy(_lpool)),
        ('lnorm', deepcopy(_lnorm)),
    ],
    [
        ('fbcorr', deepcopy(_fbcorr2)),
        ('lpool', deepcopy(_lpool)),
        ('lnorm', deepcopy(_lnorm)),
    ],
    [
        ('fbcorr', deepcopy(_fbcorr3)),
        ('lpool', deepcopy(_lpool)),
        ('lnorm', deepcopy(_lnorm)),
    ],
]
