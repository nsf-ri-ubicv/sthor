from copy import deepcopy

try:
    import pyll
    choice = pyll.scope.choice
    pyll_available = True
except ImportError:
    pyll_available = False

if pyll_available:
    # -- lnorm
    _lnorm_shape = choice([
        (3, 3),
        (5, 5),
        (9, 9),
        (17, 17),
    ])
    _lnorm = {
        'kwargs': {
            'inker_shape': _lnorm_shape,
            'outker_shape' : _lnorm_shape,
            'remove_mean' : choice([False, True]),
            'stretch' : choice([1.]),
            'threshold' : choice([1.])
        }
    }


    # -- fbcorr
    _fbcorr_shape = choice([
        (3, 3),
        (5, 5),
        (9, 9),
        (17, 17),
    ])
    _fbcorr = {
        'initialize': {
            'filter_shape': _fbcorr_shape,
            'n_filters': choice([16, 32, 64]),
            'generate': ('random:uniform', {'rseed': 42}),
        },
        'kwargs': {
            'min_out': choice([0]),
            'max_out': choice([1, None]),
        }
    }
    _fbcorr1 = deepcopy(_fbcorr)
    _fbcorr1['initialize']['n_filters'] = choice([64])
    _fbcorr2 = deepcopy(_fbcorr)
    _fbcorr2['initialize']['n_filters'] = choice([64, 128])
    _fbcorr3 = deepcopy(_fbcorr)
    _fbcorr3['initialize']['n_filters'] = choice([64, 128, 256])
    _fbcorr4 = deepcopy(_fbcorr)
    _fbcorr4['initialize']['n_filters'] = choice([64, 128, 256, 512])


    # -- lpool
    _lpool_shape = choice([
        (2, 2),
        (3, 3),
    ])
    _lpool = {
        'kwargs': {
            'ker_shape': _lpool_shape,
            'order': choice([1., 2., 10.]),
            'stride': 2,
        }
    }


    # -- Layer 4
    isbi12_l4_description_pyll = [
        [
            ('lnorm', deepcopy(_lnorm)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr1)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr2)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr3)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr4)),
            ('lpool', deepcopy(_lpool)),
        ],
    ]
    # -- Layer 3
    isbi12_l3_description_pyll = [
        [
            ('lnorm', deepcopy(_lnorm)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr1)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr2)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr3)),
            ('lpool', deepcopy(_lpool)),
        ],
    ]
    # -- Layer 2
    isbi12_l2_description_pyll = [
        [
            ('lnorm', deepcopy(_lnorm)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr1)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr2)),
            ('lpool', deepcopy(_lpool)),
        ],
    ]
    # -- Layer 4 (no lnorm)
    isbi12_l4_no_lnorm_description_pyll = [
        [
            ('fbcorr', deepcopy(_fbcorr1)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr2)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr3)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr4)),
            ('lpool', deepcopy(_lpool)),
        ],
    ]
    # -- Layer 3 (no lnorm)
    isbi12_l3_no_lnorm_description_pyll = [
        [
            ('fbcorr', deepcopy(_fbcorr1)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr2)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr3)),
            ('lpool', deepcopy(_lpool)),
        ],
    ]
    # -- Layer 2 (no lnorm)
    isbi12_l2_no_lnorm_description_pyll = [
        [
            ('fbcorr', deepcopy(_fbcorr1)),
            ('lpool', deepcopy(_lpool)),
        ],
        [
            ('fbcorr', deepcopy(_fbcorr2)),
            ('lpool', deepcopy(_lpool)),
        ],
    ]


    def get_random_isbi12_l4_description(rng=None):
        desc = pyll.stochastic.sample(isbi12_l4_description_pyll, rng=rng)
        return desc
    def get_random_isbi12_l3_description(rng=None):
        desc = pyll.stochastic.sample(isbi12_l3_description_pyll, rng=rng)
        return desc
    def get_random_isbi12_l2_description(rng=None):
        desc = pyll.stochastic.sample(isbi12_l2_description_pyll, rng=rng)
        return desc
    def get_random_isbi12_l4_no_lnorm_description(rng=None):
        desc = pyll.stochastic.sample(isbi12_l4_no_lnorm_description_pyll, rng=rng)
        return desc
    def get_random_isbi12_l3_no_lnorm_description(rng=None):
        desc = pyll.stochastic.sample(isbi12_l3_no_lnorm_description_pyll, rng=rng)
        return desc
    def get_random_isbi12_l2_no_lnorm_description(rng=None):
        desc = pyll.stochastic.sample(isbi12_l2_no_lnorm_description_pyll, rng=rng)
        return desc
