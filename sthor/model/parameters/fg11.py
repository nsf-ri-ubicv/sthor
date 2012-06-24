fg11_ht_l3_1_description = [
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


zstone12_ht_l3_prime_description = [
    [[u'lnorm',
      {u'kwargs': {u'inker_shape': [9, 9],
                   u'outker_shape': [9, 9],
                   u'remove_mean': False,
                   u'stretch': 0.1,
                   u'threshold': 1}}]],
    [[u'fbcorr',
      {'initialize': {'filter_shape': (9, 9),
                      'generate': ('random:uniform', {'rseed': 42}),
                      'n_filters': 128},
       u'kwargs': {u'max_out': None, u'min_out': 0}}],
     [u'lpool', {u'kwargs': {u'ker_shape': [9, 9], u'order': 2, u'stride': 2}}],
     [u'lnorm',
      {u'kwargs': {u'inker_shape': [5, 5],
                   u'outker_shape': [5, 5],
                   u'remove_mean': False,
                   u'stretch': 0.1,
                   u'threshold': 10}}]],
    [[u'fbcorr',
      {'initialize': {'filter_shape': (3, 3),
                      'generate': ('random:uniform', {'rseed': 42}),
                      'n_filters': 256},
       u'kwargs': {u'max_out': None, u'min_out': 0}}],
     [u'lpool', {u'kwargs': {u'ker_shape': [5, 5], u'order': 1, u'stride': 2}}],
     [u'lnorm',
      {u'kwargs': {u'inker_shape': [3, 3],
                   u'outker_shape': [3, 3],
                   u'remove_mean': False,
                   u'stretch': 10,
                   u'threshold': 1}}]],
    [[u'fbcorr',
      {'initialize': {'filter_shape': (3, 3),
                      'generate': ('random:uniform', {'rseed': 42}),
                      'n_filters': 512},
       u'kwargs': {u'max_out': None, u'min_out': 0}}],
     [u'lpool', {u'kwargs': {u'ker_shape': [9, 9], u'order': 10, u'stride': 2}}],
     [u'lnorm',
      {u'kwargs': {u'inker_shape': [5, 5],
                   u'outker_shape': [5, 5],
                   u'remove_mean': True,
                   u'stretch': 0.1,
                   u'threshold': 0.1}}]]
]


import copy
#import numpy as np

from pyll.scope import (
    choice,
    uniform,
    quniform,
    loguniform,
    qloguniform,
    )


norm_shape_choice = choice([(3,3),(5,5),(7,7),(9,9)])

lnorm = {
    'kwargs':{'inker_shape' : norm_shape_choice,
              'outker_shape' : norm_shape_choice,
              'remove_mean' : choice([0,1]),
              'stretch' : uniform(0,10),
              'threshold' : choice([.1,1,10])
             }
}

lpool = {'kwargs': {'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1, 2, 10])
         }}

lpool_sub2 = {'kwargs': {'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1, 2, 10]),
          'stride': 2,
         }}

rescale = {'kwargs': {'stride' : 2}}

activ =  {'kwargs': {'min_out' : choice([None, 0]),
                     'max_out' : choice([1, None])}}

filter1 = dict(
         initialize=dict(
            filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
            n_filters=choice([16, 32, 64]),
            generate=(
                'random:uniform',
                {'rseed': choice(range(5))})),
         kwargs={})

filter2 = copy.deepcopy(filter1)
filter2['initialize']['n_filters'] = choice([16, 32, 64, 128])
filter2['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(5,10))})

filter3 = copy.deepcopy(filter1)
filter3['initialize']['n_filters'] = choice([16, 32, 64, 128, 256])
filter3['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(10,15))})


fg11_desc = [[('lnorm', lnorm)],
            [('fbcorr', filter1),
             ('lpool', lpool_sub2),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('lpool', lpool_sub2),
             ('lnorm', lnorm)],
            [('fbcorr', filter3),
             ('lpool', lpool_sub2),
             ('lnorm', lnorm)],
           ]

lfwtop = [[('lnorm',{'kwargs':{'inker_shape': (9, 9),
                     'outker_shape': (9, 9),
                     'stretch':10,
                     'threshold': 1}})], 
         [('fbcorr', {'initialize': {'filter_shape': (3, 3),
                                     'n_filters': 64,
                                     'generate': ('random:uniform', 
                                                  {'rseed': choice(range(5))})},
                      'kwargs':{'min_out': 0,
                                'max_out': None}}),
          ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                'order': 1,
                                'stride': 2}}),
          ('lnorm', {'kwargs': {'inker_shape': (5, 5),
                                'outker_shape': (5, 5),
                                'stretch': 0.1,
                                'threshold': 1}})],
         [('fbcorr', {'initialize': {'filter_shape': (5, 5),
                                     'n_filters': 128,
                                     'generate': ('random:uniform',
                                                  {'rseed': choice(range(5))})},
                      'kwargs': {'min_out': 0,
                                 'max_out': None}}),
          ('lpool', {'kwargs': {'ker_shape': (5, 5),
                                'order': 1,
                                'stride': 2}}),
          ('lnorm', {'kwargs': {'inker_shape': (7, 7),
                                'outker_shape': (7, 7),
                                'stretch': 1,
                                'threshold': 1}})],
         [('fbcorr', {'initialize': {'filter_shape': (5, 5),
                                     'n_filters': 256,
                                     'generate': ('random:uniform',
                                                 {'rseed': choice(range(5))})},
                      'kwargs': {'min_out': 0,
                                 'max_out': None}}),
           ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                 'order': 10,
                                 'stride': 2}}),
           ('lnorm', {'kwargs': {'inker_shape': (3, 3),
                                 'outker_shape': (3, 3),
                                 'stretch': 10,
                                 'threshold': 1}})]]

fg11_top = lfwtop

cvpr_top = [[('lnorm',
       {'kwargs': {'inker_shape': [5, 5],
         'outker_shape': [5, 5],
         'remove_mean': 0,
         'stretch': 0.1,
         'threshold': 1}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [5, 5],
         'generate': ['random:uniform', {'rseed': 12}],
         'n_filters': 64},
        'kwargs': {'max_out': None, 'min_out': 0}}),
      ('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 10, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': [5, 5],
         'outker_shape': [5, 5],
         'remove_mean': 1,
         'stretch': 0.1,
         'threshold': 10}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [7, 7],
         'generate': ['random:uniform', {'rseed': 24}],
         'n_filters': 64},
        'kwargs': {'max_out': None, 'min_out': 0}}),
      ('lpool', {'kwargs': {'ker_shape': [3, 3], 'order': 1, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': [3, 3],
         'outker_shape': [3, 3],
         'remove_mean': 1,
         'stretch': 1,
         'threshold': 0.1}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [3, 3],
         'generate': ['random:uniform', {'rseed': 32}],
         'n_filters': 256},
        'kwargs': {'max_out': None, 'min_out': None}}),
      ('lpool', {'kwargs': {'ker_shape': [3, 3], 'order': 2, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': [3, 3],
         'outker_shape': [3, 3],
         'remove_mean': 1,
         'stretch': 0.1,
         'threshold': 1}})]]


crop_choice = choice([[0, 0, 250, 250],
                      [25, 25, 175, 175],
                      [88, 63, 163, 188]])

l3_params = {'slm': [[('lnorm', lnorm)],
                     [('fbcorr', filter1),
                      ('lpool', lpool_sub2),
                     ('lnorm', lnorm)],
                     [('fbcorr', filter2),
                      ('lpool', lpool_sub2),
                      ('lnorm', lnorm)],
                     [('fbcorr', filter3),
                      ('lpool', lpool_sub2),
                      ('lnorm', lnorm)],
                    ],
             'preproc': {'global_normalize': 0,
                         'crop': crop_choice,
                         'size': [200, 200]}} 

l2_params = {'slm': [[('lnorm', lnorm)],
                     [('fbcorr', filter1),
                      ('lpool', lpool_sub2),
                     ('lnorm', lnorm)],
                     [('fbcorr', filter2),
                      ('lpool', lpool_sub2),
                      ('lnorm', lnorm)]
                    ],
             'preproc': {'global_normalize': 0,
                         'crop': crop_choice,
                         'size': [100, 100]}}

main_params = choice([l3_params, l2_params])

test_params = {'slm': [[('lnorm', lnorm)]],
                          'preproc': {'global_normalize': 0,
                                      'crop': crop_choice,
                                      'size': [20, 20]}}

