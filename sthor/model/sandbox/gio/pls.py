import sys
from npprogressbar import RotatingMarker, Percentage, Bar, ETA, ProgressBar

import warnings
import numpy as np
import scipy as sp

warnings.simplefilter('always')
widgets = [RotatingMarker(), " Progress: ", Percentage(), " ",
           Bar(left='[', right=']'), ' ', ETA()]


def _pls(train_features, Y, n_components, inner_loop_max_iter=15,
         inner_loop_tol=1e-06, show_pbar=False):

    # PLS1 and PLS2 code taken from sklearn and made simpler and faster

    # -- copy X so that it can be deflated
    X = train_features.copy()

    n = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]

    # -- arrays for the projection vectors.
    all_x_weights = sp.empty((p, n_components), dtype='float32')
    all_x_loadings = sp.empty((p, n_components), dtype='float32')
    all_y_loadings = sp.empty((q, n_components), dtype='float32')

    if show_pbar:
        print "-" * 80
        print "Executing PLS2 decomposition..."
        pbar = ProgressBar(widgets=widgets, maxval=n_components, fd=sys.stdout)
        pbar.start()
        sys.stdout.flush()

    for k in xrange(n_components):
        u_old = 0
        ite = 1
        y_score = Y[:, [0]]
        while True:
            # regress each X column on y_score
            u = np.zeros(p, dtype=X.dtype)
            for i in xrange(n):
                u += X[i] * y_score[i][0]

            u.shape = u.size, -1

            # Normalize u
            u /= np.sqrt(np.dot(u.T, u))

            # Update x_score: the X latent scores
            x_score = np.zeros(n, dtype=X.dtype)
            for i in xrange(n):
                x_score[i] = np.dot(X[i], u)
            x_score.shape = x_score.size, -1

            if Y.shape[1] > 1:
                # Regress each X column on y_score
                v = np.dot(Y.T, x_score)
                # Normalize v
                v /= np.sqrt(np.dot(v.T, v))
                # Update y_score: the Y latent scores
                y_score = np.dot(Y, v)

                u_diff = u - u_old
                if np.dot(u_diff.T, u_diff) < inner_loop_tol:
                    break
                if ite == inner_loop_max_iter:
                    warnings.warn('Maximum number of iterations ' \
                                  'reached for component %d, with a ' \
                                  'square difference of %g' \
                                  % (k, np.dot(u_diff.T, u_diff)))
                    break
                u_old = u
                ite += 1
            else:
                break

        if np.dot(x_score.T, x_score) < np.finfo(np.double).eps:
            warnings.warn('X scores are null at iteration %s' % k)

        # Deflation (in place)
        # - regress X's on x_score
        x_loadings = np.zeros(p, dtype=X.dtype)
        for i in xrange(n):
            x_loadings += X[i] * x_score[i][0]
        x_loadings.shape = x_loadings.size, -1
        x_loadings /= np.dot(x_score.T, x_score)

        # - substract rank-one approximations to obtain remainder matrix
        # - Memory optmizated deflection on X
        for i in xrange(n):
            defAux = x_loadings * x_score[i]
            defAux = defAux.reshape((defAux.size, ))
            X[i] -= defAux #x_score * x_loadings[i]

        # - regress Y's on x_score, then substract rank-one approx.
        y_loadings = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)
        Y -= np.dot(x_score, y_loadings.T)

        # 3) Store weights and loadings
        all_x_weights[:, k] = u.ravel()  # W
        all_x_loadings[:, k] = x_loadings.ravel()  # P
        all_y_loadings[:, k] = y_loadings.ravel()  # Q

        if show_pbar:
            pbar.update(k + 1)
            sys.stdout.flush()

    if show_pbar:
        pbar.finish()
        print "-" * 80

    return all_x_weights, all_x_loadings, all_y_loadings


def pls(train_features, train_labels, n_components, class_specific=True):

    n_train, n_features = train_features.shape
    categories = np.unique(train_labels)
    n_categories = len(categories)

    if class_specific:
        # -- arrays for the projection vectors.
        all_x_weights = sp.empty((n_categories, n_features, n_components),
                                  dtype='float32')
        all_x_loadings = sp.empty((n_categories, n_features, n_components),
                                   dtype='float32')
        all_y_loadings = sp.empty((n_categories, n_components),
                                  dtype='float32')

        print "-" * 80
        print "Executing %d PS-PLS1 decompositions..." % n_categories
        pbar = ProgressBar(widgets=widgets, maxval=n_categories, fd=sys.stdout)
        pbar.start()
        sys.stdout.flush()
        for icat, cat in enumerate(categories):

            #Labeling according to cat
            Y = np.zeros((train_labels.size, 1))
            Y[train_labels != cat, 0] = -1
            Y[train_labels == cat, 0] = 1
            Y = Y.astype(np.double)

            all_x_weights[icat], all_x_loadings[icat], all_y_loadings[icat] = \
            _pls(train_features, Y, n_components)

            pbar.update(icat + 1)
            sys.stdout.flush()
        pbar.finish()
        print "-" * 80
    else:
        Y = np.zeros((train_labels.size, n_categories))

        #Labeling according to cat
        for icat, cat in enumerate(categories):
            Y[train_labels != cat, icat] = 0
            Y[train_labels == cat, icat] = 1
            Y = Y.astype(np.double)

        all_x_weights, all_x_loadings, all_y_loadings = \
        _pls(train_features, Y, n_components, show_pbar=True)

    return all_x_weights, all_x_loadings, all_y_loadings
