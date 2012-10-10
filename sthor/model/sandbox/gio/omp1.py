import numpy as np

BATCH_SIZE = 2000

def omp1(X, k, n_iterations, seed=63):

    n_train, n_features = X.shape

    rng = np.random.RandomState(seed)

    # initialize and normalize dictionary
    D = rng.randn(k, n_features)
    D = (D.T / np.sqrt((D ** 2).sum(axis=1) + 1e-20)).T

    for i in xrange(n_iterations):

        print 'Running GSVQ iteration %d...' % (i + 1)

        D_new = np.zeros(D.shape, dtype=D.dtype)

        # -- iterate over batches
        for b_init in xrange(0, n_train, BATCH_SIZE):

            b_end = min(b_init + BATCH_SIZE, n_train)
            b_size = b_end - b_init

            Z_b = np.dot(D, X[b_init:b_end].T)

            max_D = abs(Z_b).argmax(axis=0)

            Z_b_sparse = np.zeros(Z_b.shape, dtype=Z_b.dtype)

            for b_i in xrange(b_size):
                Z_b_sparse[max_D[b_i],b_i] = Z_b[max_D[b_i],b_i]

            D_new = D_new + np.dot(Z_b_sparse, X[b_init:b_end])

        # -- reinitialize empty atoms
        empty_atoms=(D_new ** 2).sum(axis=1) < 1e-3
        print 'empty_atoms', empty_atoms.sum()

        D_new[empty_atoms] = rng.randn(empty_atoms.sum(),n_features)

        # -- normalize dictionary
        D = (D_new.T / np.sqrt((D_new ** 2).sum(axis=1) + 1e-20)).T

    return D
