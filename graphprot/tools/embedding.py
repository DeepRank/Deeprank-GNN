from sklearn import manifold, datasets

def manifold_embedding(pos,method='tsne'):
    n_components = 2
    n_neighbors = 100

    if method == 'tsne':
        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        Y = tsne.fit_transform(pos)
    elif method == 'spectral':
        se = manifold.SpectralEmbedding(n_components=n_components,n_neighbors=n_neighbors)
        Y = se.fit_transform(pos)
    elif method == 'mds':
        mds = manifold.MDS(n_components, max_iter=100, n_init=1)
        Y = mds.fit_transform(pos)
    return Y