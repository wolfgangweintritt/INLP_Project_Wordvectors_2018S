# -*- coding: utf-8 -*-
"""
given a word and visualize near words
original source code is https://github.com/nishio/mycorpus/blob/master/vis.py
"""
import word2vec_boostpython as w2v
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager

class visWord2Vec:
    def __init__(self, filename = 'vectors.bin'):
        font = matplotlib.font_manager.FontProperties(fname='./ipag.ttc')
        FONT_SIZE = 20
        self.TEXT_KW = dict(fontsize=FONT_SIZE, fontweight='bold', fontproperties=font)

        print 'loading'
        self.data = w2v.load(filename)
        print 'loaded'

    def plot(self, query, nbest = 15):
        if ', ' not in query:
            words = [query] + w2v.search(self.data, query)[:nbest]
        else:
            words = query.split(', ')
            print ', '.join(words)
        mat = w2v.get_vectors(self.data)
        word_indexes = [w2v.get_word_index(self.data, w) for w in words]
        if word_indexes == [-1]:
            print 'not in vocabulary'
            return

        # do PCA
        X = mat[word_indexes]
        pca = PCA(n_components=2)
        pca.fit(X)
        print pca.explained_variance_ratio_
        X = pca.transform(X)
        xs = X[:, 0]
        ys = X[:, 1]

        # draw
        plt.figure(figsize=(12,8))
        plt.scatter(xs, ys, marker = 'o')
        for i, w in enumerate(words):
            plt.annotate(
                w.decode('utf-8', 'ignore'),
                xy = (xs[i], ys[i]), xytext = (3, 3),
                textcoords = 'offset points', ha = 'left', va = 'top',
                **self.TEXT_KW)

        plt.show()
